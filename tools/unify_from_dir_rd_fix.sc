// Unified JSON per shard: NODES + AST + CFG + DFG (custom intra-proc RD)
// Works with core Joern DSL (no dataflow overlays).
import io.shiftleft.semanticcpg.language._
import io.shiftleft.codepropertygraph.generated.nodes
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets
import scala.collection.mutable

def env(n:String,d:String)=Option(System.getenv(n)).filter(_.nonEmpty).getOrElse(d)

val srcDir   = env("SRC_DIR",".")                           // shard folder with .c files
val outRoot  = Paths.get(env("UNIFIED_DIR","unified_jsons"))// parent folder for outputs
val shard    = Paths.get(srcDir).getFileName.toString
val outPath  = outRoot.resolve(shard + ".json")

println(s"[i] importCode => " + srcDir)
importCode(srcDir)   // builds CPG with AST/CFG that your build already supports
Files.createDirectories(outRoot)

// ---------- NODES ----------
val nodesJson = cpg.all.toJsonPretty

// ---------- AST (parent -> child) ----------
val astPairs: List[(Long,Long)] =
  (for { m <- cpg.method.l; p <- m.ast.l; c <- p.astChildren.l } yield (p.id,c.id)).distinct
val astJson: String =
  astPairs.map{case(a,b)=>s"""{"src":$a,"dst":$b,"label":"AST"}"""}.mkString("[",",","]")

// ---------- CFG (n -> n.cfgNext) ----------
val cfgPairs: List[(Long,Long)] =
  (for { m <- cpg.method.l; n <- m.cfgNode.l; s <- n.cfgNext.l } yield (n.id,s.id)).distinct
val cfgJson: String =
  cfgPairs.map{case(a,b)=>s"""{"src":$a,"dst":$b,"label":"CFG"}"""}.mkString("[",",","]")

// ---------- DFG via simple INTRA-PROC Reaching-Definitions ----------
def lhsVars(n: nodes.CfgNode): Set[String] = {
  // Treat assignment LHS and local declarations as defs
  val fromAssign = n.ast.isCall.nameExact("<operator>.assignment").argument(1).isIdentifier.name.l.toSet
  val fromDecl   = n.ast.isLocal.name.l.toSet
  fromAssign ++ fromDecl
}
def usedVars(n: nodes.CfgNode): Set[String] = {
  val ids = n.ast.isIdentifier.name.l.toSet
  ids -- lhsVars(n)   // exclude LHS-ids so we only keep true uses
}

val dfgBuf = mutable.ArrayBuffer[String]()
for (m <- cpg.method.l) {
  val nodesInM = m.cfgNode.l
  if (nodesInM.nonEmpty) {
    val idSet  = nodesInM.map(_.id).toSet
    val preds  = nodesInM.map(n => n.id -> n.cfgPrev.l.filter(p => idSet.contains(p.id)).map(_.id).toSet).toMap
    val defs   = nodesInM.map(n => n.id -> lhsVars(n)).toMap
    val uses   = nodesInM.map(n => n.id -> usedVars(n)).toMap

    // maps: var -> set(def-node IDs)
    var IN  = nodesInM.map(n => n.id -> Map.empty[String,Set[Long]]).toMap
    var OUT = nodesInM.map(n => n.id -> Map.empty[String,Set[Long]]).toMap

    def merge(a: Map[String,Set[Long]], b: Map[String,Set[Long]]) =
      (a.keySet ++ b.keySet).map(k => k -> (a.getOrElse(k,Set()) ++ b.getOrElse(k,Set()))).toMap
    def kill(m: Map[String,Set[Long]], killed:Set[String]) = m -- killed

    var changed = true
    while (changed) {
      changed = false
      nodesInM.foreach { n =>
        val inNew  = preds(n.id).foldLeft(Map.empty[String,Set[Long]]){ (acc,p) => merge(acc, OUT(p)) }
        val gen    = if (defs(n.id).nonEmpty) defs(n.id).map(v => v -> Set(n.id)).toMap else Map.empty[String,Set[Long]]
        val outNew = merge(gen, kill(inNew, defs(n.id)))
        if (inNew != IN(n.id) || outNew != OUT(n.id)) {
          IN  = IN.updated(n.id, inNew)
          OUT = OUT.updated(n.id, outNew)
          changed = true
        }
      }
    }

    // def -> use edges for variables used at each node
    nodesInM.foreach { n =>
      uses(n.id).foreach { v =>
        IN(n.id).getOrElse(v, Set()).foreach { srcId =>
          dfgBuf += s"""{"src":$srcId,"dst":${n.id},"label":"DFG"}"""
        }
      }
    }
  }
}
val dfgJson: String = dfgBuf.mkString("[",",","]")

// ---------- UNIFIED JSON ----------
val meta =
  s"""{"tool":"joern","source":${"\"" + srcDir.replace("\\","/") + "\""},"n_nodes":${cpg.all.size},"n_edges":{"AST":${astPairs.size},"CFG":${cfgPairs.size},"DFG":${dfgBuf.size}},"dfg_status":"rd_intra_proc"}"""
val unified =
  "{\"meta\":"+meta+",\"nodes\":"+nodesJson+",\"edges\":{\"AST\":"+astJson+",\"CFG\":"+cfgJson+",\"DFG\":"+dfgJson+"}}"

Files.write(outPath, unified.getBytes(StandardCharsets.UTF_8))
println(s"[✓] $shard.json  nodes=${cpg.all.size}  AST=${astPairs.size}  CFG=${cfgPairs.size}  DFG(RD)=${dfgBuf.size}")
delete(project.name)