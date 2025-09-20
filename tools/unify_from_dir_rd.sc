// AST + CFG + DFG (intra-proc RD) → <UNIFIED_DIR>\<shard>.json
import io.shiftleft.semanticcpg.language._
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets
import scala.collection.mutable

def env(n:String,d:String)=Option(System.getenv(n)).filter(_.nonEmpty).getOrElse(d)
val srcDir   = env("SRC_DIR",".")
val outRoot  = Paths.get(env("UNIFIED_DIR","unified_jsons"))
val shard    = Paths.get(srcDir).getFileName.toString
val outPath  = outRoot.resolve(shard + ".json")

println(s"[i] importCode => " + srcDir)
importCode(srcDir)                // build CPG with AST+CFG
Files.createDirectories(outRoot)

// ---------- NODES ----------
val nodesJson = cpg.all.toJsonPretty

// ---------- AST ----------
val astPairs = (for { m <- cpg.method.l; p <- m.ast.l; c <- p.astChildren.l } yield (p.id,c.id)).distinct
val astJson  = astPairs.map{case(a,b)=>s"""{"src":$a,"dst":$b,"label":"AST"}"""}.mkString("[",",","]")

// ---------- CFG ----------
val cfgPairs = (for { m <- cpg.method.l; n <- m.cfgNode.l; s <- n.cfgNext.l } yield (n.id,s.id)).distinct
val cfgJson  = cfgPairs.map{case(a,b)=>s"""{"src":$a,"dst":$b,"label":"CFG"}"""}.mkString("[",",","]")

// ---------- DFG via simple intra-proc Reaching-Definitions ----------
val dfgBuf = mutable.ArrayBuffer[String]()
for (m <- cpg.method.l) {
  val nodes = m.cfgNode.l
  if (nodes.nonEmpty) {
    val id2node = nodes.map(n => n.id -> n).toMap
    val preds   = nodes.map(n => n.id -> n.cfgPrev.id.l.toSet).toMap

    def lhsVars(nId: Long): Set[String] =
      id2node(nId).ast.isCall.nameExact("<operator>.assignment").argument(1).isIdentifier.name.l.toSet

    def usedVars(nId: Long): Set[String] = {
      val allIds = id2node(nId).ast.isIdentifier.name.l.toSet
      allIds -- lhsVars(nId)
    }

    val defs = nodes.map(n => n.id -> lhsVars(n.id)).toMap
    val uses = nodes.map(n => n.id -> usedVars(n.id)).toMap

    // IN/OUT: var -> set(defNodeIds)
    val IN  = mutable.Map[Long, Map[String, Set[Long]]](nodes.map(n => n.id -> Map.empty[String,Set[Long]]):_*)
    val OUT = mutable.Map[Long, Map[String, Set[Long]]](nodes.map(n => n.id -> Map.empty[String,Set[Long]]):_*)

    def merge(a: Map[String, Set[Long]], b: Map[String, Set[Long]]) =
      (a.keySet ++ b.keySet).map(k => k -> (a.getOrElse(k,Set()) ++ b.getOrElse(k,Set()))).toMap
    def kill(m: Map[String, Set[Long]], killed: Set[String]) = m -- killed

    var changed = true
    while (changed) {
      changed = false
      nodes.foreach { n =>
        val inNew  = preds(n.id).foldLeft(Map.empty[String,Set[Long]]){ (acc,p) => merge(acc, OUT(p)) }
        val gen    = if (defs(n.id).nonEmpty) defs(n.id).map(v => v -> Set(n.id)).toMap else Map.empty[String,Set[Long]]
        val outNew = merge(gen, kill(inNew, defs(n.id)))
        if (inNew != IN(n.id) || outNew != OUT(n.id)) { IN(n.id)=inNew; OUT(n.id)=outNew; changed=true }
      }
    }

    // def → use edges
    nodes.foreach { n =>
      uses(n.id).foreach { v =>
        IN(n.id).getOrElse(v, Set()).foreach { srcId =>
          dfgBuf += s"""{"src":$srcId,"dst":${n.id},"label":"DFG"}"""
        }
      }
    }
  }
}

val dfgJson = dfgBuf.mkString("[",",","]")

// ---------- UNIFIED ----------
val meta = s"""{"tool":"joern","source":${"\"" + srcDir.replace("\\","/") + "\""},"n_nodes":${cpg.all.size},"n_edges":{"AST":${astPairs.size},"CFG":${cfgPairs.size},"DFG":${dfgBuf.size}},"dfg_status":"rd_intra_proc"}"""
val unified = "{\"meta\":"+meta+",\"nodes\":"+nodesJson+",\"edges\":{\"AST\":"+astJson+",\"CFG\":"+cfgJson+",\"DFG\":"+dfgJson+"}}"
Files.write(outPath, unified.getBytes(StandardCharsets.UTF_8))
println(s"[✓] Wrote ${outPath.toString}  nodes=${cpg.all.size} AST=${astPairs.size} CFG=${cfgPairs.size} DFG(RD)=${dfgBuf.size}")
