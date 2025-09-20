// tools/unify_from_dir_min.sc  — unified JSON with NODES + AST + CFG (DFG=[])
// Works on recent Joern without dataflow overlays.

import io.shiftleft.semanticcpg.language._
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets

def env(n:String, d:String) = Option(System.getenv(n)).filter(_.nonEmpty).getOrElse(d)

val srcDir   = env("SRC_DIR", ".")                       // shard folder with .c files
val outRoot  = Paths.get(env("UNIFIED_DIR", "unified"))  // parent folder for outputs
val shard    = Paths.get(srcDir).getFileName.toString
val outPath  = outRoot.resolve(shard + ".json")

println(s"[i] importCode => $srcDir")
importCode(srcDir)     // builds AST/CFG, etc.

Files.createDirectories(outRoot)

// Nodes (raw Joern dump — includes id, label, code where available)
val nodesJson = cpg.all.toJsonPretty

// AST edges (parent -> child)
val astPairs = (for {
  m <- cpg.method.l
  p <- m.ast.l
  c <- p.astChildren.l
} yield (p.id, c.id)).distinct
val astJson = astPairs
  .map { case (a,b) => s"""{"src":$a,"dst":$b,"label":"AST"}""" }
  .mkString("[", ",", "]")

// CFG edges (node -> cfgNext)
val cfgPairs = (for {
  m <- cpg.method.l
  n <- m.cfgNode.l
  s <- n.cfgNext.l
} yield (n.id, s.id)).distinct
val cfgJson = cfgPairs
  .map { case (a,b) => s"""{"src":$a,"dst":$b,"label":"CFG"}""" }
  .mkString("[", ",", "]")

// Meta + unified JSON (DFG empty for now)
val meta = s"""{"tool":"joern","source":${"\"" + srcDir.replace("\\","/") + "\""},"n_nodes":${cpg.all.size},"n_edges":{"AST":${astPairs.size},"CFG":${cfgPairs.size},"DFG":0},"dfg_status":"not_available"}"""
val unified = "{\"meta\":"+meta+",\"nodes\":"+nodesJson+",\"edges\":{\"AST\":"+astJson+",\"CFG\":"+cfgJson+",\"DFG\":[]}}"

Files.write(outPath, unified.getBytes(StandardCharsets.UTF_8))
println(s"[✓] Wrote ${outPath.toString}  nodes=${cpg.all.size} AST=${astPairs.size} CFG=${cfgPairs.size}")
