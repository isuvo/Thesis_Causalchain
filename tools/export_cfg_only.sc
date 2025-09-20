// === tools/export_cfg_only.sc ===
// Minimal, works on recent Joern (FlatGraph), no ammonite, no DFG.
import io.shiftleft.semanticcpg.language._
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets

def envOr(name: String, default: String) =
  Option(System.getenv(name)).filter(_.nonEmpty).getOrElse(default)

val cpgPath = envOr("CPG_PATH", "cpg.bin")
val outDir  = Paths.get(envOr("OUT_DIR", "export"))

println(s"[i] Importing CPG: " + cpgPath)
importCpg(cpgPath)

Files.createDirectories(outDir)
def write(fname: String, s: String): Unit =
  Files.write(outDir.resolve(fname), s.getBytes(StandardCharsets.UTF_8))

// Dump nodes (handy for ID lookups)
write("nodes.json", cpg.all.toJsonPretty)

// Build CFG edges by traversing cfgNext per method
val cfgEdges = for {
  m <- cpg.method.l
  n <- m.cfgNode.l
  s <- n.cfgNext.l
} yield s"""{"src":${n.id},"dst":${s.id},"label":"CFG"}"""

write("cfg_edges.json", cfgEdges.mkString("[", ",", "]"))
println(s"[✓] CFG edges: ${cfgEdges.size} → " + outDir.resolve("cfg_edges.json"))
