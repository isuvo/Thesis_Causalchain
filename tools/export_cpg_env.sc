// tools/export_cpg_env.sc
// Per-method export: AST + CFG + DFG (DFG via _reachingDefOut if present; else empty).
// Usage (PowerShell):
//   $env:CPG_PATH="C:\...\cpg.bin"
//   $env:OUT_DIR ="C:\...\work\cpg\json"
//   $env:LOG_PATH="C:\...\work\cpg\json\export_xxx.log"
//   "C:\tools\joern\joern.bat" --script "tools\export_cpg_env.sc"

import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.codepropertygraph.cpgloading.CpgLoader
import io.shiftleft.semanticcpg.language.*               // .ast, .cfgNode, .l, etc.
import io.shiftleft.codepropertygraph.generated.nodes.*  // Method, StoredNode

import java.nio.file.{Files, Paths, StandardOpenOption}
import java.nio.charset.StandardCharsets
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import scala.util.Try

// ---------- ENV ----------
val cpgPath = sys.env.getOrElse("CPG_PATH", throw new RuntimeException("CPG_PATH not set"))
val outDir  = sys.env.getOrElse("OUT_DIR",  throw new RuntimeException("OUT_DIR not set"))
val logPath = sys.env.getOrElse("LOG_PATH", outDir + java.io.File.separator + "export.log")
Files.createDirectories(Paths.get(outDir))

def log(msg: String): Unit = {
  val ts = DateTimeFormatter.ofPattern("uuuu-MM-dd HH:mm:ss").format(LocalDateTime.now)
  Files.writeString(
    Paths.get(logPath),
    s"[$ts] $msg\n",
    StandardCharsets.UTF_8,
    StandardOpenOption.CREATE, StandardOpenOption.APPEND
  )
}

log(s"Loading CPG: $cpgPath")
val cpg: Cpg = CpgLoader.load(cpgPath)
log("CPG loaded.")

// ---------- helpers ----------
final case class NodeRow(
  id: Long,
  label: String,
  code: String,
  line: Option[Int],
  file: Option[String]
)

def optInt(o: Option[Int])    = o.map(i => ujson.Num(i)).getOrElse(ujson.Null)
def optStr(o: Option[String]) = o.map(ujson.Str(_)).getOrElse(ujson.Null)

// Windows-safe filename
def sanitizeFileBase(s: String): String =
  s.replaceAll("[\\\\/:*?\"<>|]", "_").replaceAll("\\s+", "_")

// Prefer method.name if available; fall back to pieces of fullName; always add id+line for uniqueness
def outBaseFor(m: Method): String = {
  val nm = Option(m.name).getOrElse {
    val fn = Option(m.fullName).getOrElse("method")
    // take trailing simple name-ish chunk
    fn.split("[\\s:]+").lastOption.getOrElse("method")
  }
  val line = m.lineNumber.getOrElse(0)
  sanitizeFileBase(s"${nm}_${m.id}_${line}")
}

// Nodes
def collectNodes(m: Method): List[NodeRow] = {
  val fileName = Option(m.filename)
  m.ast.l.map { n =>
    NodeRow(
      id    = n.id,
      label = n.label,
      code  = Option(n.code).getOrElse(""),
      line  = n.lineNumber,         // Option[Int]
      file  = fileName
    )
  }.toList
}

// AST edges: parent -> child
def collectAstEdges(m: Method): List[(Long, Long)] =
  m.ast.l.flatMap { parent => parent._astOut.l.map { child => (parent.id, child.id) } }.toList

// CFG edges: use only CFG-capable nodes within this method's AST
def collectCfgEdges(m: Method): List[(Long, Long)] = {
  val cfgNodes = m.ast.isCfgNode.l
  cfgNodes.flatMap { n => n._cfgOut.l.map { succ => (n.id, succ.id) } }.toList
}

// DFG edges: via generated accessor _reachingDefOut if present; else empty
def collectDfgEdges(m: Method): List[(Long, Long)] = {
  Try {
    val ids = m.ast.id.toSet
    m.ast.l.flatMap { src =>
      src._reachingDefOut.l
        .filter(dst => ids.contains(dst.id))
        .map(dst => (src.id, dst.id))
    }.toList
  }.getOrElse(Nil)
}

// ---------- export ----------
var exported = 0
val methods = cpg.method.l
log(s"Found ${methods.size} methods to export.")

methods.foreach { m =>
  try {
    val base  = outBaseFor(m)
    val nodes = collectNodes(m)
    val astE  = collectAstEdges(m)
    val cfgE  = collectCfgEdges(m)
    val dfgE  = collectDfgEdges(m) // may be empty

    val nodesJson = ujson.Arr.from(
      nodes.map(n => ujson.Obj(
        "id"    -> n.id,
        "label" -> n.label,
        "code"  -> n.code,
        "line"  -> optInt(n.line),
        "file"  -> optStr(n.file)
      ))
    )
    val edgesJson = ujson.Arr(
      ujson.Obj("kind" -> "AST", "pairs" -> ujson.Arr.from(astE.map{ case (s,t) => ujson.Arr(s,t) })),
      ujson.Obj("kind" -> "CFG", "pairs" -> ujson.Arr.from(cfgE.map{ case (s,t) => ujson.Arr(s,t) })),
      ujson.Obj("kind" -> "DFG", "pairs" -> ujson.Arr.from(dfgE.map{ case (s,t) => ujson.Arr(s,t) }))
    )

    val obj = ujson.Obj(
      "method"    -> m.fullName,
      "signature" -> m.signature,
      "filename"  -> m.filename,
      "line"      -> optInt(m.lineNumber),
      "nodes"     -> nodesJson,
      "edges"     -> edgesJson
    )

    Files.writeString(
      Paths.get(outDir, s"${base}.cpg.json"),
      obj.render(),
      StandardCharsets.UTF_8,
      StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING
    )

    exported += 1
    if (exported % 200 == 0) log(s"Exported $exported …")
  } catch {
    case e: Throwable => log(s"[ERROR] ${m.fullName}: ${e.getMessage}")
  }
}

log(s"Done. Exported $exported methods to $outDir")
