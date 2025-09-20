// tools/verify_cpg.sc
// Verifies a single CPG by *per-method* traversals only (compatible with your build).
// Usage:
//   $env:CPG_PATH="C:\...\cpg.bin"
//   "C:\tools\joern\joern.bat" --script "tools\verify_cpg.sc"

import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.codepropertygraph.cpgloading.CpgLoader
import io.shiftleft.semanticcpg.language.*               // .method, .ast, .isCfgNode, .l
import io.shiftleft.codepropertygraph.generated.nodes.*  // Method, StoredNode
import scala.util.Try

val cpgPath = sys.env.getOrElse("CPG_PATH", throw new RuntimeException("CPG_PATH not set"))
val cpg: Cpg = CpgLoader.load(cpgPath)

// Helpers that only use per-method accessors available in your build:
def cfgEdgesIn(m: Method): Long =
  m.ast.isCfgNode.l.map(n => n._cfgOut.l.size.toLong).foldLeft(0L)(_ + _)

def dfgEdgesIn(m: Method): Long =
  // Some builds may not expose dataflow; Try swallows that and yields 0.
  Try { m.ast.l.map(n => n._reachingDefOut.l.size.toLong).foldLeft(0L)(_ + _) }.getOrElse(0L)

def astNodesIn(m: Method): Long = m.ast.size.toLong

val methods = cpg.method.l
val methodCount = methods.size
val astNodeCount = methods.map(astNodesIn).foldLeft(0L)(_ + _)
val cfgEdgeCount = methods.map(cfgEdgesIn).foldLeft(0L)(_ + _)
val dfgEdgeCount = methods.map(dfgEdgesIn).foldLeft(0L)(_ + _)

val samples = methods.take(3).map { m =>
  ujson.Obj(
    "method" -> m.fullName,
    "file"   -> m.filename,
    "line"   -> m.lineNumber,
    "cfgE"   -> cfgEdgesIn(m),
    "dfgE"   -> dfgEdgesIn(m)
  )
}

val report = ujson.Obj(
  "cpgPath"   -> cpgPath,
  "methods"   -> methodCount,
  "astNodes"  -> astNodeCount,
  "cfgEdges"  -> cfgEdgeCount,
  "dfgEdges"  -> dfgEdgeCount,
  "sample"    -> ujson.Arr.from(samples)
)
println(report.render())
