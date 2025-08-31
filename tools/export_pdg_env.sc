// tools/export_pdg_env.sc  (CDG/CFG + heuristic DDG via def–use)
// ENV VARS (PowerShell: $env:VAR="..."):
//   One of:  CPG_PATH (path to *.cpg.bin)  OR  SRC_DIR (folder with .c files)
//   Always:  OUT_DIR  (output folder for JSONs)
//   Optional: SENSI_PATH, LOG_PATH

import io.shiftleft.semanticcpg.language._
import io.shiftleft.codepropertygraph.generated.nodes
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets
import scala.util.Using
import scala.util.Try

@main def exportMain(): Unit = {
  val srcOpt    = sys.env.get("SRC_DIR").map(_.trim).filter(_.nonEmpty)
  val cpgOpt    = sys.env.get("CPG_PATH").map(_.trim).filter(_.nonEmpty)
  val outDirStr = sys.env.getOrElse("OUT_DIR", "work/pdg_out")
  val sensiStr  = sys.env.getOrElse("SENSI_PATH", "")
  val logStr    = sys.env.getOrElse("LOG_PATH", "work/pdg_export.log")

  val outDir = Paths.get(outDirStr); if (!Files.exists(outDir)) Files.createDirectories(outDir)
  val logPath = Paths.get(logStr); if (logPath.getParent != null && !Files.exists(logPath.getParent)) Files.createDirectories(logPath.getParent)

  def logLine(s: String): Unit =
    Files.write(
      logPath,
      (s + System.lineSeparator()).getBytes(StandardCharsets.UTF_8),
      java.nio.file.StandardOpenOption.CREATE,
      java.nio.file.StandardOpenOption.APPEND
    )

  (cpgOpt, srcOpt) match {
    case (Some(cpgPath), _) => logLine(s"[info] loadCpg($cpgPath)"); loadCpg(cpgPath)
    case (None, Some(src))  => logLine(s"[info] importCode($src)");  importCode(src)
    case _ =>
      System.err.println("Set either CPG_PATH or SRC_DIR in environment"); sys.exit(2)
  }

  val sinks: Set[String] =
    if (sensiStr.nonEmpty && Files.exists(Paths.get(sensiStr)))
      Using(scala.io.Source.fromFile(sensiStr, "UTF-8")) { src =>
        src.getLines().map(_.trim).filter(s => s.nonEmpty && !s.startsWith("#")).toSet
      }.getOrElse(Set.empty)
    else Set.empty

  def esc(s: String): String =
    Option(s).getOrElse("").replace("\\","\\\\").replace("\"","\\\"")

  // Robust lineNumber handling
  def lineNo(n: nodes.AstNode): Option[Int] = {
    try {
      val lnAny: Any = n.lineNumber.asInstanceOf[Any]
      lnAny match {
        case null                  => None
        case i: Int                => Some(i)
        case ji: java.lang.Integer => Some(ji.intValue)
        case opt: Option[_]        => opt.asInstanceOf[Option[Int]]
        case other =>
          val m = other.getClass.getMethods.find(_.getName == "get")
          m.flatMap { mm =>
            Option(mm.invoke(other)) match {
              case Some(ji: java.lang.Integer) => Some(ji.intValue)
              case Some(i: Int)                => Some(i)
              case _                           => None
            }
          }
      }
    } catch { case _: Throwable => None }
  }

  // CDG/CFG via label traversals (version-tolerant)
  def edgesViaLabel(stmts: List[nodes.AstNode], id2idx: Map[Long, Int], label: String): List[(Int,Int)] =
    stmts.flatMap { n =>
      val u = id2idx(n.id)
      val dests = try n.out(label).l catch { case _: Throwable => List.empty }
      dests.flatMap(d => id2idx.get(d.id).map(v => (u, v)))
    }.distinct

  // Heuristic DDG: nearest-previous def of same identifier in the same method
  def buildHeuristicDDG(stmts: List[nodes.AstNode], id2idx: Map[Long, Int]): List[(Int,Int)] = {
    // Helpers to extract names used/defined per statement
    def namesUsed(n: nodes.AstNode): Set[String] =
      try n.ast.isIdentifier.name.l.toSet catch { case _: Throwable => Set.empty }

    def namesDefined(n: nodes.AstNode): Set[String] = {
      // defs from assignments: <operator>.assignment, lhs identifiers
      val lhsFromAssign =
        try n.ast.isCall.nameExact("<operator>.assignment").argument(1).isIdentifier.name.l.toSet
        catch { case _: Throwable => Set.empty[String] }

      // defs from declarations with init: local/parameter names appear as identifiers too; keep simple
      val locals =
        try n.ast.isLocal.name.l.toSet
        catch { case _: Throwable => Set.empty[String] }

      lhsFromAssign ++ locals
    }

    // scan in source order, remember last def index for each name
    var lastDef = Map.empty[String, Int]
    val ddgBuf = scala.collection.mutable.ListBuffer.empty[(Int,Int)]

    stmts.foreach { n =>
      val idx = id2idx(n.id)
      val defs = namesDefined(n)
      val uses = namesUsed(n)

      // connect each use to nearest previous def if present
      uses.foreach { nm =>
        lastDef.get(nm).foreach { dIdx =>
          if (dIdx != idx) ddgBuf += ((dIdx, idx))
        }
      }
      // update last-defs
      defs.foreach(nm => lastDef = lastDef.updated(nm, idx))
    }
    ddgBuf.result().distinct
  }

  var total = 0; var written = 0; var skipped = 0; var errs = 0

  cpg.method.l.zipWithIndex.foreach { case (m, mi) =>
    total += 1

    Try {
      val file = m.filename

      // Keep only nodes that have a line number; stable order by (line, order)
      val stmts = m.ast.l
        .filter(n => lineNo(n).nonEmpty)
        .sortBy(n => (lineNo(n).getOrElse(Int.MaxValue), n.order))

      if (stmts.nonEmpty) {
        val id2idx = stmts.zipWithIndex.map{ case (n,i) => (n.id,i) }.toMap
        val lineContents = stmts.map(_.code).map(esc)

        val sinkIdxs =
          if (sinks.nonEmpty) stmts.zipWithIndex.collect { case (c: nodes.Call, i) if sinks.contains(c.name) => i }
          else Nil

        // CDG if available else CFG fallback
        val cdgPairs = {
          val cdg = edgesViaLabel(stmts, id2idx, "CDG")
          if (cdg.nonEmpty) cdg else edgesViaLabel(stmts, id2idx, "CFG")
        }
        // Heuristic DDG
        val ddgPairs = buildHeuristicDDG(stmts, id2idx)

        def pairsToJson(pairs: List[(Int,Int)]) =
          pairs.map{ case (u,v) => s"[$u,$v]" }.mkString(",")

        val target =
          if (file.endsWith("_t1.c")) 1
          else if (file.endsWith("_t0.c")) 0
          else 0

        val json =
          s"""{
             |"line-contents":[${lineContents.map("\""+_+"\"").mkString(",")}],
             |"control-dependences":[${pairsToJson(cdgPairs)}],
             |"data-dependences":[${pairsToJson(ddgPairs)}],
             |"target":$target,
             |"meta":{"file":"${esc(file)}","sink_lines":[${sinkIdxs.mkString(",")}]}
             |}""".stripMargin

        val stem = java.nio.file.Paths.get(file).getFileName.toString.replaceAll("\\.c$",".json")
        java.nio.file.Files.write(outDir.resolve(stem), json.getBytes(java.nio.charset.StandardCharsets.UTF_8))
        written += 1
      } else {
        skipped += 1
      }
    }.recover { case t: Throwable =>
      errs += 1
      logLine(s"[error] idx=$mi name=${m.name} file=${m.filename} -> ${t.getClass.getName}: ${t.getMessage}")
    }
  }

  logLine(s"[done] total=$total written=$written skipped=$skipped errs=$errs -> out=$outDirStr")
  println(s"Wrote PDG JSONs to $outDirStr (total=$total, written=$written, skipped=$skipped, errs=$errs)")
}
