# tools/emit_files_from_reposvul_cpp.py
# Turns ReposVul_cpp.jsonl into real source files grouped by project__commit.
# Usage:
#   py tools/emit_files_from_reposvul_cpp.py --in data/dataset/ReposVul_CPP/ReposVul_cpp.jsonl
# Outputs under: work/src_cpp_commits/<project>__<commit_id>/<file_path or file_name>

import argparse, json, re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUTROOT = PROJECT / "work" / "src_cpp_commits"

EXT_FROM_LANG = {
    "c": ".c", "cpp": ".cpp", "cc": ".cc", "cxx": ".cxx", "h": ".h", "hpp": ".hpp", "hh": ".hh"
}

def safe_seg(s: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:120] or "x"

def guess_ext(lang: str, name: str) -> str:
    if name and "." in name:
        return "." + name.split(".")[-1].lower()
    return EXT_FROM_LANG.get((lang or "").lower(), ".cpp")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    infile = Path(args.inp)
    OUTROOT.mkdir(parents=True, exist_ok=True)

    total=ok=bad=0
    with infile.open("r", encoding="utf-8", errors="ignore") as fin:
        for idx, line in enumerate(fin):
            if not line.strip(): continue
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                bad += 1; continue

            project = safe_seg(str(row.get("project", "")))
            commit  = safe_seg(str(row.get("commit_id", "")))
            details = row.get("details") or []
            if not project or not commit or not isinstance(details, list) or not details:
                bad += 1; continue

            base = OUTROOT / f"{project}__{commit}"
            for d in details:
                code   = d.get("code")
                if not isinstance(code, str) or not code.strip(): 
                    continue
                fname  = d.get("file_name") or ""
                fpath  = d.get("file_path") or fname or "file"
                lang   = d.get("file_language") or ""
                ext    = guess_ext(lang, fname)

                # normalize path segments for Windows
                segs = [safe_seg(s) for s in re.split(r"[\\/]+", fpath) if s]
                if segs and not segs[-1].lower().endswith((".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh")):
                    segs[-1] = segs[-1] + ext

                outpath = base.joinpath(*segs)
                outpath.parent.mkdir(parents=True, exist_ok=True)
                outpath.write_text(code, encoding="utf-8", newline="\n")
                ok += 1

            if args.limit and ok >= args.limit:
                break

    print(f"total_rows={total} wrote_files={ok} skipped={bad} out={OUTROOT}")

if __name__ == "__main__":
    main()
