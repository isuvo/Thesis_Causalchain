# tools/emit_functions_from_jsonl.py
# Usage:
#   py tools/emit_functions_from_jsonl.py --split train
# Creates one source file per JSONL row under work/src/<split>/shard_xxxx/
# Heuristically picks .c or .cpp. Adds a tiny preamble. No rule-based logic.

import argparse, json, os, math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "data" / "dataset" / "ReposVul_c_cpp"
OUTROOT = PROJECT / "work" / "src"

SPLIT2FILE = {
    "train": "train_c_cpp_repository2.jsonl",
    "valid": "valid_c_cpp_repository2.jsonl",
    "test":  "test_c_cpp_repository2.jsonl",
}

def guess_cpp(code: str) -> bool:
    code_l = code.lower()
    # very light heuristic for C++
    cpp_markers = ["std::", "template<", "class ", "namespace ", "new ", "delete ", "operator<<", "::"]
    return any(m in code for m in cpp_markers)

def safe_filename(fid: str) -> str:
    # keep alnum + underscore only
    s = "".join(ch if ch.isalnum() or ch in "._-"
                else "_" for ch in fid)
    return s

def write_source(root: Path, fid: str, code: str, is_cpp: bool):
    ext = ".cpp" if is_cpp else ".c"
    # shard by first 2 hex chars to distribute files (fast FS)
    shard_id = fid[:4] if len(fid) >= 4 else "root"
    shard = f"shard_{shard_id}"
    out_dir = root / shard
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{safe_filename(fid)}{ext}"

    preamble = (
        f"/* function_id: {fid} */\n"
        "/* Generated from ReposVul JSONL for CPG build. */\n"
        "#include <stddef.h>\n"
        "#include <stdint.h>\n"
        "#include <stdbool.h>\n"
    )
    if is_cpp:
        preamble += "#include <string>\n#include <vector>\n"

    # Keep code as-is; fuzzyc2cpg can handle incomplete units.
    with open(out_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(preamble)
        f.write("\n")
        f.write(code)
        f.write("\n")
    return out_file

def process_split(split: str, limit: int = None):
    in_file = DATASET / SPLIT2FILE[split]
    out_root = OUTROOT / split
    out_root.mkdir(parents=True, exist_ok=True)

    n_total, n_ok, n_bad = 0, 0, 0
    with open(in_file, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            n_total += 1
            try:
                obj = json.loads(line)
            except Exception:
                n_bad += 1
                continue

            fid = obj.get("function_id") or obj.get("id") or obj.get("uid")
            code = obj.get("function") or obj.get("code") or obj.get("src") or ""
            if not fid or not code:
                n_bad += 1
                continue

            try:
                is_cpp = guess_cpp(code)
                write_source(out_root, fid, code, is_cpp)
                n_ok += 1
            except Exception:
                n_bad += 1

            if limit and n_ok >= limit:
                break

    print(f"[{split}] total_lines={n_total} ok={n_ok} bad={n_bad} out={out_root}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train","valid","test"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="For smoke-test")
    args = ap.parse_args()
    process_split(args.split, args.limit)

if __name__ == "__main__":
    main()
