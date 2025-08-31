# step1_prepare_diversevul.py
# Purpose:
#   - Load DiverseVul JSONL
#   - Clean code (remove comments/empty lines)
#   - Tokenize lines (sanity)
#   - Dump each function to .c (for Joern later)
#   - Write a cleaned jsonl + stats + preview + manifest
#
# Usage:
#   python -m src.preprocess.step1_prepare_diversevul \
#       --in data/dataset/DiverseVul/diversevul_20230702.json \
#       --out work/diversevul_step1

import argparse, json, os, re, hashlib, random
from pathlib import Path
from typing import Dict, Any, Iterable
import pandas as pd
from tqdm import tqdm

COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.S)
COMMENT_LINE  = re.compile(r"//.*?$", re.M)
TOKEN = re.compile(r"[A-Za-z_]\w+|==|!=|<=|>=|&&|\|\||[^\s]")

def remove_comments(code: str) -> str:
    # crude but effective for C/C++; we preserve quoted strings
    # first protect string/char literals
    placeholders = []
    def protect(m):
        placeholders.append(m.group(0))
        return f'__STR{len(placeholders)-1}__'
    code = re.sub(r'\"([^\\\"]|\\.)*\"|\'.*?\'', protect, code, flags=re.S)

    code = re.sub(COMMENT_BLOCK, "", code)
    code = re.sub(COMMENT_LINE, "", code)

    # restore strings
    def restore(m):
        idx = int(m.group(0)[5:-2])
        return placeholders[idx]
    code = re.sub(r'__STR\d+__', restore, code)
    return code

def clean_code(code: str) -> str:
    code = remove_comments(code)
    # normalize whitespace, drop empty lines
    lines = [ln.rstrip() for ln in code.splitlines()]
    lines = [ln for ln in lines if ln.strip() != ""]
    return "\n".join(lines)

def tokenize_line(s: str) -> str:
    return " ".join(TOKEN.findall(s))

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to diversevul_20230702.json")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output folder (will be created)")
    ap.add_argument("--preview", type=int, default=10, help="How many rows to sample to CSV preview")
    args = ap.parse_args()

    src = Path(args.inp)
    out_dir = Path(args.out_dir)
    func_dir = out_dir / "funcs"
    out_dir.mkdir(parents=True, exist_ok=True)
    func_dir.mkdir(parents=True, exist_ok=True)

    # pass 1: load & clean; build small rows for preview; write clean.jsonl
    clean_path = out_dir / "clean.jsonl"
    with clean_path.open("w", encoding="utf-8") as fout:
        rows_for_preview = []
        total = vuln = 0
        for idx, rec in enumerate(tqdm(iter_jsonl(src), desc="Cleaning")):
            # pick the code field; DiverseVul typically uses "func"
            raw = rec.get("func") or rec.get("function") or rec.get("code") or ""
            target = int(rec.get("target", rec.get("label", 0)))
            if not raw.strip():
                continue
            cleaned = clean_code(raw)
            if not cleaned.strip():
                continue

            # basic tokenization (sanity only)
            tokens_preview = tokenize_line(cleaned.splitlines()[0]) if cleaned.splitlines() else ""

            out_rec = {
                "idx": idx,
                "target": target,
                "code": cleaned
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            # for preview table
            if len(rows_for_preview) < args.preview:
                rows_for_preview.append({
                    "idx": idx,
                    "target": target,
                    "first_line_tokens": tokens_preview,
                    "n_lines": len(cleaned.splitlines())
                })

            total += 1
            vuln += (1 if target == 1 else 0)

    # write preview CSV
    if rows_for_preview:
        pd.DataFrame(rows_for_preview).to_csv(out_dir / "sample_preview.csv", index=False, encoding="utf-8")

    # write stats
    stats = {"total": total, "vulnerable": vuln, "non_vulnerable": total - vuln}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # pass 2: dump each function to .c and build manifest
    manifest = []
    with clean_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Dumping .c"):
            row = json.loads(line)
            idx = row["idx"]; code = row["code"]; target = row["target"]
            stem = f"id{idx}_{short_hash(code)}_t{target}"
            fname = func_dir / f"{stem}.c"
            fname.write_text(code, encoding="utf-8")
            manifest.append({"idx": idx, "file": str(fname), "target": target})

    (func_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== DONE (Step 1) ===")
    print(f"Cleaned JSONL:  {clean_path}")
    print(f"Preview CSV:    {out_dir / 'sample_preview.csv'}")
    print(f"Stats JSON:     {out_dir / 'stats.json'}  -> {stats}")
    print(f"Function files: {func_dir}  (+ manifest.json)")

if __name__ == "__main__":
    main()
