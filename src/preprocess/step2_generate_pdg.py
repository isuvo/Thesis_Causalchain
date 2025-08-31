import argparse, json, os, sys, glob
from pathlib import Path
from subprocess import check_call

def run(cmd):
    print("[run]", " ".join(cmd)); check_call(cmd)

def build_manifest(pdg_dir: Path, manifest_path: Path):
    with manifest_path.open("w", encoding="utf-8") as out:
        for fp in sorted(pdg_dir.glob("*.json")):
            # label from filename suffix: *_t1.json vulnerable, *_t0.json non-vul
            target = 1 if fp.stem.endswith("_t1") else 0
            out.write(json.dumps({"path": str(fp), "target": target}) + "\n")
    print(f"[ok] wrote manifest: {manifest_path}")

def quick_stats(pdg_dir: Path, k: int = 500):
    files = sorted(pdg_dir.glob("*.json"))[:k]
    n = len(files); cdg = ddg = 0
    for f in files:
        try:
            o = json.loads(f.read_text(encoding="utf-8"))
            cdg += len(o.get("control-dependences", []))
            ddg += len(o.get("data-dependences", []))
        except Exception:
            pass
    if n:
        print(f"[stats] files(sampled)={n} avgCDG={cdg/n:.2f} avgDDG={ddg/n:.2f}")
    else:
        print("[stats] no files to sample")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--func_dir", required=True, help="Step-1 folder with .c files")
    ap.add_argument("--out_dir",  required=True, help="Output folder for PDG JSONs")
    ap.add_argument("--joern_home", default="external/joern", help="Path to joern root")
    ap.add_argument("--sensi", default="src/preprocess/external/sensiAPI.txt")
    ap.add_argument("--cpg_path", default="work/joern/diversevul.cpg.bin")
    ap.add_argument("--heap_gb", type=int, default=6)
    ap.add_argument("--verify_only", action="store_true")
    args = ap.parse_args()

    func_dir  = str(Path(args.func_dir).resolve())
    out_dir   = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    cpg_path  = Path(args.cpg_path).resolve(); cpg_path.parent.mkdir(parents=True, exist_ok=True)
    sensi     = str(Path(args.sensi).resolve())
    tools_sc  = str(Path("tools/export_pdg.sc").resolve())

    if args.verify_only:
        quick_stats(out_dir); return

    # 1) Parse -> CPG (call c2cpg directly; Windows-safe; add heap)
    joern_home = Path(args.joern_home).resolve()
    c2cpg   = str(joern_home / ("c2cpg.bat" if os.name=="nt" else "c2cpg"))
    joern   = str(joern_home / ("joern.bat"  if os.name=="nt" else "joern"))
    heap    = f"-J-Xmx{args.heap_gb}g"
    run([c2cpg, heap, func_dir, "--output", str(cpg_path)])

    # 2) Export PDGs via our Joern script (writes 1 JSON per function)
    log_path = str(out_dir / "export.log")
    run([joern, "--read", str(cpg_path), "--script", tools_sc,
         "--params", f'out="{str(out_dir)}",sensi="{sensi}",log="{log_path}"'])

    # 3) Create a manifest.jsonl for training/inference
    build_manifest(out_dir, out_dir / "manifest.jsonl")

    # 4) Quick stats
    quick_stats(out_dir)

    print("\n=== STEP 2 DONE ===")
    print("PDGs:", out_dir)
    print("CPG :", cpg_path)
    print("Log :", log_path)
    print("Manifest:", out_dir / "manifest.jsonl")

if __name__ == "__main__":
    main()
