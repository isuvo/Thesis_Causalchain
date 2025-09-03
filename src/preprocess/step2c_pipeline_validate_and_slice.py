from __future__ import annotations
import argparse, json, re, shutil, time
from pathlib import Path
from typing import List, Set, Dict, Tuple
from tqdm import tqdm
from src.utils.pdg_io import load_json_safe, normalize_pdg

CALL_NAME = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

def load_sensi_list(path: Path) -> Set[str]:
    if not path or not path.exists():
        return set()
    txt = path.read_text(encoding="utf-8", errors="ignore")
    parts = []
    if "," in txt:
        parts.extend([x.strip() for x in txt.split(",")])
    else:
        parts.extend([x.strip() for x in txt.splitlines()])
    return {x.lower() for x in parts if x and not x.startswith("#")}

def find_seed_indices(lines: List[str], sensi: Set[str]) -> Tuple[List[int], List[int]]:
    """
    Returns:
      sensi_seeds: indices of lines calling a *sensitive* API
      any_call_seeds: indices of lines calling *any* function
    """
    sensi_seeds = []
    any_call_seeds = []
    for i, s in enumerate(lines):
        names = [m.group(1).lower() for m in CALL_NAME.finditer(s)]
        if names:
            any_call_seeds.append(i)
            if sensi and any(n in sensi for n in names):
                sensi_seeds.append(i)
    # make unique + sorted
    sensi_seeds = sorted(set(sensi_seeds))
    any_call_seeds = sorted(set(any_call_seeds))
    return sensi_seeds, any_call_seeds

def combine_edges(cdg: List[List[int]], ddg: List[List[int]]) -> List[List[int]]:
    seen, out = set(), []
    for u, v in cdg + ddg:
        key = (int(u), int(v))
        if key not in seen:
            seen.add(key)
            out.append([int(u), int(v)])
    return out

def bfs_backward(n: int, edges: List[List[int]], start: int) -> List[int]:
    rev = [[] for _ in range(n)]
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n:
            rev[v].append(u)
    seen, q = {start}, [start]
    i = 0
    while i < len(q):
        cur = q[i]; i += 1
        for p in rev[cur]:
            if p not in seen:
                seen.add(p); q.append(p)
    return sorted(seen)

def bfs_forward(n: int, ddg: List[List[int]], start_set: Set[int]) -> List[int]:
    fwd = [[] for _ in range(n)]
    for u, v in ddg:
        if 0 <= u < n and 0 <= v < n:
            fwd[u].append(v)
    seen, q = set(start_set), list(start_set)
    i = 0
    while i < len(q):
        cur = q[i]; i += 1
        for nxt in fwd[cur]:
            if nxt not in seen:
                seen.add(nxt); q.append(nxt)
    return sorted(seen)

def make_slice_json(src_obj: dict, keep_nodes: List[int], combined_edges: List[List[int]], seed_idx: int) -> dict:
    idx_map = {old: i for i, old in enumerate(keep_nodes)}
    e2 = []
    for u, v in combined_edges:
        if u in idx_map and v in idx_map:
            e2.append([idx_map[u], idx_map[v]])
    return {
        "line-contents": [src_obj["line-contents"][i] for i in keep_nodes],
        "pdg_edges": e2,
        "target": int(src_obj.get("target", 0)),
        "meta": {
            "source_file": (src_obj.get("meta", {}) or {}).get("file", ""),
            "seed_src_index": seed_idx,
            "seed_local_index": idx_map.get(seed_idx, None),
            "orig_size": len(src_obj["line-contents"]),
            "slice_size": len(keep_nodes),
        }
    }

def validate_shape(obj: dict, require_ddg: bool, min_nodes: int, max_nodes: int) -> Tuple[bool, str]:
    lines = obj.get("line-contents")
    if not isinstance(lines, list) or not all(isinstance(x, str) for x in lines):
        return False, "bad_line_contents"
    n = len(lines)
    if n < min_nodes:  return False, "too_small"
    if n > max_nodes:  return False, "too_large"
    cdg = obj.get("control-dependences", [])
    ddg = obj.get("data-dependences", [])
    if not isinstance(cdg, list) or not isinstance(ddg, list):
        return False, "bad_edges_type"
    for e in cdg + ddg:
        if not (isinstance(e, list) and len(e) == 2 and all(isinstance(k, int) for k in e)):
            return False, "bad_edge_item"
    if require_ddg and len(ddg) == 0:
        return False, "no_ddg"
    return True, "ok"

def main():
    ap = argparse.ArgumentParser(description="Validate (non-destructive) + Slice PDGs in one pass.")
    ap.add_argument("--src", required=True, help="Folder with original PDG JSONs (left untouched)")
    ap.add_argument("--clean", required=True, help="Folder to write normalized/clean PDGs")
    ap.add_argument("--out", required=True, help="Folder to write slice JSONs")
    ap.add_argument("--bad", required=True, help="Folder to store copies of rejected/invalid PDGs")
    ap.add_argument("--sensi", default="src/preprocess/external/sensiAPI.txt",
                    help="Sensitive API list (comma-separated or newline-separated)")
    ap.add_argument("--sample", type=int, default=0, help="Limit number of files to process")
    ap.add_argument("--dry_run", action="store_true", help="Analyze only; do not write/copy anything")
    ap.add_argument("--require_ddg", action="store_true", help="Reject PDGs without DDG edges")
    ap.add_argument("--min_nodes", type=int, default=2)
    ap.add_argument("--max_nodes", type=int, default=5000)
    ap.add_argument("--copy_bad", action="store_true", help="Copy rejects to --bad (never move)")
    
    ap.add_argument("--fallback_any_call", action="store_true",help="If no sensitive API is found, fallback to any function call as seed(s)")
    ap.add_argument("--min_calls_for_fallback", type=int, default=1,help="Require at least N calls to attempt fallback seeding")
    ap.add_argument("--max_seeds_per_pdg", type=int, default=5, help="Limit the number of seeds per PDG to avoid explosion")
    ap.add_argument("--max_slice_size", type=int, default=512, help="Skip slices larger than this many nodes (0=disable)")
    ap.add_argument("--keep_no_seed_clean", action="store_true", help="Write cleaned PDG even if it had no seeds")
    args = ap.parse_args()

    src   = Path(args.src)
    clean = Path(args.clean)
    out   = Path(args.out)
    bad   = Path(args.bad)

    # Ensure outputs exist (we won't touch src)
    clean.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    bad.mkdir(parents=True, exist_ok=True)

    sensi = load_sensi_list(Path(args.sensi))

    files = sorted(src.glob("*.json"))
    if args.sample and args.sample < len(files):
        files = files[:args.sample]

    stats: Dict[str, int] = {
        "total_seen": 0, "json_error": 0, "normalized": 0,
        "rejected": 0, "no_seed": 0, "ok_pdgs": 0,
        "slices_written": 0, "exceptions": 0, "seeds_total": 0
    }

    t0 = time.time()
    for jf in tqdm(files, desc="validate+slice (non-destructive)"):
        stats["total_seen"] += 1
        try:
            raw = load_json_safe(jf)
            if isinstance(raw, list):
                raw = raw[0] if raw and isinstance(raw[0], dict) else raw

            if not isinstance(raw, dict):
                stats["json_error"] += 1
                if not args.dry_run and args.copy_bad:
                    shutil.copy2(str(jf), bad / jf.name)
                continue

            obj, norm_status = normalize_pdg(raw)
            if norm_status != "ok":
                stats["rejected"] += 1
                if not args.dry_run and args.copy_bad:
                    shutil.copy2(str(jf), bad / jf.name)
                continue
            stats["normalized"] += 1

            ok, why = validate_shape(obj, args.require_ddg, args.min_nodes, args.max_nodes)
            if not ok:
                stats["rejected"] += 1
                if not args.dry_run and args.copy_bad:
                    shutil.copy2(str(jf), bad / jf.name)
                continue

            # write CLEAN copy
            if not args.dry_run:
                (clean / jf.name).write_text(json.dumps(obj), encoding="utf-8")

            # slice
            lines = obj["line-contents"]
            cdg = obj.get("control-dependences", [])
            ddg = obj.get("data-dependences", [])
            combined = combine_edges(cdg, ddg)

            sensi_seeds, any_call_seeds = find_seed_indices(lines, sensi)
            seeds = sensi_seeds

            # Fallback: if no sensi seeds, use any-call seeds (optionally)
            if not seeds and args.fallback_any_call and len(any_call_seeds) >= args.min_calls_for_fallback:
                seeds = any_call_seeds

            # Cap the number of seeds per PDG
            if args.max_seeds_per_pdg > 0 and len(seeds) > args.max_seeds_per_pdg:
                seeds = seeds[:args.max_seeds_per_pdg]

            stats["seeds_total"] += len(seeds)

            # Optionally still write clean PDG even if no seeds (for later)
            if not seeds:
                stats["no_seed"] += 1
                if not args.dry_run:
                    # always write CLEAN copy (we normalized already)
                    (clean / jf.name).write_text(json.dumps(obj), encoding="utf-8")
                    if args.copy_bad:
                        shutil.copy2(str(jf), bad / jf.name)
                continue

            n = len(lines)
            wrote = False
            for si, seed_idx in enumerate(seeds):
                back = set(bfs_backward(n, combined, seed_idx))
                fwd  = set(bfs_forward(n, ddg, {seed_idx}))
                keep = sorted(back.union(fwd))
                if len(keep) < 2:
                    continue
                if args.max_slice_size > 0 and len(keep) > args.max_slice_size:
                    continue
                sl = make_slice_json(obj, keep, combined, seed_idx)
                if not args.dry_run:
                    (out / f"{jf.stem}_s{si}.json").write_text(json.dumps(sl, ensure_ascii=False), encoding="utf-8")
                stats["slices_written"] += 1
                wrote = True

            if wrote:
                stats["ok_pdgs"] += 1
            else:
                stats["rejected"] += 1
                if not args.dry_run and args.copy_bad:
                    shutil.copy2(str(jf), bad / jf.name)

        except Exception:
            stats["exceptions"] += 1
            if not args.dry_run and args.copy_bad:
                try:
                    shutil.copy2(str(jf), bad / jf.name)
                except Exception:
                    pass

    stats["elapsed_sec"] = round(time.time() - t0, 2)
    if not args.dry_run:
        (out / "validate_and_slice_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        # optional: small summary next to clean dir too
        (clean / "clean_report.json").write_text(json.dumps({
            k: stats[k] for k in ["total_seen","normalized","rejected","json_error","exceptions"]
        }, indent=2), encoding="utf-8")

    print("\n[report]")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
