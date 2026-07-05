
# src/tools/upgrade_train_gcbert.py
import os, glob, json, math, argparse, sys
import torch
from torch_geometric.data import HeteroData

try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    print("[FATAL] transformers not installed. Run: pip install transformers safetensors", file=sys.stderr)
    sys.exit(2)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

def _as_hetero(obj):
    if isinstance(obj, HeteroData):
        return obj
    raise TypeError(f"Expected HeteroData; got {type(obj)}")

def _load_aug_json(path):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    nodes = j.get("nodes", [])
    id2idx = {n.get("_id"): i for i, n in enumerate(nodes) if "_id" in n}
    return nodes, id2idx

def _node_text(n: dict) -> str:
    for k in ("code", "name", "methodFullName", "signature"):
        v = n.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _pick_node_store(data: HeteroData, preferred: str | None) -> str:
    """
    Choose which node type to embed.
    Uses string names from data.node_types (NOT data.node_stores).
    """
    node_types = tuple(getattr(data, "node_types", ()))
    if not node_types:
        # Fallback for rare cases: try to derive names from stores
        node_types = tuple(getattr(s, "_key", None) for s in getattr(data, "node_stores", []) if getattr(s, "_key", None))
    # 1) exact preference
    if preferred and preferred in node_types:
        return preferred
    # 2) common names
    for cand in ("node", "ast", "code", "token", "statement"):
        if cand in node_types:
            return cand
    # 3) first available
    if node_types:
        return node_types[0]
    # 4) last resort: HeteroData supports __iter__ over stores; pick first store key
    try:
        for key in data.keys():  # yields node type names if possible
            return key
    except Exception:
        pass
    raise RuntimeError("Could not determine a node store/type to embed.")


def _find_json(json_dir, base):
    cands = [
        os.path.join(json_dir, base + ".json"),
        os.path.join(json_dir, base + ".aug.json"),
        os.path.join(json_dir, base + ".unified.json"),
        os.path.join(json_dir, base + ".jsonl"),
    ]
    for p in cands:
        if os.path.exists(p): return p
    matches = glob.glob(os.path.join(json_dir, base + ".*"))
    return matches[0] if matches else None

@torch.no_grad()
def embed_gcbert(texts, model, tok, device="cuda", max_len=256, batch_size=64, pbar_desc="Embedding"):
    import time
    embs = []
    model.to(device).eval()
    print(f"[GCBERT] device={device}  model_dtype={next(model.parameters()).dtype}  batches≈{(len(texts)+batch_size-1)//batch_size}")

    total = math.ceil(len(texts)/batch_size) if texts else 0
    inner = tqdm(range(0, len(texts), batch_size),
                 total=total, desc=pbar_desc, unit="batch", leave=False)

    # mixed precision on GPU = faster, same quality for embeddings
    use_amp = (device == "cuda")
    start = time.time()
    for i in inner:
        chunk = texts[i:i+batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(**enc).last_hidden_state  # [B,L,H]
        else:
            out = model(**enc).last_hidden_state
        cls = out[:, 0, :].contiguous()
        embs.append(cls.detach().cpu())
    took = time.time() - start
    if total:
        print(f"[GCBERT] throughput ~{total/took:.2f} batches/s")
    return torch.cat(embs, dim=0) if embs else torch.empty(0, 768)


def process_one(pt_path, json_path, out_path, model, tok, device, node_store, max_len, batch_size, verbose=False):
    data = _as_hetero(torch.load(pt_path, map_location="cpu"))
    store_name = _pick_node_store(data, node_store)
    store = data[store_name]
    if verbose:
        print(f"[INFO] using node_type='{store_name}'  num_nodes={store.num_nodes}")


    nodes, id2idx = _load_aug_json(json_path)

    if hasattr(store, "nid"):
        nid = store.nid.view(-1).tolist()
    else:
        N = store.num_nodes
        if N != len(nodes):
            raise RuntimeError(f"node count mismatch: graph_nodes={N} json_nodes={len(nodes)}")
        nid = [nodes[i].get("_id") for i in range(len(nodes))]
        store.nid = torch.arange(len(nid), dtype=torch.long)  # stable index even if _id not int

    texts = []
    for idx, json_id in enumerate(store.nid.tolist()):
        n = nodes[id2idx.get(json_id, -1)] if json_id in id2idx else (nodes[idx] if idx < len(nodes) else {})
        texts.append(_node_text(n))
        if verbose and idx < 3:
            print(f"  sample text[{idx}]: {texts[-1][:80]!r}")

    x_text = embed_gcbert(texts, model, tok, device=device, max_len=max_len, batch_size=batch_size)
    if x_text.size(0) != len(texts):
        raise RuntimeError("GC-BERT embedding size mismatch")

    store.x_text = x_text
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(data, out_path)
    print(f"[GC-BERT] wrote {out_path}  (N={x_text.size(0)}, D={x_text.size(1)})", flush=True)

def main(in_dir, json_dir, out_dir, device, node_store, max_len, batch_size, strict, verbose, model_dir):
    os.makedirs(out_dir, exist_ok=True)
    pts = sorted(glob.glob(os.path.join(in_dir, "*.pt")))
    if not pts:
        print(f"[ERROR] No .pt files in {in_dir}", file=sys.stderr); sys.exit(2)

    model_name_or_path = model_dir or "microsoft/graphcodebert-base"
    print(f"[GC-BERT] Loading {model_name_or_path} (safetensors=True) …", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        mdl = AutoModel.from_pretrained(model_name_or_path, use_safetensors=True, torch_dtype=None)
    except Exception as e:
        print("[FATAL] Failed to load GraphCodeBERT:", e, file=sys.stderr)
        sys.exit(3)

    failures = 0
    for pt in tqdm(pts, desc="Shards", unit="file"):
        base = os.path.splitext(os.path.basename(pt))[0]
        j = _find_json(json_dir, base)
        if not j:
            print(f"[WARN] JSON not found for base '{base}' under {json_dir}", flush=True)
            failures += 1
            if strict: sys.exit(4)
            continue
        out = os.path.join(out_dir, base + ".pt")
        try:
            process_one(pt, j, out, mdl, tok, device=device, node_store=node_store,
                        max_len=max_len, batch_size=batch_size, verbose=verbose)
        except Exception as e:
            print(f"[ERROR] {os.path.basename(pt)}: {e}", file=sys.stderr)
            failures += 1
            if strict: sys.exit(5)
            continue

    if failures:
        print(f"[SUMMARY] Completed with {failures} issue(s). See messages above.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add GraphCodeBERT embeddings (CLS) to nodes as x_text")
    ap.add_argument("--in_dir",   required=True, help="e.g., Dataset/train/hetero_ready")
    ap.add_argument("--json_dir", required=True, help="e.g., Dataset/train/unified_aug")
    ap.add_argument("--out_dir",  required=True, help="e.g., Dataset/train/hetero_ready_gcbert")
    ap.add_argument("--device",   default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--node_store", default="node")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--model_dir", default=None, help="Local GraphCodeBERT path (optional)")
    args = ap.parse_args()
    main(args.in_dir, args.json_dir, args.out_dir, args.device, args.node_store,
         args.max_len, args.batch_size, args.strict, args.verbose, args.model_dir)
