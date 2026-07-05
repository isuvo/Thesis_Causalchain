# src/preprocess/upgrade_pt_with_gcbert.py
import os, glob, argparse, json, math, sys
import torch
from torch_geometric.data import HeteroData
from transformers import AutoTokenizer, AutoModel

# ---- tqdm (fallback if not installed) ----
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k): return x

def _coerce_hetero(obj):
    if isinstance(obj, HeteroData):
        return obj
    raise ValueError("Expected HeteroData in .pt")

def _load_aug_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    nodes = j.get("nodes", [])
    id2idx = {n["_id"]: i for i, n in enumerate(nodes) if "_id" in n}
    return nodes, id2idx

def _node_text(n: dict) -> str:
    return (
        n.get("code")
        or n.get("name")
        or n.get("methodFullName")
        or n.get("signature")
        or ""
    )

def embed_gcbert(texts, model, tok, device="cpu", max_len=256, batch_size=64):
    embs = []
    model.to(device).eval()
    total_batches = math.ceil(len(texts) / batch_size) if texts else 0
    for i in tqdm(range(0, len(texts), batch_size),
                  total=total_batches, desc="Embedding nodes", unit="batch", leave=False):
        chunk = texts[i:i+batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc).last_hidden_state  # [B,T,768]
            cls = out[:, 0, :].contiguous()      # [B,768]
            embs.append(cls.cpu())
    return torch.cat(embs, dim=0) if embs else torch.empty(0, 768)

def process_one(pt_path: str, json_path: str, out_path: str, model, tok, device="cpu"):
    data = _coerce_hetero(torch.load(pt_path, map_location="cpu"))
    nodes, id2idx = _load_aug_json(json_path)

    # Establish nid mapping
    if hasattr(data["node"], "nid"):
        nid = data["node"].nid.view(-1).tolist()
    else:
        N = data["node"].x.size(0) if hasattr(data["node"], "x") else None
        if N is None or N != len(nodes):
            raise RuntimeError(
                f"Cannot reconstruct nid: graph_nodes={N}, json_nodes={len(nodes)} for {pt_path}"
            )
        nid = [nodes[i]["_id"] for i in range(len(nodes))]
        data["node"].nid = torch.tensor(nid, dtype=torch.long)

    # Collect texts
    texts = []
    for g_idx, json_id in enumerate(nid):
        jidx = id2idx.get(json_id, None)
        n = nodes[jidx] if jidx is not None else {}
        t = _node_text(n)
        texts.append(t if isinstance(t, str) else "")

    # Embed with GC-BERT
    x_text = embed_gcbert(texts, model, tok, device=device, max_len=256, batch_size=64)  # [N,768]
    if x_text.size(0) != len(texts):
        raise RuntimeError(f"Embedding size mismatch for {pt_path}")

    data["node"].x_text = x_text  # preferred feature for training/inference

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(data, out_path)
    print(f"[GC-BERT] wrote {out_path}  (N={x_text.size(0)}, D=768)", flush=True)

def main(in_dir, json_dir, out_dir, device="cpu"):
    os.makedirs(out_dir, exist_ok=True)
    pts = sorted(glob.glob(os.path.join(in_dir, "*.pt")))
    if not pts:
        print(f"No .pt files in {in_dir}", file=sys.stderr); sys.exit(1)

    print("[GC-BERT] loading model 'microsoft/graphcodebert-base' ...", flush=True)
    tok = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    mdl = AutoModel.from_pretrained("microsoft/graphcodebert-base")

    for pt in tqdm(pts, desc="Shards", unit="file"):
        base = os.path.splitext(os.path.basename(pt))[0]
        j = os.path.join(json_dir, base + ".json")
        if not os.path.exists(j):
            print(f"[WARN] missing JSON for {pt}; skipping", flush=True)
            continue
        out = os.path.join(out_dir, base + ".pt")
        process_one(pt, j, out, mdl, tok, device=device)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    main(args.in_dir, args.json_dir, args.out_dir, device=args.device)
