# src/infer_one_slice_gcbert.py
import argparse, json, math, re
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch_geometric.data import Data, Batch

# ---- reuse helpers from your Step-3 exporter ----
try:
    from src.preprocess.slices_to_pyg_gcbert import (
        OnDemandEmbedder, degree_features, dist_to_seed
    )
except Exception:
    from preprocess.slices_to_pyg_gcbert import (
        OnDemandEmbedder, degree_features, dist_to_seed
    )

# ---- model from Step-4 ----
try:
    from src.step4_train_gnn_attn_v1 import GNNSeedAttention
except Exception:
    from src.step4_train_gnn_attn_v1 import GNNSeedAttention

# simple tokenizer (for 1 small structural feature)
#OP = re.compile(r"[A-Za-z_]\w+|==|!=|<=|>=|&&|\|\||<<|>>|[{}()\[\].,;:+\\-*/%&|^!=<>?~]")

OP = re.compile(
    r"[A-Za-z_]\w+|==|!=|<=|>=|\&\&|\|\||<<|>>|[{}\(\)\[\]\.,;:\+\*/%&\|^!=<>?~\-]"
)
def _tok(line: str) -> List[str]:
    return OP.findall(line) or [line.strip()]

def _build_features(obj: dict, embedder: OnDemandEmbedder, include_sensi: bool, sensi: set|None):
    lines = obj.get("line-contents") or []
    edges = obj.get("pdg_edges") or []
    assert len(lines) >= 2 and len(edges) > 0, "Slice must have >=2 lines and >=1 edge"
    n = len(lines)

    Xtxt = embedder.embed(lines)  # [n, 768] GraphCodeBERT

    indeg, outdeg = degree_features(n, edges)
    meta = obj.get("meta", {}) or {}
    seed_idx = meta.get("seed_local_index", -1)
    dseed = dist_to_seed(n, edges, seed_idx if isinstance(seed_idx, int) else -1)
    node_idx = [i / max(n - 1, 1) for i in range(n)]
    tok_norm = [math.log1p(len(_tok(s))) for s in lines]
    extra = np.stack([indeg, outdeg, node_idx, dseed, tok_norm], axis=1).astype(np.float32)

    if include_sensi and sensi:
        sensi_flag = np.array([1.0 if any(w in s for w in sensi) else 0.0 for s in lines],
                              dtype=np.float32).reshape(-1, 1)
        X = np.hstack([Xtxt, extra, sensi_flag])
    else:
        X = np.hstack([Xtxt, extra])

    E = []
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n:
            E.append((u, v)); E.append((v, u))
    assert E, "No valid edges"

    data = Data(
        x=torch.from_numpy(X),
        edge_index=torch.tensor(E, dtype=torch.long).t().contiguous(),
        y=torch.tensor([int(obj.get("target", 0))], dtype=torch.long)
    )
    data.seed_idx = torch.tensor([seed_idx if isinstance(seed_idx, int) and 0 <= seed_idx < n else -1], dtype=torch.long)
    data.num_nodes = n
    data.lines = lines  # keep raw code lines for printing
    data.src = meta.get("source_file", "")
    return data

def _best_path_from_seed(edge_index, attn_vec, seed_idx, max_len=8):
    import collections
    E = edge_index.t().tolist()
    adj = collections.defaultdict(list)
    for u, v in E:
        adj[u].append(v); adj[v].append(u)
    if seed_idx < 0 or seed_idx >= int(attn_vec.numel()):
        seed_idx = int(torch.argmax(attn_vec).item())
    path = [seed_idx]; visited = {seed_idx}; cur = seed_idx
    for _ in range(max_len - 1):
        nbrs = [n for n in adj[cur] if n not in visited]
        if not nbrs: break
        nxt = max(nbrs, key=lambda n: float(attn_vec[n]))
        path.append(nxt); visited.add(nxt); cur = nxt
    return path

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice_json", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--sensi_file", default="")
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--print_topk", type=int, default=8)
    args = ap.parse_args()

    sensi = set()
    if args.sensi_file and Path(args.sensi_file).exists():
        sensi = {l.strip() for l in Path(args.sensi_file).read_text(encoding="utf-8").splitlines() if l.strip()}

    #obj = json.loads(Path(args.slice_json).read_text(encoding="utf-8"))
    txt = Path(args.slice_json).read_text(encoding="utf-8-sig")  # handles UTF-8 BOM
    obj = json.loads(txt.lstrip("\ufeff"))
    
    embedder = OnDemandEmbedder("microsoft/graphcodebert-base", device="cpu", max_len=args.max_len)

    # try with sensi flag first to match your trained in_dim=774; fallback if mismatch
    data = _build_features(obj, embedder, include_sensi=True, sensi=sensi)
    cfg = json.loads(Path(args.cfg).read_text(encoding="utf-8"))
    in_dim, hidden, attn_hidden = cfg["in_dim"], cfg["hidden"], cfg["attn_hidden"]
    if data.x.size(1) != in_dim:
        data = _build_features(obj, embedder, include_sensi=False, sensi=None)
        assert data.x.size(1) == in_dim, f"feature dim {data.x.size(1)} != model in_dim {in_dim}"

    device = torch.device("cpu")
    model = GNNSeedAttention(in_dim, hidden=hidden, attn_hidden=attn_hidden).to(device)
    try:
        state = torch.load(args.model, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)

    batch = Batch.from_data_list([data]).to(device)
    logits, attn_list = model.forward_batch(batch)
    prob = torch.sigmoid(logits[0]).item()
    attn = attn_list[0]
    seed = int(data.seed_idx.item())
    path = _best_path_from_seed(data.edge_index, attn, seed_idx=seed, max_len=8)

    print(f"SRC: {data.src or '(custom slice)'}")
    print(f"P(vuln): {prob:.4f}")
    print(f"seed_idx: {seed}")
    if args.print_topk > 0:
        k = min(args.print_topk, attn.numel())
        topi = torch.topk(attn, k=k).indices.tolist()
        print("Top nodes (idx, attn):")
        for i in topi:
            print(f"  {i:>3}  {attn[i]:.3f}  |  {data.lines[i]}")
    print("Chain (idx → code):")
    for i in path:
        print(f"  {i}: {data.lines[i]}")

if __name__ == "__main__":
    main()
