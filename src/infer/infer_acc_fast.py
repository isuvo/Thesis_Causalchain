# src/infer/infer_acc_fast.py
import os, sys, json, glob, argparse, traceback
import torch
from torch_geometric.data import HeteroData
from src.models.ca_gat_acc import CAGAT_ACC_Model, ACCGate, _pick_x

def coerce_to_hetero(obj) -> HeteroData:
    if isinstance(obj, HeteroData): return obj
    if isinstance(obj, list) and obj: return coerce_to_hetero(obj[0])
    if isinstance(obj, dict) and "graphs" in obj and obj["graphs"]: return coerce_to_hetero(obj["graphs"][0])
    if isinstance(obj, dict):
        from torch_geometric.data import HeteroData as HD
        data = HD()
        # node features
        for k in ("x_text","x_tfidf","x","feat","features","node_x"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                data["node"].x = obj[k]; break
        # edges (tuple or UPPER string)
        for k,v in list(obj.items()):
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                if isinstance(k, tuple) and len(k) == 3:
                    data[k].edge_index = v
                elif isinstance(k, str) and k.isupper():
                    data[("node", k, "node")].edge_index = v
        return data
    raise ValueError(f"Unsupported PT content: {type(obj)}")

def compute_degrees(data: HeteroData, N: int, device):
    outdeg = torch.zeros(N, dtype=torch.long, device=device)
    indeg  = torch.zeros(N, dtype=torch.long, device=device)
    for et in data.edge_types:
        ei = data[et].edge_index.to(device)
        ones = torch.ones(ei.size(1), dtype=torch.long, device=device)
        outdeg.index_add_(0, ei[0], ones)
        indeg.index_add_(0, ei[1], ones)
    return outdeg, indeg

@torch.no_grad()
def build_alpha_logs_only(acc: ACCGate, data: HeteroData, H: int, device):
    N = _pick_x(data).size(0)
    outdeg, indeg = compute_degrees(data, N, device)
    alpha = {}
    for h in range(H):
        alpha_h = {}
        for et in data.edge_types:
            ei = data[et].edge_index.to(device)  # [2, E]
            src, dst = ei
            is_inter = 1.0 if et[1] in {"CALL","ARG2PARAM","RET2CALL","RET2LHS"} else 0.0
            inter = torch.full_like(src, is_inter, dtype=torch.float, device=device)
            g = acc(et, h, outdeg[src], indeg[dst], inter)  # [E] in (0,1)
            alpha_h[et] = g.cpu()
        alpha[h] = alpha_h
    return alpha

def pick_sinks_from_alpha(alpha, data: HeteroData, topk=3):
    # sinkiness = total incoming gate mass per node (averaged across hops and edge types)
    N = _pick_x(data).size(0)
    score = torch.zeros(N)
    H = len(alpha)
    for h in range(H):
        for et, g in alpha[h].items():
            ei = data[et].edge_index
            dst = ei[1]
            score.index_add_(0, dst, g)
    score = score / max(1, H)
    k = min(topk, N)
    return torch.topk(score, k=k).indices.tolist(), score

def backtrace_chain(alpha, data: HeteroData, sink: int):
    # greedy: for each hop (reverse), pick incoming edge with max gate
    path = [sink]
    cur = sink
    H = len(alpha)
    for h in range(H-1, -1, -1):
        best_prev, best_w = None, -1.0
        for et, g in alpha[h].items():
            ei = data[et].edge_index
            src, dst = ei
            mask = (dst == cur)
            if not mask.any(): continue
            gm = g[mask]
            if gm.numel() == 0: continue
            idx = torch.argmax(gm)
            u, w = src[mask][idx].item(), gm[idx].item()
            if w > best_w:
                best_w, best_prev = w, u
        if best_prev is None: break
        path.append(best_prev)
        cur = best_prev
    return list(reversed(path))

@torch.no_grad()
def main(pt: str, ckpt: str, out_json: str, H: int = 3):
    # allow passing a directory
    path = pt
    if os.path.isdir(path):
        pts = sorted(glob.glob(os.path.join(path, "*.pt")))
        if not pts:
            raise SystemExit(f"No .pt files under: {path}")
        path = pts[0]

    print(f"[fast] loading graph: {path}", flush=True)
    obj = torch.load(path, map_location="cpu")
    data = coerce_to_hetero(obj)
    ets = tuple(data.edge_types)
    N = _pick_x(data).size(0)
    print(f"[fast] nodes={N}, edge_types={len(ets)}", flush=True)

    # Build a model shell simply to load ACC params
    model = CAGAT_ACC_Model(hidden=128, heads=4, H=H, edge_types=ets, num_classes=2)
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
        print(f"[fast] loaded ckpt: {ckpt}", flush=True)
    else:
        print(f"[fast][warn] checkpoint not found: {ckpt} (using random ACC params)", flush=True)

    # Compute alpha logs without running GAT
    alpha = build_alpha_logs_only(model.acc, data, H=H, device=torch.device("cpu"))

    # Pick sinks and build chains
    sinks, sink_scores = pick_sinks_from_alpha(alpha, data, topk=3)
    chains = [backtrace_chain(alpha, data, s) for s in sinks]

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "pt": path,
            "sinks": sinks,
            "sink_scores": [float(x) for x in sink_scores.tolist()],
            "chains": chains
        }, f, indent=2)
    print(f"[fast] wrote {out_json}", flush=True)
    if chains:
        print(f"[fast] example chain: {chains[0]}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="a .pt file or a directory containing .pt files")
    ap.add_argument("--ckpt", default="work/logs_acc_small/ckpt_acc.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--H", type=int, default=3)
    args = ap.parse_args()
    main(args.pt, args.ckpt, args.out, args.H)
