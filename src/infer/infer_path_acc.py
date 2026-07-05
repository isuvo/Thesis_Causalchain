# src/infer/infer_acc.py
import os, json, argparse
import torch
from torch_geometric.data import HeteroData
from src.models.ca_gat_acc import CAGAT_ACC_Model, _pick_x

def coerce_to_hetero(obj) -> HeteroData:
    if isinstance(obj, HeteroData): return obj
    if isinstance(obj, list) and obj: return coerce_to_hetero(obj[0])
    if isinstance(obj, dict) and "graphs" in obj and obj["graphs"]: return coerce_to_hetero(obj["graphs"][0])
    if isinstance(obj, dict):
        from torch_geometric.data import HeteroData as HD
        data = HD()
        for k in ("x_text","x_tfidf","x","feat","features","node_x"):
            if k in obj: data["node"].x = obj[k]; break
        for k in ("y_node","y","labels"):
            if k in obj: data["node"].y = obj[k]; break
        for k,v in list(obj.items()):
            if isinstance(k, tuple) and len(k)==3 and hasattr(v, "shape"): data[k].edge_index = v
            elif isinstance(k,str) and k.isupper() and hasattr(v,"shape"): data[("node",k,"node")].edge_index = v
        return data
    raise ValueError(f"Unsupported PT content: {type(obj)}")

@torch.no_grad()
def extract_chains(out, data: HeteroData, topk_sinks=3, tau=None):
    logits = out["logits"]
    sink_scores = logits.softmax(-1)[:, -1]
    k = min(topk_sinks, sink_scores.numel())
    sinks = torch.topk(sink_scores, k=k).indices.tolist()
    chains = []
    H = len(out["alpha"])
    for sink in sinks:
        path = [sink]
        cur = sink
        for h in range(H-1, -1, -1):
            best_prev, best_w = None, -1.0
            for et, gates in out["alpha"][h].items():
                ei = data[et].edge_index
                src, dst = ei
                mask = (dst == cur)
                if not mask.any(): continue
                w = gates[mask]
                if tau is not None: w = w * (w >= tau).float()
                if w.numel()==0: continue
                idx = torch.argmax(w)
                u, wv = src[mask][idx].item(), w[idx].item()
                if wv > best_w: best_w, best_prev = wv, u
            if best_prev is None: break
            path.append(best_prev); cur = best_prev
        chains.append(list(reversed(path)))
    return chains, sink_scores

@torch.no_grad()
def main(pt_path: str, ckpt_path: str, out_json: str, H: int = 3):
    obj = torch.load(pt_path, map_location="cpu")
    data = coerce_to_hetero(obj)
    edge_types = tuple(data.edge_types)
    model = CAGAT_ACC_Model(hidden=128, heads=4, H=H, edge_types=edge_types, num_classes=2)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
    model.eval()
    out = model(data)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    chains, scores = extract_chains(out, data, topk_sinks=3)
    with open(out_json, "w") as f:
        json.dump({"pt": pt_path,
                   "topk_sink_scores": [float(x) for x in scores.tolist()],
                   "chains": chains}, f, indent=2)
    print(f"Wrote {out_json}")
    print("Example chain:", chains[0] if chains else "—")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--ckpt", default="work\\logs_acc_small\\ckpt_acc.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--H", type=int, default=3)
    args = ap.parse_args()
    main(args.pt, args.ckpt, args.out, args.H)
