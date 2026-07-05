# src/infer/infer_acc.py
import os, sys, json, glob, argparse, traceback
import torch
from torch_geometric.data import HeteroData
from src.models.ca_gat_acc import CAGAT_ACC_Model, _pick_x  # uses lazy input + align

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
        # labels if present
        for k in ("y_node","y","labels"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                data["node"].y = obj[k]; break
        # edges: tuple keys or UPPER string keys
        for k,v in list(obj.items()):
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                if isinstance(k, tuple) and len(k) == 3:
                    data[k].edge_index = v
                elif isinstance(k, str) and k.isupper():
                    data[("node", k, "node")].edge_index = v
        return data
    raise ValueError(f"Unsupported PT content: {type(obj)}")

@torch.no_grad()
def extract_chains(out, data: HeteroData, topk_sinks=3, tau=None):
    logits = out["logits"]                       # [N, C]
    sink_scores = logits.softmax(-1)[:, -1]      # prob(class=1)
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
                if not mask.any(): 
                    continue
                w = gates[mask]
                if tau is not None:
                    w = w * (w >= tau).float()
                if w.numel() == 0:
                    continue
                idx = torch.argmax(w)
                u, wv = src[mask][idx].item(), w[idx].item()
                if wv > best_w:
                    best_w, best_prev = wv, u
            if best_prev is None:
                break
            path.append(best_prev)
            cur = best_prev
        chains.append(list(reversed(path)))
    return chains, sink_scores

@torch.no_grad()
def main(pt_path: str, ckpt_path: str, out_json: str, H: int = 3, debug: bool = True):
    try:
        # Allow directory for --pt
        path = pt_path
        if os.path.isdir(path):
            pts = sorted(glob.glob(os.path.join(path, "*.pt")))
            if not pts:
                raise SystemExit(f"No .pt files under directory: {path}")
            path = pts[0]

        print(f"[infer] loading graph: {path}", flush=True)
        obj = torch.load(path, map_location="cpu")
        data = coerce_to_hetero(obj)
        N = _pick_x(data).size(0)
        ets = tuple(data.edge_types)
        print(f"[infer] nodes={N}, edge_types={len(ets)} -> {ets[:6]}{'...' if len(ets)>6 else ''}", flush=True)

        model = CAGAT_ACC_Model(hidden=128, heads=4, H=H, edge_types=ets, num_classes=2)
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[infer] loading ckpt: {ckpt_path}", flush=True)
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state.get("model", state), strict=False)
        else:
            print(f"[infer][warn] checkpoint not found: {ckpt_path} (running with fresh weights)", flush=True)
        model.eval()

        print("[infer] running model forward ...", flush=True)
        out = model(data)
        print("[infer] forward done.", flush=True)
        # quick debug print
        if debug:
            alphas = out["alpha"]
            hops = list(alphas.keys())
            et0 = next(iter(alphas[hops[0]].keys())) if hops and alphas[hops[0]] else None
            if et0 is not None:
                print(f"[infer] alpha[h0][{et0[1]}] length = {alphas[hops[0]][et0].numel()}", flush=True)

        chains, scores = extract_chains(out, data, topk_sinks=3)
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        payload = {
            "pt": path,
            "topk_sink_scores": [float(x) for x in scores.tolist()],
            "chains": chains
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[infer] wrote {out_json}", flush=True)
        if chains:
            print(f"[infer] example chain: {chains[0]}", flush=True)
        else:
            print("[infer] no chain found (try another shard or higher H)", flush=True)

    except Exception as e:
        print("[infer][ERROR]", e, file=sys.stderr, flush=True)
        traceback.print_exc()
        # still try to write some diagnostics so you have something to inspect
        try:
            os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
            with open(out_json or "infer_error.json", "w") as f:
                json.dump({"error": str(e)}, f, indent=2)
            print(f"[infer] wrote error file: {out_json}", flush=True)
        except Exception:
            pass
        raise

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="path to a .pt file or a directory that contains .pt files")
    ap.add_argument("--ckpt", default="work/logs_acc_small/ckpt_acc.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--no-debug", dest="debug", action="store_false")
    args = ap.parse_args()
    main(args.pt, args.ckpt, args.out, args.H, args.debug)
