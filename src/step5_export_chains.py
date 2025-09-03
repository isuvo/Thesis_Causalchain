# src/step5_export_chains.py
import argparse, json, os, collections, warnings
import torch
from torch_geometric.loader import DataLoader

# robust import
try:
    from .step4_train_gnn_attn_v1 import GNNSeedAttention, load_shards
except ImportError:
    from src.step4_train_gnn_attn_v1 import GNNSeedAttention, load_shards

def best_paths_from_seed(edge_index, attn_vec, seed_idx, k=3, max_len=8, beam=4):
    E = edge_index.t().tolist()
    adj = collections.defaultdict(list)
    for u, v in E:
        adj[u].append(v); adj[v].append(u)
    if seed_idx is None or seed_idx < 0 or seed_idx >= int(attn_vec.numel()):
        seed_idx = int(torch.argmax(attn_vec).item())

    # beams: (path, visited_set, score_sum)
    beams = [([seed_idx], {seed_idx}, float(attn_vec[seed_idx].item() if hasattr(attn_vec[seed_idx], "item") else float(attn_vec[seed_idx])))]
    finished = []  # (path, score_sum)

    for _ in range(max_len - 1):
        new_beams = []
        for path, vis, score in beams:
            cur = path[-1]
            nbrs = [n for n in adj[cur] if n not in vis]
            if not nbrs:
                finished.append((path, score))
                continue
            nbrs_sorted = sorted(
                nbrs,
                key=lambda n: float(attn_vec[n].item() if hasattr(attn_vec[n], "item") else float(attn_vec[n])),
                reverse=True
            )[:beam]
            for n in nbrs_sorted:
                new_score = score + float(attn_vec[n].item() if hasattr(attn_vec[n], "item") else float(attn_vec[n]))
                new_beams.append((path + [n], vis | {n}, new_score))
        if not new_beams:
            break
        new_beams.sort(key=lambda t: t[2], reverse=True)
        beams = new_beams[:beam]

    candidates = finished + [(p, s) for (p, _vis, s) in beams]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [p for (p, _s) in candidates[:k]]

def _normalize_lines_attr(g):
    """
    With batch_size=1, PyG stores non-tensors as [value]; unwrap and stringify.
    Also join list-of-tokens into a single string per node line.
    """
    lines = getattr(g, "lines", None)
    if lines is None:
        return None
    if isinstance(lines, (list, tuple)) and len(lines) == 1 and isinstance(lines[0], (list, tuple)):
        lines = lines[0]
    norm = []
    for s in lines:
        if isinstance(s, (list, tuple)):
            norm.append(" ".join(map(str, s)))
        else:
            norm.append(str(s))
    return norm

def _normalize_src_attr(g):
    """
    Unwrap src to a plain string: ["file.c"] -> "file.c"
    """
    src = getattr(g, "src", "")
    if isinstance(src, (list, tuple)) and len(src) == 1 and isinstance(src[0], str):
        return src[0]
    return src if isinstance(src, str) else str(src)

@torch.no_grad()
def node_attention(model, batch, device):
    model.eval()
    batch = batch.to(device)
    logits, attn = model.forward_batch(batch)
    return logits.sigmoid().cpu(), attn  # list of [n_g] tensors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="work/pyg_diversevul_tf")
    ap.add_argument("--model", required=True, help="work/models/gcbert_gnn_attn.pt")
    ap.add_argument("--cfg",   required=True, help="work/models/gcbert_gnn_attn.cfg.json")
    ap.add_argument("--out",   default="work/chains/sample_chains.jsonl")
    ap.add_argument("--num_graphs", type=int, default=20)
    ap.add_argument("--max_len", type=int, default=8)
    ap.add_argument("--topk_nodes", type=int, default=10)
    ap.add_argument("--k_paths", type=int, default=3)
    ap.add_argument("--clean_out", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.clean_out and os.path.exists(args.out):
        os.remove(args.out)

    # load graphs (silence harmless FutureWarning)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`.*", category=FutureWarning)
        graphs = load_shards(args.data_dir)[:args.num_graphs]

    loader = DataLoader(graphs, batch_size=1, shuffle=False)

    cfg = json.load(open(args.cfg, "r", encoding="utf-8"))
    in_dim, hidden, attn_hidden = cfg["in_dim"], cfg["hidden"], cfg["attn_hidden"]

    device = torch.device("cpu")
    model = GNNSeedAttention(in_dim, hidden=hidden, attn_hidden=attn_hidden)
    try:
        state = torch.load(args.model, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    with open(args.out, "w", encoding="utf-8") as fw:
        for batch in loader:
            probs, attn_list = node_attention(model, batch, device)
            g = batch
            attn_vec = attn_list[0]
            seed = int(g.seed_idx[0].item())

            # unwrap attrs
            lines = _normalize_lines_attr(g)
            src_str = _normalize_src_attr(g)

            # top-k nodes
            topk = min(args.topk_nodes, attn_vec.numel())
            idxs = torch.topk(attn_vec, k=topk).indices.tolist()
            top_nodes = []
            for i in idxs:
                code = lines[i] if (lines and i < len(lines)) else "<no_line_available>"
                top_nodes.append([int(i), float(attn_vec[i]), code])

            # paths with extra metrics
            paths = best_paths_from_seed(g.edge_index, attn_vec, seed_idx=seed, k=args.k_paths, max_len=args.max_len)
            paths_readable = []
            for path in paths:
                parts = []
                for i in path:
                    code = lines[i] if (lines and i < len(lines)) else "<no_line_available>"
                    parts.append({"idx": int(i), "attn": float(attn_vec[i]), "code": code})
                score_sum = sum(p["attn"] for p in parts)
                chain_len = len(parts)
                conf_mean = (score_sum / chain_len) if chain_len else 0.0
                chain_str = " -> ".join(f"{p['idx']}: {p['code']}" for p in parts)
                paths_readable.append({
                    "score_sum": score_sum,
                    "confidence_mean_attn": conf_mean,
                    "chain_len": chain_len,
                    "chain": parts,
                    "chain_string": chain_str
                })

            fw.write(json.dumps({
                "src": src_str,
                "prob_vuln": float(probs[0].item()),
                "seed_idx": seed,
                "top_nodes": top_nodes,
                "paths": paths_readable
            }) + "\n")

    print(f"[ok] wrote {args.out}")

if __name__ == "__main__":
    main()
