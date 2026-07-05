# src/infer/pretty_chain.py
import os, json, argparse
import torch
from torch_geometric.data import HeteroData

def coerce_to_hetero(obj):
    if isinstance(obj, HeteroData): return obj
    if isinstance(obj, list) and obj: return coerce_to_hetero(obj[0])
    if isinstance(obj, dict) and "graphs" in obj and obj["graphs"]: return coerce_to_hetero(obj["graphs"][0])
    if isinstance(obj, dict):
        from torch_geometric.data import HeteroData as HD
        data = HD()
        for k in ("x_text","x_tfidf","x","feat","features","node_x"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                data["node"].x = obj[k]; break
        for k,v in list(obj.items()):
            if isinstance(v, torch.Tensor) and v.ndim==2:
                if isinstance(k, tuple) and len(k)==3: data[k].edge_index = v
                elif isinstance(k, str) and k.isupper(): data[("node",k,"node")].edge_index = v
        for name, val in obj.items():
            if name not in ("x","x_text","x_tfidf","feat","features") and not isinstance(val, torch.Tensor):
                setattr(data["node"], name, val)
        return data
    raise ValueError(f"Unsupported PT type: {type(obj)}")

def guess_mapping_tensor(node_store):
    for cand in ("json_id","orig_id","_id","nid","node_id","raw_id","origid","orig_ids"):
        if hasattr(node_store, cand):
            t = getattr(node_store, cand)
            if isinstance(t, torch.Tensor) and t.numel()==node_store.x.size(0):
                return t
    return None

def safe_get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d: return d[k]
    return default

def main(pt, chains_json, aug_json=None, out_md=None):
    data = torch.load(pt, map_location="cpu")
    data = coerce_to_hetero(data)
    chains = json.load(open(chains_json,"r"))["chains"]
    id_map = guess_mapping_tensor(data["node"])

    id2node = {}
    if aug_json:
        aug = json.load(open(aug_json,"r"))
        id2node = {n["_id"]: n for n in aug.get("nodes", [])}

    def render_node(idx):
        node = data["node"]
        file_str = None; line_no = None; code = None
        for cand in ("filename","file","fname"):
            if hasattr(node, cand):
                arr = getattr(node, cand)
                if isinstance(arr, list) and idx < len(arr): file_str = arr[idx]
        for cand in ("line","lineno","lineNumber"):
            if hasattr(node, cand):
                t = getattr(node, cand)
                if isinstance(t, torch.Tensor) and idx < t.numel(): line_no = int(t[idx])
        if hasattr(node, "code"):
            arr = getattr(node, "code")
            if isinstance(arr, list) and idx < len(arr): code = arr[idx]
        if (file_str is None or code is None or line_no is None) and id_map is not None and id2node:
            json_id = int(id_map[idx].item())
            nj = id2node.get(json_id, {})
            file_str = file_str or safe_get(nj, "filename", "file")
            line_no  = line_no  or safe_get(nj, "lineNumber", "lineno", default=None)
            code     = code     or safe_get(nj, "code", default=None)
        file_str = file_str or "<unknown-file>"
        line_no  = line_no if line_no is not None else -1
        code     = code or "<code unavailable in .pt; pass --aug_json to pretty-print from augmented JSON>"
        snippet = " ".join((code if isinstance(code,str) else str(code)).strip().splitlines()[:2] + (["..."] if isinstance(code,str) and len(code.splitlines())>2 else []))
        return f"{file_str}:{line_no}", snippet

    lines = ["# Causal Chains (root → … → sink)", ""]
    for ci, chain in enumerate(chains):
        lines.append(f"### Chain {ci+1}  (len={len(chain)})")
        for i, nid in enumerate(chain):
            loc, snippet = render_node(nid)
            role = "ROOT" if i==0 else ("SINK" if i==len(chain)-1 else f"hop{i}")
            lines.append(f"- **{role}**  node={nid}  @ `{loc}`")
            lines.append(f"  - `{snippet}`")
        lines.append("")
    out_md = out_md or os.path.splitext(chains_json)[0]+"_pretty.md"
    with open(out_md,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[pretty] wrote {out_md}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--chains", required=True, help="JSON from infer_acc.py or infer_acc_fast.py")
    ap.add_argument("--aug_json", default=None, help="matching augmented JSON for the same shard")
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()
    main(args.pt, args.chains, args.aug_json, args.out_md)
