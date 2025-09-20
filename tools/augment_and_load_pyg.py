# tools/augment_and_load_pyg.py  (robust: handles nodes without "id", cleans edges)
import json, os, re, argparse, glob
from collections import defaultdict

try:
    import torch
    from torch_geometric.data import HeteroData
except Exception:
    torch = None
    HeteroData = None

EDGE_KINDS = ["AST","CFG","DFG","CALL","ARG2PARAM","RET2CALL","RET2LHS"]

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def get_first(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default

def get_node_id(n):
    v = get_first(n, "id", "_id", "nodeId", "_nodeId")
    if v is None: return None
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None

def get_node_label(n):
    return get_first(n, "label", "_label", "nodeLabel", "TYPE", "type", default="UNKNOWN")

def as_list(x):
    return x if isinstance(x, list) else []

def build_node_maps(nodes):
    by_id, by_label = {}, defaultdict(list)
    for n in nodes:
        if not isinstance(n, dict):  # skip junk
            continue
        nid = get_node_id(n)
        if nid is None:
            continue
        by_id[nid] = n
        by_label[get_node_label(n)].append(n)
    return by_id, by_label

def normalize_edges(edges, id_set):
    """Keep only edges with src/dst present; normalize src/dst keys."""
    norm = {}
    for et, elist in edges.items():
        kept = []
        for e in as_list(elist):
            if not isinstance(e, dict): 
                continue
            s = get_first(e, "src", "out", "from"); d = get_first(e, "dst", "in", "to")
            try:
                s = int(s); d = int(d)
            except Exception:
                continue
            if s in id_set and d in id_set:
                kept.append({"src": s, "dst": d, "label": et})
        norm[et] = kept
    return norm

def children_map(edges):
    ch = defaultdict(list)
    for e in edges:
        ch[e["src"]].append(e["dst"])
    return ch

def parent_map(edges):
    par = defaultdict(list)
    for e in edges:
        par[e["dst"]].append(e["src"])
    return par

def add_call_and_interproc_edges(graph):
    """Augment with CALL / ARG2PARAM / RET2CALL / RET2LHS using AST+nodes only."""
    nodes = graph["nodes"]
    edges = graph["edges"]
    ast   = edges.get("AST", [])
    by_id, by_label = build_node_maps(nodes)
    ast_children = children_map(ast)
    ast_parents  = parent_map(ast)

    # CALL -> METHOD (resolved by methodFullName / fullName / name)
    call_edges = []
    m_by_fullname = {}
    for m in by_label.get("METHOD", []):
        fn = get_first(m, "fullName", "methodFullName", "name", default=None)
        mid = get_node_id(m)
        if fn and mid is not None:
            m_by_fullname[str(fn)] = mid

    for c in by_label.get("CALL", []):
        c_id = get_node_id(c)
        if c_id is None: 
            continue
        callee_name = get_first(c, "methodFullName", "name", default=None)
        if callee_name and str(callee_name) in m_by_fullname:
            call_edges.append({"src": c_id, "dst": m_by_fullname[str(callee_name)], "label": "CALL"})

    # Build PARAM_IN map: methodId -> {argIndex: paramNodeId}
    params_of_method = defaultdict(dict)
    for p in by_label.get("METHOD_PARAMETER_IN", []):
        pid = get_node_id(p)
        if pid is None:
            continue
        mid = get_first(p, "methodId")
        if mid is None:  # fallback via AST parent METHOD
            parents = ast_parents.get(pid, [])
            mid = parents[0] if parents else None
        if mid is None:
            continue
        i = get_first(p, "order", "index")
        if i is None:
            continue
        try:
            params_of_method[int(mid)][int(i)] = pid
        except Exception:
            continue

    # ARG2PARAM: CALL.argument(i) -> callee.PARAM_IN(i)
    arg2param = []
    for c in by_label.get("CALL", []):
        c_id = get_node_id(c)
        if c_id is None: 
            continue
        callee_name = get_first(c, "methodFullName", "name", default=None)
        mid = m_by_fullname.get(str(callee_name))
        if mid is None:
            continue
        for ch in ast_children.get(c_id, []):
            chn = by_id.get(ch)
            if not chn:
                continue
            arg_idx = get_first(chn, "argumentIndex", "order")
            if arg_idx is None:
                continue
            pid = params_of_method.get(int(mid), {}).get(int(arg_idx))
            if pid:
                arg2param.append({"src": ch, "dst": pid, "label": "ARG2PARAM"})

    # RET2CALL and RET2LHS
    ret_nodes_of_method = {}
    for r in by_label.get("METHOD_RETURN", []):
        rid = get_node_id(r)
        if rid is None:
            continue
        mid = get_first(r, "methodId")
        if mid is None:
            parents = ast_parents.get(rid, [])
            mid = parents[0] if parents else None
        if mid is not None:
            ret_nodes_of_method[int(mid)] = rid

    ret2call, ret2lhs = [], []
    for c in by_label.get("CALL", []):
        c_id = get_node_id(c)
        if c_id is None:
            continue
        callee_name = get_first(c, "methodFullName", "name")
        mid = m_by_fullname.get(str(callee_name))
        if mid is None:
            continue
        ret_id = ret_nodes_of_method.get(int(mid))
        if not ret_id:
            continue
        ret2call.append({"src": ret_id, "dst": c_id, "label": "RET2CALL"})
        # If parent is <operator>.assignment, link to LHS (order=1)
        parents = ast_parents.get(c_id, [])
        if parents:
            p = by_id.get(parents[0])
            if p and get_first(p, "name") == "<operator>.assignment":
                for ch in ast_children.get(get_node_id(p), []):
                    chn = by_id.get(ch)
                    if not chn: 
                        continue
                    ordv = get_first(chn, "order", "argumentIndex")
                    if ordv == 1:
                        ret2lhs.append({"src": ret_id, "dst": ch, "label": "RET2LHS"})
                        break

    edges.setdefault("CALL", []).extend(call_edges)
    edges.setdefault("ARG2PARAM", []).extend(arg2param)
    edges.setdefault("RET2CALL", []).extend(ret2call)
    edges.setdefault("RET2LHS", []).extend(ret2lhs)
    return {"added": {"CALL": len(call_edges), "ARG2PARAM": len(arg2param), "RET2CALL": len(ret2call), "RET2LHS": len(ret2lhs)}}

def to_heterodata(graph):
    if HeteroData is None:
        return None, {"error":"PyTorch Geometric not installed"}
    nodes = as_list(graph.get("nodes", []))
    edges = graph.get("edges", {})

    # robust id + label extraction
    filtered_nodes = [n for n in nodes if isinstance(n, dict) and get_node_id(n) is not None]
    id_list = [get_node_id(n) for n in filtered_nodes]
    id2idx  = {nid:i for i,nid in enumerate(id_list)}
    num_nodes = len(id_list)

    # one-hot node types
    labels = [get_node_label(n) for n in filtered_nodes]
    uniq   = sorted(set(labels))
    import torch
    x = torch.zeros((num_nodes, len(uniq)), dtype=torch.float32)
    for i, lab in enumerate(labels):
        x[i, uniq.index(lab)] = 1.0

    # normalize/clean edges first
    norm_edges = normalize_edges({k: edges.get(k, []) for k in EDGE_KINDS}, set(id_list))

    data = HeteroData()
    data["node"].x = x
    data["node"].num_nodes = num_nodes
    data["node"].node_type_vocab = uniq

    for et, elist in norm_edges.items():
        if not elist:
            continue
        src = [id2idx[e["src"]] for e in elist]
        dst = [id2idx[e["dst"]] for e in elist]
        ei = torch.tensor([src, dst], dtype=torch.long)
        data[("node", et, "node")].edge_index = ei

    return data, {"num_nodes": num_nodes, "edge_types": {k: len(v) for k,v in norm_edges.items()}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="work/src/test/unified_jsons")
    ap.add_argument("--out", default="work/src/test/unified_aug")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.inp, "*.json")))[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f"[i] Found {len(files)} JSON(s). Augmenting and building HeteroData ...")

    summaries = []
    for fp in files:
        g = load_json(fp)

        # Ensure edges dict exists
        g.setdefault("edges", {})
        for et in EDGE_KINDS:
            g["edges"].setdefault(et, [])

        add = add_call_and_interproc_edges(g)

        # Clean edges against node id set (drops dangling endpoints)
        by_id, _ = build_node_maps(as_list(g.get("nodes", [])))
        g["edges"] = normalize_edges(g["edges"], set(by_id.keys()))

        # Save augmented JSON (same shard name) and (optionally) a .pt HeteroData
        shard = os.path.splitext(os.path.basename(fp))[0]
        outp  = os.path.join(args.out, f"{shard}.json")
        g.setdefault("meta", {})
        g["meta"]["cpg_augmented"] = True
        g["meta"]["added_edges"] = add["added"]
        save_json(g, outp)

        hd, stats = to_heterodata(g)
        ok_nodes = isinstance(stats, dict) and stats.get("num_nodes", 0) > 0
        counts   = g.get("meta", {}).get("n_edges", {})
        sanity   = {
            "ok_nodes": ok_nodes,
            "has_AST_or_CFG": (len(g["edges"].get("AST",[]))>0 or len(g["edges"].get("CFG",[]))>0),
            "added": add["added"]
        }
        summaries.append((shard, counts, sanity))

        if hd is not None:
            torch.save(hd, os.path.join(args.out, f"{shard}.pt"))

    for shard, counts, sanity in summaries:
        astc = counts.get("AST", "?"); cfgc = counts.get("CFG","?"); dfgc = counts.get("DFG","?")
        print(f"{shard}: AST={astc} CFG={cfgc} DFG={dfgc} "
              f"CALL+ARG2PARAM+RET={sum(sanity['added'].values())} "
              f"ok_nodes={sanity['ok_nodes']} ok_edges={sanity['has_AST_or_CFG']}")

if __name__ == "__main__":
    main()
