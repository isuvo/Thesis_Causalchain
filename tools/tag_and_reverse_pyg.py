# tools/tag_and_reverse_pyg.py  — tag sources/sinks + add reverse edges (robust + heuristics)
import os, re, glob, json, argparse
from collections import defaultdict

try:
    import torch
    from torch_geometric.data import HeteroData
except Exception:
    torch, HeteroData = None, None

EDGE_KINDS = ["AST","CFG","DFG","CALL","ARG2PARAM","RET2CALL","RET2LHS"]

def load_json(p):  return json.load(open(p, "r", encoding="utf-8"))
def save_json(o,p): os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(o, open(p,"w",encoding="utf-8"), indent=2)

def get_first(d,*ks,default=None):
    for k in ks:
        if isinstance(d,dict) and k in d and d[k] is not None: return d[k]
    return default

def get_node_id(n):
    v = get_first(n, "id","_id","nodeId","_nodeId"); 
    if v is None: return None
    try: return int(v)
    except: 
        try: return int(float(v))
        except: return None

def get_node_label(n):
    return get_first(n, "label","_label","nodeLabel","TYPE","type", default="UNKNOWN")

def normalize_edges(edges, idset):
    out = {}
    for et in EDGE_KINDS:
        kept = []
        for e in edges.get(et, []):
            if not isinstance(e, dict): continue
            s = get_first(e,"src","out","from"); d = get_first(e,"dst","in","to")
            try: s = int(s); d = int(d)
            except: continue
            if s in idset and d in idset: kept.append({"src":s,"dst":d,"label":et})
        out[et] = kept
    return out

def children_map(ast):
    ch = defaultdict(list)
    for e in ast: ch[e["src"]].append(e["dst"])
    return ch

def parent_map(ast):
    par = defaultdict(list)
    for e in ast: par[e["dst"]].append(e["src"])
    return par

def to_heterodata(g):
    if HeteroData is None: return None, None, None
    nodes = [n for n in g.get("nodes",[]) if isinstance(n,dict) and get_node_id(n) is not None]
    ids   = [get_node_id(n) for n in nodes]
    id2ix = {nid:i for i,nid in enumerate(ids)}
    labels= [get_node_label(n) for n in nodes]
    vocab = sorted(set(labels))

    x = torch.zeros((len(ids), len(vocab)), dtype=torch.float32)
    for i, lab in enumerate(labels):
        x[i, vocab.index(lab)] = 1.0

    data = HeteroData()
    data["node"].x = x
    data["node"].num_nodes = len(ids)
    data["node"].node_type_vocab = vocab

    norm_edges = normalize_edges(g.get("edges",{}), set(ids))
    for et, el in norm_edges.items():
        if not el: continue
        src = [id2ix[e["src"]] for e in el]; dst = [id2ix[e["dst"]] for e in el]
        data[("node", et, "node")].edge_index = torch.tensor([src,dst], dtype=torch.long)
    return data, ids, id2ix

def add_reverse_edges(data):
    for (st, et, dt) in list(data.edge_types):
        ei = data[(st, et, dt)].edge_index
        data[(dt, et+"_REV", st)].edge_index = torch.stack((ei[1], ei[0]), dim=0)

def make_masks(g, ids, id2ix, src_rx, sink_rx, params_as_sources):
    import torch
    by_id = {get_node_id(n): n for n in g["nodes"] if isinstance(n,dict) and get_node_id(n) is not None}
    ast   = g["edges"].get("AST", [])
    ast_children = children_map(ast)
    # masks
    N = len(ids)
    is_call  = torch.zeros(N, dtype=torch.bool)
    is_src   = torch.zeros(N, dtype=torch.bool)
    is_sink  = torch.zeros(N, dtype=torch.bool)

    # CALL-based tagging
    for nid in ids:
        n = by_id[nid]; lab = get_node_label(n)
        if lab != "CALL": continue
        i = id2ix[nid]; is_call[i] = True
        name = (get_first(n, "methodFullName","name", default="") or "").lower()
        if src_rx and src_rx.search(name):  is_src[i]  = True
        # sinks: regex OR assignment-to-pointer/char-array lhs
        if sink_rx and sink_rx.search(name): is_sink[i] = True
        if get_first(n,"name") == "<operator>.assignment":
            # find LHS child (order/argumentIndex == 1)
            for ch in ast_children.get(nid, []):
                cn = by_id.get(ch); 
                if not cn: continue
                ordv = get_first(cn,"order","argumentIndex")
                if ordv == 1:
                    tfn = (get_first(cn,"typeFullName","TYPE_FULL_NAME", default="") or "").lower()
                    # heuristics: pointer or char buffer likely
                    if "*" in tfn or "char[" in tfn or "wchar_t[" in tfn:
                        is_sink[i] = True
                    break

    # PARAMETER_IN as sources (weak/agnostic supervision)
    if params_as_sources:
        for nid in ids:
            n = by_id[nid]
            if get_node_label(n) == "METHOD_PARAMETER_IN":
                is_src[id2ix[nid]] = True

    return {"call_mask": is_call, "source_mask": is_src, "sink_mask": is_sink}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="work/src/test/unified_aug")
    ap.add_argument("--out", default="work/src/test/hetero_ready")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--src-rx", default=r"(read|recv|recvfrom|recvmsg|fgets|gets|getline|scanf|fscanf|readline|cin|istream|ifstream|fread)")
    ap.add_argument("--sink-rx", default=r"(memcpy|memmove|strcpy|strcat|sprintf|vsprintf|snprintf|system|popen|execl|execv|execve|strncpy|strncat|write|fwrite|copy|append|assign|insert|push_back|emplace)")
    ap.add_argument("--params-as-sources", action="store_true", help="mark METHOD_PARAMETER_IN nodes as taint sources")
    ap.add_argument("--undirected", action="store_true", help="make every relation undirected instead of adding *_REV")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.inp, "*.json")))[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    print(f"[i] Tagging on {len(files)} file(s)…")

    SRC_RX  = re.compile(args.src_rx.lower())
    SINK_RX = re.compile(args.sink_rx.lower())

    for fp in files:
        g = load_json(fp)
        g.setdefault("edges", {k:[] for k in EDGE_KINDS})

        data, ids, id2ix = to_heterodata(g)
        shard = os.path.splitext(os.path.basename(fp))[0]

        if data is None:
            print(f"{shard}: PyG not installed → skipped .pt"); continue

        masks = make_masks(g, ids, id2ix, SRC_RX, SINK_RX, args.params_as_sources)
        data["node"].call_mask   = masks["call_mask"]
        data["node"].source_mask = masks["source_mask"]
        data["node"].sink_mask   = masks["sink_mask"]

        if args.undirected:
            from torch_geometric.utils import to_undirected
            for et in list(data.edge_types):
                ei = data[et].edge_index
                data[et].edge_index = to_undirected(ei)
        else:
            add_reverse_edges(data)

        out_pt = os.path.join(args.out, f"{shard}.pt")
        torch.save(data, out_pt)

        print(f"{shard}: nodes={int(data['node'].num_nodes)} calls={int(data['node'].call_mask.sum())} "
              f"sources={int(data['node'].source_mask.sum())} sinks={int(data['node'].sink_mask.sum())} "
              f"edge_types={len(data.edge_types)}")

if __name__ == "__main__":
    main()
