# tools/make_labels.py
import torch, glob, os
EDGE_TYPES = (
    "AST","CFG","DFG","CALL","ARG2PARAM","RET2CALL","RET2LHS",
    "AST_REV","CFG_REV","DFG_REV","CALL_REV","ARG2PARAM_REV","RET2CALL_REV","RET2LHS_REV"
)

def reachable_mask_from_sources(data, edge_types=EDGE_TYPES, cap=200000):
    N = int(data["node"].num_nodes)
    if N == 0: return torch.zeros(N, dtype=torch.bool)
    src = torch.nonzero(data["node"].source_mask).flatten().tolist()
    if not src: return torch.zeros(N, dtype=torch.bool)

    adj = [[] for _ in range(N)]
    for et in edge_types:
        k = ("node", et, "node")
        if k in data.edge_types:
            ei = data[k].edge_index
            for u, v in zip(ei[0].tolist(), ei[1].tolist()):
                adj[u].append(v); adj[v].append(u)

    from collections import deque
    seen = [False]*N
    q = deque()
    for s in src:
        if 0 <= s < N and not seen[s]:
            seen[s] = True; q.append(s)
    steps = 0
    while q and steps < cap:
        u = q.popleft()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True; q.append(v)
        steps += 1
    return torch.tensor(seen, dtype=torch.bool)

inp = r"work\src\test\hetero_ready"
for p in sorted(glob.glob(os.path.join(inp, "*.pt"))):
    data = torch.load(p)
    reach = reachable_mask_from_sources(data)
    # label only CALL nodes: y=1 iff sink & reachable
    call_mask  = data["node"].call_mask
    sink_mask  = data["node"].sink_mask
    y = (sink_mask & reach) & call_mask
    train_mask = call_mask  # simple: train on all CALL nodes
    data["node"].y = y.to(torch.long)
    data["node"].train_mask = train_mask
    torch.save(data, p)  # overwrite with labels
    print(os.path.basename(p), f"calls={int(call_mask.sum())} positives={int(y.sum())}")
