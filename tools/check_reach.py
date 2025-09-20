import torch
from torch_geometric.data import HeteroData

EDGE_TYPES = (
    "AST","CFG","DFG","CALL","ARG2PARAM","RET2CALL","RET2LHS",
    "AST_REV","CFG_REV","DFG_REV","CALL_REV","ARG2PARAM_REV","RET2CALL_REV","RET2LHS_REV"
)

def reaches_source_to_sink(data, edge_types=EDGE_TYPES, max_steps=200000):
    N = int(data["node"].num_nodes)
    if N == 0:
        return False
    src = torch.nonzero(data["node"].source_mask, as_tuple=False).flatten().tolist()
    snk = set(torch.nonzero(data["node"].sink_mask, as_tuple=False).flatten().tolist())
    if not src or not snk:
        return False
    adj = [[] for _ in range(N)]
    for et in edge_types:
        key = ("node", et, "node")
        if key in data.edge_types:
            ei = data[key].edge_index
            for u, v in zip(ei[0].tolist(), ei[1].tolist()):
                adj[u].append(v); adj[v].append(u)
    from collections import deque
    seen = [False]*N; q = deque([i for i in src if 0 <= i < N])
    for i in q: seen[i] = True
    steps = 0
    while q and steps < max_steps:
        u = q.popleft()
        if u in snk: return True
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True; q.append(v)
        steps += 1
    return False

if __name__ == "__main__":
    for shard in ["shard_0003","shard_0031","shard_0038","shard_0048","shard_0051"]:
        data = torch.load(rf"work\src\test\hetero_ready\{shard}.pt")
        print(shard, reaches_source_to_sink(data))
