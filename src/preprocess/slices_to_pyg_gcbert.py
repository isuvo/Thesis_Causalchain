# src/preprocess/slices_to_pyg_gcbert.py
import argparse, json, math, re, os, shutil
from pathlib import Path
from typing import List, Tuple
from collections import deque

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data

# ---- cache ----------------------------------------------------
try:
    from .transformer_cache import EmbeddingCache
except ImportError:
    from transformer_cache import EmbeddingCache

# ---- simple tokenization (for token-count feature) ------------
OP = re.compile(r"[A-Za-z_]\w+|==|!=|<=|>=|&&|\|\||<<|>>|[{}()\[\].,;:+\-*/%&|^!=<>?~]")
def tokenize(line: str) -> List[str]:
    return OP.findall(line) or [line.strip()]

# ---- structural features --------------------------------------
def degree_features(n: int, edges: List[Tuple[int,int]]):
    indeg = [0]*n; outdeg=[0]*n
    for u,v in edges:
        if 0<=u<n and 0<=v<n:
            outdeg[u]+=1; indeg[v]+=1
    indeg = [math.log1p(x) for x in indeg]
    outdeg= [math.log1p(x) for x in outdeg]
    return indeg, outdeg

def dist_to_seed(n: int, edges, seed_idx: int):
    adj=[[] for _ in range(n)]
    for u,v in edges:
        if 0<=u<n and 0<=v<n:
            adj[u].append(v); adj[v].append(u)
    INF=10**9; dist=[INF]*n
    if 0<=seed_idx<n:
        q=deque([seed_idx]); dist[seed_idx]=0
        while q:
            x=q.popleft()
            for y in adj[x]:
                if dist[y]==INF:
                    dist[y]=dist[x]+1; q.append(y)
    mx=max([d for d in dist if d<INF]+[1])
    return [ (d/mx if d<INF else 1.0) for d in dist ]

# ---- GraphCodeBERT embedder w/ on-demand + persistent cache ---
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    s = (last_hidden_state * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-9)
    return s / d

class OnDemandEmbedder:
    def __init__(self, model_name="microsoft/graphcodebert-base", device="cpu", max_len=320):
        self.model_name = model_name
        self.device = device
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mdl = AutoModel.from_pretrained(model_name).eval().to(device)

    def embed(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            last = self.mdl(**enc).last_hidden_state
            vec = mean_pool(last, enc["attention_mask"]).float().cpu().numpy()
            return vec  # [B, 768]

def lookup_with_cache(lines: List[str], cache: EmbeddingCache, embedder: OnDemandEmbedder) -> np.ndarray:
    keys, found = cache.get_many(embedder.model_name, lines)
    miss_idx = [i for i,k in enumerate(keys) if k not in found]
    if miss_idx:
        todo = [lines[i] for i in miss_idx]
        vecs = embedder.embed(todo)
        cache.put_many(embedder.model_name, todo, list(vecs))
        for k, v in zip([keys[i] for i in miss_idx], vecs):
            found[k] = v
    dim = next(iter(found.values())).shape[0]
    X = np.zeros((len(lines), dim), dtype=np.float32)
    for i,k in enumerate(keys):
        X[i] = found[k]
    return X

# ---- build one PyG Data ---------------------------------------
def build_data(obj: dict, cache: EmbeddingCache, embedder: OnDemandEmbedder, sensi: set|None, store_lines: bool):
    lines = obj.get("line-contents") or []
    edges = obj.get("pdg_edges") or []
    if len(lines) < 2 or not edges:
        return None
    n = len(lines)

    Xtxt = lookup_with_cache(lines, cache, embedder)  # [n, 768]
    indeg,outdeg = degree_features(n, edges)
    meta = obj.get("meta", {}) or {}
    seed_idx = meta.get("seed_local_index", -1)
    dseed = dist_to_seed(n, edges, seed_idx if isinstance(seed_idx,int) else -1)
    node_idx = [i/max(n-1,1) for i in range(n)]
    tok_norm = [math.log1p(len(tokenize(s))) for s in lines]
    extra = np.stack([indeg,outdeg,node_idx,dseed,tok_norm], axis=1).astype(np.float32)

    if sensi:
        sensi_flag = np.array([1.0 if any(w in s for w in sensi) else 0.0 for s in lines],
                              dtype=np.float32).reshape(-1,1)
        X = np.hstack([Xtxt, extra, sensi_flag])
    else:
        X = np.hstack([Xtxt, extra])

    E=[]
    for u,v in edges:
        if 0<=u<n and 0<=v<n:
            E.append((u,v)); E.append((v,u))
    if not E: return None

    data = Data(
        x=torch.from_numpy(X),
        edge_index=torch.tensor(E, dtype=torch.long).t().contiguous(),
        y=torch.tensor([int(obj.get("target",0))], dtype=torch.long)
    )
    data.seed_idx = torch.tensor([seed_idx if isinstance(seed_idx,int) and 0<=seed_idx<n else -1], dtype=torch.long)
    data.num_nodes = n
    data.src = meta.get("source_file","")
    if store_lines:
        data.lines = lines
    return data

# ---- main ------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True)
    ap.add_argument("--sensi_file", default="")
    ap.add_argument("--out_dir", default="work/pyg_diversevul_tf")
    ap.add_argument("--cache_db", default="work/embeddings/line_cache.sqlite")
    ap.add_argument("--shard_size", type=int, default=50000)
    ap.add_argument("--max_len", type=int, default=320)     # a little larger than 256
    ap.add_argument("--store_lines", action="store_true")   # <— keep original code lines
    ap.add_argument("--clean_out", action="store_true")     # <— remove old shards first
    args = ap.parse_args()

    sd = Path(args.slices)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    if args.clean_out:
        for p in out.glob("diversevul_tf_*.pt"): p.unlink(missing_ok=True)
        (out/"pyg_transformer_report.txt").unlink(missing_ok=True)

    cache = EmbeddingCache(args.cache_db)
    sensi = set()
    if args.sensi_file and Path(args.sensi_file).exists():
        sensi = {l.strip() for l in Path(args.sensi_file).read_text(encoding="utf-8").splitlines() if l.strip()}

    embedder = OnDemandEmbedder("microsoft/graphcodebert-base", device="cpu", max_len=args.max_len)

    kept=bad=0; shard=0; buf=[]
    files = list(sd.glob("*.json"))
    for jf in files:
        try:
            obj = json.loads(jf.read_text(encoding="utf-8"))
            d = build_data(obj, cache, embedder, sensi if sensi else None, store_lines=args.store_lines)
            if d is None: bad+=1; continue
            buf.append(d); kept+=1
            if len(buf) >= args.shard_size:
                torch.save(buf, out / f"diversevul_tf_{shard:03d}.pt"); buf.clear(); shard+=1
        except Exception:
            bad+=1

    if buf:
        torch.save(buf, out / f"diversevul_tf_{shard:03d}.pt")

    (out / "pyg_transformer_report.txt").write_text(
        f"kept={kept} bad={bad} shards={shard+1}\n", encoding="utf-8"
    )
    print({"kept": kept, "bad": bad, "shards": shard+1, "out_dir": str(out)})

if __name__ == "__main__":
    main()
