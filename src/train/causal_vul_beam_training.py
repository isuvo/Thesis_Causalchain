#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal-Vul: Node+Path training with learned beam guidance, GraphCodeBERT (optional),
hard-negative mining, path-aligned objectives, pass-through edges, CCS/CFAM,
and per-source calibration. Supports augmented JSON shards and PyG HeteroData .pt files.
"""

import os, sys, json, math, time, random, hashlib, argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: PyG for .pt HeteroData
try:
    from torch_geometric.data import HeteroData
except Exception:
    HeteroData = None

# Optional: HuggingFace for GraphCodeBERT
try:
    from transformers import AutoTokenizer, AutoModel
    HF_OK = True
except Exception:
    HF_OK = False

# ---------------------------
# Repro
# ---------------------------
def set_seed(seed=23):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------------------------
# Small utilities
# ---------------------------
EDGE_TYPES_CANON = ["AST","CFG","DFG","CALL","ARG2PARAM","RET2CALL","RET2LHS","PASS"]  # PASS = cheap inter-proc

def md5_bow(text: str, dim: int, ngram=(3,5)) -> torch.Tensor:
    if not text: return torch.zeros(dim)
    text = str(text); low, high = ngram
    v = torch.zeros(dim)
    s = f"^{text}$"
    for n in range(low, high+1):
        for i in range(0, max(0, len(s)-n+1)):
            g = s[i:i+n]
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % dim] += 1.0
    v = torch.log1p(v)
    return v / (v.norm()+1e-8)

# ---------------------------
# Graph containers
# ---------------------------
@dataclass
class Graph:
    x: torch.Tensor
    y: torch.Tensor                  # [N] in [0,1] (weak OK)
    node_ids: List[int]
    edge_index: Dict[str, torch.Tensor]  # t -> [2,E]
    source_name: str
    split_tag: str
    meta: Dict[str, Any]

# ---------------------------
# GraphCodeBERT embedder (optional)
# ---------------------------
class GCBEmbedder:
    def __init__(self, model_name: str = "microsoft/graphcodebert-base", device="cpu"):
        if not HF_OK:
            raise RuntimeError("transformers not installed")
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mdl = AutoModel.from_pretrained(model_name).to(self.device)
        self.mdl.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size=16, max_len=256) -> torch.Tensor:
        outs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
            h = self.mdl(**enc).last_hidden_state[:,0,:]  # CLS
            outs.append(h.cpu())
        return torch.cat(outs, dim=0)

# ---------------------------
# Loaders: JSON shards & PyG .pt
# ---------------------------
def _canon_edges_from_json(obj, id2idx):
    out = {t: [] for t in EDGE_TYPES_CANON}
    edges = obj.get("edges", {})
    for t in ["AST","CFG","DFG","CALL","ARG2PARAM","RET2CALL","RET2LHS"]:
        for e in edges.get(t, []):
            s = e.get("src") or e.get("from") or e.get("out")
            d = e.get("dst") or e.get("to")   or e.get("in")
            if s in id2idx and d in id2idx:
                out[t].append((id2idx[s], id2idx[d]))
    # cheap PASS edges: copy inter-proc carriers
    for t in ("ARG2PARAM","RET2CALL","RET2LHS"):
        for (u,v) in out[t]:
            out["PASS"].append((u,v))
    return {t:(torch.tensor(v).t().contiguous() if v else torch.empty(2,0,dtype=torch.long))
            for t,v in out.items()}

def _canon_edges_from_pyg(g: "HeteroData") -> Dict[str, torch.Tensor]:
    out = {t:[] for t in EDGE_TYPES_CANON}
    if not hasattr(g, "edge_types"):
        return {t: torch.empty(2,0,dtype=torch.long) for t in EDGE_TYPES_CANON}
    for (s,r,t) in g.edge_types:
        store = g[(s,r,t)]
        ei = getattr(store, "edge_index", None)
        if ei is None and hasattr(store, "adj_t") and store.adj_t is not None:
            row, col, _ = store.adj_t.coo()
            ei = torch.stack([col,row], dim=0)
        if ei is None: continue
        if r in out:
            out[r].append(ei.cpu())
    out = {k: (torch.cat(v, dim=1) if v else torch.empty(2,0,dtype=torch.long)) for k,v in out.items()}
    for r in ("ARG2PARAM","RET2CALL","RET2LHS"):
        if out[r].numel(): out["PASS"] = torch.cat([out["PASS"], out[r]], dim=1) if out["PASS"].numel() else out[r]
    for k in EDGE_TYPES_CANON:
        if k not in out:
            out[k] = torch.empty(2,0,dtype=torch.long)
    return out

def expand_files(maybe_paths: List[str], exts=(".json",".pt")) -> List[str]:
    files = []
    for p in maybe_paths:
        if not p: continue
        if os.path.isdir(p):
            for root,_,fnames in os.walk(p):
                for f in fnames:
                    if f.lower().endswith(exts):
                        files.append(os.path.join(root,f))
        else:
            files.append(p)
    return files

def load_graphs_from_json(paths: List[str], use_gcb=False, gcb: Optional[GCBEmbedder]=None,
                          use_ccs=False, use_cfam=False, split_tag="", src_tag="json") -> List[Graph]:
    graphs = []
    for path in paths:
        with open(path,"r",encoding="utf-8") as f:
            obj = json.load(f)
        nodes = obj.get("nodes", [])
        N = len(nodes)
        node_ids = [int(n.get("_id", i)) for i,n in enumerate(nodes)]
        id2idx = {nid:i for i,nid in enumerate(node_ids)}
        # labels
        y = torch.zeros(N, dtype=torch.float32)
        for i,n in enumerate(nodes):
            for key in ("y","label","vulnerable","vuln","is_vuln","weak_label","P"):
                if key in n:
                    try: y[i] = float(n[key]); break
                    except: pass
        # features (GCB→CLS or hashing fallback)
        texts = [(n.get("code") or n.get("name") or n.get("signature") or "") for n in nodes]
        if use_gcb and gcb is not None:
            x = gcb.encode_texts(texts, batch_size=16, max_len=256)
        else:
            x = torch.stack([md5_bow(t, 768) for t in texts], dim=0)
        # edges
        eidx = _canon_edges_from_json(obj, id2idx)
        meta = obj.get("meta", {})
        # stash any provided path/causal info for supervision & CFAM/CCS
        if "vulnerable_paths" in obj: meta["vulnerable_paths"] = obj["vulnerable_paths"]
        if "causal_nodes" in obj: meta["causal_nodes"] = obj["causal_nodes"]
        if "spurious_nodes" in obj: meta["spurious_nodes"] = obj["spurious_nodes"]
        src = meta.get("source", os.path.basename(path))
        graphs.append(Graph(x=x, y=y, node_ids=node_ids, edge_index=eidx,
                            source_name=str(src), split_tag=split_tag, meta=meta))
    return graphs

def load_graphs_from_pt(paths: List[str], use_gcb=False, gcb: Optional[GCBEmbedder]=None,
                        split_tag="", src_tag="pt") -> List[Graph]:
    if HeteroData is None:
        raise RuntimeError("torch_geometric not available to load .pt graphs")
    graphs = []
    for p in paths:
        g = torch.load(p, map_location="cpu")
        node_types = getattr(g, "node_types", ["node"])
        node_type = "node" if "node" in node_types else node_types[0]
        st = g[node_type]
        feats = []
        if hasattr(st, "x_text") and st.x_text is not None:
            feats.append(st.x_text.float().cpu())
        if hasattr(st, "x") and st.x is not None:
            feats.append(st.x.float().cpu())
        if feats:
            x = torch.cat(feats, dim=1)
        else:
            nid = st.nid.cpu().view(-1).tolist() if hasattr(st,"nid") else list(range(st.num_nodes))
            x = torch.stack([md5_bow(str(i), 256) for i in nid], dim=0)
        y = torch.zeros(x.size(0), dtype=torch.float32)
        if hasattr(st,"y") and st.y is not None:
            y = st.y.float().view(-1).cpu()
        eidx = _canon_edges_from_pyg(g)
        meta = {"source": os.path.basename(p)}
        graphs.append(Graph(x=x, y=y, node_ids=list(range(x.size(0))), edge_index=eidx,
                            source_name=os.path.basename(os.path.dirname(p)) or "pt",
                            split_tag=split_tag, meta=meta))
    return graphs

# ---------------------------
# Model (encoder + RGCN + heads)
# ---------------------------
class MLP(nn.Module):
    def __init__(self, dims: List[int], act=nn.SiLU, dropout=0.1, lazy_first=False):
        super().__init__()
        layers=[]
        for i in range(len(dims)-1):
            if lazy_first and i==0:
                lin = nn.LazyLinear(dims[i+1])
            else:
                lin = nn.Linear(dims[i], dims[i+1])
            layers.append(lin)
            if i < len(dims)-2:
                layers += [act(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class RelLayer(nn.Module):
    def __init__(self, hidden: int, etypes: List[str]):
        super().__init__()
        self.etypes = etypes
        self.self_lin = nn.Linear(hidden, hidden)
        self.msg_lin = nn.ModuleDict({t: nn.Linear(hidden, hidden) for t in etypes})
        self.ln = nn.LayerNorm(hidden)
    def forward(self, h: torch.Tensor, edge_index: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.self_lin(h)
        for t, ei in edge_index.items():
            if ei.numel()==0: continue
            src, dst = ei[0].to(h.device), ei[1].to(h.device)
            m = self.msg_lin[t](h[src])
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            deg = torch.zeros(h.size(0), device=h.device)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
            out = out + agg / (deg.clamp(min=1).unsqueeze(-1))
        return F.silu(self.ln(out))

class EdgeGuidance(nn.Module):
    """Learned per-edge-type guidance (replaces fixed bonuses)."""
    def __init__(self, hidden: int, etypes: List[str]):
        super().__init__()
        self.etypes = etypes
        self.emb = nn.Embedding(len(etypes), hidden)
        self.log_prior = nn.Parameter(torch.tensor([
            0.1 if t=="AST" else 0.2 if t=="CFG" else 0.5 if t in ("CALL","ARG2PARAM","RET2CALL","RET2LHS") else 0.7
            for t in etypes
        ]))
        self.score = MLP([hidden*3, hidden, 1], dropout=0.1)
    def forward(self, h: torch.Tensor, edge_index: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out={}
        for i,t in enumerate(self.etypes):
            ei = edge_index.get(t)
            if ei is None or ei.numel()==0:
                out[t] = torch.empty(0, device=h.device); continue
            src, dst = ei[0].to(h.device), ei[1].to(h.device)
            te = self.emb.weight[i].view(1,-1).expand(src.numel(), -1)
            s = self.score(torch.cat([h[src], h[dst], te], dim=-1)).squeeze(-1) + self.log_prior[i]
            out[t] = s
        return out

class CausalVul(nn.Module):
    def __init__(self, hidden=128, layers=3, etypes: List[str]=EDGE_TYPES_CANON):
        super().__init__()
        self.encoder = MLP([256, hidden], dropout=0.0, lazy_first=True)  # Lazy first fixes dim mismatches
        self.rgcn = nn.ModuleList([RelLayer(hidden, etypes) for _ in range(layers)])
        # Heads
        self.node_head = MLP([hidden, hidden, 1], dropout=0.1)
        self.edge_guidance = EdgeGuidance(hidden, etypes)           # for beam
        self.edge_participation = MLP([hidden*2, hidden, 1], dropout=0.1)  # supervised with path edges
        self.etypes = etypes
    def forward(self, x: torch.Tensor, edge_index: Dict[str, torch.Tensor]):
        h = self.encoder(x)
        for layer in self.rgcn:
            h = layer(h, edge_index)
        node_p = torch.sigmoid(self.node_head(h).squeeze(-1))
        guide_scores = self.edge_guidance(h, edge_index)
        # edge participation score per (u,v)
        part = {}
        for t, ei in edge_index.items():
            if ei.numel()==0: part[t]=torch.empty(0, device=h.device); continue
            src, dst = ei[0].to(h.device), ei[1].to(h.device)
            s = self.edge_participation(torch.cat([h[src], h[dst]], dim=-1)).squeeze(-1)
            part[t] = torch.sigmoid(s)
        return node_p, h, guide_scores, part

# ---------------------------
# Losses & miners
# ---------------------------
class FocalBCELoss(nn.Module):
    def __init__(self, alpha_pos=0.97, gamma=2.0): super().__init__(); self.a=alpha_pos; self.g=gamma
    def forward(self, p, y, w=None):
        p = p.clamp(1e-6, 1-1e-6)
        pt = torch.where(y>0, p, 1-p)
        base = -(y*torch.log(p) + (1-y)*torch.log(1-p))
        foc = ((1-pt)**self.g) * base
        aw = (self.a*(y>0).float() + (1-self.a)*(y<=0).float())
        if w is not None: aw = aw * w
        return (foc*aw).mean()

def class_weights(y: torch.Tensor):
    pos = y.sum(); neg = y.numel()-pos
    w = torch.ones_like(y)
    if pos>0: w[y>0.5] = (neg+1e-6)/(pos+1e-6)
    return w

def hard_negative_indices(p: torch.Tensor, y: torch.Tensor, k: int = 256):
    mask_neg = (y<=0.5)
    if mask_neg.sum()==0: return torch.tensor([], dtype=torch.long, device=p.device)
    scores = p.clone(); scores[~mask_neg] = -1
    k = min(int(k), int(mask_neg.sum().item()))
    idx = torch.topk(scores, k=k).indices
    return idx

def make_path_edge_labels(paths: List[List[int]], edge_index: Dict[str, torch.Tensor], N: int):
    labels = {t: torch.zeros(ei.size(1), dtype=torch.float32) for t,ei in edge_index.items()}
    if not paths: return labels
    edge_sets = {t: set(zip(ei[0].tolist(), ei[1].tolist())) for t,ei in edge_index.items()}
    for path in paths:
        for a,b in zip(path[:-1], path[1:]):
            for t in labels:
                if (a,b) in edge_sets[t]:
                    ei = edge_index[t]
                    m = (ei[0]==a) & (ei[1]==b)
                    labels[t][m] = 1.0
    return labels

def mine_paths_simple(edge_index: Dict[str, torch.Tensor], roots: List[int], sinks: List[int], max_hops=6, limit=8):
    from collections import deque
    sinks = set(sinks)
    order = ["PASS","CALL","DFG","ARG2PARAM","RET2CALL","RET2LHS","CFG","AST"]
    adj = {}
    for t in order:
        ei = edge_index.get(t)
        if ei is None or ei.numel()==0: continue
        for u,v in zip(ei[0].tolist(), ei[1].tolist()):
            adj.setdefault(u, []).append(v)
    paths=[]
    for s in roots:
        q=deque([(s,[s])])
        while q and len(paths)<limit:
            u, path = q.popleft()
            if len(path)-1>max_hops: continue
            if u in sinks and len(path)>1:
                paths.append(path[:]); continue
            for v in adj.get(u, [])[:64]:
                if v in path: continue
                q.append((v, path+[v]))
    return paths

# ---------------------------
# Calibration
# ---------------------------
def platt_fit(p: torch.Tensor, y: torch.Tensor):
    eps=1e-6; p=p.clamp(eps,1-eps); x=torch.log(p/(1-p)).unsqueeze(-1); X=torch.cat([x, torch.ones_like(x)],1)
    y=y.unsqueeze(-1); lam=1e-3
    a= torch.linalg.solve(X.t()@X+lam*torch.eye(2), X.t()@y).squeeze(1)
    a0,a1=float(a[0]), float(a[1])
    if not math.isfinite(a0): a0=1.0
    if not math.isfinite(a1): a1=0.0
    return a0,a1

def platt_apply(p,a,b):
    p=p.clamp(1e-6,1-1e-6)
    return torch.sigmoid(a*torch.log(p/(1-p))+b)

def f1_at(p, y, thr):
    pred = (p >= thr).float()
    tp = (pred*y).sum().item()
    fp = (pred*(1-y)).sum().item()
    fn = ((1-pred)*y).sum().item()
    prec = tp/(tp+fp+1e-9)
    rec  = tp/(tp+fn+1e-9)
    f1   = 2*prec*rec/(prec+rec+1e-9)
    return {"precision":prec,"recall":rec,"f1":f1}

# ---------------------------
# Beam search (learned guidance)
# ---------------------------
@dataclass
class BeamPath:
    score: float
    nodes: List[int]

def run_beam(p, guide, edge_index, seeds, beam_width=32, max_hops=6, alpha_node=0.7):
    look = {}
    for t, ei in edge_index.items():
        if ei.numel()==0: continue
        key={}
        for i in range(ei.size(1)):
            u=int(ei[0,i]); v=int(ei[1,i])
            key.setdefault(u, []).append((v, float(guide[t][i].item())))
        look[t]=key
    beams=[BeamPath(score=float(torch.log(p[s].clamp(1e-9,1-1e-9)).item()), nodes=[s]) for s in seeds]
    finished=[]
    order=["PASS","CALL","DFG","ARG2PARAM","RET2CALL","RET2LHS","CFG","AST"]
    for _ in range(max_hops):
        cand=[]
        for b in beams:
            u=b.nodes[-1]
            for t in order:
                local=look.get(t)
                if not local or u not in local: continue
                for v,es in local[u]:
                    ns = b.score + alpha_node*float(torch.log(p[v].clamp(1e-9,1-1e-9)).item()) + (1-alpha_node)*es
                    cand.append(BeamPath(score=ns, nodes=b.nodes+[v]))
        cand.sort(key=lambda z:z.score, reverse=True)
        beams=cand[:beam_width]
        if not beams: break
        finished.extend(beams[:beam_width//2])
    finished.sort(key=lambda z:z.score, reverse=True)
    uniq=[]; seen=set()
    for bp in finished:
        key=tuple(bp.nodes[-min(4,len(bp.nodes)):])
        if key in seen: continue
        seen.add(key); uniq.append(bp)
        if len(uniq)>=beam_width: break
    return uniq

# ---------------------------
# CCS / CFAM
# ---------------------------
def grad_attribution(model: "CausalVul", g: Graph, device) -> torch.Tensor:
    model.eval()
    x = g.x.to(device).detach().requires_grad_(True)
    p, h, guide, part = model(x, g.edge_index)
    if (g.y>0.5).sum()>0:
        tgt = (p*(g.y.to(device)>0.5).float()).sum()
    else:
        tgt = torch.topk(p, k=min(8, p.numel())).values.sum()
    tgt.backward()
    sal = x.grad.detach().abs().sum(dim=1).cpu()
    return sal

def compute_cfam(model: "CausalVul", g: Graph, causal_nodes: List[int], spurious_nodes: List[int], device) -> float:
    sal = grad_attribution(model, g, device)
    num = float(sal[causal_nodes].sum().item()) if causal_nodes else 0.0
    den = num + float(sal[spurious_nodes].sum().item()) if spurious_nodes else max(1e-9, num)
    return num / max(den, 1e-9)

def compute_ccs(model: "CausalVul", g: Graph, causal_nodes: List[int], device) -> float:
    model.eval()
    x = g.x.to(device)
    with torch.no_grad():
        p0, *_ = model(x, g.edge_index)
        base = float(p0.mean().item())
    x_cf = x.clone()
    if causal_nodes:
        x_cf[torch.tensor(causal_nodes, dtype=torch.long, device=device)] = 0.0
    with torch.no_grad():
        p1, *_ = model(x_cf, g.edge_index)
        cf = float(p1.mean().item())
    return (base - cf)**2

# ---------------------------
# Training / Eval
# ---------------------------
@dataclass
class TrainCfg:
    epochs:int=5; lr:float=2e-4; wd:float=1e-4; hidden:int=128; layers:int=3
    beam_width:int=32; max_hops:int=6; alpha_node:float=0.7
    dataset_weights:Dict[str,float]=None
    thresholds_by_source:Dict[str,float]=None

def metrics_for_graph(p, y, thr):
    pred=(p>=thr).float(); tp=(pred*y).sum().item(); fp=(pred*(1-y)).sum().item(); fn=((1-pred)*y).sum().item()
    prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9); f1=2*prec*rec/(prec+rec+1e-9)
    return {"precision":prec,"recall":rec,"f1":f1}

def train_loop(model: CausalVul, train: List[Graph], valid: List[Graph], cfg: TrainCfg, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    focal = FocalBCELoss(alpha_pos=0.98, gamma=2.0)
    model.to(device)

    for ep in range(1, cfg.epochs+1):
        model.train(); total=0.0
        for g in train:
            x, y = g.x.to(device), g.y.to(device)
            p, h, guide, part = model(x, g.edge_index)
            # 1) Node focal + class weights
            w = class_weights(y).to(device)
            loss_node = focal(p, y, w)
            # Hard-negative OHEM
            idx_hn = hard_negative_indices(p.detach(), y.detach(), k=min(512, y.numel()))
            if idx_hn.numel()>0:
                loss_hn = F.binary_cross_entropy(p[idx_hn].clamp(1e-6,1-1e-6), y[idx_hn], reduction="mean")
            else:
                loss_hn = torch.tensor(0.0, device=device)
            # 2) Path supervision (from meta or weak miner)
            paths = g.meta.get("vulnerable_paths") or g.meta.get("paths") or []
            if not paths:
                roots = torch.topk(p, k=min(8, p.numel())).indices.tolist()
                sinks = roots
                paths = mine_paths_simple(g.edge_index, roots, sinks, max_hops=cfg.max_hops, limit=8)
            # (a) Monotonicity: enforce P to increase along path
            mono = torch.tensor(0.0, device=device)
            for path in paths:
                if len(path)<2: continue
                pv = p[torch.tensor(path, device=device)]
                mono = mono + sum(F.relu(pv[i]-pv[i+1]) for i in range(len(path)-1)) / (len(path)-1)
            mono = 0.1*mono / max(1,len(paths))
            # (b) Edge participation BCE
            edge_lbl = make_path_edge_labels(paths, g.edge_index, x.size(0))
            part_loss = torch.tensor(0.0, device=device)
            for t, lbl in edge_lbl.items():
                if lbl.numel()==0: continue
                lbl = lbl.to(device)
                part_loss = part_loss + F.binary_cross_entropy(part[t].clamp(1e-6,1-1e-6), lbl)
            part_loss = 0.5*part_loss
            # (c) Path ranking: correct path score > corrupted
            pranking = torch.tensor(0.0, device=device)
            for path in paths[:6]:
                if len(path)<2: continue
                logp = torch.log(p[torch.tensor(path, device=device)].clamp(1e-6,1-1e-6)).sum()
                es_list=[]
                for a,b in zip(path[:-1], path[1:]):
                    smax = 0.0
                    for t, ei in g.edge_index.items():
                        if ei.numel()==0: continue
                        m = (ei[0].to(device)==a) & (ei[1].to(device)==b)
                        if m.any(): smax = max(smax, float(guide[t][m].max().item()))
                    es_list.append(smax)
                es = torch.tensor(es_list, device=device).mean() if es_list else torch.tensor(0.0, device=device)
                pos_score = logp + 0.5*es
                if len(path)>=3:
                    corrupt = path[:-1] + [random.choice(path[:-1])]
                else:
                    corrupt = path[:1] + [random.randint(0, x.size(0)-1)]
                logp_c = torch.log(p[torch.tensor(corrupt, device=device)].clamp(1e-6,1-1e-6)).sum()
                pranking = pranking + F.relu(1.0 - (pos_score - logp_c))
            pranking = 0.2*pranking / max(1,len(paths))
            # total
            loss = loss_node + loss_hn + mono + part_loss + pranking
            sw = cfg.dataset_weights.get(g.source_name, cfg.dataset_weights.get("default", 1.0)) if cfg.dataset_weights else 1.0
            loss = loss * sw
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.item())
        # per-source calibration on valid
        thr_by_src={}
        if valid:
            with torch.no_grad():
                coll={}
                for vg in valid:
                    pv, *_ = model(vg.x.to(device), vg.edge_index)
                    coll.setdefault(vg.source_name, {"p":[], "y":[]})
                    coll[vg.source_name]["p"].append(pv.cpu()); coll[vg.source_name]["y"].append(vg.y.cpu())
                for k,xy in coll.items():
                    P=torch.cat(xy["p"]); Y=torch.cat(xy["y"])
                    a,b=platt_fit(P,Y); Pcal=platt_apply(P,a,b); grid=torch.linspace(0.05,0.95,19)
                    best=(0.5,-1.0)
                    for t in grid:
                        m=f1_at(Pcal,Y,float(t))
                        if m["f1"]>best[1]: best=(float(t), m["f1"])
                    thr_by_src[k]=best[0]
        cfg.thresholds_by_source = thr_by_src or cfg.thresholds_by_source
        print(f"[Epoch {ep}/{cfg.epochs}] loss={total/ max(1,len(train)) :.4f}")
    return model, (cfg.thresholds_by_source or {})

def evaluate(model: CausalVul, graphs: List[Graph], cfg: TrainCfg, split_tag: str, device):
    model.eval(); out={"split":split_tag,"per_source":{},"overall":{}}
    with torch.no_grad():
        for g in graphs:
            p, *_ = model(g.x.to(device), g.edge_index)
            thr = cfg.thresholds_by_source.get(g.source_name, 0.5) if cfg.thresholds_by_source else 0.5
            m = metrics_for_graph(p.cpu(), g.y.cpu(), thr)
            bucket = out["per_source"].setdefault(g.source_name, {"n":0,"metrics":[]})
            bucket["n"]+=1; bucket["metrics"].append(m)
    P=R=F1=N=0
    for k,b in out["per_source"].items():
        avg={kk: sum(mm[kk] for mm in b["metrics"])/len(b["metrics"]) for kk in b["metrics"][0]}
        b["avg"]=avg; P+=avg["precision"]; R+=avg["recall"]; F1+=avg["f1"]; N+=1
    if N>0: out["overall"]={"precision":P/N,"recall":R/N,"f1":F1/N}
    return out

# ---------------------------
# CLI / Main
# ---------------------------
def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", default=[], help="Train files (.json or .pt)")
    ap.add_argument("--valid", nargs="*", default=[], help="Valid files")
    ap.add_argument("--test",  nargs="*", default=[], help="Test files")
    ap.add_argument("--train_dir", type=str, default="", help="Directory for train graphs")
    ap.add_argument("--valid_dir", type=str, default="", help="Directory for valid graphs")
    ap.add_argument("--test_dir",  type=str, default="", help="Directory for test graphs")
    ap.add_argument("--use_gcb", action="store_true")
    ap.add_argument("--gcb_model", type=str, default="microsoft/graphcodebert-base")
    ap.add_argument("--gcb_device", type=str, default=None)
    ap.add_argument("--use_ccs", action="store_true")
    ap.add_argument("--use_cfam", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--beam_width", type=int, default=32)
    ap.add_argument("--max_hops", type=int, default=6)
    ap.add_argument("--alpha_node", type=float, default=0.7)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=23)
    return ap.parse_args()

def main():
    args=parse_args(); set_seed(args.seed)
    device=torch.device(args.device)
    # Optional GCB
    gcb=None
    if args.use_gcb:
        use_dev = args.gcb_device if args.gcb_device else args.device
        gcb = GCBEmbedder(args.gcb_model, device=use_dev)

    # Collect files
    train_files = expand_files(args.train) + (expand_files([args.train_dir]) if args.train_dir else [])
    valid_files = expand_files(args.valid) + (expand_files([args.valid_dir]) if args.valid_dir else [])
    test_files  = expand_files(args.test)  + (expand_files([args.test_dir])  if args.test_dir  else [])

    # Split by extension
    tr_json = [p for p in train_files if p.lower().endswith(".json")]
    tr_pt   = [p for p in train_files if p.lower().endswith(".pt")]
    va_json = [p for p in valid_files if p.lower().endswith(".json")]
    va_pt   = [p for p in valid_files if p.lower().endswith(".pt")]
    te_json = [p for p in test_files  if p.lower().endswith(".json")]
    te_pt   = [p for p in test_files  if p.lower().endswith(".pt")]

    # Load graphs
    graphs_train = []
    if tr_json: graphs_train += load_graphs_from_json(tr_json, use_gcb=args.use_gcb, gcb=gcb, use_ccs=args.use_ccs, use_cfam=args.use_cfam, split_tag="train")
    if tr_pt:   graphs_train += load_graphs_from_pt(tr_pt, use_gcb=args.use_gcb, gcb=gcb, split_tag="train")
    graphs_valid = []
    if va_json: graphs_valid += load_graphs_from_json(va_json, use_gcb=args.use_gcb, gcb=gcb, use_ccs=args.use_ccs, use_cfam=args.use_cfam, split_tag="valid")
    if va_pt:   graphs_valid += load_graphs_from_pt(va_pt, use_gcb=args.use_gcb, gcb=gcb, split_tag="valid")
    graphs_test = []
    if te_json: graphs_test += load_graphs_from_json(te_json, use_gcb=args.use_gcb, gcb=gcb, use_ccs=args.use_ccs, use_cfam=args.use_cfam, split_tag="test")
    if te_pt:   graphs_test += load_graphs_from_pt(te_pt, use_gcb=args.use_gcb, gcb=gcb, split_tag="test")

    # Dataset weights (provenance)
    dset_w = {"default":1.0}
    for g in graphs_train: dset_w.setdefault(g.source_name, 1.0)

    cfg = TrainCfg(epochs=args.epochs, lr=args.lr, wd=args.wd, hidden=args.hidden, layers=args.layers,
                   beam_width=args.beam_width, max_hops=args.max_hops, alpha_node=args.alpha_node,
                   dataset_weights=dset_w, thresholds_by_source=None)

    model = CausalVul(hidden=args.hidden, layers=args.layers, etypes=EDGE_TYPES_CANON)

    # ---- Train
    model, thr_by_src = train_loop(model, graphs_train, graphs_valid, cfg, device)

    # ---- Evaluate
    valid_report = evaluate(model, graphs_valid, cfg, split_tag="valid", device=device) if graphs_valid else {}
    test_report  = evaluate(model, graphs_test,  cfg, split_tag="test",  device=device) if graphs_test else {}

    # ---- Demo beam on first train graph
    demo_paths=[]; demo_top_p=[]
    if graphs_train:
        g0 = graphs_train[0]
        with torch.no_grad():
            p, h, guide, part = model(g0.x.to(device), g0.edge_index)
        demo_top_p = torch.topk(p, k=min(10,p.numel())).values.tolist()
        seeds = (g0.y>0.5).nonzero(as_tuple=False).view(-1).tolist() or torch.topk(p, k=min(8,p.numel())).indices.tolist()
        beams = run_beam(p, guide, g0.edge_index, seeds, beam_width=cfg.beam_width, max_hops=cfg.max_hops, alpha_node=cfg.alpha_node)
        demo_paths = [bp.nodes for bp in beams[:10]]

    # ---- CFAM/CCS demo (if requested)
    cfam_demo=None; ccs_demo=None
    if (len(graphs_train)>0):
        g=g0
        provided_causal = g.meta.get("causal_nodes", [])
        provided_spur   = g.meta.get("spurious_nodes", [])
        if demo_paths and not provided_causal:
            causal = demo_paths[0]
            spurious = [i for i in range(g.x.size(0)) if i not in set(causal)]
        else:
            causal = [int(x) for x in provided_causal]
            spurious = [int(x) for x in provided_spur] if provided_spur else [i for i in range(g.x.size(0)) if i not in set(causal)]
        if args.use_cfam:
            cfam_demo = compute_cfam(model, g, causal, spurious, device)
        if args.use_ccs:
            ccs_demo = compute_ccs(model, g, causal, device)

    # ---- Save
    out_dir = os.path.join("out_models", str(int(time.time())))
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir,"model.pt"))
    cfg_json = {
        "device": str(device), "epochs": args.epochs,
        "train_files": train_files, "valid_files": valid_files, "test_files": test_files,
        "hidden": args.hidden, "layers": args.layers,
        "use_gcb": bool(args.use_gcb), "gcb_model": args.gcb_model,
        "use_ccs": bool(args.use_ccs), "use_cfam": bool(args.use_cfam),
        "thresholds_by_source": thr_by_src,
        "valid_overall": valid_report.get("overall", {}),
        "test_overall": test_report.get("overall", {}),
        "demo_top_p_first10": demo_top_p,
        "demo_paths": demo_paths,
        "cfam_demo": cfam_demo,
        "ccs_demo": ccs_demo,
    }
    with open(os.path.join(out_dir,"report.json"),"w",encoding="utf-8") as f:
        json.dump(cfg_json, f, indent=2)
    print("[SAVED]", out_dir)
    return out_dir

if __name__=="__main__":
    main()
