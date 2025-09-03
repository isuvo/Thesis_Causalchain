# src/infer_one_slice_pretty.py
# Pretty, labeled single-slice inference with clear progress prints.
import argparse, json, os, re, warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, BatchNorm

from transformers import AutoTokenizer, AutoModel

# ---------------- Pretty-print helpers ----------------
SINK_APIS_DEFAULT = [
    "strcpy","strcat","memcpy","memmove","sprintf","vsprintf","gets",
    "scanf","recv","read","system","popen"
]
LENGTH_HINTS = ["strlen","sizeof","strnlen"]
GUARD_HINTS  = ["if","assert","check","validate","verify"]

DECL_PAT    = re.compile(r"^\s*(?:[A-Za-z_][\w:<>]*\s+)+[A-Za-z_]\w*\s*(?:\[[^\]]*\])?(?:\s*=\s*[^;]*)?;\s*$")
BRACE_PAT   = re.compile(r"^\s*[{}]\s*$")
INCLUDE_PAT = re.compile(r"^\s*#\s*include\b")

def classify_line(line: str, sensi:list):
    low = line.lower().strip()
    if not low or BRACE_PAT.match(line) or INCLUDE_PAT.match(line):             return "TRIVIAL"
    if any(api in low for api in sensi):
        return "SINK" if ("strcpy" in low or "sprintf" in low) else "SOURCE"
    if any(h in low for h in LENGTH_HINTS):                                     return "LENGTH"
    if any(h in low for h in GUARD_HINTS) or (" if " in f" {low} "):            return "GUARD"
    if DECL_PAT.match(line):
        if re.search(r"\[\s*(?:8|16|32|64)\s*\]", line):                        return "BUF"
        return "DECL"
    if "(" in line and ");" in line:                                            return "CALL"
    return "OTHER"

def is_trivial_for_topn(label:str, line:str)->bool:
    if label in ("TRIVIAL",): return True
    if label == "DECL": return True
    if re.match(r"^\s*(?:[A-Za-z_]\w*\s+[A-Za-z_]\w*(;\s*))*\s*$", line): return True
    return False

def tag(label:str)->str:
    return {
        "SINK":"[SINK]","SOURCE":"[SOURCE]","GUARD":"[GUARD]",
        "LENGTH":"[LENGTH]","BUF":"[BUF]","CALL":"[CALL]"
    }.get(label,"")

def shorten(s: str, maxlen=90):
    s = s.strip().replace("\n"," ")
    return s if len(s)<=maxlen else s[:maxlen-3]+"..."

# ------------- Model (must match training) -------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

class GNNSeedAttention(nn.Module):
    def __init__(self, in_dim, hidden=192, attn_hidden=192, num_layers=3, dropout=0.2):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden)
        self.gnns = nn.ModuleList()
        self.bns  = nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(hidden, hidden, hidden, dropout)
            self.gnns.append(GINConv(mlp, train_eps=True))
            self.bns.append(BatchNorm(hidden))
        self.dropout = nn.Dropout(dropout)
        # seed-conditioned attention
        self.W1 = nn.Linear(hidden, attn_hidden, bias=False)
        self.W2 = nn.Linear(hidden, attn_hidden, bias=False)
        self.w  = nn.Linear(attn_hidden, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        self.cls = nn.Linear(hidden, 1)

    def forward_graph(self, g: Data):
        x, edge_index = g.x, g.edge_index
        h = F.relu(self.proj_in(x))
        for conv, bn in zip(self.gnns, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)
        h_g = h
        seed = int(getattr(g, "seed_idx", -1))
        h_seed = h_g[seed] if 0<=seed<h_g.size(0) else h_g.mean(dim=0)
        z = torch.tanh(self.W1(h_g) + self.W2(h_seed))
        e = self.leaky(self.w(z)).squeeze(-1)  # [n]
        alpha = torch.softmax(e, dim=0)
        readout = (alpha.unsqueeze(-1) * h_g).sum(dim=0)
        logit = self.cls(readout)
        return logit, alpha

# ------------- Embedding & features (774 = 768 + 6) -------------
def embed_lines_gcbert(lines, model_name="microsoft/graphcodebert-base", device="cpu", local_only=True):
    print("[1/5] Loading GraphCodeBERT tokenizer/model...", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
        mdl = AutoModel.from_pretrained(model_name, local_files_only=local_only)
    except Exception as e:
        if local_only:
            print("  → Local cache not found; trying to download (this can take a while)...", flush=True)
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModel.from_pretrained(model_name)
        else:
            raise
    mdl.to(device).eval()
    outs=[]
    print("[2/5] Embedding lines...", flush=True)
    with torch.no_grad():
        for ln in lines:
            enc = tok(ln, return_tensors="pt", truncation=True, max_length=256).to(device)
            rep = mdl(**enc).last_hidden_state.mean(dim=1).squeeze(0)  # [768]
            outs.append(rep.cpu())
    return torch.stack(outs, dim=0) if outs else torch.zeros((0,768))

def simple_feats(lines, edges, seed_idx):
    n = len(lines)
    deg_in  = [0]*n
    deg_out = [0]*n
    for u,v in edges:
        if 0<=u<n and 0<=v<n:
            deg_out[u]+=1; deg_in[v]+=1
    feats=[]
    for i,ln in enumerate(lines):
        low=ln.lower()
        is_call = 1.0 if ("(" in ln and ");" in ln) else 0.0
        has_num = 1.0 if re.search(r"\d", ln) else 0.0
        is_cond = 1.0 if (" if " in f" {low} " or "while" in low or "for(" in low) else 0.0
        feats.append([
            float(deg_in[i]), float(deg_out[i]),
            1.0 if i==seed_idx else 0.0,
            is_call, has_num, is_cond
        ])
    return torch.tensor(feats, dtype=torch.float32)

def build_pyg_from_slice(obj, device="cpu"):
    print("[3/5] Building PyG graph...", flush=True)
    lines = obj.get("line-contents", [])
    edges = obj.get("pdg_edges", [])
    n = len(lines)
    # bidirectional edges
    seen=set(); e2=[]
    for (u,v) in edges:
        if 0<=u<n and 0<=v<n and (u,v) not in seen:
            e2.append([u,v]); seen.add((u,v))
        if 0<=u<n and 0<=v<n and (v,u) not in seen:
            e2.append([v,u]); seen.add((v,u))
    edge_index = torch.tensor(e2, dtype=torch.long).t().contiguous() if e2 else torch.empty((2,0), dtype=torch.long)

    seed = int(obj.get("meta",{}).get("seed_local_index", -1))
    return lines, e2, edge_index, seed

def greedy_seed_to_root(edge_index, attn, seed_idx, max_len=8):
    """
    Build a greedy causal path: start at seed, at each step pick the
    highest-attention predecessor not already in the path.
    Returns a list of node indices: [seed, ..., root].
    """
    if edge_index.numel() == 0 or seed_idx < 0:
        return [seed_idx]

    src, dst = edge_index  # tensors shape [E]
    # Determine #nodes safely
    max_u = int(src.max().item()) if src.numel() else -1
    max_v = int(dst.max().item()) if dst.numel() else -1
    n_nodes = max(max_u, max_v) + 1

    # Build predecessor lists
    preds = {i: [] for i in range(n_nodes)}
    for u, v in zip(src.tolist(), dst.tolist()):
        preds[v].append(u)

    path = [seed_idx]
    used = {seed_idx}
    cur = seed_idx
    for _ in range(max_len - 1):
        cand = preds.get(cur, [])
        best = None
        best_a = -1.0
        for p in cand:
            a = float(attn[p].item()) if p < len(attn) else 0.0
            if p not in used and a > best_a:
                best, best_a = p, a
        if best is None:
            break
        path.append(best)
        used.add(best)
        cur = best

    return path



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice_json", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--sensi_file", default="src/preprocess/external/sensiAPI.txt")
    ap.add_argument("--max_len", type=int, default=8)
    ap.add_argument("--print_topk", type=int, default=12)
    ap.add_argument("--hide_trivial_topn", action="store_true")
    ap.add_argument("--min_attn", type=float, default=0.03)
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--hf_local_only", action="store_true", help="Only use local HF cache (faster).")
    args = ap.parse_args()

    print("[0/5] Loading slice JSON...", flush=True)
    try:
        raw = Path(args.slice_json).read_text(encoding="utf-8")
        obj = json.loads(raw)
    except UnicodeDecodeError:
        raw = Path(args.slice_json).read_text(encoding="utf-8-sig")
        obj = json.loads(raw)

    try:
        sensi_low = [s.strip().lower() for s in Path(args.sensi_file).read_text(encoding="utf-8").splitlines() if s.strip()]
        if not sensi_low: sensi_low = SINK_APIS_DEFAULT
    except FileNotFoundError:
        sensi_low = SINK_APIS_DEFAULT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lines, e2, edge_index, seed_idx = build_pyg_from_slice(obj, device=device)

    # Embedding + features
    emb = embed_lines_gcbert(lines, device=device, local_only=args.hf_local_only)  # [n,768]
    feats = simple_feats(lines, e2, seed_idx)                                      # [n,6]
    x = torch.cat([emb, feats], dim=1) if emb.size(0)>0 else torch.zeros((0,774))
    g = Data(x=x, edge_index=edge_index)
    g.seed_idx = torch.tensor(seed_idx if 0<=seed_idx<len(lines) else 0, dtype=torch.long)
    g.y = torch.tensor([obj.get("target",0)], dtype=torch.long)

    # Load model
    cfg = json.load(open(args.cfg, "r", encoding="utf-8"))
    hidden = int(cfg.get("hidden", 192))
    attn_hidden = int(cfg.get("attn_hidden", 192))
    model = GNNSeedAttention(in_dim=x.size(1), hidden=hidden, attn_hidden=attn_hidden).to(device)
    print("[4/5] Loading trained GNN...", flush=True)
    try:
        state = torch.load(args.model, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("[5/5] Running inference...", flush=True)
    with torch.no_grad():
        logit, alpha = model.forward_graph(g.to(device))
        prob = torch.sigmoid(logit).item()

    src_name = obj.get("meta",{}).get("source_file", Path(args.slice_json).name)

    # ---------- Top-N ----------
    tops = sorted([(i, float(alpha[i].item())) for i in range(len(lines))], key=lambda x:x[1], reverse=True)
    cleaned=[]
    for i,a in tops:
        lab = classify_line(lines[i], sensi_low)
        if a < args.min_attn: break
        if args.hide_trivial_topn and is_trivial_for_topn(lab, lines[i]): 
            continue
        cleaned.append((i,a,lab,lines[i]))
        if len(cleaned)>=args.print_topk: break

    # ---------- Chain ----------
    path = greedy_seed_to_root(g.edge_index, alpha, int(g.seed_idx.item()), max_len=args.max_len)
    path = [p for i,p in enumerate(path) if p not in path[:i]]  # unique

    verdict = "VULNERABLE" if prob >= args.thr else "likely SAFE"

    # ---------- Print ----------
    print(f"\nSRC: {src_name}")

    print(f"P(vuln): {prob:.4f}  Decision@{args.thr:.2f}: {verdict}")
    print(f"seed_idx: {int(g.seed_idx.item())}")

    print("\nTop nodes (idx, attn, tag):")
    for i,a,lab,ln in cleaned:
        print(f"  {i:3d}  {a:0.3f}  {tag(lab):7s} |  {shorten(ln)}")

    print("\nSeed → Root (causal chain):")
    for idx in path:
        lab = classify_line(lines[idx], sensi_low)
        print(f"  {idx:3d}: {tag(lab):7s} {lines[idx]}")

    print("\nChain (one-line):")
    print("  " + " -> ".join(shorten(lines[i]) for i in path))

    print("\nRoot → Seed (reversed route):")
    for idx in reversed(path):
        lab = classify_line(lines[idx], sensi_low)
        print(f"  {idx:3d}: {tag(lab):7s} {lines[idx]}")

    print("\nRoute (one-line):")
    print("  " + " -> ".join(shorten(lines[i]) for i in reversed(path)))

if __name__ == "__main__":
    # reduce noisy warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main()
