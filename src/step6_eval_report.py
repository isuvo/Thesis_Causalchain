# src/step6_eval_report.py
import argparse, os, json, glob, warnings
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

# --- safe shard loader (torch>=2.6) ---
def load_shards_safe(data_dir):
    import torch_geometric.data as pyg_data
    files = sorted(glob.glob(os.path.join(data_dir, "diversevul_tf_*.pt")))
    assert files, f"No shards found in {data_dir}"
    graphs = []
    try:
        from torch.serialization import safe_globals
        allow = []
        for mod in (pyg_data, getattr(pyg_data, "data", None)):
            if mod is None: continue
            for name in ("Data", "DataEdgeAttr"):
                if hasattr(mod, name):
                    allow.append(getattr(mod, name))
        for f in files:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                    message="You are using `torch.load` with `weights_only=False`.*",
                    category=FutureWarning,
                )
            with safe_globals(allow):
                graphs.extend(torch.load(f, map_location="cpu", weights_only=False))
    except Exception:
        for f in files:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                    message="You are using `torch.load` with `weights_only=False`.*",
                    category=FutureWarning,
                )
            graphs.extend(torch.load(f, map_location="cpu", weights_only=False))
    return graphs

def split_by_src(graphs, train=0.8, val=0.1, seed=7):
    import random
    rng = random.Random(seed)
    by_src = defaultdict(list)
    for i, g in enumerate(graphs):
        by_src[getattr(g, "src", "")].append(i)
    srcs = list(by_src.keys()); rng.shuffle(srcs)
    n = len(srcs); n_tr = int(train*n); n_va = int(val*n)
    trS = set(srcs[:n_tr]); vaS = set(srcs[n_tr:n_tr+n_va]); teS = set(srcs[n_tr+n_va:])
    def idxs(S):
        out = []
        for s, li in by_src.items():
            if s in S: out.extend(li)
        return out
    return idxs(trS), idxs(vaS), idxs(teS)

# robust model import
try:
    from .step4_train_gnn_attn import GNNSeedAttention
except ImportError:
    from step4_train_gnn_attn import GNNSeedAttention

@torch.no_grad()
def infer_all(model, loader, device):
    model.eval()
    probs, ys, srcs = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model.forward_batch(batch)
        p = torch.sigmoid(logits).view(-1).cpu().tolist()
        y = batch.y.view(-1).cpu().tolist()
        # unwrap src attr
        for i in range(len(p)):
            s = getattr(batch, "src", "")
            if isinstance(s, (list, tuple)) and len(s)==len(p):
                si = s[i]
                if isinstance(si, (list, tuple)) and len(si)==1: si = si[0]
                srcs.append(str(si))
            else:
                srcs.append(str(s))
        probs.extend(p); ys.extend(y)
    return probs, ys, srcs

def compute_metrics(probs, ys, thr=0.5):
    import math
    from math import isfinite
    import numpy as np
    y_true = np.array(ys, dtype=int)
    y_prob = np.array(probs, dtype=float)
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred==1)&(y_true==1)).sum())
    tn = int(((y_pred==0)&(y_true==0)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    acc = (tp+tn)/max(len(y_true),1)
    prec = tp/max(tp+fp,1)
    rec = tp/max(tp+fn,1)
    f1 = 2*prec*rec/max(prec+rec,1e-9)
    # optional: AUCs via sklearn if available
    roc_auc = pr_auc = None
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pass
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=acc, prec=prec, rec=rec, f1=f1, roc_auc=roc_auc, pr_auc=pr_auc)

def best_threshold(probs, ys):
    import numpy as np
    best = {"thr":0.5,"f1":-1}
    for thr in np.linspace(0.05, 0.95, 19):
        m = compute_metrics(probs, ys, thr)
        if m["f1"] > best["f1"]:
            best = {"thr": float(thr), "f1": float(m["f1"])}
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out_dir", default="work/eval")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    graphs = load_shards_safe(args.data_dir)
    in_dim = graphs[0].x.size(1)

    # split
    idx_tr, idx_va, idx_te = split_by_src(graphs, seed=args.seed)
    test_graphs = [graphs[i] for i in idx_te]
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    # model
    cfg = json.load(open(args.cfg, "r", encoding="utf-8"))
    hidden, attn_hidden = cfg["hidden"], cfg["attn_hidden"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNSeedAttention(in_dim, hidden=hidden, attn_hidden=attn_hidden).to(device)
    try:
        state = torch.load(args.model, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)

    # infer
    probs, ys, srcs = infer_all(model, test_loader, device)
    base = compute_metrics(probs, ys, thr=0.5)
    best = best_threshold(probs, ys)
    tuned = compute_metrics(probs, ys, thr=best["thr"])

    # write report
    rep = {
        "n_test": len(ys),
        "default_thr": 0.5,
        "metrics@0.5": base,
        "best_threshold_by_f1": best,
        "metrics@best_thr": tuned,
    }
    with open(os.path.join(args.out_dir, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    # save predictions
    import csv
    with open(os.path.join(args.out_dir, "test_preds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src","y_true","prob_vuln"])
        for s,y,p in zip(srcs, ys, probs):
            w.writerow([s, int(y), float(p)])

    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
