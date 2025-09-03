# src/step4_train_gnn_attn.py
import argparse, glob, json, os, random, shutil
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, BatchNorm
import warnings


def load_shards(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "diversevul_tf_*.pt")))
    assert files, f"No shards found in {data_dir}"
    graphs = []
    for f in files:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`.*",
                category=FutureWarning,
            )
            graphs.extend(torch.load(f, map_location="cpu"))
    return graphs

def load_shards(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "diversevul_tf_*.pt")))
    assert files, f"No shards found in {data_dir}"
    graphs = []
    for f in files:
        graphs.extend(torch.load(f))
    return graphs

def split_by_src(graphs, train=0.8, val=0.1, seed=42):
    rng = random.Random(seed)
    by_src = defaultdict(list)
    for i,g in enumerate(graphs):
        s = getattr(g, "src", "")
        by_src[s].append(i)
    srcs = list(by_src.keys()); rng.shuffle(srcs)
    n = len(srcs); n_train = int(train*n); n_val = int(val*n)
    train_srcs = set(srcs[:n_train]); val_srcs=set(srcs[n_train:n_train+n_val]); test_srcs=set(srcs[n_train+n_val:])
    idx_train= []; idx_val=[]; idx_test=[]
    for s, idxs in by_src.items():
        if s in train_srcs: idx_train.extend(idxs)
        elif s in val_srcs: idx_val.extend(idxs)
        else: idx_test.extend(idxs)
    return Subset(graphs, idx_train), Subset(graphs, idx_val), Subset(graphs, idx_test)

def class_pos_weight(ds):
    pos = sum(int(ds[i].y.item()==1) for i in range(len(ds)))
    neg = len(ds) - pos
    return torch.tensor([(neg / max(pos,1))], dtype=torch.float32)

# -----------------------
# Model
# -----------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

class GNNSeedAttention(nn.Module):
    """
    GNN encoder + seed-conditioned node attention.
    - 3× GIN layers + BN (more expressive for PDGs)
    - Attention: score_i = LeakyReLU( w^T tanh(W1 h_i + W2 h_seed) ), softmax over nodes.
    - Graph logit = Linear( sum_i alpha_i * h_i ).
    """
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

    def forward_batch(self, batch):
        x, edge_index, graph_id = batch.x, batch.edge_index, batch.batch
        h = F.relu(self.proj_in(x))
        for conv, bn in zip(self.gnns, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)

        B = int(graph_id.max().item()) + 1
        logits = []; attn_all = []
        for gid in range(B):
            mask = (graph_id == gid)
            h_g = h[mask]
            seed_local = int(batch.seed_idx[gid].item()) if hasattr(batch,"seed_idx") else -1
            h_seed = h_g[seed_local] if 0 <= seed_local < h_g.size(0) else h_g.mean(dim=0)

            z = torch.tanh(self.W1(h_g) + self.W2(h_seed))
            e = self.leaky(self.w(z)).squeeze(-1)          # [n_g]
            alpha = torch.softmax(e, dim=0)                 # [n_g]
            readout = (alpha.unsqueeze(-1) * h_g).sum(dim=0)
            logits.append(self.cls(readout))
            attn_all.append(alpha.detach().cpu())
        return torch.vstack(logits), attn_all

# -----------------------
# Train / Eval
# -----------------------
def train_one_epoch(model, loader, optim, loss_fn, device, max_norm=1.0):
    model.train(); tot=0.0
    for batch in loader:
        batch = batch.to(device)
        optim.zero_grad()
        logits, _ = model.forward_batch(batch)
        y = batch.y.float().view(-1,1)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optim.step()
        tot += loss.item()*y.size(0)
    return tot / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval(); tot=0.0; correct=0; N=0
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model.forward_batch(batch)
        y = batch.y.float().view(-1,1)
        loss = loss_fn(logits, y)
        tot += loss.item()*y.size(0)
        pred = (torch.sigmoid(logits) >= 0.5).long().view(-1)
        correct += (pred == y.long().view(-1)).sum().item()
        N += y.size(0)
    return tot/len(loader.dataset), correct/max(N,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="work/pyg_diversevul_tf")
    ap.add_argument("--out_dir", default="work/models")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--attn_hidden", type=int, default=192)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--clean_out", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.clean_out:
        for f in Path(args.out_dir).glob("*"): f.unlink(missing_ok=True)

    torch.manual_seed(args.seed); random.seed(args.seed)

    graphs = load_shards(args.data_dir); assert len(graphs)>0
    in_dim = graphs[0].x.size(1)
    train_ds, val_ds, test_ds = split_by_src(graphs, train=0.8, val=0.1, seed=args.seed)
    pos_w = class_pos_weight(train_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cpu")
    model = GNNSeedAttention(in_dim, hidden=args.hidden, attn_hidden=args.attn_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))

    # warmup + cosine
    warmup_epochs = max(2, min(5, args.epochs//5))
    total_steps = args.epochs
    sched1 = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_epochs)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[sched1, sched2], milestones=[warmup_epochs])

    best = {"val_loss": 1e9}
    cfg_path = os.path.join(args.out_dir, "gcbert_gnn_attn.cfg.json")
    ckpt_path = os.path.join(args.out_dir, "gcbert_gnn_attn.pt")

    for ep in range(1, args.epochs+1):
        tr = train_one_epoch(model, train_loader, opt, loss_fn, device)
        vl, vacc = eval_epoch(model, val_loader, loss_fn, device)
        sched.step()
        print(f"[{ep:02d}] train_loss={tr:.4f}  val_loss={vl:.4f}  val_acc={vacc:.3f}  lr={opt.param_groups[0]['lr']:.2e}")

        if vl < best["val_loss"]:
            best.update({"val_loss": vl, "epoch": ep})
            torch.save(model.state_dict(), ckpt_path)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({"in_dim": in_dim, "hidden": args.hidden, "attn_hidden": args.attn_hidden}, f)

    # final test
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    tl, tacc = eval_epoch(model, test_loader, loss_fn, device)
    print(f"[TEST] loss={tl:.4f} acc={tacc:.3f}")

if __name__ == "__main__":
    main()
