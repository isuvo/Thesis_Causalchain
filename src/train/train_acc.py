# src/train/train_acc.py
import os, glob, json, argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData
from src.models.ca_gat_acc import CAGAT_ACC_Model

# ---------- robust coercion to HeteroData ----------

def coerce_to_hetero(obj) -> HeteroData:
    if isinstance(obj, HeteroData):
        return obj
    if isinstance(obj, list) and obj:
        return coerce_to_hetero(obj[0])
    if isinstance(obj, dict) and "graphs" in obj and obj["graphs"]:
        return coerce_to_hetero(obj["graphs"][0])
    if isinstance(obj, dict):
        data = HeteroData()
        # node features
        for k in ("x_text","x_tfidf","x","feat","features","node_x"):
            if k in obj:
                data["node"].x = obj[k]
                break
        # labels if present
        for k in ("y_node","y","labels"):
            if k in obj:
                data["node"].y = obj[k]
                break
        # edges: tuple keys or UPPER string keys
        for k,v in list(obj.items()):
            if isinstance(k, tuple) and len(k)==3 and isinstance(v, torch.Tensor) and v.ndim==2:
                data[k].edge_index = v
            elif isinstance(k, str) and k.isupper() and isinstance(v, torch.Tensor) and v.ndim==2:
                data[("node", k, "node")].edge_index = v
        if len(data.edge_types)==0:
            raise ValueError("Could not find any edges in dict PT")
        if not hasattr(data["node"], "x") or data["node"].x is None:
            raise ValueError("Could not find node features (x/x_text/feat) in dict PT")
        return data
    raise ValueError(f"Unsupported PT content: {type(obj)}")

class ShardDataset(Dataset):
    def __init__(self, folder: str):
        self.paths = sorted(glob.glob(os.path.join(folder, "*.pt")))
        if not self.paths:
            raise FileNotFoundError(f"No .pt files under {folder}")
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        obj = torch.load(self.paths[i], map_location="cpu")
        return coerce_to_hetero(obj)

def collate(batch):  # single-graph batches (simplest for PoC)
    return batch[0]

# ---------- training ----------

def build_model_from_sample(sample: HeteroData, H: int) -> CAGAT_ACC_Model:
    edge_types = tuple(sample.edge_types)
    return CAGAT_ACC_Model(hidden=128, heads=4, H=H, edge_types=edge_types, num_classes=2)

def train_loop(model, dl, device, out_dir, save_every=5, max_steps=None, verbose=False):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.abspath(os.path.join(out_dir, "ckpt_acc.pt"))
    log_path  = os.path.abspath(os.path.join(out_dir, "train_log.json"))

    model.train()
    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    total, steps = 0.0, 0

    def save_now(tag="PERIODIC"):
        torch.save({"model": model.state_dict()}, ckpt_path)
        with open(log_path, "w") as f:
            json.dump({"loss_avg": total/max(1,steps), "steps": steps}, f, indent=2)
        print(f"[{tag} SAVE] -> {ckpt_path}", flush=True)

    try:
        for data in dl:
            data = data.to(device)
            out = model(data)
            logits = out["logits"]                     # [N,2]
            y = getattr(data["node"], "y", None)
            if y is None:
                y = torch.zeros(logits.size(0), dtype=torch.long, device=device)
            elif y.ndim > 1:
                y = y.argmax(-1)
            loss = crit(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            steps += 1
            total += float(loss.detach().cpu())
            if verbose:
                print(f"[step {steps}] loss={total/steps:.6f}, N={logits.size(0)}", flush=True)

            if save_every and steps % save_every == 0:
                save_now("PERIODIC")

            if max_steps and steps >= max_steps:
                break
    finally:
        save_now("FINAL")

def main(train_dir: str, logdir: str, H: int = 3, save_every=5, max_steps=None, verbose=False):
        paths = sorted(glob.glob(os.path.join(train_dir, "*.pt")))
        if not paths:
            raise FileNotFoundError(f"No .pt files under {train_dir}")
        sample = coerce_to_hetero(torch.load(paths[0], map_location="cpu"))
        model  = build_model_from_sample(sample, H)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        dl = DataLoader(ShardDataset(train_dir), batch_size=1, shuffle=True, collate_fn=collate)
        train_loop(model, dl, device, logdir, save_every=save_every, max_steps=max_steps, verbose=verbose)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--save_every", type=int, default=5, help="save checkpoint every N steps")
    ap.add_argument("--max_steps", type=int, default=None, help="stop after N steps (for quick PoC)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    main(args.train_dir, args.logdir, args.H, args.save_every, args.max_steps, args.verbose)
