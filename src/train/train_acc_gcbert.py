# src/train/train_acc_gcbert.py
import os, glob, json, argparse, atexit, traceback
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData
from src.models.ca_gat_acc import CAGAT_ACC_Model

try:
    import faulthandler; faulthandler.enable()
except Exception:
    pass

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k): return x

def coerce_hetero(obj):
    if isinstance(obj, HeteroData): return obj
    raise ValueError("Expected HeteroData")

class ShardDS(Dataset):
    def __init__(self, folder):
        self.paths = sorted(glob.glob(os.path.join(folder, "*.pt")))
        if not self.paths:
            raise FileNotFoundError(f"No .pt under {folder}")
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return coerce_hetero(torch.load(self.paths[i], map_location="cpu"))

def collate(b): return b[0]

def build_model(sample: HeteroData, H: int):
    edge_types = tuple(sample.edge_types)
    return CAGAT_ACC_Model(hidden=128, heads=4, H=H, edge_types=edge_types, num_classes=2)

def train_loop(model, dl, device, out_dir, save_every=100, max_steps=None, verbose=False):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "ckpt_acc.pt")
    logp = os.path.join(out_dir, "train_log.json")
    errp = os.path.join(out_dir, "fatal_error.log")

    model.train()
    opt  = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    steps, total = 0, 0.0
    pbar = tqdm(dl, total=(max_steps or len(dl)), desc="Training", unit="shard", leave=True)

    def save(tag="SAVE"):
        torch.save({"model": model.state_dict()}, ckpt)
        with open(logp, "w") as f:
            json.dump({"loss_avg": (total / max(1, steps)), "steps": steps}, f, indent=2)
        pbar.write(f"[{tag}] -> {ckpt}")

    # save on normal exit too (if possible)
    atexit.register(lambda: os.path.exists(out_dir) and save("ATEXIT"))

    try:
        for i, data in enumerate(pbar, 1):
            data = data.to(device)
            out = model(data)
            logits = out["logits"]
            y = getattr(data["node"], "y", None)
            if y is None: y = torch.zeros(logits.size(0), dtype=torch.long, device=device)
            elif y.ndim > 1: y = y.argmax(-1)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()

            steps += 1
            total += float(loss.detach().cpu())
            if verbose:
                pbar.set_postfix(step=steps, N=int(logits.size(0)), loss=f"{total/steps:.5f}")

            if save_every and steps % save_every == 0:
                save("PERIODIC SAVE")

            if max_steps and steps >= max_steps:
                break

    except Exception as e:
        with open(errp, "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        pbar.write(f"[FATAL] {type(e).__name__}: {e}. See {errp}")
        save("FINAL (EXC)")
        raise
    finally:
        save("FINAL")

def main(train_dir, logdir, H=3, save_every=100, max_steps=None, verbose=False):
    paths = sorted(glob.glob(os.path.join(train_dir, "*.pt")))
    if not paths: raise FileNotFoundError(f"No .pt in {train_dir}")
    sample = coerce_hetero(torch.load(paths[0], map_location="cpu"))
    model = build_model(sample, H)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dl = DataLoader(ShardDS(train_dir), batch_size=1, shuffle=True, collate_fn=collate)
    train_loop(model, dl, device, logdir, save_every=save_every, max_steps=max_steps, verbose=verbose)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    main(args.train_dir, args.logdir, args.H, args.save_every, args.max_steps, args.verbose)
