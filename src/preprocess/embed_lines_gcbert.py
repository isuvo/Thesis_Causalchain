# src/preprocess/embed_lines_gcbert.py
import argparse, json, torch, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
try:
    from .transformer_cache import EmbeddingCache
except ImportError:
    from transformer_cache import EmbeddingCache

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    s = (last_hidden_state * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-9)
    return s / d

def collect_unique_lines(slices_dir: Path, limit: int = 0):
    uniq = set()
    files = list(slices_dir.glob("*.json"))
    for jf in tqdm(files, desc="scan"):
        try:
            obj = json.loads(jf.read_text(encoding="utf-8"))
            for s in obj.get("line-contents") or []:
                s = s.strip()
                if s:
                    uniq.add(s)
                    if limit and len(uniq) >= limit:
                        return list(uniq)
        except Exception:
            continue
    return list(uniq)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True)
    ap.add_argument("--model", default="microsoft/graphcodebert-base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=16)   # CPU-friendly
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--limit_unique", type=int, default=0)  # 0 = all
    ap.add_argument("--db", default="work/embeddings/line_cache.sqlite")
    args = ap.parse_args()

    sd = Path(args.slices)
    cache = EmbeddingCache(args.db)

    lines = collect_unique_lines(sd, args.limit_unique)
    keys, found = cache.get_many(args.model, lines)
    todo = [l for l, k in zip(lines, keys) if k not in found]
    if not todo:
        print("[ok] all unique lines already cached"); return

    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModel.from_pretrained(args.model).eval().to(args.device)

    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(todo), args.batch_size), desc="embed"):
            chunk = todo[i:i+args.batch_size]
            enc = tok(chunk, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
            enc = {k: v.to(args.device) for k, v in enc.items()}
            last = mdl(**enc).last_hidden_state
            vec = mean_pool(last, enc["attention_mask"]).float().cpu().numpy()
            out.extend(list(vec))

    cache.put_many(args.model, todo, out)
    print(f"[ok] cached {len(todo)} new lines")

if __name__ == "__main__":
    main()
