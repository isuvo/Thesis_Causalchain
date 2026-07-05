import os, sys, glob, torch

def find_call_labels(g):
  
    call_index = None

    # common canonical layout
    if 'node' in g.node_types:
        # direct y_call on 'node'
        if 'y_call' in g['node']:
            y = g['node']['y_call']
            call_index = g['node'].get('call_idx', g['node'].get('call_index', None))
            return 'node', 'y_call', y, call_index
        # some builds put per-node labels in 'y' and a boolean mask 'is_call'
        if 'y' in g['node'] and 'is_call' in g['node']:
            return 'node', 'y', g['node']['y'], (g['node']['is_call'].nonzero(as_tuple=True)[0])

    # separate 'call' node store
    if 'call' in getattr(g, 'node_types', []):
        if 'y' in g['call']:
            return 'call', 'y', g['call']['y'], None
        if 'y_call' in g['call']:
            return 'call', 'y_call', g['call']['y_call'], None

    # generic sweep for plausible 1-D label tensors
    for ntype in g.node_types:
        for k in list(g[ntype].keys()):
            v = g[ntype][k]
            if torch.is_tensor(v) and v.dim() == 1 and v.dtype in (torch.long, torch.int64, torch.bool):
                # heuristics: treat small-class integers/bools as labels
                if v.numel() > 0 and (v.dtype == torch.bool or int(v.max()) <= 2):
                    return ntype, k, v, None

    return None, None, None, None

def main(root):
    pts = sorted(glob.glob(os.path.join(root, "*.pt")))
    if not pts:
        print(f"[!] no .pt under {root}")
        sys.exit(1)

    ok = 0; empty = 0; bad = 0
    for p in pts:
        try:
            # Your files are locally generated; pickle is fine here.
            g = torch.load(p, map_location='cpu')  # keep default weights_only=False for HeteroData
            store, key, y, call_idx = find_call_labels(g)
            if store is None:
                print(f"{os.path.basename(p)}  NO_LABELS_FOUND")
                bad += 1
                continue

            num_calls = (len(call_idx) if call_idx is not None else (y.numel()))
            pos = int(y.sum().item()) if y.dtype == torch.bool else int((y > 0).sum().item())

            if num_calls == 0:
                print(f"{os.path.basename(p)}  calls=0  labels=0  (ok)")
                empty += 1
            else:
                print(f"{os.path.basename(p)}  store={store}:{key}  calls={num_calls}  positives={pos}")
                ok += 1

        except Exception as e:
            print(f"{os.path.basename(p)}  ERROR: {e}")
            bad += 1

    print(f"\nSummary: ok={ok}  empty_calls={empty}  missing/err={bad}  total={len(pts)}")
    # Exit non-zero only if many are missing; tweak threshold if you like.
    if bad > 0 and bad > 0.01 * len(pts):
        sys.exit(2)

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else r"F:\work\src\train\hetero_ready"
    main(root)
