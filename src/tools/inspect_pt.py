# src/tools/inspect_pt.py
import argparse, torch, pprint
from torch_geometric.data import HeteroData

def main(path):
    obj = torch.load(path, map_location="cpu")
    print("PT type:", type(obj))
    if isinstance(obj, HeteroData):
        hd = obj
    elif isinstance(obj, list) and obj:
        hd = obj[0]
    elif isinstance(obj, dict) and "graphs" in obj and obj["graphs"]:
        hd = obj["graphs"][0]
    elif isinstance(obj, dict):
        hd = obj
    else:
        print("Unsupported container")
        return
    # print keys
    if isinstance(hd, HeteroData):
        print("HeteroData node features present:", [k for k in ("x_text","x","feat","features") if hasattr(hd['node'], k)])
        print("Num edge types:", len(hd.edge_types))
        for et in list(hd.edge_types)[:8]:
            ei = hd[et].edge_index
            print("  ET:", et, "E=", ei.size(1))
        if hasattr(hd["node"], "x"):
            print("Node feature dim:", hd["node"].x.size(1))
        elif hasattr(hd["node"], "x_text"):
            print("Node feature dim:", hd["node"].x_text.size(1))
    else:
        print("Dict keys:", list(hd.keys())[:20])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    args = ap.parse_args()
    main(args.pt)
