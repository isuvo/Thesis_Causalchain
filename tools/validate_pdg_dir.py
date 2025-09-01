# tools/validate_pdg_dir.py
import json, os, glob
from pathlib import Path

SRC = Path("work/diversevul_pdg")          # your PDG dir
DST = Path("work/diversevul_pdg_clean")    # cleaned output (non-destructive)
DST.mkdir(parents=True, exist_ok=True)

tot = ok = fixed = dropped = 0
for jf in glob.glob(str(SRC / "*.json")):
    try:
        obj = json.load(open(jf, encoding="utf-8"))
        lines = obj.get("line-contents") or obj.get("line_contents") or []
        n = len(lines)
        cd = obj.get("control-dependences", [])
        dd = obj.get("data-dependences", [])

        def clamp_edges(edges):
            good = []
            for e in edges:
                if not (isinstance(e, list) and len(e)==2 and all(isinstance(x,int) for x in e)):
                    continue
                u,v = e
                if 0 <= u < n and 0 <= v < n:
                    good.append((u,v))
            return list(sorted(set(good)))

        cd2 = clamp_edges(cd)
        dd2 = clamp_edges(dd)

        changed = (len(cd2)!=len(cd)) or (len(dd2)!=len(dd))
        obj["control-dependences"] = [[u,v] for u,v in cd2]
        obj["data-dependences"]    = [[u,v] for u,v in dd2]
        if "target" not in obj: obj["target"] = 0

        with open(DST / Path(jf).name, "w", encoding="utf-8") as w:
            json.dump(obj, w)

        ok += 1
        if changed:
            fixed += 1
    except Exception:
        dropped += 1
    tot += 1

print({"total": tot, "ok": ok, "fixed_edges": fixed, "failed_to_parse": dropped,
       "dst": str(DST)})
