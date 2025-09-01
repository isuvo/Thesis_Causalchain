from __future__ import annotations
import json, ast
from pathlib import Path
from typing import Any, Dict, List, Tuple

def _decode_bytes(p: Path) -> str:
    raw = p.read_bytes()
    return raw.decode("utf-8", errors="ignore")

def load_json_safe(p: Path) -> Any:
    """UTF-8 tolerant JSON loader."""
    txt = _decode_bytes(p)
    return json.loads(txt)

def _coerce_edges(val, n: int) -> List[List[int]]:
    """Accept lists, dicts {u,v}, or strings like '[3,7]'. Clamp to 0..n-1 and dedupe."""
    out: List[List[int]] = []
    if isinstance(val, dict):
        iterable = val.values()
    elif isinstance(val, list):
        iterable = val
    else:
        return out
    for e in iterable:
        try:
            if isinstance(e, str):
                e = ast.literal_eval(e)
            if isinstance(e, dict) and "u" in e and "v" in e:
                u, v = int(e["u"]), int(e["v"])
            else:
                u, v = int(e[0]), int(e[1])
            if 0 <= u < n and 0 <= v < n:
                out.append([u, v])
        except Exception:
            continue
    # dedupe
    seen = set()
    dedup = []
    for u, v in out:
        if (u, v) not in seen:
            seen.add((u, v))
            dedup.append([u, v])
    return dedup

def normalize_pdg(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Ensure we get:
      - line-contents: List[str]
      - control-dependences: List[[u,v]]
      - data-dependences: List[[u,v]]
      - target: int (default 0)
    Returns (normalized_obj, "ok" | reason)
    """
    try:
        lines = obj.get("line-contents") or obj.get("line_contents")
        if not isinstance(lines, list) or not all(isinstance(x, str) for x in lines):
            return obj, "bad_line_contents"
        n = len(lines)

        cdg_raw = (
            obj.get("control-dependences")
            or obj.get("controlDependencies")
            or obj.get("cdgEdges")
            or []
        )
        ddg_raw = (
            obj.get("data-dependences")
            or obj.get("dataDependencies")
            or obj.get("ddgEdges")
            or []
        )

        cdg = _coerce_edges(cdg_raw, n)
        ddg = _coerce_edges(ddg_raw, n)

        obj["line-contents"] = lines
        obj["control-dependences"] = cdg
        obj["data-dependences"] = ddg
        obj["target"] = int(obj.get("target", 0))
        return obj, "ok"
    except Exception as e:
        return obj, f"exc:{type(e).__name__}"
