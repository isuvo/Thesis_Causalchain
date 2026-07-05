# src/models/ca_gat_acc.py
from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, Linear

__all__ = ["CAGAT_ACC_Model", "ACCGate", "_pick_x"]

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _pick_x(data: HeteroData) -> torch.Tensor:
    """
    Robustly pick a node feature tensor from various shard conventions.
    Tries: x_text, x_tfidf, x, feat, features (under the single node type 'node').
    """
    node = data["node"]
    for k in ("x_text", "x_tfidf", "x", "feat", "features"):
        if hasattr(node, k) and getattr(node, k) is not None:
            return getattr(node, k)
    raise AttributeError("No node features on data['node'] (tried x_text, x_tfidf, x, feat, features)")

def _compute_degrees(data: HeteroData, N: int, device: torch.device):
    """
    Compute out-/in-degree per node over *all* edge types.
    Per-edge-type 'ones' avoids size mismatches across heterogeneous relations.
    """
    outdeg = torch.zeros(N, dtype=torch.long, device=device)
    indeg  = torch.zeros(N, dtype=torch.long, device=device)
    for et in data.edge_types:
        ei = data[et].edge_index.to(device)           # [2, E]
        ones_e = torch.ones(ei.size(1), dtype=torch.long, device=device)
        outdeg.index_add_(0, ei[0], ones_e)
        indeg.index_add_(0, ei[1], ones_e)
    return outdeg, indeg

# ---------------------------------------------------------------------
# ACC (Adaptive Causal Contextualization) gate
# ---------------------------------------------------------------------

class ACCGate(nn.Module):
    """
    Per-hop, per-edge gate g_e in (0,1). Conditions on:
      - hop index h,
      - edge type (embedding),
      - lightweight structure: deg_out(src), deg_in(dst),
      - interprocedural flag for edges {CALL, ARG2PARAM, RET2CALL, RET2LHS}.
    """
    def __init__(self, edge_types: Tuple[Tuple[str, str, str], ...], H: int, hidden: int = 32):
        super().__init__()
        self.edge_types = edge_types
        self.H = H
        self.t2idx = {t: i for i, t in enumerate(edge_types)}
        self.type_emb = nn.Embedding(len(edge_types), hidden)
        self.h_emb    = nn.Embedding(H, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2 + 3, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        # per-edge-type base (init=0 -> sigmoid ~ 0.5 baseline)
        self.type_logit = nn.Parameter(torch.zeros(len(edge_types)))

    def forward(
        self,
        edge_type: Tuple[str, str, str],
        hop_idx: int,
        deg_src: torch.Tensor,          # [E] long
        deg_dst: torch.Tensor,          # [E] long
        interproc_flag: torch.Tensor,   # [E] float
    ) -> torch.Tensor:
        t_idx = torch.full_like(deg_src, self.t2idx[edge_type], dtype=torch.long)
        h_idx = torch.full_like(deg_src, hop_idx, dtype=torch.long)
        te, he = self.type_emb(t_idx), self.h_emb(h_idx)  # [E, Hdim] each
        s = torch.stack([
            torch.tanh(deg_src.float() / 8.0),
            torch.tanh(deg_dst.float() / 8.0),
            interproc_flag.float()
        ], dim=-1)  # [E, 3]
        z = torch.cat([te, he, s], dim=-1)               # [E, 2*Hdim+3]
        base = torch.sigmoid(self.type_logit[self.t2idx[edge_type]])  # scalar in (0,1)
        g = torch.sigmoid(self.mlp(z)).squeeze(-1)                    # [E] in (0,1)
        # Blend base with MLP prediction for stability
        return torch.clamp(0.25 * base + 0.75 * g, 0.0, 1.0)          # [E]

# ---------------------------------------------------------------------
# CA-GAT + ACC model
# ---------------------------------------------------------------------

class CAGAT_ACC_Model(nn.Module):
    """
    Heterogeneous GAT with ACC gates:
      - Lazy input projection (Linear(-1, hidden)) so variable feature widths across shards are OK.
      - H hops of HeteroConv(GATConv); per-hop we compute ACC gates for each edge and
        modulate aggregation. We *log* the per-edge gates so inference can backtrace paths.
      - Output: per-node logits (2 classes by default) and alpha logs {hop: {etype: gates}}
    """
    def __init__(
        self,
        hidden: int = 128,
        heads: int = 4,
        H: int = 3,
        edge_types: Tuple[Tuple[str, str, str], ...] = None,
        num_classes: int = 2,
    ):
        super().__init__()
        assert edge_types and len(edge_types) > 0, "edge_types must be provided"
        self.H = H
        self.heads = heads
        self.edge_types = edge_types
        self.acc = ACCGate(edge_types, H)

        # Lazy projections tolerate variable Din across shards
        self.lin_in  = Linear(-1, hidden)
        self.lin_out = nn.Sequential(
            Linear(-1, hidden), nn.ReLU(), nn.Dropout(0.1),
            Linear(hidden, num_classes)
        )

        # H hops of hetero-GAT
        self.layers = nn.ModuleList()
        for _ in range(H):
            convs = {}
            for et in edge_types:
                # GAT with lazy in_channels for hetero inputs
                convs[et] = GATConv(
                    (-1, -1),
                    hidden // heads,
                    heads=heads,
                    add_self_loops=False,
                    dropout=0.1
                )
            self.layers.append(HeteroConv(convs, aggr="sum"))

    # -------- helper: align features to the initialized input projection --------
    def _align_in_features(self, x0: torch.Tensor) -> torch.Tensor:
        """
        After the first forward, self.lin_in gets initialized with a fixed input width.
        For subsequent graphs with different widths, pad/trim x0 to that learned width.
        If still lazy (in_channels == -1), do nothing.
        """
        in_ch = getattr(self.lin_in, "in_channels", -1)
        try:
            in_ch = int(in_ch)
        except Exception:
            in_ch = -1
        if in_ch == -1:  # still lazy/uninitialized
            return x0
        target = in_ch
        cur = x0.size(1)
        if cur == target:
            return x0
        if cur < target:
            pad = x0.new_zeros(x0.size(0), target - cur)
            return torch.cat([x0, pad], dim=1)
        # cur > target
        return x0[:, :target]

    # -------------------------------- forward -----------------------------------
    def forward(self, data: HeteroData):
        # 1) Input projection (lazy) with on-the-fly width alignment
        x0 = _pick_x(data).to(torch.float)   # ensure float for Linear
        x0 = self._align_in_features(x0)
        N  = x0.size(0)
        x  = self.lin_in(x0)                 # [N, hidden]
        xs = {"node": x}

        # 2) Precompute degrees once on device
        out_alpha: Dict[int, Dict[Tuple[str, str, str], torch.Tensor]] = {}
        outdeg, indeg = _compute_degrees(data, N, x.device)

        # 3) H hops of CA-GAT with ACC gates
        for h, conv in enumerate(self.layers):
            out_dict: Dict[str, torch.Tensor] = {}
            alpha_h: Dict[Tuple[str, str, str], torch.Tensor] = {}

            for et, gconv in conv.convs.items():
                ei = data[et].edge_index.to(x.device)  # [2, E]
                src, dst = ei

                # Interprocedural edges flagged (CALL/ARG2PARAM/RET2CALL/RET2LHS)
                is_inter = 1.0 if et[1] in {"CALL", "ARG2PARAM", "RET2CALL", "RET2LHS"} else 0.0
                inter_flag = torch.full_like(src, is_inter, dtype=torch.float)

                # ACC gate per edge (0..1)
                g = self.acc(et, h, outdeg[src], indeg[dst], inter_flag)  # [E], float

                # GAT message passing (no direct edge weights; modulate aggregated dst activations)
                out = gconv((xs[et[0]], xs[et[2]]), ei)                   # [N, hidden]

                # Average gate per destination node (dtype-safe)
                dst_gate = torch.zeros(N, device=x.device, dtype=g.dtype)
                cnt      = torch.zeros(N, device=x.device, dtype=g.dtype)
                ones_e   = torch.ones(dst.size(0), device=x.device, dtype=g.dtype)
                dst_gate.index_add_(0, dst, g)        # sum of gates to each dst
                cnt.index_add_(0, dst, ones_e)        # number of incoming edges per dst
                cnt = cnt.clamp(min=1.0)
                scale = (0.5 + 0.5 * (dst_gate / cnt)).unsqueeze(-1)  # smooth [0.5,1.0]
                out = out * scale

                # Aggregate per destination node type (single 'node' in your graphs)
                out_dict.setdefault(et[2], 0)
                out_dict[et[2]] = out_dict[et[2]] + out

                # Log ACC gates for inference-time causal chain extraction
                alpha_h[et] = g  # [E]

            # Residual + activation per type
            new_xs = xs.copy()
            for k, v in out_dict.items():
                new_xs[k] = F.elu(v + xs.get(k, 0))
            xs = new_xs
            out_alpha[h] = alpha_h

        # 4) Node-level logits + per-edge gate logs
        logits = self.lin_out(xs["node"])  # [N, num_classes]
        return {"logits": logits, "alpha": out_alpha}
