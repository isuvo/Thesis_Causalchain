import torch
from src.models.ca_gat_acc import CAGAT_ACC_Model, _pick_x

@torch.no_grad()
def has_acc_params(model):
    return any("type_logit" in n for n,_ in model.named_parameters())

def test_grad_flow(sample):
    model = CAGAT_ACC_Model(in_dim=_pick_x(sample).size(-1), hidden=64, heads=2, H=2, edge_types=tuple(sample.edge_types))
    logits = model(sample)["logits"]
    y = torch.zeros(logits.size(0), dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    # PROOF 1: ACC parameters receive gradients
    acc_grads = [p.grad.norm().item() for n,p in model.named_parameters() if "acc" in n or "type_logit" in n]
    assert any(g>0 for g in acc_grads), "ACC parameters did not receive gradients"
    # PROOF 2: alpha masks exist and are in (0,1)
    out = model(sample)
    for h, d in out["alpha"].items():
        for g in d.values():
            assert (g>=0).all() and (g<=1).all()
