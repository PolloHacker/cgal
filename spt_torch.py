"""Learned GraphSAGE layers and masked soft-target losses (PyTorch, no PyG)."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from spt_learning import SCHEMA, graph_operator


def torch_graph(graph, device):
    matrix = graph_operator(len(graph["X"]), graph["edge_u"], graph["edge_v"], graph["sp_pos"]).tocoo()
    index = torch.from_numpy(np.vstack([matrix.row, matrix.col]).astype(np.int64))
    with torch.sparse.check_sparse_tensor_invariants():
        return torch.sparse_coo_tensor(index, torch.from_numpy(matrix.data.astype(np.float32)),
                                       matrix.shape, device=device, check_invariants=True).coalesce()


class SuperpointGraphSAGE(nn.Module):
    def __init__(self, feature_mean, feature_std, hidden_dim=64):
        super().__init__()
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(feature_std, dtype=torch.float32))
        n = self.feature_mean.numel()
        self.self_1 = nn.Linear(n, hidden_dim)
        self.neighbor_1 = nn.Linear(n, hidden_dim, bias=False)
        self.self_2 = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, 3)

    def forward(self, x, adjacency):
        h = torch.clamp((x-self.feature_mean) / self.feature_std, -5.0, 5.0)
        h = F.relu(self.self_1(h) + self.neighbor_1(torch.sparse.mm(adjacency, h)))
        h = F.relu(self.self_2(h) + self.neighbor_2(torch.sparse.mm(adjacency, h)))
        return self.output(h)

    def export(self, path, feature_names, metadata):
        def array(tensor):
            return tensor.detach().cpu().numpy()
        import json
        values = {"schema": np.asarray(SCHEMA), "feature_names": feature_names,
                  "metadata": np.asarray(json.dumps(metadata)),
                  "feature_mean": array(self.feature_mean), "feature_std": array(self.feature_std),
                  "output_weight": array(self.output.weight), "output_bias": array(self.output.bias)}
        for layer in (1, 2):
            values[f"self_{layer}"] = array(getattr(self, f"self_{layer}").weight)
            values[f"neighbor_{layer}"] = array(getattr(self, f"neighbor_{layer}").weight)
            values[f"bias_{layer}"] = array(getattr(self, f"self_{layer}").bias)
        np.savez_compressed(path, **values)


def lovasz_softmax(probabilities, labels):
    """Multiclass Lovasz extension on known pure nodes; labels -100 are ignored."""
    valid = labels != -100
    probabilities, labels = probabilities[valid], labels[valid]
    terms = []
    for c in range(3):
        foreground = (labels == c).to(probabilities.dtype)
        if not foreground.any():
            continue
        errors = torch.abs(foreground-probabilities[:, c])
        errors, order = torch.sort(errors, descending=True)
        foreground = foreground[order]
        total = foreground.sum()
        intersection = total-foreground.cumsum(0)
        union = total+(1-foreground).cumsum(0)
        gradient = 1-intersection/union
        gradient = torch.cat([gradient[:1], gradient[1:]-gradient[:-1]])
        terms.append(torch.dot(errors, gradient))
    return torch.stack(terms).mean() if terms else probabilities.sum()*0


def masked_purity_loss(logits, targets, purity, counts, class_weights=None, lovasz_weight=0.1,
                       label_smoothing=0.08, kl_clip=2.0):
    """Unknown targets contribute zero loss; impure nodes keep their full soft distribution."""
    valid = (targets.sum(dim=1) > 0) & (purity > 0)
    if not valid.any():
        return logits.sum()*0
    base_weight = purity * torch.sqrt(torch.clamp(counts, min=1))
    weight = base_weight * (targets*class_weights).sum(dim=1) if class_weights is not None else base_weight
    q_target = targets[valid]
    active = q_target.sum(dim=0) > 0
    if int(active.sum()) <= 1:
        active = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
        if logits.shape[-1] == 3 and q_target[:, 2].sum() == 0:
            active[2] = False
    n_active = float(active.sum())
    smooth_prior = active.to(q_target.dtype) / max(n_active, 1.0)
    q_smoothed = (1.0 - label_smoothing) * q_target + label_smoothing * smooth_prior
    log_probs = F.log_softmax(logits[valid], dim=-1)
    per_node_kl = F.kl_div(log_probs, q_smoothed, reduction="none").sum(dim=-1)
    robust_kl = torch.clamp(per_node_kl, max=kl_clip)
    loss = (robust_kl * weight[valid]).sum() / base_weight[valid].sum().clamp_min(1e-8)
    if lovasz_weight:
        # Do not invent hard majority labels for mixed nodes just to use Lovasz.
        pure = valid & (targets.max(dim=1).values >= 1-1e-6) & (purity >= 1-1e-6)
        hard = torch.full((len(logits),), -100, dtype=torch.long, device=logits.device)
        hard[pure] = targets[pure].argmax(dim=1)
        has_multiple_classes = int((q_target.sum(dim=0) > 0).sum()) >= 2
        lw = lovasz_weight if has_multiple_classes else lovasz_weight * 0.2
        loss = loss + lw*lovasz_softmax(logits.softmax(dim=1), hard)
    return loss


def train_graphs(graphs, epochs=60, lr=0.003, hidden_dim=64, device="cpu", seed=42):
    if not graphs or not any(g["valid_mask"].any() for g in graphs):
        raise ValueError("No supervised nodes: supply pure interiors or canopy/CutMix graphs")
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    names = graphs[0]["feature_names"]
    if any(not np.array_equal(g["feature_names"], names) for g in graphs):
        raise ValueError("Training graphs use different feature schemas")
    known = np.vstack([g["X"][g["valid_mask"]] for g in graphs])
    model = SuperpointGraphSAGE(known.mean(axis=0), np.maximum(known.std(axis=0), 0.01), hidden_dim).to(device)
    class_mass = sum((g["histogram"][:, :3].sum(axis=0) for g in graphs), np.zeros(3))
    if class_mass[0] == 0 or class_mass[1] == 0:
        raise ValueError(f"Missing required cavern/vegetation training classes: {np.flatnonzero(class_mass[:2] == 0).tolist()}")
    present = class_mass > 0
    balancing = np.zeros(3, dtype=np.float32)
    balancing[present] = np.clip(class_mass[present].sum() / (present.sum() * class_mass[present]), 0.5, 2.5)
    class_weights = torch.as_tensor(balancing, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tensor_graphs = []
    for graph in graphs:
        tensor_graphs.append((torch.as_tensor(graph["X"], dtype=torch.float32),
                              torch_graph(graph, "cpu"),
                              torch.as_tensor(graph["soft_q"], dtype=torch.float32),
                              torch.as_tensor(graph["purity_weight"], dtype=torch.float32),
                              torch.as_tensor(graph["sp_counts"], dtype=torch.float32)))
    for epoch in range(epochs):
        model.train()
        losses = []
        for idx in np.random.default_rng(seed+epoch).permutation(len(tensor_graphs)):
            x, adjacency, targets, purity, counts = tensor_graphs[idx]
            if device != "cpu":
                x = x.to(device, non_blocking=True)
                adjacency = adjacency.to(device)
                targets = targets.to(device, non_blocking=True)
                purity = purity.to(device, non_blocking=True)
                counts = counts.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_purity_loss(model(x, adjacency), targets, purity, counts, class_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch+1) % 10 == 0 or epoch+1 == epochs:
            print(f"[Epoch {epoch+1}/{epochs}] masked KL + Lovasz: {np.mean(losses):.5f}", flush=True)
    model.eval()
    return model
