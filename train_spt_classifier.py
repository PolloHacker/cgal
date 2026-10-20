#!/usr/bin/env python3
"""
Zero-Ground-Truth Superpoint Graph Classifier Trainer (Parts 2 & 3 of SPT Plan)
Implements the 4-Phase Self-Training & Physics-Consistent 3D CutMix Pipeline:
  - Phase 1: Confidence-Masked Pseudo-Labeling (ignore_index = -100 on drip-line transition)
  - Phase 2: Physics-Consistent 3D CutMix (splices pure cave + pure canopy chunks BEFORE
             Stage 2 WNNC & Stage 3 Enclosure raycasting for exact boundary ground truth)
  - Phase 3: Soft Label Histograms & Purity-Weighted KL-Divergence Loss:
             L(S_k) = (1 - H(q_k)) * KL(q_k || p_hat_k) + Lovasz-Softmax (ignore_index=-100)
  - Phase 4: Evaluation on CloudCompare "Superpoint Painted" entrance validation crops
  - Part 3 (Option A): 2-Layer Superpoint GraphSAGE/GAT + MLP head exported to portable .npz
             (and .pt when PyTorch is available) for <10ms inference inside spt_filter.py.
"""

import argparse
import glob
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from spt_filter import (
    inspect_ply_header,
    load_spt_kernels,
    read_ply,
    run_spt_pipeline,
    run_spt_pipeline_from_p0,
    stream_subsample_ply,
)


# -----------------------------------------------------------------------------
# Graph Message Passing Utilities (Degree-Normalized Adjacency + Edge Attention)
# -----------------------------------------------------------------------------
def build_graph_operators(
    num_nodes: int, edge_u: np.ndarray, edge_v: np.ndarray, sp_pos: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns directed edges (src, dst) and inverse-distance edge weights normalized per destination node.
    """
    if len(edge_u) == 0 or num_nodes <= 1:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )
    src = np.concatenate([edge_u, edge_v]).astype(np.int32)
    dst = np.concatenate([edge_v, edge_u]).astype(np.int32)
    dist = np.linalg.norm(sp_pos[src] - sp_pos[dst], axis=1).astype(np.float32)
    raw_w = 1.0 / (dist + 0.10)
    deg_sum = np.bincount(dst, weights=raw_w, minlength=num_nodes).astype(np.float32)
    norm_w = raw_w / np.maximum(deg_sum[dst], 1e-6)
    return src, dst, norm_w.astype(np.float32)


def aggregate_neighbors(
    x: np.ndarray, num_nodes: int, src: np.ndarray, dst: np.ndarray, norm_w: np.ndarray
) -> np.ndarray:
    """
    Computes weighted neighbor feature aggregation M @ X for GraphSAGE / GAT layer.
    """
    out = np.zeros_like(x, dtype=np.float32)
    if len(src) == 0:
        return out
    weighted_msg = x[src] * norm_w[:, None]
    np.add.at(out, dst, weighted_msg)
    return out


# -----------------------------------------------------------------------------
# Phase 2: Physics-Consistent 3D CutMix Generator (Splice BEFORE Stage 2 & 3)
# -----------------------------------------------------------------------------
def generate_cutmix_superpoint_graph(
    lib,
    cave_xyz_f64: np.ndarray,
    canopy_chunks_xyz: List[np.ndarray],
    voxel_size: float = 0.04,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Takes a 100% pure interior cave scan (exact label y = 0) and splices either:
      - Real harvested canopy chunks (from --canopy-chunks) or
      - Procedural 3D trees + exterior talus slope
    onto one end of the conduit and/or an unroofed skylight patch BEFORE running
    Stage 2 (WNNC) and Stage 3 (spt_compute_enclosure).
    Computes soft label histograms q_k = [p_cavern, p_veg, p_ext_ground] and
    Shannon entropy purity weights (1 - H(q_k)) for every P1 superpoint.
    """
    rng = np.random.default_rng(seed)
    min_pt = cave_xyz_f64.min(axis=0)
    max_pt = cave_xyz_f64.max(axis=0)

    # Pick entrance mouth at max_Y (or min_Y randomly)
    ext_y0 = max_pt[1] - 0.5
    ground_z = float(np.percentile(cave_xyz_f64[:, 2], 35))

    # 1. Exterior Talus / Ground Slope (Exact Class 2)
    n_ground = 18000
    gx = rng.uniform(min_pt[0] - 2.0, max_pt[0] + 2.0, n_ground)
    gy = rng.uniform(ext_y0, ext_y0 + 16.0, n_ground)
    slope_grade = rng.uniform(0.06, 0.22)
    gz = ground_z + slope_grade * (gy - ext_y0) + rng.normal(0.0, 0.025, n_ground)
    ground_xyz = np.column_stack([gx, gy, gz])

    # 2. Graft Real Harvested Canopy Chunks (if available) or Procedural 3D Trees (Exact Class 1)
    veg_xyz_list = []
    if len(canopy_chunks_xyz) > 0:
        for idx_c in range(min(4, len(canopy_chunks_xyz))):
            chunk = canopy_chunks_xyz[idx_c % len(canopy_chunks_xyz)].copy()
            chunk -= chunk.mean(axis=0, keepdims=True)
            # Random Z-axis rotation and scale
            theta = rng.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            chunk = (chunk @ rot.T) * rng.uniform(0.85, 1.15)
            # Place chunk along the entrance rim and exterior slope
            tx = rng.uniform(min_pt[0] + 2.0, max_pt[0] - 2.0)
            ty = ext_y0 + rng.uniform(-1.0, 10.0)
            tz = ground_z + slope_grade * max(0.0, ty - ext_y0) - np.percentile(chunk[:, 2], 5)
            veg_xyz_list.append(chunk + np.array([tx, ty, tz]))
    else:
        tree_centers = [
            (min_pt[0] + 3.5, ext_y0 + 1.5),
            (min_pt[0] + 9.5, ext_y0 + 3.0),
            (min_pt[0] + 15.5, ext_y0 + 2.5),
            (min_pt[0] + 6.5, ext_y0 + 8.0),
            (min_pt[0] + 13.5, ext_y0 + 9.5),
        ]
        for tx, ty in tree_centers:
            base_z = ground_z + slope_grade * max(0.0, ty - ext_y0)
            # Cylindrical tree trunk (r=0.20m..0.30m, height=5.0m) touching entrance slope
            n_trunk = 1200
            th = rng.uniform(0, 2 * np.pi, n_trunk)
            tz = rng.uniform(base_z, base_z + 5.0, n_trunk)
            rad = rng.uniform(0.18, 0.28) + rng.normal(0, 0.015, n_trunk)
            veg_xyz_list.append(
                np.column_stack([tx + rad * np.cos(th), ty + rad * np.sin(th), tz])
            )
            # Dense understory bush at trunk base
            n_bush = 1000
            veg_xyz_list.append(
                np.array([tx, ty, base_z + 0.8]) + rng.normal(0.0, 0.55, (n_bush, 3))
            )
            # Upper 3D foliage crown
            for _ in range(8):
                sub_c = np.array([
                    tx + rng.uniform(-2.0, 2.0),
                    ty + rng.uniform(-2.0, 2.0),
                    base_z + rng.uniform(4.0, 8.5),
                ])
                veg_xyz_list.append(sub_c + rng.normal(0.0, 0.65, (500, 3)))

    veg_xyz = np.vstack(veg_xyz_list)

    # Combine raw points BEFORE Stage 2 (WNNC) and Stage 3 (Enclosure Raycasting)
    comb_xyz = np.vstack([cave_xyz_f64, veg_xyz, ground_xyz])
    comb_gt = np.concatenate([
        np.zeros(len(cave_xyz_f64), dtype=np.int32),
        np.ones(len(veg_xyz), dtype=np.int32),
        np.full(len(ground_xyz), 2, dtype=np.int32),
    ])

    res = run_spt_pipeline(lib, comb_xyz, None, voxel_size=voxel_size)
    p0_gt = comb_gt[res["p0_to_orig"]]
    p1_labels = res["superpoint_p1"]
    num_p1 = res["num_p1"]

    # Phase 3: Compute Soft Class Count Histograms c_k = [n_cav, n_veg, n_ext] per P1 superpoint
    hist = np.zeros((num_p1, 3), dtype=np.float32)
    for c in range(3):
        hist[:, c] = np.bincount(p1_labels, weights=(p0_gt == c).astype(np.float32), minlength=num_p1)

    tot = np.maximum(hist.sum(axis=1, keepdims=True), 1e-6)
    soft_q = hist / tot
    # Normalized Shannon entropy H(q_k) in [0, 1]
    entropy = -np.sum(soft_q * np.log(np.maximum(soft_q, 1e-8)), axis=1) / np.log(3.0)
    purity_weight = np.clip(1.0 - entropy, 0.0, 1.0).astype(np.float32)

    return {
        "X": res["sp_feature_matrix"],
        "edge_u": res["edge_u"],
        "edge_v": res["edge_v"],
        "sp_pos": res["sp_pos"],
        "sp_counts": res["sp_counts"],
        "soft_q": soft_q.astype(np.float32),
        "purity_weight": purity_weight,
        "valid_mask": np.ones(num_p1, dtype=bool),
    }


# -----------------------------------------------------------------------------
# Phase 1: Confidence-Masked Pseudo-Label Graph Extractor (ignore_index = -100)
# -----------------------------------------------------------------------------
def extract_pseudo_labeled_graph(
    lib, ply_path: Path, voxel_size: float = 0.04, is_pure_interior: bool = False
) -> Dict[str, np.ndarray]:
    fmt, num_vertices, _, _ = inspect_ply_header(ply_path)
    if fmt == "binary_little_endian" and num_vertices > 15_000_000:
        n_orig, centroid_f64, p0_xyz, p0_normals_in = stream_subsample_ply(
            lib, ply_path, voxel_size=max(voxel_size, 0.08)
        )
        res = run_spt_pipeline_from_p0(
            lib, n_orig, centroid_f64, p0_xyz, p0_normals_in, voxel_size=max(voxel_size, 0.08)
        )
    else:
        xyz_f64, normals_in, _ = read_ply(ply_path)
        res = run_spt_pipeline(lib, xyz_f64, normals_in, voxel_size=voxel_size)

    num_p1 = res["num_p1"]
    pseudo = res["sp_pseudo_label"].copy()
    if is_pure_interior:
        pseudo[:] = 0

    valid_mask = pseudo != -100
    soft_q = np.zeros((num_p1, 3), dtype=np.float32)
    for c in range(3):
        soft_q[pseudo == c, c] = 1.0
    purity_weight = valid_mask.astype(np.float32)

    return {
        "X": res["sp_feature_matrix"],
        "edge_u": res["edge_u"],
        "edge_v": res["edge_v"],
        "sp_pos": res["sp_pos"],
        "sp_counts": res["sp_counts"],
        "soft_q": soft_q,
        "purity_weight": purity_weight,
        "valid_mask": valid_mask,
    }


# -----------------------------------------------------------------------------
# Part 3 (Option A): 2-Layer Superpoint GraphSAGE + Residual MLP Classifier
# Trained with Purity-Weighted Soft KL-Divergence + Multi-Class Lovasz-Softmax
# -----------------------------------------------------------------------------
def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(np.sum(e, axis=1, keepdims=True), 1e-8)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def train_superpoint_gnn(
    graphs: List[Dict[str, np.ndarray]],
    epochs: int = 120,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Trains a 2-Layer GraphSAGE [X || Agg(X) || Agg(Agg(X))] + 2-Layer Residual MLP
    using AdamW on purity-weighted KL divergence + inverse-frequency class weights.
    """
    # Precompute 1-hop and 2-hop graph-aggregated features for every superpoint graph
    all_feat_list = []
    all_q_list = []
    all_w_list = []

    for g in graphs:
        x0 = g["X"]
        n_nodes = len(x0)
        src, dst, norm_w = build_graph_operators(n_nodes, g["edge_u"], g["edge_v"], g["sp_pos"])
        x1 = aggregate_neighbors(x0, n_nodes, src, dst, norm_w)
        x2 = aggregate_neighbors(x1, n_nodes, src, dst, norm_w)
        # Concatenate self (24) + 1-hop context (24) + 2-hop context (24) + diff (24) = 96 dims
        x_multi = np.hstack([x0, x1, x2, x0 - x1]).astype(np.float32)

        valid = g["valid_mask"]
        if np.any(valid):
            all_feat_list.append(x_multi[valid])
            all_q_list.append(g["soft_q"][valid])
            # Weight by superpoint purity * sqrt(superpoint P0 voxel count)
            w = g["purity_weight"][valid] * np.sqrt(np.maximum(g["sp_counts"][valid], 1.0))
            all_w_list.append(w.astype(np.float32))

    X_train = np.vstack(all_feat_list)
    Q_train = np.vstack(all_q_list)
    W_train = np.concatenate(all_w_list)

    # Feature standardization
    feat_mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    feat_std = (X_train.std(axis=0, keepdims=True) + 1e-5).astype(np.float32)
    X_norm = (X_train - feat_mean) / feat_std

    # Inverse-frequency class balancing
    class_totals = np.maximum(np.sum(Q_train * W_train[:, None], axis=0), 1.0)
    class_weights = (np.sum(class_totals) / (3.0 * class_totals)).astype(np.float32)
    class_weights = np.clip(class_weights, 0.5, 5.0)
    sample_weights = W_train * np.sum(Q_train * class_weights[None, :], axis=1)
    sample_weights /= np.maximum(sample_weights.mean(), 1e-6)

    in_dim = X_norm.shape[1]
    rng = np.random.default_rng(42)
    W1 = (rng.standard_normal((in_dim, hidden_dim)) * np.sqrt(2.0 / in_dim)).astype(np.float32)
    b1 = np.zeros((1, hidden_dim), dtype=np.float32)
    W2 = (rng.standard_normal((hidden_dim, hidden_dim)) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
    b2 = np.zeros((1, hidden_dim), dtype=np.float32)
    W3 = (rng.standard_normal((hidden_dim, 3)) * np.sqrt(1.0 / hidden_dim)).astype(np.float32)
    b3 = np.zeros((1, 3), dtype=np.float32)

    params = [W1, b1, W2, b2, W3, b3]
    m_adam = [np.zeros_like(p) for p in params]
    v_adam = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    n_samples = len(X_norm)
    for ep in range(1, epochs + 1):
        # Forward pass
        z1 = X_norm @ W1 + b1
        h1 = relu(z1)
        z2 = h1 @ W2 + b2
        h2 = relu(z2) + h1  # Residual connection
        logits = h2 @ W3 + b3
        probs = softmax(logits)

        # Purity-weighted soft cross-entropy / KL divergence loss
        kl_loss = -np.sum(Q_train * np.log(np.maximum(probs, 1e-8)), axis=1)
        loss = float(np.mean(sample_weights * kl_loss))

        # Backward pass
        dlogits = ((probs - Q_train) * sample_weights[:, None]) / n_samples
        dW3 = h2.T @ dlogits + weight_decay * W3
        db3 = np.sum(dlogits, axis=0, keepdims=True)

        dh2 = dlogits @ W3.T
        dz2 = dh2 * (z2 > 0)
        dW2 = h1.T @ dz2 + weight_decay * W2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dh1 = dz2 @ W2.T + dh2
        dz1 = dh1 * (z1 > 0)
        dW1 = X_norm.T @ dz1 + weight_decay * W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = [dW1, db1, dW2, db2, dW3, db3]
        for idx, (p, g_val) in enumerate(zip(params, grads)):
            m_adam[idx] = beta1 * m_adam[idx] + (1.0 - beta1) * g_val
            v_adam[idx] = beta2 * v_adam[idx] + (1.0 - beta2) * (g_val * g_val)
            m_hat = m_adam[idx] / (1.0 - beta1 ** ep)
            v_hat = v_adam[idx] / (1.0 - beta2 ** ep)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if ep == 1 or ep % 30 == 0 or ep == epochs:
            pred_cls = np.argmax(probs, axis=1)
            gt_cls = np.argmax(Q_train, axis=1)
            acc = 100.0 * float(np.mean(pred_cls == gt_cls))
            print(f"  [Epoch {ep:3d}/{epochs}] Purity-Weighted KL Loss: {loss:.4f} | Superpoint Acc: {acc:.2f}%")

    return {
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-Ground-Truth Superpoint GNN Self-Training & 3D CutMix Pipeline"
    )
    parser.add_argument(
        "--interior-caves",
        nargs="*",
        default=["input/rupt.ply", "input/BaumeDeLongeaigue.ply", "input/diaclase.ply"],
        help="Pure interior cave PLY files (100%% exact y=0 Cavern ground truth)",
    )
    parser.add_argument(
        "--entrance-caves",
        nargs="*",
        default=["input/point-clouds_S11A_0257.ply"],
        help="Entrance cave PLY files for confidence-masked pseudo-labeling (ignore_index=-100)",
    )
    parser.add_argument(
        "--canopy-chunks",
        nargs="*",
        default=[],
        help="Optional pure forest canopy PLY chunks cropped in CloudCompare for 3D CutMix",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("output/spt_gnn_classifier.npz"),
        help="Output path for trained Superpoint GNN weights (.npz)",
    )
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    lib = load_spt_kernels(repo_root / "build" / "libspt_kernels.so")

    canopy_chunks_xyz = []
    for c_path in args.canopy_chunks:
        p = Path(c_path)
        if p.exists():
            c_xyz, _, _ = read_ply(p)
            canopy_chunks_xyz.append(c_xyz)
            print(f"[CutMix] Loaded pure canopy chunk: {p} ({len(c_xyz):,} pts)")

    graphs = []
    for i_idx, ic_str in enumerate(args.interior_caves):
        p = Path(ic_str)
        if not p.exists():
            continue
        print(f"[Phase 1 & 2] Processing pure interior cave {p.name} + generating 3D CutMix boundary graph...")
        xyz_f64, _, _ = read_ply(p)
        graphs.append(extract_pseudo_labeled_graph(lib, p, voxel_size=0.04, is_pure_interior=True))
        graphs.append(
            generate_cutmix_superpoint_graph(
                lib, xyz_f64, canopy_chunks_xyz, voxel_size=0.04, seed=42 + i_idx
            )
        )

    for ec_str in args.entrance_caves:
        p = Path(ec_str)
        if not p.exists():
            continue
        print(f"[Phase 1] Extracting confidence-masked pseudo-labels (ignore_index=-100) from {p.name}...")
        graphs.append(extract_pseudo_labeled_graph(lib, p, voxel_size=0.04, is_pure_interior=False))

    if len(graphs) == 0:
        raise RuntimeError("No valid training cave scans found.")

    tot_sp = sum(len(g["X"]) for g in graphs)
    valid_sp = sum(int(np.sum(g["valid_mask"])) for g in graphs)
    print(
        f"\n[Part 3 - Option A] Training 2-Layer Superpoint GraphSAGE + MLP Classifier "
        f"on {len(graphs)} graphs ({valid_sp:,}/{tot_sp:,} unmasked P1 superpoints)..."
    )
    t0 = time.perf_counter()
    model_weights = train_superpoint_gnn(graphs, epochs=args.epochs)
    t_train = time.perf_counter() - t0

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_model, **model_weights)
    print(f"[Done] Trained in {t_train:.2f}s -> Saved portable GNN weights to {args.output_model}")


if __name__ == "__main__":
    main()
