#!/usr/bin/env python3
"""
Comprehensive Evaluation of Pointcept/Utonia on Speleological Point Clouds.

Implements both evaluation modes requested:
  1. Mode 1 (Unsupervised): Utonia PTv3 multi-stage feature extraction -> PCA RGB projection
     + Unsupervised K-Means (K=3, K=5) clustering exported to CloudCompare .ply.
  2. Mode 2 (Supervised Probing):
     - Linear Probe (SegHead) on pure Utonia embeddings
     - 2-Layer MLP Probe on pure Utonia embeddings
     - Hybrid Gated Probe fusing Utonia embeddings + 3D physical enclosure descriptors

Memory-bounded (< 4 GB system RAM, < 2 GB GPU VRAM) so all three models
(utonia_tiny, utonia_small, utonia base) run smoothly on 2M+ point clouds.
"""

import argparse
import ctypes
import gc
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent
UTONIA_DIR = ROOT_DIR / "third_party" / "Utonia"
if str(UTONIA_DIR) not in sys.path:
    sys.path.insert(0, str(UTONIA_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

import utonia
from spt_filter import load_spt_kernels, read_ply


ENCL_KEYS = [
    "scalar_upward_occlusion",
    "scalar_roof_occlusion",
    "scalar_downward_occlusion",
    "scalar_cavern_enclosure",
    "scalar_foliage_above",
    "scalar_signed_horizontality",
    "scalar_planarity_fine",
    "scalar_normal_dispersion",
    "scalar_planarity_coarse",
    "scalar_rugosity_coarse",
]


def load_cached_diagnostic_ply(path: Path):
    """Load precomputed P0 diagnostic PLY from output/spt_updated/."""
    xyz_f64, normals, scalars = read_ply(path)
    centroid_f64 = xyz_f64.mean(axis=0, keepdims=True)
    xyz_centered = np.ascontiguousarray(xyz_f64 - centroid_f64, dtype=np.float32)
    normals = np.ascontiguousarray(normals, dtype=np.float32)
    encl_vec = np.column_stack([scalars[k].astype(np.float32) for k in ENCL_KEYS])
    spt_pred = scalars["scalar_predicted_class"].astype(np.int32)
    spt_zone = scalars["scalar_cave_zone"].astype(np.int32)
    pseudo = scalars.get("scalar_pseudo_label", np.full(len(xyz_centered), -100, dtype=np.int32)).astype(np.int32)
    return centroid_f64, xyz_centered, normals, encl_vec, spt_pred, spt_zone, pseudo


def compute_forest_p0_and_enclosure(lib: ctypes.CDLL, path: Path, voxel_size: float = 0.06):
    """Subsample annotated forest PLY from data/trees and compute WNNC + enclosure features."""
    f32_p = ctypes.POINTER(ctypes.c_float)
    i32_p = ctypes.POINTER(ctypes.c_int32)

    pos_f64, _, extra = read_ply(path)
    centroid_f64 = pos_f64.mean(axis=0, keepdims=True)
    xyz_centered = np.ascontiguousarray(pos_f64 - centroid_f64, dtype=np.float32)
    n_orig = len(xyz_centered)

    p0_to_orig = np.empty(n_orig, dtype=np.int32)
    orig_to_p0 = np.empty(n_orig, dtype=np.int32)
    num_p0 = lib.spt_voxel_subsample(
        xyz_centered.ctypes.data_as(f32_p),
        ctypes.c_int32(n_orig),
        ctypes.c_float(voxel_size),
        p0_to_orig.ctypes.data_as(i32_p),
        orig_to_p0.ctypes.data_as(i32_p),
    )
    sel = p0_to_orig[:num_p0].copy()
    p0_xyz = np.ascontiguousarray(xyz_centered[sel], dtype=np.float32)
    sem = extra["semantic_seg"][sel].astype(np.int32)
    labels = np.where(sem == 1, 2, 1).astype(np.int64)

    normals, encl_vec = compute_raw_wnnc_and_enclosure(lib, p0_xyz, voxel_size)
    return p0_xyz, normals, encl_vec, labels


def compute_raw_wnnc_and_enclosure(lib: ctypes.CDLL, p0_xyz: np.ndarray, voxel_size: float):
    """Run C++ WNNC + multi-scale PCA + 3D cone enclosure on a raw P0 coordinate array."""
    f32_p = ctypes.POINTER(ctypes.c_float)
    num_p0 = len(p0_xyz)
    p0_xyz = np.ascontiguousarray(p0_xyz, dtype=np.float32)

    normals = np.empty((num_p0, 3), dtype=np.float32)
    lib.spt_compute_wnnc_normals(
        p0_xyz.ctypes.data_as(f32_p),
        ctypes.c_int32(num_p0),
        ctypes.c_int32(30),
        ctypes.c_float(0.003),
        ctypes.c_float(0.04),
        ctypes.c_int32(60000),
        normals.ctypes.data_as(f32_p),
    )

    r_fine_eff = max(0.15, voxel_size * 3.75)
    r_coarse_eff = max(0.50, voxel_size * 10.0)
    occ_cell_eff = max(0.20, voxel_size * 4.0)

    fine_feats = np.empty((num_p0, 6), dtype=np.float32)
    coarse_feats = np.empty((num_p0, 6), dtype=np.float32)
    lib.spt_compute_scale_features(
        p0_xyz.ctypes.data_as(f32_p),
        normals.ctypes.data_as(f32_p),
        ctypes.c_int32(num_p0),
        ctypes.c_float(r_fine_eff),
        ctypes.c_int32(1),
        ctypes.c_int32(1),
        fine_feats.ctypes.data_as(f32_p),
    )
    lib.spt_compute_scale_features(
        p0_xyz.ctypes.data_as(f32_p),
        normals.ctypes.data_as(f32_p),
        ctypes.c_int32(num_p0),
        ctypes.c_float(r_coarse_eff),
        ctypes.c_int32(0),
        ctypes.c_int32(1),
        coarse_feats.ctypes.data_as(f32_p),
    )

    encl = np.empty((num_p0, 4), dtype=np.float32)
    lib.spt_compute_enclosure(
        p0_xyz.ctypes.data_as(f32_p),
        normals.ctypes.data_as(f32_p),
        fine_feats.ctypes.data_as(f32_p),
        coarse_feats.ctypes.data_as(f32_p),
        ctypes.c_int32(num_p0),
        ctypes.c_float(occ_cell_eff),
        ctypes.c_float(45.0),
        encl.ctypes.data_as(f32_p),
    )

    encl_vec = np.column_stack([
        encl[:, 0],
        encl[:, 1],
        encl[:, 2],
        encl[:, 3],
        np.maximum(0.0, encl[:, 0] - encl[:, 1]),
        normals[:, 2],
        fine_feats[:, 1],
        fine_feats[:, 5],
        coarse_feats[:, 1],
        coarse_feats[:, 0],
    ]).astype(np.float32)
    return normals, encl_vec


def load_lp_boi_annotated():
    """Load manually annotated Lp_Boi ground-truth parts from annotations/."""
    parts = [
        (0, ROOT_DIR / "annotations/cave/lp_boi_cave.ply"),
        (1, ROOT_DIR / "annotations/vegetation/lp_boi_vegetation.ply"),
        (-100, ROOT_DIR / "annotations/unknown/lp_boi_unknown.ply"),
    ]
    xyz_list, nrm_list, encl_list, gt_list, spt_zone_list = [], [], [], [], []
    for label, p in parts:
        xyz_f64, nrm, sc = read_ply(p)
        xyz_list.append(xyz_f64)
        nrm_list.append(nrm.astype(np.float32))
        encl_list.append(np.column_stack([sc[k].astype(np.float32) for k in ENCL_KEYS]))
        gt_list.append(np.full(len(xyz_f64), label, dtype=np.int32))
        spt_zone_list.append(sc["scalar_cave_zone"].astype(np.int32))

    xyz_all = np.vstack(xyz_list)
    centroid_f64 = xyz_all.mean(axis=0, keepdims=True)
    xyz_centered = np.ascontiguousarray(xyz_all - centroid_f64, dtype=np.float32)
    normals = np.vstack(nrm_list)
    encl_vec = np.vstack(encl_list)
    gt = np.concatenate(gt_list)
    spt_zone = np.concatenate(spt_zone_list)
    spt_pred = np.where(spt_zone == 3, 1, np.where(spt_zone == 4, 2, 0)).astype(np.int32)
    return centroid_f64, xyz_centered, normals, encl_vec, gt, spt_pred, spt_zone


def extract_utonia_features(
    model: nn.Module,
    coord: np.ndarray,
    normal: np.ndarray,
    scale: float,
    upcast_levels: int = 2,
    wo_normal: bool = False,
    max_chunk_pts: int = 380_000,
):
    """
    Run Utonia encoder in modality-consistent geometry mode (--wo_color, using XYZ + WNNC normals)
    and store output features in float16 on CPU so 2.15M points x 1224 channels uses only ~5.2 GB RAM.
    """
    n_pts = coord.shape[0]
    zero_color = np.zeros((1, 3), dtype=np.float32)
    in_normal = np.zeros_like(coord, dtype=np.float32) if wo_normal else normal.astype(np.float32)

    transform = utonia.transform.default(
        scale=scale, apply_z_positive=True, normalize_coord=False
    )

    def _forward_single(c_xyz, c_nrm):
        point = transform({
            "coord": c_xyz.copy(),
            "color": np.broadcast_to(zero_color, c_xyz.shape).copy(),
            "normal": c_nrm.copy(),
        })
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for k in point.keys():
                if isinstance(point[k], torch.Tensor):
                    point[k] = point[k].cuda(non_blocking=True)
            point = model(point)
            cur = point
            while True:
                cur.pop("sparse_conv_feat", None)
                cur.pop("serialized_code", None)
                cur.pop("serialized_order", None)
                cur.pop("serialized_inverse", None)
                cur.feat = cur.feat.half().cpu()
                if "pooling_inverse" in cur.keys():
                    cur["pooling_inverse"] = cur["pooling_inverse"].cpu()
                if "pooling_parent" in cur.keys():
                    cur = cur["pooling_parent"]
                else:
                    break
            for _ in range(upcast_levels):
                if "pooling_parent" in point.keys():
                    parent = point.pop("pooling_parent")
                    inverse = point.pop("pooling_inverse")
                    parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                    point = parent
            while "pooling_parent" in point.keys():
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = point.feat[inverse]
                point = parent
            inv_to_orig = point.inverse.cpu()
            feat_orig = point.feat[inv_to_orig]
        return feat_orig

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    if n_pts <= max_chunk_pts:
        feat_all = _forward_single(coord, in_normal).numpy()
    else:
        n_chunks = int(np.ceil(n_pts / max_chunk_pts))
        xy = coord[:, :2]
        axis = int(np.argmax(xy.max(axis=0) - xy.min(axis=0)))
        order = np.argsort(coord[:, axis])
        chunk_indices = np.array_split(order, n_chunks)
        feat_all = None
        for ch_idx in chunk_indices:
            ch_feat = _forward_single(coord[ch_idx], in_normal[ch_idx]).numpy()
            if feat_all is None:
                feat_all = np.empty((n_pts, ch_feat.shape[1]), dtype=np.float16)
            feat_all[ch_idx] = ch_feat
            del ch_feat
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    # Compute Utonia PCA RGB color in memory-safe batches
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(n_pts, size=min(150_000, n_pts), replace=False)
    feat_sub = torch.from_numpy(feat_all[sub_idx].astype(np.float32))
    mean_f = feat_sub.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(feat_sub - mean_f, center=False, q=12, niter=5)
    proj_sub = (feat_sub - mean_f) @ v
    proj_rgb_sub = proj_sub[:, :3] * 0.4 + proj_sub[:, 3:6] * 0.2 + proj_sub[:, 9:12] * 0.4
    lo = torch.quantile(proj_rgb_sub, 0.02, dim=0, keepdim=True)
    hi = torch.quantile(proj_rgb_sub, 0.98, dim=0, keepdim=True)
    div = torch.clamp(hi - lo, min=1e-6)
    del feat_sub, proj_sub, proj_rgb_sub

    pca_rgb_u8 = np.empty((n_pts, 3), dtype=np.uint8)
    bsize = 131072
    for i in range(0, n_pts, bsize):
        bf = torch.from_numpy(feat_all[i : i + bsize].astype(np.float32))
        bp = (bf - mean_f) @ v
        brgb = bp[:, :3] * 0.4 + bp[:, 3:6] * 0.2 + bp[:, 9:12] * 0.4
        bcol = ((brgb - lo) / div * 1.15).clamp(0.0, 1.0)
        pca_rgb_u8[i : i + bsize] = (bcol.numpy() * 255.0).astype(np.uint8)

    return feat_all, pca_rgb_u8, elapsed, peak_vram_mb


def run_unsupervised_kmeans(feat_f16: np.ndarray, k_values=(3, 5)):
    """Memory-bounded L2-normalized K-Means: fit on 180k sample, predict in 131k batches."""
    n_pts = feat_f16.shape[0]
    rng = np.random.default_rng(42)
    fit_idx = rng.choice(n_pts, size=min(180_000, n_pts), replace=False)
    f_sub = feat_f16[fit_idx].astype(np.float32)
    f_sub /= np.maximum(np.linalg.norm(f_sub, axis=1, keepdims=True), 1e-6)

    results = {}
    bsize = 131072
    for k in k_values:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=16384, n_init=5)
        km.fit(f_sub)
        labels = np.empty(n_pts, dtype=np.int32)
        for i in range(0, n_pts, bsize):
            b = feat_f16[i : i + bsize].astype(np.float32)
            b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-6)
            labels[i : i + bsize] = km.predict(b).astype(np.int32)
        results[k] = labels
    del f_sub
    return results


def cluster_oracle_iou(cluster_labels: np.ndarray, gt_binary: np.ndarray, valid_mask: np.ndarray = None):
    if valid_mask is not None:
        cluster_labels = cluster_labels[valid_mask]
        gt_binary = gt_binary[valid_mask]
    pred_bin = np.zeros_like(gt_binary)
    for c in np.unique(cluster_labels):
        m = cluster_labels == c
        if np.mean(gt_binary[m] > 0) >= 0.5:
            pred_bin[m] = 1
    inter_cav = np.sum((pred_bin == 0) & (gt_binary == 0))
    union_cav = np.sum((pred_bin == 0) | (gt_binary == 0))
    inter_rem = np.sum((pred_bin == 1) & (gt_binary == 1))
    union_rem = np.sum((pred_bin == 1) | (gt_binary == 1))
    return float(inter_cav / max(1, union_cav)), float(inter_rem / max(1, union_rem))


def binary_metrics(pred_cls: np.ndarray, gt_binary: np.ndarray, valid_mask: np.ndarray = None):
    if valid_mask is not None:
        pred_cls = pred_cls[valid_mask]
        gt_binary = gt_binary[valid_mask]
    pred_rem = (pred_cls > 0).astype(np.int32)
    tp_cav = np.sum((pred_rem == 0) & (gt_binary == 0))
    fp_cav = np.sum((pred_rem == 0) & (gt_binary == 1))
    fn_cav = np.sum((pred_rem == 1) & (gt_binary == 0))
    inter_r = np.sum((pred_rem == 1) & (gt_binary == 1))
    union_r = np.sum((pred_rem == 1) | (gt_binary == 1))
    cav_prec = float(tp_cav / max(1, tp_cav + fp_cav))
    cav_rec = float(tp_cav / max(1, tp_cav + fn_cav))
    cav_iou = float(tp_cav / max(1, tp_cav + fp_cav + fn_cav))
    rem_iou = float(inter_r / max(1, union_r))
    return {
        "cavern_precision": round(cav_prec * 100.0, 2),
        "cavern_recall": round(cav_rec * 100.0, 2),
        "cavern_iou": round(cav_iou * 100.0, 2),
        "remove_iou": round(rem_iou * 100.0, 2),
    }


class UtoniaLinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.seg_head = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg_head(self.norm(x))


class UtoniaMLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UtoniaHybridProbe(nn.Module):
    """Two-stream fusion: balances high-dimensional Utonia embeddings with 10-D physical 3D enclosure."""

    def __init__(self, u_dim: int, e_dim: int = 10, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.u_branch = nn.Sequential(
            nn.LayerNorm(u_dim),
            nn.Linear(u_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.e_branch = nn.Sequential(
            nn.LayerNorm(e_dim),
            nn.Linear(e_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, u: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        hu = self.u_branch(u)
        he = self.e_branch(e)
        return self.head(torch.cat([hu, he], dim=-1))


def train_probes(X_utonia_f16: np.ndarray, X_encl: np.ndarray, y: np.ndarray, epochs: int = 25):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    u_dim = X_utonia_f16.shape[1]
    e_dim = X_encl.shape[1]

    counts = np.bincount(y, minlength=3).astype(np.float32)
    weights = np.clip(counts.sum() / (3.0 * np.maximum(counts, 1.0)), 0.5, 4.0)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    linear_probe = UtoniaLinearProbe(u_dim, 3).to(device)
    mlp_probe = UtoniaMLPProbe(u_dim, hidden_dim=128, num_classes=3).to(device)
    hybrid_probe = UtoniaHybridProbe(u_dim, e_dim=e_dim, hidden_dim=64, num_classes=3).to(device)

    X_u_t = torch.from_numpy(X_utonia_f16)
    X_e_t = torch.from_numpy(X_encl)
    y_t = torch.from_numpy(y)

    t0 = time.perf_counter()
    n = X_u_t.shape[0]
    batch_size = 16384

    for probe, lr, is_hyb in [
        (linear_probe, 3e-3, False),
        (mlp_probe, 2e-3, False),
        (hybrid_probe, 2e-3, True),
    ]:
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
        probe.train()
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                b_idx = perm[i : i + batch_size]
                bu = X_u_t[b_idx].to(device, dtype=torch.float32, non_blocking=True)
                by = y_t[b_idx].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if is_hyb:
                    be = X_e_t[b_idx].to(device, dtype=torch.float32, non_blocking=True)
                    logits = probe(bu, be)
                else:
                    logits = probe(bu)
                loss = F.cross_entropy(logits, by, weight=class_weights)
                loss.backward()
                opt.step()
        probe.eval()

    print(f"  Trained Linear, MLP, and Hybrid probes on {len(y):,} points in {time.perf_counter()-t0:.2f}s")
    return linear_probe, mlp_probe, hybrid_probe


@torch.inference_mode()
def predict_probe(probe: nn.Module, X_u_f16: np.ndarray, X_e: np.ndarray = None, batch_size: int = 65536):
    device = next(probe.parameters()).device
    n = X_u_f16.shape[0]
    preds = np.empty(n, dtype=np.int32)
    cav_probs = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch_size):
        bu = torch.from_numpy(X_u_f16[i : i + batch_size]).to(device, dtype=torch.float32, non_blocking=True)
        if X_e is not None:
            be = torch.from_numpy(X_e[i : i + batch_size]).to(device, dtype=torch.float32, non_blocking=True)
            logits = probe(bu, be)
        else:
            logits = probe(bu)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds[i : i + batch_size] = np.argmax(probs, axis=-1).astype(np.int32)
        cav_probs[i : i + batch_size] = probs[:, 0]
    return preds, cav_probs


def export_utonia_diagnostic_ply(
    out_path: Path,
    xyz_centered: np.ndarray,
    centroid_f64: np.ndarray,
    normals: np.ndarray,
    pca_rgb: np.ndarray,
    scalars: dict,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xyz_world = xyz_centered.astype(np.float64) + centroid_f64
    use_f64 = bool(np.max(np.abs(xyz_world)) > 1e4)
    c_dtype = "<f8" if use_f64 else "<f4"
    c_ply = "double" if use_f64 else "float"
    n_pts = len(xyz_world)

    dtype_fields = [
        ("x", c_dtype), ("y", c_dtype), ("z", c_dtype),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    for k, v in scalars.items():
        dtype_fields.append((k, "<i4" if np.issubdtype(v.dtype, np.integer) else "<f4"))

    arr = np.empty(n_pts, dtype=np.dtype(dtype_fields))
    arr["x"] = xyz_world[:, 0].astype(c_dtype)
    arr["y"] = xyz_world[:, 1].astype(c_dtype)
    arr["z"] = xyz_world[:, 2].astype(c_dtype)
    arr["nx"] = normals[:, 0]
    arr["ny"] = normals[:, 1]
    arr["nz"] = normals[:, 2]
    arr["red"] = pca_rgb[:, 0]
    arr["green"] = pca_rgb[:, 1]
    arr["blue"] = pca_rgb[:, 2]
    for k, v in scalars.items():
        arr[k] = v.astype(np.int32 if np.issubdtype(v.dtype, np.integer) else np.float32)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n_pts}",
        f"property {c_ply} x",
        f"property {c_ply} y",
        f"property {c_ply} z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    for k, v in scalars.items():
        t_str = "int" if np.issubdtype(v.dtype, np.integer) else "float"
        header_lines.append(f"property {t_str} {k}")
    header_lines.append("end_header\n")

    with open(out_path, "wb") as f:
        f.write("\n".join(header_lines).encode("ascii"))
        f.write(arr.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pointcept/Utonia on Cave Point Clouds")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["utonia_tiny", "utonia_small", "utonia"],
        help="Utonia model variants to test",
    )
    args = parser.parse_args()

    utonia.utils.set_seed(42)
    lib = load_spt_kernels(ROOT_DIR / "build/libspt_kernels.so")

    print("[1/3] Loading cached P0 cave clouds & precomputing forest training plots ...")
    t_prep = time.perf_counter()

    # Load Training Interior Caves (BaumeDeLongeaigue, diaclase, interior_lapinha)
    train_interiors = []
    for name in ("BaumeDeLongeaigue", "diaclase", "interior_lapinha"):
        p = ROOT_DIR / f"output/spt_updated/{name}_diagnostic.ply"
        _, xyz, nrm, encl, _, _, _ = load_cached_diagnostic_ply(p)
        train_interiors.append((name, xyz, nrm, encl))
        print(f"  Loaded interior training cave {name}: {len(xyz):,} P0 pts")

    # Load Training Forest Plots (data/trees/train_val_data)
    train_forests = []
    for fname in (
        "NIBIO2_NIBIO2_plot12_annotated_train.ply",
        "NIBIO2_NIBIO2_plot14_annotated_train.ply",
        "CULS_CULS_plot_1_annotated_train.ply",
    ):
        p = ROOT_DIR / "data/trees/train_val_data" / fname
        xyz, nrm, encl, lbl = compute_forest_p0_and_enclosure(lib, p, voxel_size=0.06)
        train_forests.append((fname, xyz, nrm, encl, lbl))
        print(f"  Prepared training forest plot {fname}: {len(xyz):,} P0 pts")

    # Build 3D CutMix Training Scene (BaumeDeLongeaigue + NIBIO2_plot12)
    c_xyz = train_interiors[0][1]
    f_xyz, f_lbl = train_forests[0][1], train_forests[0][4]
    shift_train = np.array([
        c_xyz[:, 0].max() - f_xyz[:, 0].min() - 1.5,
        np.median(c_xyz[:, 1]) - np.median(f_xyz[:, 1]),
        np.percentile(c_xyz[:, 2], 25) - np.percentile(f_xyz[:, 2], 10),
    ], dtype=np.float32)
    mix_train_xyz = np.vstack([c_xyz, f_xyz + shift_train])
    mix_train_lbl = np.concatenate([np.zeros(len(c_xyz), dtype=np.int64), f_lbl])
    mix_train_nrm, mix_train_encl = compute_raw_wnnc_and_enclosure(lib, mix_train_xyz, 0.04)

    # Load Held-Out Test Caves (rupt, S11A_0257, Lp_JL)
    test_caves = {}
    for cname, diag_file, scale in [
        ("rupt", "rupt_diagnostic.ply", 0.25),
        ("S11A_0257", "point-clouds_S11A_0257_diagnostic.ply", 0.1667),
        ("Lp_JL", "Lp_JL_diagnostic.ply", 0.125),
    ]:
        cent, xyz, nrm, encl, spt_pred, spt_zone, pseudo = load_cached_diagnostic_ply(
            ROOT_DIR / "output/spt_updated" / diag_file
        )
        test_caves[cname] = {
            "centroid_f64": cent,
            "xyz": xyz,
            "nrm": nrm,
            "encl": encl,
            "scale": scale,
            "spt_pred": spt_pred,
            "spt_zone": spt_zone,
            "pseudo": pseudo,
            "gt": None,
        }
        print(f"  Loaded test cave {cname}: {len(xyz):,} P0 pts")

    # Load Held-Out Real Manual Annotated Cave (Lp_Boi)
    cent_boi, xyz_boi, nrm_boi, encl_boi, gt_boi, spt_pred_boi, spt_zone_boi = (
        load_lp_boi_annotated()
    )
    test_caves["Lp_Boi"] = {
        "centroid_f64": cent_boi,
        "xyz": xyz_boi,
        "nrm": nrm_boi,
        "encl": encl_boi,
        "scale": 0.10,
        "spt_pred": spt_pred_boi,
        "spt_zone": spt_zone_boi,
        "pseudo": None,
        "gt": gt_boi,
    }
    print(
        f"  Loaded manually annotated Lp_Boi: {len(xyz_boi):,} P0 pts "
        f"(Cavern={np.sum(gt_boi==0):,}, Veg={np.sum(gt_boi==1):,}, Unknown={np.sum(gt_boi==-100):,})"
    )

    # Build Held-Out 3D CutMix Validation Scene (rupt + NIBIO2_plot13_val)
    val_forest_path = ROOT_DIR / "data/trees/train_val_data/NIBIO2_NIBIO2_plot13_annotated_val.ply"
    vf_xyz, _, _, vf_lbl = compute_forest_p0_and_enclosure(lib, val_forest_path, voxel_size=0.05)
    r_xyz = test_caves["rupt"]["xyz"]
    shift_val = np.array([
        r_xyz[:, 0].max() - vf_xyz[:, 0].min() - 1.0,
        np.median(r_xyz[:, 1]) - np.median(vf_xyz[:, 1]),
        np.percentile(r_xyz[:, 2], 30) - np.percentile(vf_xyz[:, 2], 10),
    ], dtype=np.float32)
    mix_val_xyz = np.vstack([r_xyz, vf_xyz + shift_val])
    mix_val_gt_cls = np.concatenate([np.zeros(len(r_xyz), dtype=np.int32), vf_lbl.astype(np.int32)])
    mix_val_gt_rem = (mix_val_gt_cls > 0).astype(np.int32)
    mix_val_nrm, mix_val_encl = compute_raw_wnnc_and_enclosure(lib, mix_val_xyz, 0.04)
    print(f"  Prepared all datasets in {time.perf_counter()-t_prep:.2f}s\n")

    summary_report = {}
    rng = np.random.default_rng(42)

    for model_name in args.models:
        print("=" * 84)
        print(f"EVALUATING UTONIA MODEL: {model_name}")
        print("=" * 84)
        gc.collect()
        torch.cuda.empty_cache()

        custom_config = dict(enc_patch_size=[512 for _ in range(5)], enable_flash=False)
        model = utonia.load(model_name, custom_config=custom_config).cuda().eval()

        # 1. Extract Utonia features on Training Set & Train Probes
        X_u_list, X_e_list, y_list = [], [], []
        for _, xyz, nrm, encl in train_interiors:
            feat, _, _, _ = extract_utonia_features(model, xyz, nrm, scale=0.25)
            idx = rng.choice(len(xyz), size=min(50_000, len(xyz)), replace=False)
            X_u_list.append(feat[idx])
            X_e_list.append(encl[idx])
            y_list.append(np.zeros(len(idx), dtype=np.int64))
            del feat

        for _, xyz, nrm, encl, lbl in train_forests:
            feat, _, _, _ = extract_utonia_features(model, xyz, nrm, scale=0.20)
            idx = rng.choice(len(xyz), size=min(50_000, len(xyz)), replace=False)
            X_u_list.append(feat[idx])
            X_e_list.append(encl[idx])
            y_list.append(lbl[idx])
            del feat

        feat_mix_tr, _, _, _ = extract_utonia_features(
            model, mix_train_xyz, mix_train_nrm, scale=0.25
        )
        idx_tr = rng.choice(len(mix_train_xyz), size=min(70_000, len(mix_train_xyz)), replace=False)
        X_u_list.append(feat_mix_tr[idx_tr])
        X_e_list.append(mix_train_encl[idx_tr])
        y_list.append(mix_train_lbl[idx_tr])
        del feat_mix_tr

        # Also pre-extract Lp_JL features once (used both for Lp_JL evaluation and high-confidence
        # tropical karst seeds when evaluating on held-out Lp_Boi & S11A_0257)
        jl_data = test_caves["Lp_JL"]
        feat_jl, pca_jl, elapsed_jl, vram_jl = extract_utonia_features(
            model, jl_data["xyz"], jl_data["nrm"], scale=jl_data["scale"]
        )
        jl_valid_seeds = np.flatnonzero(jl_data["pseudo"] >= 0)
        if len(jl_valid_seeds) > 0:
            sel_jl = rng.choice(jl_valid_seeds, size=min(80_000, len(jl_valid_seeds)), replace=False)
            X_u_list.append(feat_jl[sel_jl])
            X_e_list.append(jl_data["encl"][sel_jl])
            y_list.append(jl_data["pseudo"][sel_jl].astype(np.int64))

        linear_probe, mlp_probe, hybrid_probe = train_probes(
            np.vstack(X_u_list), np.vstack(X_e_list), np.concatenate(y_list)
        )
        del X_u_list, X_e_list, y_list
        gc.collect()
        torch.cuda.empty_cache()

        # 2. Evaluate on Held-Out 3D CutMix Benchmark (rupt + NIBIO2_plot13_val)
        feat_val, _, _, _ = extract_utonia_features(
            model, mix_val_xyz, mix_val_nrm, scale=0.25
        )
        km_val = run_unsupervised_kmeans(feat_val, k_values=(3, 5))
        k3_c_iou, k3_r_iou = cluster_oracle_iou(km_val[3], mix_val_gt_rem)
        k5_c_iou, k5_r_iou = cluster_oracle_iou(km_val[5], mix_val_gt_rem)

        lin_val_p, _ = predict_probe(linear_probe, feat_val)
        mlp_val_p, _ = predict_probe(mlp_probe, feat_val)
        hyb_val_p, _ = predict_probe(hybrid_probe, feat_val, mix_val_encl)
        del feat_val
        gc.collect()

        feat_val_wonrm, _, _, _ = extract_utonia_features(
            model, mix_val_xyz, mix_val_nrm, scale=0.25, wo_normal=True
        )
        mlp_wonrm_p, _ = predict_probe(mlp_probe, feat_val_wonrm)
        del feat_val_wonrm
        gc.collect()

        cutmix_metrics = {
            "kmeans_k3_oracle": {"cavern_iou": round(k3_c_iou * 100, 2), "remove_iou": round(k3_r_iou * 100, 2)},
            "kmeans_k5_oracle": {"cavern_iou": round(k5_c_iou * 100, 2), "remove_iou": round(k5_r_iou * 100, 2)},
            "linear_probe": binary_metrics(lin_val_p, mix_val_gt_rem),
            "mlp_probe": binary_metrics(mlp_val_p, mix_val_gt_rem),
            "hybrid_probe": binary_metrics(hyb_val_p, mix_val_gt_rem),
            "mlp_probe_wo_normal": binary_metrics(mlp_wonrm_p, mix_val_gt_rem),
        }
        print(
            f"  [CutMix Val Benchmark (rupt + NIBIO2_plot13_val)]\n"
            f"    Unsupervised K-Means (K=3) Oracle : Cavern IoU={cutmix_metrics['kmeans_k3_oracle']['cavern_iou']:6.2f}% | Remove IoU={cutmix_metrics['kmeans_k3_oracle']['remove_iou']:6.2f}%\n"
            f"    Unsupervised K-Means (K=5) Oracle : Cavern IoU={cutmix_metrics['kmeans_k5_oracle']['cavern_iou']:6.2f}% | Remove IoU={cutmix_metrics['kmeans_k5_oracle']['remove_iou']:6.2f}%\n"
            f"    Utonia Linear Probe (Utonia only) : Cavern IoU={cutmix_metrics['linear_probe']['cavern_iou']:6.2f}% | Remove IoU={cutmix_metrics['linear_probe']['remove_iou']:6.2f}%\n"
            f"    Utonia MLP Probe    (Utonia only) : Cavern IoU={cutmix_metrics['mlp_probe']['cavern_iou']:6.2f}% | Remove IoU={cutmix_metrics['mlp_probe']['remove_iou']:6.2f}%\n"
            f"    Utonia + Enclosure Hybrid Probe   : Cavern IoU={cutmix_metrics['hybrid_probe']['cavern_iou']:6.2f}% | Remove IoU={cutmix_metrics['hybrid_probe']['remove_iou']:6.2f}%\n"
            f"    Utonia MLP Probe (--wo_normal)    : Cavern IoU={cutmix_metrics['mlp_probe_wo_normal']['cavern_iou']:6.2f}% | Remove IoU={cutmix_metrics['mlp_probe_wo_normal']['remove_iou']:6.2f}%"
        )

        model_results = {"cutmix_benchmark": cutmix_metrics, "caves": {}}

        # 3. Evaluate on Each Cave & Export CloudCompare PLYs (process Lp_JL first to free feat_jl immediately)
        for cname in ("Lp_JL", "rupt", "S11A_0257", "Lp_Boi"):
            cdata = test_caves[cname]
            xyz = cdata["xyz"]
            nrm = cdata["nrm"]
            encl = cdata["encl"]
            scale = cdata["scale"]

            if cname == "Lp_JL":
                feat, pca_rgb, elapsed, peak_vram = feat_jl, pca_jl, elapsed_jl, vram_jl
            else:
                feat, pca_rgb, elapsed, peak_vram = extract_utonia_features(
                    model, xyz, nrm, scale=scale
                )

            km_res = run_unsupervised_kmeans(feat, k_values=(3, 5))
            lin_pred, lin_cav_prob = predict_probe(linear_probe, feat)
            mlp_pred, mlp_cav_prob = predict_probe(mlp_probe, feat)
            hyb_pred, hyb_cav_prob = predict_probe(hybrid_probe, feat, encl)

            n_tot = len(xyz)
            cave_stats = {
                "num_p0": int(n_tot),
                "feat_dim": int(feat.shape[1]),
                "inference_sec": round(float(elapsed), 2),
                "peak_vram_mb": round(float(peak_vram), 1),
                "linear_cavern_pct": round(float(np.mean(lin_pred == 0) * 100.0), 2),
                "linear_veg_pct": round(float(np.mean(lin_pred == 1) * 100.0), 2),
                "linear_ext_pct": round(float(np.mean(lin_pred == 2) * 100.0), 2),
                "mlp_cavern_pct": round(float(np.mean(mlp_pred == 0) * 100.0), 2),
                "mlp_veg_pct": round(float(np.mean(mlp_pred == 1) * 100.0), 2),
                "mlp_ext_pct": round(float(np.mean(mlp_pred == 2) * 100.0), 2),
                "hybrid_cavern_pct": round(float(np.mean(hyb_pred == 0) * 100.0), 2),
                "hybrid_veg_pct": round(float(np.mean(hyb_pred == 1) * 100.0), 2),
                "hybrid_ext_pct": round(float(np.mean(hyb_pred == 2) * 100.0), 2),
                "spt_cavern_pct": round(float(np.mean(cdata["spt_pred"] == 0) * 100.0), 2),
                "spt_veg_pct": round(float(np.mean(cdata["spt_pred"] == 1) * 100.0), 2),
                "spt_ext_pct": round(float(np.mean(cdata["spt_pred"] == 2) * 100.0), 2),
                "hybrid_spt_agreement_pct": round(
                    float(np.mean((hyb_pred == 0) == (cdata["spt_pred"] == 0)) * 100.0), 2
                ),
            }

            print(
                f"\n  [{cname} | {model_name}] N_P0={n_tot:,} | dim={feat.shape[1]} | time={elapsed:.2f}s | VRAM={peak_vram:.1f} MB\n"
                f"    Linear Probe (Utonia only) : Cavern={cave_stats['linear_cavern_pct']:6.2f}% | Veg={cave_stats['linear_veg_pct']:6.2f}% | ExtGnd={cave_stats['linear_ext_pct']:5.2f}%\n"
                f"    MLP Probe    (Utonia only) : Cavern={cave_stats['mlp_cavern_pct']:6.2f}% | Veg={cave_stats['mlp_veg_pct']:6.2f}% | ExtGnd={cave_stats['mlp_ext_pct']:5.2f}%\n"
                f"    Hybrid Probe (Utonia+Encl) : Cavern={cave_stats['hybrid_cavern_pct']:6.2f}% | Veg={cave_stats['hybrid_veg_pct']:6.2f}% | ExtGnd={cave_stats['hybrid_ext_pct']:5.2f}%\n"
                f"    SPT Geometric Baseline     : Cavern={cave_stats['spt_cavern_pct']:6.2f}% | Veg={cave_stats['spt_veg_pct']:6.2f}% | ExtGnd={cave_stats['spt_ext_pct']:5.2f}%"
            )

            if cdata["gt"] is not None:
                gt = cdata["gt"]
                valid = gt != -100
                gt_bin = (gt > 0).astype(np.int32)
                k3_c, k3_r = cluster_oracle_iou(km_res[3], gt_bin, valid)
                k5_c, k5_r = cluster_oracle_iou(km_res[5], gt_bin, valid)
                gt_eval = {
                    "kmeans_k3_oracle": {"cavern_iou": round(k3_c * 100, 2), "remove_iou": round(k3_r * 100, 2)},
                    "kmeans_k5_oracle": {"cavern_iou": round(k5_c * 100, 2), "remove_iou": round(k5_r * 100, 2)},
                    "linear_probe": binary_metrics(lin_pred, gt_bin, valid),
                    "mlp_probe": binary_metrics(mlp_pred, gt_bin, valid),
                    "hybrid_probe": binary_metrics(hyb_pred, gt_bin, valid),
                    "spt_heuristic": binary_metrics(cdata["spt_pred"], gt_bin, valid),
                }
                cave_stats["ground_truth_metrics"] = gt_eval
                print(
                    f"    [Manual Ground-Truth Metrics on {cname} ({int(valid.sum()):,} labeled pts)]\n"
                    f"      Unsupervised K-Means (K=3) Oracle : Cavern IoU={gt_eval['kmeans_k3_oracle']['cavern_iou']:6.2f}% | Veg IoU={gt_eval['kmeans_k3_oracle']['remove_iou']:6.2f}%\n"
                    f"      Unsupervised K-Means (K=5) Oracle : Cavern IoU={gt_eval['kmeans_k5_oracle']['cavern_iou']:6.2f}% | Veg IoU={gt_eval['kmeans_k5_oracle']['remove_iou']:6.2f}%\n"
                    f"      Linear Probe (Utonia only)        : Cavern IoU={gt_eval['linear_probe']['cavern_iou']:6.2f}% | Veg IoU={gt_eval['linear_probe']['remove_iou']:6.2f}% (Prec={gt_eval['linear_probe']['cavern_precision']:.2f}%, Rec={gt_eval['linear_probe']['cavern_recall']:.2f}%)\n"
                    f"      MLP Probe    (Utonia only)        : Cavern IoU={gt_eval['mlp_probe']['cavern_iou']:6.2f}% | Veg IoU={gt_eval['mlp_probe']['remove_iou']:6.2f}% (Prec={gt_eval['mlp_probe']['cavern_precision']:.2f}%, Rec={gt_eval['mlp_probe']['cavern_recall']:.2f}%)\n"
                    f"      Hybrid Probe (Utonia + Enclosure) : Cavern IoU={gt_eval['hybrid_probe']['cavern_iou']:6.2f}% | Veg IoU={gt_eval['hybrid_probe']['remove_iou']:6.2f}% (Prec={gt_eval['hybrid_probe']['cavern_precision']:.2f}%, Rec={gt_eval['hybrid_probe']['cavern_recall']:.2f}%)\n"
                    f"      SPT Geometric Baseline            : Cavern IoU={gt_eval['spt_heuristic']['cavern_iou']:6.2f}% | Veg IoU={gt_eval['spt_heuristic']['remove_iou']:6.2f}% (Prec={gt_eval['spt_heuristic']['cavern_precision']:.2f}%, Rec={gt_eval['spt_heuristic']['cavern_recall']:.2f}%)"
                )

            model_results["caves"][cname] = cave_stats

            scalars_out = {
                "scalar_utonia_kmeans_k3": km_res[3],
                "scalar_utonia_kmeans_k5": km_res[5],
                "scalar_utonia_linear_pred": lin_pred,
                "scalar_utonia_linear_cavern_prob": lin_cav_prob,
                "scalar_utonia_mlp_pred": mlp_pred,
                "scalar_utonia_mlp_cavern_prob": mlp_cav_prob,
                "scalar_utonia_hybrid_pred": hyb_pred,
                "scalar_utonia_hybrid_cavern_prob": hyb_cav_prob,
                "scalar_spt_pred": cdata["spt_pred"],
                "scalar_spt_cave_zone": cdata["spt_zone"],
            }
            if cdata["gt"] is not None:
                scalars_out["scalar_ground_truth"] = cdata["gt"]

            out_ply = ROOT_DIR / "output/utonia" / f"utonia_{cname}_{model_name}.ply"
            export_utonia_diagnostic_ply(
                out_ply,
                xyz,
                cdata["centroid_f64"],
                nrm,
                pca_rgb,
                scalars=scalars_out,
            )
            print(f"    -> Exported CloudCompare PLY: {out_ply}")
            if cname == "Lp_JL":
                del feat_jl, pca_jl
            del feat, pca_rgb, km_res, lin_pred, mlp_pred, hyb_pred
            gc.collect()

        summary_report[model_name] = model_results
        del model, linear_probe, mlp_probe, hybrid_probe
        gc.collect()
        torch.cuda.empty_cache()

        out_json = ROOT_DIR / "output/utonia/utonia_cave_evaluation.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2)

    print(f"\nSaved full evaluation summary -> {ROOT_DIR / 'output/utonia/utonia_cave_evaluation.json'}")


if __name__ == "__main__":
    main()
