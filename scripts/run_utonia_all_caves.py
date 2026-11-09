#!/usr/bin/env python3
"""
Run Pointcept/Utonia across all 25 caves in our speleological dataset for a direct,
point-for-point comparison against the Fine-Tuned SPT SuperpointGraphSAGE Classifier
and the Stage 5 SPT Geometric Heuristic.

Evaluation & Export Protocol:
  1. Strict Leave-One-Cave-Out (LOCO) probes for the 4 held-out forested caves
     (S11A_0257, Lp_JL, Lp_Boi, gruta_do_janelao) — zero cave-system leakage.
  2. Production probes (trained on all annotated caves + FOR-instance forestry donors)
     for the remaining 21 caves.
  3. Three Utonia segmentation heads evaluated per cave:
     - Utonia Linear Probe (SegHead) : Linear layer on pure Utonia PTv3 embeddings (wo_color=True)
     - Utonia MLP Probe              : 2-layer MLP on pure Utonia PTv3 embeddings (wo_color=True)
     - Utonia + Enclosure Hybrid     : Two-stream probe fusing Utonia embeddings + 10-D 3D enclosure
  4. Exports full point-level labeled PLYs for every cave in output/utonia/:
     - output/utonia/<cave>_utonia_diagnostic.ply
     - output/utonia/<cave>_utonia_cleaned.ply
     - output/utonia/<cave>_utonia_full_labeled.ply (for annotated full-res caves)
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

ROOT = Path(__file__).resolve().parents[1]
UTONIA_DIR = ROOT / "third_party" / "Utonia"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(UTONIA_DIR) not in sys.path:
    sys.path.insert(0, str(UTONIA_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utonia
from spt_filter import load_spt_kernels, read_ply, write_ply_binary
from spt_learning import semantic_metrics
from scripts.run_spt_finetuned_all_caves import CAVE_CONFIGS


ENCL_DIAG_KEYS = [
    "scalar_roof_occlusion",
    "scalar_downward_occlusion",
    "scalar_cavern_enclosure",
    "scalar_drip_line_dist",
    "scalar_verticality",
    "scalar_horizontality",
    "scalar_signed_horizontality",
    "scalar_planarity_fine",
    "scalar_scattering_fine",
    "scalar_normal_dispersion",
]


def extract_encl_from_diag_scalars(scalars: dict) -> np.ndarray:
    """Build sanitized 10-D physical enclosure + local geometry matrix from diagnostic PLY scalars."""
    cols = []
    for k in ENCL_DIAG_KEYS:
        v = scalars[k].astype(np.float32)
        if k == "scalar_drip_line_dist":
            v = np.tanh(v / 5.0)
        cols.append(np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0))
    return np.column_stack(cols).astype(np.float32)


def compute_forest_p0_and_enclosure(lib: ctypes.CDLL, path: Path, voxel_size: float = 0.06):
    """Subsample annotated FOR-instance forest PLY and compute 10-D features matching ENCL_DIAG_KEYS."""
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
    # In FOR-instance: semantic_seg == 1 is terrain/ground, > 1 is tree/stem/crown
    # For binary/3-class cave vs non-cave segmentation, map non-cave vegetation/ground to 1 (Vegetation/Exterior)
    labels = np.ones(num_p0, dtype=np.int64)

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

    nz = normals[:, 2]
    horiz = np.abs(nz)
    vert = 1.0 - horiz
    # Open-sky forest has drip_line_dist > +10m -> tanh(10/5) ~ +0.964
    drip_tanh = np.full(num_p0, np.tanh(2.0), dtype=np.float32)

    encl_vec = np.column_stack([
        encl[:, 1],        # scalar_roof_occlusion
        encl[:, 2],        # scalar_downward_occlusion
        encl[:, 3],        # scalar_cavern_enclosure
        drip_tanh,         # tanh(scalar_drip_line_dist / 5.0)
        vert,              # scalar_verticality
        horiz,             # scalar_horizontality
        nz,                # scalar_signed_horizontality
        fine_feats[:, 1],  # scalar_planarity_fine
        fine_feats[:, 2],  # scalar_scattering_fine
        fine_feats[:, 5],  # scalar_normal_dispersion
    ]).astype(np.float32)
    return p0_xyz, normals, encl_vec, labels


def forward_utonia_single_chunk(
    model: nn.Module,
    transform,
    c_xyz: np.ndarray,
    c_nrm: np.ndarray,
    upcast_levels: int = 2,
) -> torch.Tensor:
    """Run Utonia encoder on a single chunk (< 300k points) and return float16 CPU features."""
    zero_color = np.zeros((1, 3), dtype=np.float32)
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


def get_spatial_chunks(coord: np.ndarray, max_chunk_pts: int = 280_000):
    """Split point cloud into spatial slabs along its longest horizontal axis."""
    n_pts = coord.shape[0]
    if n_pts <= max_chunk_pts:
        return [np.arange(n_pts, dtype=np.int64)]
    n_chunks = int(np.ceil(n_pts / max_chunk_pts))
    xy = coord[:, :2]
    axis = int(np.argmax(xy.max(axis=0) - xy.min(axis=0)))
    order = np.argsort(coord[:, axis])
    return [np.asarray(ch, dtype=np.int64) for ch in np.array_split(order, n_chunks)]


def extract_training_samples_from_cloud(
    model: nn.Module,
    coord: np.ndarray,
    normal: np.ndarray,
    encl: np.ndarray,
    gt: np.ndarray,
    scale: float,
    rng: np.random.Generator,
    max_per_class: int = 45_000,
    max_chunk_pts: int = 280_000,
    max_train_chunks: int = 4,
):
    """
    Extract Utonia features for a training cloud in memory-bounded spatial slabs,
    retaining only a balanced random sample of known (gt >= 0) points so host RAM stays minimal.
    """
    chunks = get_spatial_chunks(coord, max_chunk_pts=max_chunk_pts)
    if len(chunks) > max_train_chunks:
        # Select the most informative full-density spatial slabs (richest in Cavern & Vegetation)
        c0_counts = [int(np.sum(gt[ch] == 0)) for ch in chunks]
        c1_counts = [int(np.sum(gt[ch] == 1)) for ch in chunks]
        chosen = set(np.argsort(c0_counts)[-(max_train_chunks // 2):].tolist())
        chosen.update(np.argsort(c1_counts)[-(max_train_chunks - len(chosen)):].tolist())
        chunks = [chunks[i] for i in sorted(chosen)]

    candidate_mask = np.zeros(len(coord), dtype=bool)
    for ch in chunks:
        candidate_mask[ch] = True

    selected_list = []
    for cls_id in (0, 1):
        cls_idx = np.flatnonzero((gt == cls_id) & candidate_mask)
        if len(cls_idx) > 0:
            take = min(max_per_class, len(cls_idx))
            selected_list.append(rng.choice(cls_idx, size=take, replace=False))
    if not selected_list:
        return None, None, None
    selected_idx = np.sort(np.concatenate(selected_list))

    n_pts = len(coord)
    keep_pos = np.full(n_pts, -1, dtype=np.int32)
    keep_pos[selected_idx] = np.arange(len(selected_idx), dtype=np.int32)

    transform = utonia.transform.default(
        scale=scale, apply_z_positive=True, normalize_coord=False
    )
    feat_sampled = None

    for ch_idx in chunks:
        local_keep = keep_pos[ch_idx]
        if not np.any(local_keep >= 0):
            continue
        ch_feat = forward_utonia_single_chunk(model, transform, coord[ch_idx], normal[ch_idx]).numpy()
        if feat_sampled is None:
            feat_sampled = np.empty((len(selected_idx), ch_feat.shape[1]), dtype=np.float16)
        mask = local_keep >= 0
        feat_sampled[local_keep[mask]] = ch_feat[mask]
        del ch_feat
        torch.cuda.empty_cache()

    return feat_sampled, encl[selected_idx].copy(), gt[selected_idx].astype(np.int64)


class UtoniaLinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.seg_head = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg_head(self.norm(x))


class UtoniaMLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 2):
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
    """Two-stream fusion: balances Utonia PTv3 embeddings with 10-D physical 3D enclosure."""

    def __init__(self, u_dim: int, e_dim: int = 10, hidden_dim: int = 64, num_classes: int = 2):
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


def train_probe_trio(X_utonia_f16: np.ndarray, X_encl: np.ndarray, y: np.ndarray, epochs: int = 24):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    u_dim = X_utonia_f16.shape[1]
    e_dim = X_encl.shape[1]

    y_bin = (y > 0).astype(np.int64)
    counts = np.bincount(y_bin, minlength=2).astype(np.float32)
    weights = np.clip(counts.sum() / (2.0 * np.maximum(counts, 1.0)), 0.5, 3.0)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    linear_probe = UtoniaLinearProbe(u_dim, 2).to(device)
    mlp_probe = UtoniaMLPProbe(u_dim, hidden_dim=128, num_classes=2).to(device)
    hybrid_probe = UtoniaHybridProbe(u_dim, e_dim=e_dim, hidden_dim=64, num_classes=2).to(device)

    X_u_t = torch.from_numpy(X_utonia_f16)
    X_e_t = torch.from_numpy(X_encl)
    y_t = torch.from_numpy(y_bin)

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
                loss = F.cross_entropy(logits, by, weight=class_weights, label_smoothing=0.05)
                loss.backward()
                opt.step()
        probe.eval()

    return linear_probe, mlp_probe, hybrid_probe


@torch.inference_mode()
def predict_probes_on_chunk(
    linear_probe: nn.Module,
    mlp_probe: nn.Module,
    hybrid_probe: nn.Module,
    feat_f16: np.ndarray,
    encl_f32: np.ndarray,
    batch_size: int = 65536,
):
    device = next(linear_probe.parameters()).device
    n = feat_f16.shape[0]
    lin_pred = np.empty(n, dtype=np.int32)
    mlp_pred = np.empty(n, dtype=np.int32)
    hyb_pred = np.empty(n, dtype=np.int32)
    mlp_prob = np.empty(n, dtype=np.float32)
    hyb_prob = np.empty(n, dtype=np.float32)

    for i in range(0, n, batch_size):
        bu = torch.from_numpy(feat_f16[i : i + batch_size]).to(device, dtype=torch.float32, non_blocking=True)
        be = torch.from_numpy(encl_f32[i : i + batch_size]).to(device, dtype=torch.float32, non_blocking=True)

        p_lin = F.softmax(linear_probe(bu), dim=-1).cpu().numpy()
        p_mlp = F.softmax(mlp_probe(bu), dim=-1).cpu().numpy()
        p_hyb = F.softmax(hybrid_probe(bu, be), dim=-1).cpu().numpy()

        lin_pred[i : i + batch_size] = np.argmax(p_lin, axis=-1).astype(np.int32)
        mlp_pred[i : i + batch_size] = np.argmax(p_mlp, axis=-1).astype(np.int32)
        hyb_pred[i : i + batch_size] = np.argmax(p_hyb, axis=-1).astype(np.int32)
        mlp_prob[i : i + batch_size] = p_mlp[:, 0]
        hyb_prob[i : i + batch_size] = p_hyb[:, 0]

    return lin_pred, mlp_pred, hyb_pred, mlp_prob, hyb_prob


def run_utonia_inference_on_cave(
    model: nn.Module,
    coord: np.ndarray,
    normal: np.ndarray,
    encl: np.ndarray,
    scale: float,
    linear_probe: nn.Module,
    mlp_probe: nn.Module,
    hybrid_probe: nn.Module,
    max_chunk_pts: int = 280_000,
):
    """
    Stream Utonia feature extraction + probe inference + PCA RGB projection chunk-by-chunk
    so even 6.6M-point P0 clouds use < 800 MB CPU RAM and < 1.6 GB GPU VRAM.
    """
    n_pts = coord.shape[0]
    transform = utonia.transform.default(
        scale=scale, apply_z_positive=True, normalize_coord=False
    )
    chunks = get_spatial_chunks(coord, max_chunk_pts=max_chunk_pts)

    lin_pred_all = np.empty(n_pts, dtype=np.int32)
    mlp_pred_all = np.empty(n_pts, dtype=np.int32)
    hyb_pred_all = np.empty(n_pts, dtype=np.int32)
    mlp_prob_all = np.empty(n_pts, dtype=np.float32)
    hyb_prob_all = np.empty(n_pts, dtype=np.float32)
    pca_rgb_all = np.empty((n_pts, 3), dtype=np.uint8)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    pca_basis = None
    feat_dim = 0

    for ch_idx in chunks:
        ch_feat_t = forward_utonia_single_chunk(model, transform, coord[ch_idx], normal[ch_idx])
        ch_feat = ch_feat_t.numpy()
        feat_dim = int(ch_feat.shape[1])

        # Fit PCA basis on the first chunk (up to 120k points) for consistent RGB coloring
        if pca_basis is None:
            rng = np.random.default_rng(42)
            sub_n = min(120_000, len(ch_feat))
            sub_i = rng.choice(len(ch_feat), size=sub_n, replace=False)
            f_sub = torch.from_numpy(ch_feat[sub_i].astype(np.float32))
            mean_f = f_sub.mean(dim=0, keepdim=True)
            _, _, v = torch.pca_lowrank(f_sub - mean_f, center=False, q=12, niter=4)
            proj_sub = (f_sub - mean_f) @ v
            rgb_sub = proj_sub[:, :3] * 0.4 + proj_sub[:, 3:6] * 0.2 + proj_sub[:, 9:12] * 0.4
            lo = torch.quantile(rgb_sub, 0.02, dim=0, keepdim=True)
            hi = torch.quantile(rgb_sub, 0.98, dim=0, keepdim=True)
            div = torch.clamp(hi - lo, min=1e-6)
            pca_basis = (mean_f, v, lo, div)
            del f_sub, proj_sub, rgb_sub

        mean_f, v, lo, div = pca_basis
        bsize = 131072
        ch_rgb = np.empty((len(ch_idx), 3), dtype=np.uint8)
        for i in range(0, len(ch_idx), bsize):
            bf = torch.from_numpy(ch_feat[i : i + bsize].astype(np.float32))
            bp = (bf - mean_f) @ v
            brgb = bp[:, :3] * 0.4 + bp[:, 3:6] * 0.2 + bp[:, 9:12] * 0.4
            bcol = ((brgb - lo) / div * 1.15).clamp(0.0, 1.0)
            ch_rgb[i : i + bsize] = (bcol.numpy() * 255.0).astype(np.uint8)
        pca_rgb_all[ch_idx] = ch_rgb

        lp, mp, hp, mprob, hprob = predict_probes_on_chunk(
            linear_probe, mlp_probe, hybrid_probe, ch_feat, encl[ch_idx]
        )
        lin_pred_all[ch_idx] = lp
        mlp_pred_all[ch_idx] = mp
        hyb_pred_all[ch_idx] = hp
        mlp_prob_all[ch_idx] = mprob
        hyb_prob_all[ch_idx] = hprob

        del ch_feat_t, ch_feat, ch_rgb
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
    return (
        lin_pred_all,
        mlp_pred_all,
        hyb_pred_all,
        mlp_prob_all,
        hyb_prob_all,
        pca_rgb_all,
        feat_dim,
        elapsed,
        peak_vram_mb,
    )


def export_utonia_ply_with_rgb(
    out_path: Path,
    xyz_world: np.ndarray,
    normals: np.ndarray,
    pca_rgb: np.ndarray,
    scalars: dict,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
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


def compute_orig_to_p0_map(lib: ctypes.CDLL, xyz_f64: np.ndarray, voxel_size: float) -> np.ndarray:
    """Recompute deterministic orig_to_p0 voxel index mapping in C++ (<0.15s for 10M points)."""
    f32_p = ctypes.POINTER(ctypes.c_float)
    i32_p = ctypes.POINTER(ctypes.c_int32)
    n_orig = len(xyz_f64)
    centroid_f64 = xyz_f64.mean(axis=0, keepdims=True)
    xyz_centered = np.ascontiguousarray(xyz_f64 - centroid_f64, dtype=np.float32)
    p0_to_orig = np.empty(n_orig, dtype=np.int32)
    orig_to_p0 = np.empty(n_orig, dtype=np.int32)
    lib.spt_voxel_subsample(
        xyz_centered.ctypes.data_as(f32_p),
        ctypes.c_int32(n_orig),
        ctypes.c_float(voxel_size),
        p0_to_orig.ctypes.data_as(i32_p),
        orig_to_p0.ctypes.data_as(i32_p),
    )
    return orig_to_p0


def main():
    parser = argparse.ArgumentParser(description="Run Pointcept/Utonia across all 25 caves")
    parser.add_argument("--model", default="utonia", help="Utonia checkpoint name (default: utonia)")
    args = parser.parse_args()

    t_start = time.perf_counter()
    utonia.utils.set_seed(42)
    rng = np.random.default_rng(42)

    lib = load_spt_kernels(ROOT / "build/libspt_kernels.so")
    spt_dir = ROOT / "output/spt_finetuned"
    out_dir = ROOT / "output/utonia"
    out_dir.mkdir(parents=True, exist_ok=True)

    spt_results_path = spt_dir / "all_caves_results.json"
    spt_all_results = json.loads(spt_results_path.read_text()) if spt_results_path.exists() else {}

    print("=" * 90)
    print(f"LOADING UTONIA BACKBONE ({args.model}) ON GPU")
    print("=" * 90, flush=True)
    custom_config = dict(enc_patch_size=[512 for _ in range(5)], enable_flash=False)
    model = utonia.load(args.model, custom_config=custom_config).cuda().eval()

    # -------------------------------------------------------------------------
    # STEP 1: Build Training Feature Bank from Annotated Caves + FOR-instance
    # -------------------------------------------------------------------------
    print("\n[Step 1/3] Extracting Utonia Training Feature Bank (9 annotated caves + 3 FOR-instance plots)...", flush=True)
    training_bank = {}

    for cfg in CAVE_CONFIGS:
        if not cfg["annotated"]:
            continue
        name = cfg["name"]
        vs = float(cfg["voxel_size"])
        scale = float(np.clip(0.01 / max(vs, 0.02), 0.05, 0.25))
        diag_path = spt_dir / f"{name}_diagnostic.ply"
        xyz_f64, nrm, sc = read_ply(diag_path)
        xyz_c = np.ascontiguousarray(xyz_f64 - xyz_f64.mean(axis=0, keepdims=True), dtype=np.float32)
        nrm = np.ascontiguousarray(nrm, dtype=np.float32)
        encl = extract_encl_from_diag_scalars(sc)
        gt = sc["scalar_ground_truth"].astype(np.int32)

        u_sub, e_sub, y_sub = extract_training_samples_from_cloud(
            model, xyz_c, nrm, encl, gt, scale=scale, rng=rng, max_per_class=45_000
        )
        if u_sub is not None:
            training_bank[name] = (u_sub, e_sub, y_sub)
            print(
                f"  + Cave {name:20s}: sampled {len(y_sub):7,d} pts "
                f"(Cavern={int(np.sum(y_sub==0)):,}, Veg={int(np.sum(y_sub==1)):,})",
                flush=True,
            )
        del xyz_f64, xyz_c, nrm, sc, encl, gt
        gc.collect()

    for fname in (
        "NIBIO2_NIBIO2_plot12_annotated_train.ply",
        "NIBIO2_NIBIO2_plot14_annotated_train.ply",
        "CULS_CULS_plot_1_annotated_train.ply",
    ):
        fpath = ROOT / "data/trees/train_val_data" / fname
        f_xyz, f_nrm, f_encl, f_lbl = compute_forest_p0_and_enclosure(lib, fpath, voxel_size=0.06)
        u_sub, e_sub, y_sub = extract_training_samples_from_cloud(
            model, f_xyz, f_nrm, f_encl, f_lbl.astype(np.int32), scale=0.20, rng=rng, max_per_class=40_000
        )
        if u_sub is not None:
            training_bank[fname] = (u_sub, e_sub, y_sub)
            print(f"  + Forest {fname:34s}: sampled {len(y_sub):7,d} pts (Veg/Ext={len(y_sub):,})", flush=True)
        del f_xyz, f_nrm, f_encl, f_lbl
        gc.collect()

    # -------------------------------------------------------------------------
    # STEP 2: Train Held-Out LOCO Probes (4 Forested Caves) + Production Probes
    # -------------------------------------------------------------------------
    print("\n[Step 2/3] Training Held-Out LOCO Probes & Production Probes...", flush=True)
    loco_caves = ["S11A_0257", "Lp_JL", "Lp_Boi", "gruta_do_janelao"]
    probe_sets = {}

    for held_out in loco_caves:
        u_list, e_list, y_list = [], [], []
        for k, (u_arr, e_arr, y_arr) in training_bank.items():
            if k == held_out:
                continue
            u_list.append(u_arr)
            e_list.append(e_arr)
            y_list.append(y_arr)
        t_p0 = time.perf_counter()
        probe_sets[held_out] = train_probe_trio(
            np.vstack(u_list), np.vstack(e_list), np.concatenate(y_list)
        )
        print(
            f"  Trained LOCO probe set (held-out: {held_out:18s}) on {sum(len(y) for y in y_list):,} pts in {time.perf_counter()-t_p0:.2f}s",
            flush=True,
        )

    u_all = np.vstack([v[0] for v in training_bank.values()])
    e_all = np.vstack([v[1] for v in training_bank.values()])
    y_all = np.concatenate([v[2] for v in training_bank.values()])
    t_p0 = time.perf_counter()
    probe_sets["production"] = train_probe_trio(u_all, e_all, y_all)
    print(
        f"  Trained Production probe set (all sources)        on {len(y_all):,} pts in {time.perf_counter()-t_p0:.2f}s",
        flush=True,
    )
    del training_bank, u_all, e_all, y_all
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # STEP 3: Run Point-Level Utonia Inference Across All 25 Caves
    # -------------------------------------------------------------------------
    print("\n[Step 3/3] Running Point-Level Utonia Segmentation Across All 25 Caves...", flush=True)
    all_utonia_reports = {}

    for idx, cfg in enumerate(CAVE_CONFIGS, 1):
        name = cfg["name"]
        vs = float(cfg["voxel_size"])
        scale = float(np.clip(0.01 / max(vs, 0.02), 0.05, 0.25))
        probe_key = name if name in probe_sets else "production"
        linear_probe, mlp_probe, hybrid_probe = probe_sets[probe_key]

        t_c0 = time.perf_counter()
        diag_in_path = spt_dir / f"{name}_diagnostic.ply"
        p0_xyz_f64, p0_normals, sc = read_ply(diag_in_path)
        n_p0 = len(p0_xyz_f64)
        centroid_f64 = p0_xyz_f64.mean(axis=0, keepdims=True)
        p0_xyz_c = np.ascontiguousarray(p0_xyz_f64 - centroid_f64, dtype=np.float32)
        p0_normals = np.ascontiguousarray(p0_normals, dtype=np.float32)
        p0_encl = extract_encl_from_diag_scalars(sc)

        (
            p0_lin_pred,
            p0_mlp_pred,
            p0_hyb_pred,
            p0_mlp_prob,
            p0_hyb_prob,
            p0_pca_rgb,
            feat_dim,
            utonia_sec,
            peak_vram_mb,
        ) = run_utonia_inference_on_cave(
            model,
            p0_xyz_c,
            p0_normals,
            p0_encl,
            scale=scale,
            linear_probe=linear_probe,
            mlp_probe=mlp_probe,
            hybrid_probe=hybrid_probe,
        )

        # Export P0 Utonia Diagnostic PLY (with PCA RGB + all Utonia & SPT predictions)
        utonia_diag_path = out_dir / f"{name}_utonia_diagnostic.ply"
        scalars_diag = {
            "scalar_utonia_linear_class": p0_lin_pred,
            "scalar_utonia_mlp_class": p0_mlp_pred,
            "scalar_utonia_hybrid_class": p0_hyb_pred,
            "scalar_utonia_mlp_cavern_prob": p0_mlp_prob,
            "scalar_utonia_hybrid_cavern_prob": p0_hyb_prob,
            "scalar_spt_finetuned_class": sc["scalar_predicted_class"].astype(np.int32),
            "scalar_spt_heuristic_class": sc["scalar_heuristic_class"].astype(np.int32),
            "scalar_spt_cave_zone": sc["scalar_cave_zone"].astype(np.int32),
            "scalar_cavern_enclosure": sc["scalar_cavern_enclosure"].astype(np.float32),
        }
        if "scalar_ground_truth" in sc:
            scalars_diag["scalar_ground_truth"] = sc["scalar_ground_truth"].astype(np.int32)

        export_utonia_ply_with_rgb(
            utonia_diag_path,
            p0_xyz_f64,
            p0_normals,
            p0_pca_rgb,
            scalars_diag,
        )

        # Map P0 predictions to full-resolution original point cloud (n_orig points)
        orig_ply_path = ROOT / cfg["path"]
        if cfg["already_p0"]:
            xyz_orig_f64 = p0_xyz_f64
            orig_normals = p0_normals
            n_orig = n_p0
            orig_to_p0 = np.arange(n_orig, dtype=np.int32)
            truth_orig = sc.get("scalar_ground_truth", None)
        else:
            xyz_orig_f64, _, extras_orig = read_ply(orig_ply_path)
            n_orig = len(xyz_orig_f64)
            orig_to_p0 = compute_orig_to_p0_map(lib, xyz_orig_f64, vs)
            orig_normals = p0_normals[orig_to_p0]
            truth_orig = extras_orig.get("scalar_ground_truth", None)

        orig_lin_pred = p0_lin_pred[orig_to_p0]
        orig_mlp_pred = p0_mlp_pred[orig_to_p0]
        orig_hyb_pred = p0_hyb_pred[orig_to_p0]
        orig_mlp_prob = p0_mlp_prob[orig_to_p0]
        orig_hyb_prob = p0_hyb_prob[orig_to_p0]
        orig_spt_pred = sc["scalar_predicted_class"].astype(np.int32)[orig_to_p0]

        # Export Full-Resolution Utonia Cleaned PLY (using pure Utonia MLP Probe & Hybrid Probe)
        utonia_clean_path = out_dir / f"{name}_utonia_cleaned.ply"
        keep_mask = orig_mlp_pred == 0
        write_ply_binary(
            utonia_clean_path,
            xyz_orig_f64[keep_mask],
            orig_normals[keep_mask],
            {
                "scalar_utonia_mlp_cavern_prob": orig_mlp_prob[keep_mask],
                "scalar_utonia_hybrid_class": orig_hyb_pred[keep_mask],
            },
        )

        full_labeled_path = None
        if cfg["annotated"] and not cfg["already_p0"] and truth_orig is not None:
            full_labeled_path = out_dir / f"{name}_utonia_full_labeled.ply"
            write_ply_binary(
                full_labeled_path,
                xyz_orig_f64,
                orig_normals,
                {
                    "scalar_utonia_linear_class": orig_lin_pred,
                    "scalar_utonia_mlp_class": orig_mlp_pred,
                    "scalar_utonia_hybrid_class": orig_hyb_pred,
                    "scalar_utonia_mlp_cavern_prob": orig_mlp_prob,
                    "scalar_utonia_hybrid_cavern_prob": orig_hyb_prob,
                    "scalar_spt_finetuned_class": orig_spt_pred,
                    "scalar_ground_truth": truth_orig.astype(np.int32),
                },
            )

        metrics_lin = None
        metrics_mlp = None
        metrics_hyb = None
        if cfg["annotated"] and truth_orig is not None:
            truth_i32 = truth_orig.astype(np.int32)
            metrics_lin = semantic_metrics(truth_i32, orig_lin_pred)
            metrics_mlp = semantic_metrics(truth_i32, orig_mlp_pred)
            metrics_hyb = semantic_metrics(truth_i32, orig_hyb_pred)

        spt_rep = spt_all_results.get(name, {})
        elapsed_total = time.perf_counter() - t_c0

        report = {
            "name": name,
            "eval_mode": cfg["eval_mode"],
            "voxel_size": vs,
            "original_points": int(n_orig),
            "p0_points": int(n_p0),
            "feat_dim": feat_dim,
            "utonia_inference_sec": round(float(utonia_sec), 2),
            "total_elapsed_sec": round(float(elapsed_total), 2),
            "peak_vram_mb": round(float(peak_vram_mb), 1),
            # Point-level Cavern retention percentages across all 4 methods
            "utonia_linear_cavern_points": int(np.sum(orig_lin_pred == 0)),
            "utonia_linear_cavern_pct": round(100.0 * float(np.mean(orig_lin_pred == 0)), 2),
            "utonia_mlp_cavern_points": int(np.sum(orig_mlp_pred == 0)),
            "utonia_mlp_cavern_pct": round(100.0 * float(np.mean(orig_mlp_pred == 0)), 2),
            "utonia_hybrid_cavern_points": int(np.sum(orig_hyb_pred == 0)),
            "utonia_hybrid_cavern_pct": round(100.0 * float(np.mean(orig_hyb_pred == 0)), 2),
            "spt_finetuned_cavern_points": int(np.sum(orig_spt_pred == 0)),
            "spt_finetuned_cavern_pct": round(100.0 * float(np.mean(orig_spt_pred == 0)), 2),
            # Agreement with Fine-Tuned SPT
            "mlp_spt_agreement_pct": round(100.0 * float(np.mean((orig_mlp_pred == 0) == (orig_spt_pred == 0))), 2),
            "hybrid_spt_agreement_pct": round(100.0 * float(np.mean((orig_hyb_pred == 0) == (orig_spt_pred == 0))), 2),
            # Ground-truth metrics on annotated caves
            "utonia_linear_metrics": metrics_lin,
            "utonia_mlp_metrics": metrics_mlp,
            "utonia_hybrid_metrics": metrics_hyb,
            "spt_finetuned_metrics": spt_rep.get("finetuned_metrics"),
            "spt_heuristic_metrics": spt_rep.get("heuristic_metrics"),
            # Output file paths
            "diagnostic_ply": str(utonia_diag_path.relative_to(ROOT)),
            "cleaned_ply": str(utonia_clean_path.relative_to(ROOT)),
            "full_labeled_ply": (
                str(full_labeled_path.relative_to(ROOT))
                if full_labeled_path
                else str(utonia_diag_path.relative_to(ROOT))
            ),
        }
        all_utonia_reports[name] = report

        msg = (
            f"  [{idx:2d}/25] {name:24s} | Orig={n_orig:9,d} | "
            f"UtoniaMLP Cav={report['utonia_mlp_cavern_pct']:5.1f}% | "
            f"UtoniaHyb Cav={report['utonia_hybrid_cavern_pct']:5.1f}% | "
            f"SPT-FT Cav={report['spt_finetuned_cavern_pct']:5.1f}% ({utonia_sec:.1f}s)"
        )
        if metrics_mlp and report["spt_finetuned_metrics"]:
            u_ciou = (metrics_mlp["iou"]["cavern"] or 0.0) * 100.0
            h_ciou = (metrics_hyb["iou"]["cavern"] or 0.0) * 100.0
            s_ciou = (report["spt_finetuned_metrics"]["iou"]["cavern"] or 0.0) * 100.0
            msg += f" | GT CavIoU: UtoniaMLP={u_ciou:.2f}%, UtoniaHyb={h_ciou:.2f}%, SPT-FT={s_ciou:.2f}%"
        print(msg, flush=True)

        del p0_xyz_f64, p0_xyz_c, p0_normals, sc, p0_encl, xyz_orig_f64, orig_normals
        gc.collect()
        torch.cuda.empty_cache()

    out_json = out_dir / "all_caves_utonia_results.json"
    out_json.write_text(json.dumps(all_utonia_reports, indent=2) + "\n")
    print(
        f"\n[Done] Completed Utonia evaluation across all {len(CAVE_CONFIGS)} caves in {time.perf_counter()-t_start:.1f}s -> {out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
