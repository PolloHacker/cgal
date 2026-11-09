#!/usr/bin/env python3
"""
Train and evaluate the Fine-Tuned SPT SuperpointGraphSAGE Classifier across all
annotated caves (pure interiors + 50 FOR-instance canopy donor chunks + forested caves)
using strict Cave-System Held-Out / Leave-One-Cave-Out (LOCO) validation, and export
CloudCompare diagnostic PLYs for visual inspection.
"""

import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spt_filter import read_ply, write_ply_binary
from spt_learning import (
    CLASS_NAMES,
    SCHEMA,
    cave_system_id,
    filter_cavern_manifold,
    load_graph,
    portable_predict,
    semantic_metrics,
)
from spt_torch import train_graphs


def sanitize_cross_scale_features(graph):
    """Ensure shape_radius and Stage-5 elevation anchors are scale-invariant across 4cm-35cm scans."""
    g = dict(graph)
    X = g["X"].copy()
    names = g["feature_names"]
    vs = float(g["metadata"].get("voxel_size", 0.04))

    sr_idx = int(np.flatnonzero(names == "shape_radius")[0])
    r = np.arctanh(np.clip(X[:, sr_idx], -0.999, 0.999)) * 5.0
    X[:, sr_idx] = np.tanh(r / (25.0 * vs))

    # Fix Stage-5 drip_tanh on 2km-wide canyon scans (voxel_size >= 0.25m) where hardcoded 5m radius fragmented anchors
    if vs >= 0.25:
        dr_idx = int(np.flatnonzero(names == "drip_tanh")[0])
        enc_idx = int(np.flatnonzero(names == "mean_enclosure")[0])
        ro_idx = int(np.flatnonzero(names == "mean_roof_occlusion")[0])
        is_enclosed = (X[:, enc_idx] > 0.24) | (X[:, ro_idx] > 0.20)
        X[:, dr_idx] = np.where(is_enclosed, np.minimum(X[:, dr_idx], 0.08), X[:, dr_idx])

    g["X"] = X
    return g


def main():
    t0 = time.perf_counter()
    cache_dir = ROOT / "output/spt_graph_cache"
    out_dir = ROOT / "output/spt_finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_graphs = []
    for p in sorted(cache_dir.glob("*.npz")):
        with np.load(p, allow_pickle=False) as a:
            if str(a["schema"]) != SCHEMA:
                continue
        all_graphs.append(sanitize_cross_scale_features(load_graph(p)))

    by_csys = {
        g["metadata"]["cave_system"]: g
        for g in all_graphs
        if g["metadata"]["role"] == "entrance"
    }
    cutmix_list = [g for g in all_graphs if g["metadata"]["role"] == "cutmix"]
    interior_list = [
        g for g in all_graphs
        if g["metadata"]["role"] == "interior"
        and g["metadata"]["cave_system"] in ("rupt", "baume_de_longeaigue", "diaclase", "lapinha")
    ]

    forested_order = ["S11A_0257", "lp_jl", "lp_boi", "gruta_do_janelao"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load previous heuristic metrics from validation JSON for comparison
    val_json_path = out_dir / "spt_graphsage.validation.json"
    prev_heur = {}
    if val_json_path.exists():
        val_data = json.loads(val_json_path.read_text())
        for case in val_data.get("cases", []):
            prev_heur[case["cave_system"]] = case["heuristic"]

    loco_results = {}
    for held_out in forested_order:
        print(f"\n=== [Held-Out Cave: {held_out}] Training on remaining caves + FOR-instance CutMix ===", flush=True)
        train_set = (
            [g for c, g in by_csys.items() if c != held_out]
            + cutmix_list
            + interior_list
        )
        # Verify zero cave-system leakage
        train_groups = {cave_system_id(g["metadata"]["cave_system"]) for g in train_set}
        assert cave_system_id(held_out) not in train_groups, f"Leakage detected for {held_out}!"

        # For caves without downslope roof-arch ambiguity, zero out CutMix-sensitive height anchors
        names = train_set[0]["feature_names"]
        ar_idx = int(np.flatnonzero(names == "above_roof_tanh")[0])
        fh_idx = int(np.flatnonzero(names == "floor_height_tanh")[0])
        fold_train = []
        for g in train_set:
            gc = dict(g)
            gc["X"] = g["X"].copy()
            gc["X"][:, [ar_idx, fh_idx]] = 0.0
            fold_train.append(gc)

        val_g = dict(by_csys[held_out])
        val_g["X"] = val_g["X"].copy()
        val_g["X"][:, [ar_idx, fh_idx]] = 0.0

        model = train_graphs(fold_train, epochs=60, device=device, seed=42)
        fold_model_path = out_dir / f"spt_graphsage_loco_{held_out}.npz"
        meta = {
            "training_groups": sorted(train_groups),
            "held_out": held_out,
            "epochs": 60,
            "device": device,
        }
        model.export(fold_model_path, val_g["feature_names"], meta)
        torch.save(
            {"state_dict": model.cpu().state_dict(), "metadata": meta, "feature_names": val_g["feature_names"].tolist()},
            fold_model_path.with_suffix(".pt"),
        )

        # Portable CPU inference + 3D connected cavern manifold filter
        probs = portable_predict(
            val_g["X"], val_g["edge_u"], val_g["edge_v"], val_g["sp_pos"],
            fold_model_path, val_g["feature_names"]
        )
        sp_pred = filter_cavern_manifold(
            probs.argmax(axis=1).astype(np.int32),
            val_g["edge_u"], val_g["edge_v"], val_g["sp_pos"], val_g["sp_counts"],
            probabilities=probs,
        )

        # Evaluate on the exact annotated point cloud
        ann_path = Path(val_g["metadata"]["source"]["path"])
        xyz, normals, fields = read_ply(ann_path)
        truth = fields["scalar_ground_truth"].astype(np.int32)

        # Compute exact point-level metrics from superpoint histogram (since partition_oracle == 1.0)
        hist = val_g["histogram"]
        mat = np.zeros((3, 3), dtype=np.int64)
        for c_pred in range(3):
            mask = sp_pred == c_pred
            if np.any(mask):
                mat[:, c_pred] = hist[mask, :3].sum(axis=0).astype(np.int64)
        union = mat.sum(axis=0) + mat.sum(axis=1) - np.diag(mat)
        iou = np.divide(np.diag(mat), union, out=np.full(3, np.nan), where=union > 0)
        cav_rec = float(mat[0, 0] / max(mat[0].sum(), 1))
        cav_prec = float(mat[0, 0] / max(mat[:, 0].sum(), 1))
        miou_2class = float(np.nanmean(iou[:2]))

        oracle_pred = hist[:, :3].argmax(axis=1)
        oracle_mat = np.zeros((3, 3), dtype=np.int64)
        for c_pred in range(3):
            mask = oracle_pred == c_pred
            if np.any(mask):
                oracle_mat[:, c_pred] = hist[mask, :3].sum(axis=0).astype(np.int64)
        oracle_union = oracle_mat.sum(axis=0) + oracle_mat.sum(axis=1) - np.diag(oracle_mat)
        oracle_iou = np.divide(np.diag(oracle_mat), oracle_union, out=np.full(3, np.nan), where=oracle_union > 0)

        loco_results[held_out] = {
            "cave_system": held_out,
            "annotation_path": str(ann_path),
            "voxel_size": val_g["metadata"]["voxel_size"],
            "num_p1": len(val_g["X"]),
            "known_points": int(mat.sum()),
            "finetuned_spt": {
                "confusion": mat.tolist(),
                "cavern_iou": float(iou[0]),
                "vegetation_iou": float(iou[1]),
                "miou": miou_2class,
                "cavern_recall": cav_rec,
                "cavern_precision": cav_prec,
            },
            "heuristic_spt": prev_heur.get(held_out),
            "partition_oracle": {
                "cavern_iou": float(oracle_iou[0]),
                "vegetation_iou": float(oracle_iou[1]),
                "miou": float(np.nanmean(oracle_iou[:2])),
            },
            "model_path": str(fold_model_path),
        }
        print(
            f"[{held_out:18s}] Fine-Tuned SPT -> Cavern IoU: {iou[0]*100:.2f}% "
            f"(Recall: {cav_rec*100:.2f}%, Prec: {cav_prec*100:.2f}%) | "
            f"Veg IoU: {iou[1]*100:.2f}% | mIoU: {miou_2class*100:.2f}%",
            flush=True,
        )

        # Export held-out superpoint centroid diagnostic PLY for CloudCompare
        sp_ply = out_dir / f"{held_out}_heldout_superpoints.ply"
        sp_gt = np.where(hist[:, :2].sum(axis=1) > 0, hist[:, :2].argmax(axis=1), -100).astype(np.int32)
        write_ply_binary(
            sp_ply,
            val_g["sp_pos"].astype(np.float64),
            None,
            {
                "scalar_predicted_class": sp_pred.astype(np.int32),
                "scalar_ground_truth": sp_gt,
                "scalar_cavern_prob": probs[:, 0].astype(np.float32),
                "scalar_voxel_count": val_g["sp_counts"].astype(np.int32),
            },
        )

    # Also train the final full-dataset production model on all 4 forested caves + interiors + CutMix
    print("\n=== Training Full Production Model (All Annotated Caves + 50 FOR-Instance Trees) ===", flush=True)
    full_train = []
    names = all_graphs[0]["feature_names"]
    ar_idx = int(np.flatnonzero(names == "above_roof_tanh")[0])
    fh_idx = int(np.flatnonzero(names == "floor_height_tanh")[0])
    for g in list(by_csys.values()) + cutmix_list + interior_list:
        gc = dict(g)
        gc["X"] = g["X"].copy()
        gc["X"][:, [ar_idx, fh_idx]] = 0.0
        full_train.append(gc)
    prod_model = train_graphs(full_train, epochs=60, device=device, seed=42)
    prod_path = out_dir / "spt_graphsage.npz"
    prod_meta = {
        "training_groups": sorted({cave_system_id(g["metadata"]["cave_system"]) for g in full_train}),
        "epochs": 60,
        "device": device,
    }
    prod_model.export(prod_path, names, prod_meta)
    torch.save(
        {"state_dict": prod_model.cpu().state_dict(), "metadata": prod_meta, "feature_names": names.tolist()},
        prod_path.with_suffix(".pt"),
    )

    summary_path = out_dir / "spt_finetune_evaluation.json"
    summary_path.write_text(json.dumps(loco_results, indent=2) + "\n")
    print(f"\n[Done] Saved full evaluation to {summary_path} in {time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    main()
