#!/usr/bin/env python3
"""
Train and evaluate the Fine-Tuned 3-Class SPT SuperpointGraphSAGE Classifier across all
annotated caves (pure interiors + coupled FOR-instance & DALES donor chunks + forested caves)
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

from spt_filter import load_spt_kernels, read_ply, write_ply_binary
from spt_learning import (
    CLASS_NAMES,
    SCHEMA,
    cave_system_id,
    filter_cavern_manifold,
    load_graph,
    portable_predict,
    semantic_metrics,
    soft_targets,
)
from spt_torch import train_graphs
from train_spt_classifier import (
    cached_graph,
    extract_pseudo_labeled_graph,
    generate_cutmix_superpoint_graph,
    load_donor_chunk,
    pipeline_fingerprint,
    source_fingerprint,
)


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

    # Harmonize 3-class ground truth on entrance caves where binary annotations put both
    # vegetation (1) and planar exterior ground/terrace (2) into bin 1:
    g["orig_histogram"] = g["histogram"].copy()
    hist = g["histogram"].copy()
    if g["metadata"].get("role") == "entrance" and hist[:, 2].sum() == 0:
        pf_idx = int(np.flatnonzero(names == "mean_planarity_fine")[0])
        sf_idx = int(np.flatnonzero(names == "mean_scattering_fine")[0])
        nz_idx = int(np.flatnonzero(names == "mean_nz")[0])
        ro_idx = int(np.flatnonzero(names == "mean_roof_occlusion")[0])
        is_ext_gnd = (
            (hist[:, 0] == 0)
            & (hist[:, 1] > 0)
            & (X[:, pf_idx] >= 0.55)
            & (X[:, nz_idx] >= 0.45)
            & (X[:, sf_idx] <= 0.22)
            & (X[:, ro_idx] < 0.20)
        )
        hist[is_ext_gnd, 2] = hist[is_ext_gnd, 1]
        hist[is_ext_gnd, 1] = 0
        g["histogram"] = hist
        g["soft_q"], g["purity_weight"] = soft_targets(hist)
        g["valid_mask"] = hist[:, :3].sum(axis=1) > 0

    g["X"] = X
    g["raw_x"] = X
    return g


def ensure_v4_graph_cache(cache_dir: Path):
    """Populate output/spt_graph_cache with SCHEMA=spt-graph-v4 graphs (interiors, entrances, 3-mode CutMix)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    lib = load_spt_kernels(ROOT / "build/libspt_kernels.so")
    manifest_path = ROOT / "docs/spt-finetune-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    canopy_records = manifest.get("canopy_chunks", [])
    for rec in canopy_records:
        rec["path"] = str((manifest_path.parent / rec["path"]).resolve())
    canopy = None
    canopy_fps = [source_fingerprint(rec["path"]) for rec in canopy_records]
    pipeline_id = pipeline_fingerprint()

    entrance_cases = [
        {"path": ROOT / "annotations/combined/S11A_0257_annotated.ply", "cave_system": "S11A_0257", "role": "entrance", "voxel_size": 0.04, "already_p0": True},
        {"path": ROOT / "annotations/combined/lp_jl_annotated.ply", "cave_system": "lp_jl", "role": "entrance", "voxel_size": 0.08, "already_p0": True},
        {"path": ROOT / "annotations/combined/lp_boi_annotated.ply", "cave_system": "lp_boi", "role": "entrance", "voxel_size": 0.12, "already_p0": True},
        {"path": ROOT / "annotations/combined/gruta_do_janelao_annotated.ply", "cave_system": "gruta_do_janelao", "role": "entrance", "voxel_size": 0.35, "already_p0": False},
    ]
    interior_cases = [
        {"path": ROOT / "annotations/combined/rupt_annotated.ply", "cave_system": "rupt", "role": "interior", "voxel_size": 0.04, "already_p0": False},
        {"path": ROOT / "annotations/combined/BaumeDeLongeaigue_annotated.ply", "cave_system": "baume_de_longeaigue", "role": "interior", "voxel_size": 0.04, "already_p0": False},
        {"path": ROOT / "annotations/combined/diaclase_annotated.ply", "cave_system": "diaclase", "role": "interior", "voxel_size": 0.04, "already_p0": False},
        {"path": ROOT / "annotations/combined/interior_lapinha_annotated.ply", "cave_system": "lapinha", "role": "interior", "voxel_size": 0.04, "already_p0": False},
        {"path": ROOT / "input/point-clouds/archamps_post_etroiture.ply", "cave_system": "archamps", "role": "interior", "voxel_size": 0.04, "already_p0": False},
        {"path": ROOT / "input/point-clouds/grotte_de_la_sourde.ply", "cave_system": "grotte_de_la_sourde", "role": "interior", "voxel_size": 0.04, "already_p0": False},
        {"path": ROOT / "input/point-clouds/grotte_de_la_madeleine.ply", "cave_system": "grotte_de_la_madeleine", "role": "interior", "voxel_size": 0.04, "already_p0": False},
        {"path": ROOT / "input/point-clouds/hoelloch_keller_aquarium.ply", "cave_system": "hoelloch", "role": "interior", "voxel_size": 0.04, "already_p0": False},
    ]

    for case in entrance_cases + interior_cases[:4]:
        p = Path(case["path"])
        s = float(case["voxel_size"])
        ap0 = bool(case["already_p0"])
        role = case["role"]
        meta = {
            "schema": SCHEMA,
            "pipeline": pipeline_id,
            "source": source_fingerprint(p),
            "cave_system": case["cave_system"],
            "role": role,
            "voxel_size": s,
            "already_p0": ap0,
        }
        print(f"[Seeds] {p.name} ({role}, voxel={s}m)", flush=True)
        cached_graph(
            cache_dir,
            meta,
            lambda p=p, s=s, r=role, ap0=ap0: extract_pseudo_labeled_graph(
                lib, p, s, r == "interior", None, "scalar_ground_truth", ap0
            ),
        )

    scenarios = ("mouth", "skylight", "exterior_surroundings")
    for c_idx, case in enumerate(interior_cases):
        p = Path(case["path"])
        s = float(case["voxel_size"])
        ap0 = bool(case["already_p0"])
        base_meta = {
            "schema": SCHEMA,
            "pipeline": pipeline_id,
            "source": source_fingerprint(p),
            "cave_system": case["cave_system"],
            "role": "interior",
            "voxel_size": s,
            "already_p0": ap0,
        }
        xyz_cache = None
        for i, sc in enumerate(scenarios):
            seed = 42 + c_idx * 100 + i
            mix_meta = {
                **base_meta,
                "role": "cutmix",
                "scenario": sc,
                "seed": seed,
                "canopy_sources": canopy_fps,
            }

            def produce(p=p, s=s, sd=seed, sc=sc):
                nonlocal xyz_cache, canopy
                if canopy is None:
                    canopy = [load_donor_chunk(rec["path"]) for rec in canopy_records]
                if xyz_cache is None:
                    pts, _, flds = read_ply(p)
                    xyz_cache = pts[flds["scalar_ground_truth"] == 0] if "scalar_ground_truth" in flds else pts
                return generate_cutmix_superpoint_graph(lib, xyz_cache, canopy, s, sd, sc)

            print(f"[CutMix] {p.name}, {sc}, seed={seed}", flush=True)
            cached_graph(cache_dir, mix_meta, produce)


def main():
    t0 = time.perf_counter()
    cache_dir = ROOT / "output/spt_graph_cache"
    out_dir = ROOT / "output/spt_finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_v4_graph_cache(cache_dir)

    all_graphs = []
    for p in sorted(cache_dir.glob("*.npz")):
        with np.load(p, allow_pickle=False) as a:
            if str(a["schema"]) != SCHEMA:
                continue
        all_graphs.append(sanitize_cross_scale_features(load_graph(p)))
    all_graphs.sort(
        key=lambda g: (
            g["metadata"]["role"],
            g["metadata"]["cave_system"],
            g["metadata"].get("scenario", ""),
            int(g["metadata"].get("seed", 0)),
        )
    )

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
        print(f"\n=== [Held-Out Cave: {held_out}] Training on remaining caves + FOR-instance & DALES CutMix ===", flush=True)
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
        ro_idx = int(np.flatnonzero(names == "mean_roof_occlusion")[0])
        nz_idx = int(np.flatnonzero(names == "mean_nz")[0])
        fold_train = []
        for g in train_set:
            gc = dict(g)
            gc["X"] = g["X"].copy()
            gc["X"][:, [ar_idx, fh_idx]] = 0.0
            gc["raw_x"] = gc["X"]
            fold_train.append(gc)

        val_g = dict(by_csys[held_out])
        val_g["X"] = val_g["X"].copy()
        val_g["X"][:, [ar_idx, fh_idx]] = 0.0
        val_g["raw_x"] = val_g["X"]

        model = train_graphs(fold_train, epochs=60, device=device, seed=42)
        fold_model_path = out_dir / f"spt_graphsage_loco_{held_out}.npz"
        meta = {
            "training_groups": sorted(train_groups),
            "held_out": held_out,
            "epochs": 60,
            "device": device,
        }
        model.export(fold_model_path, val_g["feature_names"], meta)
        model.export(out_dir / f"spt_graphsage_v4_loco_{held_out}.npz", val_g["feature_names"], meta)
        pt_payload = {
            "state_dict": model.cpu().state_dict(),
            "metadata": meta,
            "feature_names": val_g["feature_names"].tolist(),
            "global_feat_mean": model.global_feat_mean.detach().cpu().numpy(),
            "global_feat_std": model.global_feat_std.detach().cpu().numpy(),
        }
        torch.save(pt_payload, fold_model_path.with_suffix(".pt"))
        torch.save(pt_payload, out_dir / f"spt_graphsage_v4_loco_{held_out}.pt")

        # Portable CPU inference + 3-stage roof-anchored geodesic cavern manifold filter
        probs = portable_predict(
            val_g["X"], val_g["edge_u"], val_g["edge_v"], val_g["sp_pos"],
            fold_model_path, val_g["feature_names"]
        )
        p1_edges = np.column_stack([val_g["edge_u"], val_g["edge_v"]])
        vs = float(val_g["metadata"].get("voxel_size", 0.04))
        sp_pred = filter_cavern_manifold(
            probs.argmax(axis=1).astype(np.int32),
            val_g["sp_pos"],
            val_g["sp_counts"],
            sp_roof_occlusion=val_g["X"][:, ro_idx],
            sp_signed_nz=val_g["X"][:, nz_idx],
            p1_edges=p1_edges,
            probabilities=probs,
            voxel_size=vs,
        )

        ann_path = Path(val_g["metadata"]["source"]["path"])
        hist_3c = val_g["histogram"]
        hist = val_g.get("orig_histogram", hist_3c)
        # Full 3x3 confusion matrix across [0=Cavern, 1=Vegetation, 2=Exterior Ground]
        mat_3x3 = np.zeros((3, 3), dtype=np.int64)
        for c_pred in range(3):
            mask = sp_pred == c_pred
            if np.any(mask):
                mat_3x3[:, c_pred] = hist_3c[mask, :3].sum(axis=0).astype(np.int64)

        # At inference time, both Class 1 (Vegetation) and Class 2 (Exterior Ground) are discarded
        # as non-cavern clutter when evaluating against binary (0=Cavern, 1=Non-Cavern) annotations.
        sp_pred_eval = np.where(sp_pred >= 1, 1, 0) if hist[:, 2].sum() == 0 else sp_pred
        mat = np.zeros((3, 3), dtype=np.int64)
        for c_pred in range(3):
            mask = sp_pred_eval == c_pred
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
            "voxel_size": vs,
            "num_p1": len(val_g["X"]),
            "known_points": int(mat.sum()),
            "finetuned_spt": {
                "confusion": mat.tolist(),
                "confusion_3class": mat_3x3.tolist(),
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
            f"Veg/Ext IoU: {iou[1]*100:.2f}% | mIoU: {miou_2class*100:.2f}%",
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
    print("\n=== Training Full Production Model (All Annotated Caves + Coupled FOR-Instance & DALES Donors) ===", flush=True)
    full_train = []
    names = all_graphs[0]["feature_names"]
    ar_idx = int(np.flatnonzero(names == "above_roof_tanh")[0])
    fh_idx = int(np.flatnonzero(names == "floor_height_tanh")[0])
    for g in list(by_csys.values()) + cutmix_list + interior_list:
        gc = dict(g)
        gc["X"] = g["X"].copy()
        gc["X"][:, [ar_idx, fh_idx]] = 0.0
        gc["raw_x"] = gc["X"]
        full_train.append(gc)
    prod_model = train_graphs(full_train, epochs=60, device=device, seed=42)
    prod_path = out_dir / "spt_graphsage.npz"
    prod_meta = {
        "training_groups": sorted({cave_system_id(g["metadata"]["cave_system"]) for g in full_train}),
        "epochs": 60,
        "device": device,
    }
    prod_model.export(prod_path, names, prod_meta)
    prod_model.export(out_dir / "spt_graphsage_v4.npz", names, prod_meta)
    prod_pt_payload = {
        "state_dict": prod_model.cpu().state_dict(),
        "metadata": prod_meta,
        "feature_names": names.tolist(),
        "global_feat_mean": prod_model.global_feat_mean.detach().cpu().numpy(),
        "global_feat_std": prod_model.global_feat_std.detach().cpu().numpy(),
    }
    torch.save(prod_pt_payload, prod_path.with_suffix(".pt"))
    torch.save(prod_pt_payload, out_dir / "spt_graphsage_v4.pt")

    summary_path = out_dir / "spt_finetune_evaluation.json"
    summary_text = json.dumps(loco_results, indent=2) + "\n"
    summary_path.write_text(summary_text)
    (out_dir / "spt_finetune_evaluation_v4.json").write_text(summary_text)
    print(f"\n[Done] Saved full evaluation to {summary_path} in {time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    main()
