#!/usr/bin/env python3
"""
Run the Fine-Tuned SPT SuperpointGraphSAGE Classifier on every cave in the dataset,
exporting full point-level labeled PLY files (not superpoint centroids):
  - output/spt_finetuned/<cave>_diagnostic.ply : Every P0 point with all features,
    scalar_predicted_class, scalar_cave_zone, scalar_cavern_score, scalar_heuristic_class,
    and scalar_ground_truth (when annotated).
  - output/spt_finetuned/<cave>_cleaned.ply    : Full-resolution Cavern-only point cloud
    (scalar_predicted_class == 0) with oriented normals and scalar_cave_zone (0=Floor, 1=Wall, 2=Ceiling).
"""

import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spt_filter import (
    load_spt_kernels,
    read_ply,
    run_spt_pipeline,
    run_spt_pipeline_from_p0,
    write_ply_binary,
)
from spt_learning import apply_model, semantic_metrics


CAVE_CONFIGS = [
    # 1. All 9 Manually Annotated Caves (using held-out LOCO models for the 4 forested caves)
    {
        "name": "S11A_0257",
        "path": "annotations/combined/S11A_0257_annotated.ply",
        "voxel_size": 0.04,
        "already_p0": True,
        "annotated": True,
        "model": "output/spt_finetuned/spt_graphsage_loco_S11A_0257.npz",
        "eval_mode": "held-out (LOCO)",
    },
    {
        "name": "Lp_JL",
        "path": "annotations/combined/lp_jl_annotated.ply",
        "voxel_size": 0.08,
        "already_p0": True,
        "annotated": True,
        "model": "output/spt_finetuned/spt_graphsage_loco_lp_jl.npz",
        "eval_mode": "held-out (LOCO)",
    },
    {
        "name": "Lp_Boi",
        "path": "annotations/combined/lp_boi_annotated.ply",
        "voxel_size": 0.12,
        "already_p0": True,
        "annotated": True,
        "model": "output/spt_finetuned/spt_graphsage_loco_lp_boi.npz",
        "eval_mode": "held-out (LOCO)",
    },
    {
        "name": "gruta_do_janelao",
        "path": "annotations/combined/gruta_do_janelao_annotated.ply",
        "voxel_size": 0.35,
        "already_p0": False,
        "annotated": True,
        "model": "output/spt_finetuned/spt_graphsage_loco_gruta_do_janelao.npz",
        "eval_mode": "held-out (LOCO)",
    },
    {
        "name": "S11D_0061",
        "path": "annotations/combined/S11D_0061_annotated.ply",
        "voxel_size": 0.08,
        "already_p0": False,
        "annotated": True,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "annotated-interior",
    },
    {
        "name": "interior_lapinha",
        "path": "annotations/combined/interior_lapinha_annotated.ply",
        "voxel_size": 0.04,
        "already_p0": False,
        "annotated": True,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "annotated-interior",
    },
    {
        "name": "rupt",
        "path": "annotations/combined/rupt_annotated.ply",
        "voxel_size": 0.04,
        "already_p0": False,
        "annotated": True,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "annotated-interior",
    },
    {
        "name": "BaumeDeLongeaigue",
        "path": "annotations/combined/BaumeDeLongeaigue_annotated.ply",
        "voxel_size": 0.04,
        "already_p0": False,
        "annotated": True,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "annotated-interior",
    },
    {
        "name": "diaclase",
        "path": "annotations/combined/diaclase_annotated.ply",
        "voxel_size": 0.04,
        "already_p0": False,
        "annotated": True,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "annotated-interior",
    },
    # 2. Additional Unannotated Caves across the dataset (Brazilian & European scans)
    {
        "name": "S11D_0001",
        "path": "input/point-clouds/S11D_0001.ply",
        "voxel_size": 0.04,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
    {
        "name": "N5S_0021_0027",
        "path": "input/point-clouds/N5S_0021_0027.ply",
        "voxel_size": 0.06,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
    {
        "name": "BR_0034",
        "path": "input/point-clouds/BR_0034.ply",
        "voxel_size": 0.06,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
    {
        "name": "CAV04",
        "path": "input/point-clouds/CAV04.ply",
        "voxel_size": 0.04,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
    {
        "name": "lapa_dos_desenhos",
        "path": "input/point-clouds/lapa_dos_desenhos.ply",
        "voxel_size": 0.10,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
    {
        "name": "lapa_da_onca",
        "path": "input/point-clouds/lapa_da_onça.ply",
        "voxel_size": 0.12,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
    {
        "name": "lapa_dos_anjos",
        "path": "input/point-clouds/lapa_dos_anjos.ply",
        "voxel_size": 0.12,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
    {
        "name": "lapa_dos_troncos",
        "path": "input/point-clouds/lapa_dos_troncos.ply",
        "voxel_size": 0.12,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
    {
        "name": "arco_do_andre",
        "path": "input/point-clouds/arco_do_andre.ply",
        "voxel_size": 0.18,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
    {
        "name": "casa_de_pedra",
        "path": "input/point-clouds/casa_de_pedra.ply",
        "voxel_size": 0.22,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
    {
        "name": "cascade_de_motiers",
        "path": "input/point-clouds/cascade_de_motiers.ply",
        "voxel_size": 0.10,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
    {
        "name": "lauiloch",
        "path": "input/point-clouds/lauiloch.ply",
        "voxel_size": 0.06,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
    {
        "name": "archamps_post_etroiture",
        "path": "input/point-clouds/archamps_post_etroiture.ply",
        "voxel_size": 0.04,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
    {
        "name": "grotte_de_la_sourde",
        "path": "input/point-clouds/grotte_de_la_sourde.ply",
        "voxel_size": 0.04,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
    {
        "name": "grotte_de_la_madeleine",
        "path": "input/point-clouds/grotte_de_la_madeleine.ply",
        "voxel_size": 0.04,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
    {
        "name": "hoelloch_keller_aquarium",
        "path": "input/point-clouds/hoelloch_keller_aquarium.ply",
        "voxel_size": 0.04,
        "already_p0": False,
        "annotated": False,
        "model": "output/spt_finetuned/spt_graphsage.npz",
        "eval_mode": "unannotated-inference",
    },
]


def main():
    t_start = time.perf_counter()
    lib = load_spt_kernels(ROOT / "build/libspt_kernels.so")
    out_dir = ROOT / "output/spt_finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove old superpoint-centroid-only PLYs so only full point clouds are present
    for old_sp in out_dir.glob("*_heldout_superpoints.ply"):
        old_sp.unlink()

    all_reports = {}
    for idx, cfg in enumerate(CAVE_CONFIGS, 1):
        name = cfg["name"]
        ply_path = ROOT / cfg["path"]
        vs = float(cfg["voxel_size"])
        model_path = ROOT / cfg["model"]
        t_c0 = time.perf_counter()
        print(f"\n[{idx}/{len(CAVE_CONFIGS)}] Processing {name} ({ply_path.name}, voxel={vs}m) ...", flush=True)

        xyz_f64, normals_in, extras = read_ply(ply_path)
        n_orig = len(xyz_f64)

        if cfg["already_p0"]:
            centroid = xyz_f64.mean(axis=0, keepdims=True)
            p0_xyz = (xyz_f64 - centroid).astype(np.float32)
            res = run_spt_pipeline_from_p0(
                lib, n_orig, centroid, p0_xyz, normals_in, voxel_size=vs, pure_interior=False
            )
        else:
            res = run_spt_pipeline(
                lib, xyz_f64, normals_in, voxel_size=vs, pure_interior=False
            )

        # Apply Fine-Tuned SPT GraphSAGE Model (pure model inference + 3D cavern manifold check)
        apply_model(res, model_path, min_confidence=0.0)

        p0_pred = res["predicted_class"]
        p0_heur = res["heuristic_class"]
        p0_zone = res["cave_zone"]
        n_p0 = len(p0_pred)

        # Map predictions to every original point (exact full resolution)
        orig_to_p0 = res["orig_to_p0"]
        if orig_to_p0 is None or len(orig_to_p0) != n_orig:
            orig_to_p0 = np.arange(n_orig, dtype=np.int32)
        p0_to_orig = res["p0_to_orig"]
        if p0_to_orig is None or len(p0_to_orig) != n_p0:
            p0_to_orig = np.arange(n_p0, dtype=np.int32)

        orig_pred = p0_pred[orig_to_p0]
        orig_heur = p0_heur[orig_to_p0]
        orig_zone = p0_zone[orig_to_p0]
        orig_score = res["cavern_score"][orig_to_p0]
        orig_conf = res["model_confidence"][orig_to_p0]
        orig_normals = res["p0_normals"][orig_to_p0]

        n_cav_orig = int(np.sum(orig_pred == 0))
        n_veg_only_orig = int(np.sum(orig_pred == 1))
        n_ext_gnd_orig = int(np.sum(orig_pred == 2))
        n_veg_orig = int(np.sum((orig_pred == 1) | (orig_pred == 2)))
        n_floor_orig = int(np.sum(orig_zone == 0))
        n_wall_orig = int(np.sum(orig_zone == 1))
        n_ceil_orig = int(np.sum(orig_zone == 2))

        # Export P0 Diagnostic PLY (every P0 point with all scalars)
        diag_path = out_dir / f"{name}_diagnostic.ply"
        scalars_diag = {
            "scalar_predicted_class": p0_pred.astype(np.int32),
            "scalar_cave_zone": p0_zone.astype(np.int32),
            "scalar_cavern_score": res["cavern_score"].astype(np.float32),
            "scalar_model_confidence": res["model_confidence"].astype(np.float32),
            "scalar_heuristic_class": p0_heur.astype(np.int32),
            "scalar_superpoint_p1": res["superpoint_p1"].astype(np.int32),
            "scalar_superpoint_p2": res["superpoint_p2"].astype(np.int32),
            "scalar_verticality": res["verticality"].astype(np.float32),
            "scalar_horizontality": res["horizontality"].astype(np.float32),
            "scalar_signed_horizontality": res["signed_horizontality"].astype(np.float32),
            "scalar_planarity_fine": res["planarity_fine"].astype(np.float32),
            "scalar_scattering_fine": res["scattering_fine"].astype(np.float32),
            "scalar_normal_dispersion": res["normal_dispersion_fine"].astype(np.float32),
            "scalar_roof_occlusion": res["roof_occlusion"].astype(np.float32),
            "scalar_downward_occlusion": res["downward_occlusion"].astype(np.float32),
            "scalar_cavern_enclosure": res["cavern_enclosure"].astype(np.float32),
            "scalar_drip_line_dist": res["drip_line_dist"].astype(np.float32),
        }

        metrics_model = None
        metrics_heur = None
        if cfg["annotated"] and "scalar_ground_truth" in extras:
            truth_orig = extras["scalar_ground_truth"].astype(np.int32)
            # Map ground truth to P0 via p0_to_orig for diagnostic PLY
            scalars_diag["scalar_ground_truth"] = truth_orig[p0_to_orig].astype(np.int32)
            eval_pred = np.where(orig_pred >= 1, 1, 0) if not np.any(truth_orig == 2) else orig_pred
            eval_heur = np.where(orig_heur >= 1, 1, 0) if not np.any(truth_orig == 2) else orig_heur
            metrics_model = semantic_metrics(truth_orig, eval_pred)
            metrics_heur = semantic_metrics(truth_orig, eval_heur)

        write_ply_binary(diag_path, res["p0_xyz_f64"], res["p0_normals"], scalars_diag)

        # Export Full-Resolution Cleaned PLY (every Cavern point at original resolution)
        clean_path = out_dir / f"{name}_cleaned.ply"
        keep_mask = orig_pred == 0
        write_ply_binary(
            clean_path,
            xyz_f64[keep_mask],
            orig_normals[keep_mask],
            {
                "scalar_cave_zone": orig_zone[keep_mask].astype(np.int32),
                "scalar_cavern_score": orig_score[keep_mask].astype(np.float32),
            },
        )

        # Also export Full-Resolution Labeled PLY if N_orig != N_P0 on annotated caves
        full_labeled_path = None
        if cfg["annotated"] and not cfg["already_p0"]:
            full_labeled_path = out_dir / f"{name}_full_labeled.ply"
            write_ply_binary(
                full_labeled_path,
                xyz_f64,
                orig_normals,
                {
                    "scalar_predicted_class": orig_pred.astype(np.int32),
                    "scalar_cave_zone": orig_zone.astype(np.int32),
                    "scalar_cavern_score": orig_score.astype(np.float32),
                    "scalar_heuristic_class": orig_heur.astype(np.int32),
                    "scalar_ground_truth": extras["scalar_ground_truth"].astype(np.int32),
                },
            )

        elapsed = time.perf_counter() - t_c0
        report = {
            "name": name,
            "input_path": cfg["path"],
            "eval_mode": cfg["eval_mode"],
            "voxel_size": vs,
            "original_points": n_orig,
            "p0_points": n_p0,
            "p1_superpoints": int(res["num_p1"]),
            "p2_superpoints": int(res["num_p2"]),
            "cavern_points": n_cav_orig,
            "cavern_pct": round(100.0 * n_cav_orig / max(n_orig, 1), 2),
            "vegetation_points": n_veg_orig,
            "vegetation_pct": round(100.0 * n_veg_orig / max(n_orig, 1), 2),
            "vegetation_only_points": n_veg_only_orig,
            "exterior_ground_points": n_ext_gnd_orig,
            "exterior_ground_pct": round(100.0 * n_ext_gnd_orig / max(n_orig, 1), 2),
            "floor_points": n_floor_orig,
            "wall_points": n_wall_orig,
            "ceiling_points": n_ceil_orig,
            "diagnostic_ply": str(diag_path.relative_to(ROOT)),
            "cleaned_ply": str(clean_path.relative_to(ROOT)),
            "full_labeled_ply": str(full_labeled_path.relative_to(ROOT)) if full_labeled_path else str(diag_path.relative_to(ROOT)),
            "finetuned_metrics": metrics_model,
            "heuristic_metrics": metrics_heur,
            "elapsed_sec": round(elapsed, 2),
        }
        all_reports[name] = report

        msg = (
            f"      -> {name:24s} | Orig={n_orig:9,d} | Cavern={n_cav_orig:9,d} ({report['cavern_pct']:5.1f}%) | "
            f"Veg={n_veg_orig:8,d} ({report['vegetation_pct']:4.1f}%) | "
            f"Floor={n_floor_orig:,} Wall={n_wall_orig:,} Ceil={n_ceil_orig:,} ({elapsed:.1f}s)"
        )
        if metrics_model:
            ciou = (metrics_model["iou"]["cavern"] or 0.0) * 100.0
            viou = metrics_model["iou"]["vegetation"]
            viou_str = f"{viou*100:.2f}%" if viou is not None else "N/A (pure interior)"
            msg += f" | GT CavIoU={ciou:.2f}%, VegIoU={viou_str}, Rec={metrics_model['cavern_recall']*100:.2f}%"
        print(msg, flush=True)

    summary_file = out_dir / "all_caves_results.json"
    summary_file.write_text(json.dumps(all_reports, indent=2) + "\n")
    print(f"\n[Done] Processed all {len(CAVE_CONFIGS)} caves in {time.perf_counter()-t_start:.1f}s -> {summary_file}", flush=True)


if __name__ == "__main__":
    main()
