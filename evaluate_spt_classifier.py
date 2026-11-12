#!/usr/bin/env python3
"""Evaluate held-out entrance annotations using full-cloud geometric context."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from spt_filter import (inspect_ply_header, load_spt_kernels, read_ply,
                        run_spt_pipeline, run_spt_pipeline_from_p0, stream_subsample_ply)
from spt_learning import (cave_system_id, class_histograms, filter_cavern_manifold,
                          portable_predict, sanitize_inference_features, semantic_metrics)


def evaluate_manifest(lib, entries, model_path=None, voxel_size=0.04):
    training_groups = set()
    if model_path:
        with np.load(model_path, allow_pickle=False) as model:
            training_groups = set(json.loads(str(model["metadata"]))["training_groups"])
    overlap = training_groups & {cave_system_id(r["cave_system"]) for r in entries}
    if overlap:
        raise ValueError(f"Held-out evaluation overlaps model training sources: {sorted(overlap)}")
    reports = []
    previous_source = None
    for entry in entries:
        annotations, _, fields = read_ply(Path(entry["path"]))
        label_field = entry.get("label_field", "scalar_ground_truth")
        if label_field in ("scalar_predicted_class", "scalar_cave_zone", "scalar_pseudo_label"):
            raise ValueError("Use a separate ground-truth scalar field; predictions are not annotations")
        if label_field not in fields:
            raise ValueError(f"Missing annotation field {label_field} in {entry['path']}")
        truth = fields[label_field]
        if not np.isin(truth, [-100, 0, 1, 2]).all() or not np.any(truth != -100):
            raise ValueError("Annotations need known semantic labels 0..2; use -100 for unknown")
        source = Path(entry.get("source", entry["path"]))
        spacing = float(entry.get("voxel_size", voxel_size))
        already_p0 = bool(entry.get("already_p0", False))
        source_key = (source.resolve(), spacing, already_p0)
        if source_key != previous_source:
            fmt, count, _, _ = inspect_ply_header(source)
            if fmt == "binary_little_endian" and count > 15_000_000:
                n, center, xyz, normals = stream_subsample_ply(lib, source, spacing)
                result = run_spt_pipeline_from_p0(lib, n, center, xyz, normals, voxel_size=spacing)
            elif already_p0:
                xyz, normals, _ = read_ply(source)
                center = xyz.mean(axis=0, keepdims=True)
                p0_xyz = (xyz - center).astype(np.float32)
                result = run_spt_pipeline_from_p0(lib, len(xyz), center, p0_xyz, normals, voxel_size=spacing)
            else:
                xyz, normals, _ = read_ply(source)
                result = run_spt_pipeline(lib, xyz, normals, voxel_size=spacing)
            tree = None if (already_p0 and source.resolve() == Path(entry["path"]).resolve()) else cKDTree(result["p0_xyz_f64"])
            if model_path:
                X_sanitized = sanitize_inference_features(result["sp_feature_matrix"], result["feature_names"], spacing)
                probabilities = portable_predict(X_sanitized, result["edge_u"],
                                                  result["edge_v"], result["sp_pos"], model_path,
                                                  result["feature_names"])
                labels_p1 = result["superpoint_p1"]
                n_p1 = result["num_p1"]
                counts_p1 = result["sp_counts"]
                sp_roof = np.bincount(labels_p1, weights=result["roof_occlusion"], minlength=n_p1) / counts_p1
                sp_nz = np.bincount(labels_p1, weights=result["signed_horizontality"], minlength=n_p1) / counts_p1
                p1_edges = np.column_stack([result["edge_u"], result["edge_v"]])
                sp_pred = filter_cavern_manifold(
                    probabilities.argmax(axis=1).astype(np.int32),
                    result["sp_pos"],
                    counts_p1,
                    sp_roof_occlusion=sp_roof,
                    sp_signed_nz=sp_nz,
                    p1_edges=p1_edges,
                    probabilities=probabilities,
                    voxel_size=spacing,
                )
                prediction = sp_pred[result["superpoint_p1"]]
            else:
                prediction = result["predicted_class"]
            previous_source = source_key
        if tree is None:
            mapping = np.arange(len(annotations), dtype=np.int64)
            distance = np.zeros(len(annotations), dtype=np.float64)
        else:
            distance, mapping = tree.query(annotations, k=1, workers=-1)
        tolerance = float(entry.get("mapping_tolerance", np.sqrt(3)*spacing*1.01))
        if np.any(distance > tolerance):
            raise ValueError(f"Annotation coordinates do not match {source}: {int(np.sum(distance > tolerance))} points "
                             f"exceed {tolerance:.4f} m mapping tolerance")
        hist = class_histograms(truth, result["superpoint_p1"][mapping], result["num_p1"])
        oracle = hist[:, :3].argmax(axis=1)[result["superpoint_p1"][mapping]]
        pred_mapped = prediction[mapping]
        if not np.any(truth == 2):
            pred_mapped = np.where(pred_mapped >= 1, 1, 0)
        report = {
            "path": entry["path"], "source": str(source), "cave_system": entry["cave_system"],
            "voxel_size": spacing, "model": str(model_path) if model_path else None,
            "prediction": semantic_metrics(truth, pred_mapped),
            "heuristic": semantic_metrics(truth, np.where(result["predicted_class"][mapping] >= 1, 1, 0) if not np.any(truth == 2) else result["predicted_class"][mapping]),
            "partition_oracle": semantic_metrics(truth, oracle),
            "max_mapping_distance": float(distance.max()),
        }
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    return {"cases": reports, "evaluation": "held-out real annotations, full source-cloud context"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--output", type=Path, default=Path("output/spt_validation/held_out.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    entries = manifest.get("validation", [])
    if not entries:
        parser.error("The manifest has no validation entries")
    for record in entries:
        for key in ("path", "source"):
            if key in record:
                record[key] = str((args.manifest.parent/record[key]).resolve())
    from train_spt_classifier import validate_splits
    validate_splits(manifest)
    lib = load_spt_kernels(Path(__file__).resolve().parent/"build/libspt_kernels.so")
    report = evaluate_manifest(lib, entries, args.model, args.voxel_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
