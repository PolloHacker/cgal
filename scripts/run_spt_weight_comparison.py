#!/usr/bin/env python3
"""Rerun the two cave comparisons and export full-point CloudCompare differences."""
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spt_filter import (load_spt_kernels, read_ply, run_spt_pipeline,
                        run_spt_pipeline_from_p0, write_ply_binary)
from spt_learning import apply_model, semantic_metrics
from scripts.run_spt_finetuned_all_caves import CAVE_CONFIGS


def preview(path, xyz, baseline, edited, change):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(42)
    changed = np.flatnonzero(change)
    same = np.flatnonzero(change == 0)
    changed = rng.choice(changed, min(len(changed), 60000), replace=False)
    same = rng.choice(same, min(len(same), 100000), replace=False)
    sample = np.r_[same, changed]
    p = xyz[sample] - np.median(xyz, axis=0)
    # Shared orthographic views and identical point order/bounds for all panels.
    views = [(p[:, 0], p[:, 1], "Plan (X / Y)"),
             ((p[:, 0]-p[:, 1])/np.sqrt(2),
              p[:, 2]*.8660254-(p[:, 0]+p[:, 1])*.3535534, "Oblique")]
    semantic_colors = np.array(["#8b929d", "#408d47", "#c2a66f"])
    change_colors = np.array(["#d9dde2", "#ee762d", "#008bc2", "#aa47ba"])
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="white")
    for row, (x, y, label) in enumerate(views):
        for col, (title, colors) in enumerate([
                ("Original weights", semantic_colors[baseline[sample]]),
                ("Edited weights", semantic_colors[edited[sample]]),
                ("Changes: orange = removed, blue = retained", change_colors[change[sample]])]):
            ax = axes[row, col]
            ax.scatter(x, y, c=colors, s=.13, linewidths=0, rasterized=True)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel(label+" · meters, centered for preview")
            ax.set_facecolor("#fafafa")
    fig.suptitle(path.parent.name+" · cavern gray / vegetation green / exterior ground tan\n"
                 "Preview samples up to 100k unchanged + 60k changed points; PLY exports contain every point",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_case(lib, cfg, output_root):
    started = time.perf_counter()
    name = cfg["name"]
    key = "lp_jl" if name == "Lp_JL" else name
    out = output_root / key
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{key}] Loading {cfg['path']}", flush=True)
    xyz, normals, extras = read_ply(ROOT / cfg["path"])
    if cfg["already_p0"]:
        center = xyz.mean(axis=0, keepdims=True)
        result = run_spt_pipeline_from_p0(lib, len(xyz), center,
            (xyz-center).astype(np.float32), normals, voxel_size=cfg["voxel_size"])
    else:
        result = run_spt_pipeline(lib, xyz, normals, voxel_size=cfg["voxel_size"])
    mapping = result["orig_to_p0"]
    if mapping is None:
        if len(xyz) != len(result["predicted_class"]):
            raise ValueError("Missing original-point mapping")
        mapping = np.arange(len(xyz))
    truth = extras["scalar_ground_truth"].astype(np.int32)
    point_normals = result["p0_normals"][mapping]
    predictions = {}
    scores = {}
    metrics = {}
    for version, folder in [("baseline", "spt_finetuned"), ("edited", "spt_programmed")]:
        model = ROOT / "output" / folder / f"spt_graphsage_loco_{key}.npz"
        print(f"[{key}] Inference {version}", flush=True)
        inference = apply_model(dict(result), model)
        prediction = inference["predicted_class"][mapping]
        predictions[version] = prediction
        scores[version] = inference["cavern_score"][mapping]
        metrics[version] = semantic_metrics(truth, prediction)
        keep = prediction == 0
        write_ply_binary(out / f"{version}_cleaned.ply", xyz[keep], point_normals[keep],
                         {"scalar_cave_zone": inference["cave_zone"][mapping][keep]})
        del inference
    a, b = predictions["baseline"], predictions["edited"]
    changed = a != b
    change = np.zeros(len(xyz), dtype=np.int32)
    change[changed] = 3
    change[(a == 0) & (b != 0)] = 1
    change[(a != 0) & (b == 0)] = 2
    known = truth != -100
    review = np.zeros(len(xyz), dtype=np.int32)
    review[changed & known & (a != truth) & (b == truth)] = 1
    review[changed & known & (a == truth) & (b != truth)] = 2
    review[changed & ~known] = 3
    review[changed & known & (a != truth) & (b != truth)] = 4
    fields = {"scalar_baseline_class": a, "scalar_edited_class": b,
              "scalar_change": change, "scalar_review": review,
              "scalar_ground_truth": truth,
              "scalar_baseline_cavern_prob": scores["baseline"],
              "scalar_edited_cavern_prob": scores["edited"]}
    print(f"[{key}] Exporting {len(xyz):,} points, {changed.sum():,} changed", flush=True)
    write_ply_binary(out / "comparison.ply", xyz, point_normals, fields)
    if changed.any():
        write_ply_binary(out / "changes_only.ply", xyz[changed], point_normals[changed],
                         {k: v[changed] for k, v in fields.items()})
    preview(out / "preview.png", xyz, a, b, change)
    report = {"source": cfg["path"], "models": "cave-held-out original and edited (scale 0.775)",
              "points": len(xyz), "voxel_size": cfg["voxel_size"],
              "changed_points": int(changed.sum()),
              "newly_removed_from_cavern": int((change == 1).sum()),
              "newly_retained_as_cavern": int((change == 2).sum()),
              "corrected_labeled_points": int((review == 1).sum()),
              "regressed_labeled_points": int((review == 2).sum()),
              "changed_unknown_points": int((review == 3).sum()),
              "metrics": metrics, "seconds": time.perf_counter()-started}
    (out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    output = ROOT / "output/spt_programmed/comparison"
    output.mkdir(parents=True, exist_ok=True)
    lib = load_spt_kernels(ROOT / "build/libspt_kernels.so")
    report = {}
    for cfg in CAVE_CONFIGS:
        if cfg["name"] in ("Lp_JL", "gruta_do_janelao"):
            report[cfg["name"]] = run_case(lib, cfg, output)
            gc.collect()
    (output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    (output / "README.md").write_text(
        "# Original versus edited SPT weights\n\n"
        "Each folder contains `baseline_cleaned.ply`, `edited_cleaned.ply`, "
        "`comparison.ply` (every input point), `changes_only.ply`, `preview.png`, and `report.json`.\n\n"
        "Open comparison.ply in CloudCompare and select scalar_baseline_class or scalar_edited_class "
        "(0 cavern, 1 vegetation, 2 exterior ground). Use scalar_change: 0 unchanged, "
        "1 newly removed from cavern, 2 newly retained as cavern, 3 other class change.\n\n"
        "scalar_review: 0 unchanged, 1 corrected, 2 regressed, 3 changed but unknown truth, "
        "4 changed but both incorrect. Ground truth -100 means unreviewed. "
        "Coordinates are preserved. Both models use identical freshly computed geometry. "
        "Inputs are the same annotated scan versions as the benchmark; LP_JL is already P0. "
        "Preview images are sampled, whereas PLY exports include all corresponding points.\n")


if __name__ == "__main__":
    main()
