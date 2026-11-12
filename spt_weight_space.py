#!/usr/bin/env python3
"""Program saved GraphSAGE neighbor weights; audit against cached cave annotations.

This is validation-guided weight editing, not a new independent test. Geometry,
normalization, architecture and the other learned tensors remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from spt_learning import (CLASS_NAMES, SCHEMA, cave_system_id,
                          filter_cavern_manifold, load_graph, portable_predict,
                          sanitize_inference_features)

FORESTED = ("S11A_0257", "lp_jl", "lp_boi", "gruta_do_janelao")


def edit_weights(source, destination, scale):
    """Scale the second neighbor map in its existing neuron basis; never merge folds."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Keep the original checkpoint; choose a separate destination")
    if not np.isfinite(scale) or not 0 < scale <= 2:
        raise ValueError("Neighbor scale must be finite and in (0, 2]")
    with np.load(source, allow_pickle=False) as archive:
        values = {k: archive[k].copy() for k in archive.files}
    if str(values["schema"]) != SCHEMA:
        raise ValueError("Checkpoint schema differs from this pipeline")
    metadata = json.loads(str(values["metadata"]))
    if "weight_space_edit" in metadata:
        raise ValueError("Use an original checkpoint to avoid compounding edits")
    metadata["weight_space_edit"] = {
        "method": "second GraphSAGE neighbor-map attenuation",
        "neighbor_2_scale": float(scale),
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selection": "validation-guided exploratory search on four LOCO cave folds",
        "limitations": "No fresh independent test; no real exterior-ground labels",
    }
    values["neighbor_2"] *= np.float32(scale)
    values["metadata"] = np.asarray(json.dumps(metadata))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **values)
    # Derive the torch artifact from these exact portable tensors, even if a
    # sibling source .pt is stale. Both formats describe the same edited model.
    import torch
    from spt_torch import SuperpointGraphSAGE
    model = SuperpointGraphSAGE(values["feature_mean"], values["feature_std"],
                               values["self_1"].shape[0])
    state = model.state_dict()
    for layer in (1, 2):
        state[f"self_{layer}.weight"] = torch.from_numpy(values[f"self_{layer}"])
        state[f"self_{layer}.bias"] = torch.from_numpy(values[f"bias_{layer}"])
        state[f"neighbor_{layer}.weight"] = torch.from_numpy(values[f"neighbor_{layer}"])
    state["output.weight"] = torch.from_numpy(values["output_weight"])
    state["output.bias"] = torch.from_numpy(values["output_bias"])
    model.load_state_dict(state)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata,
                "feature_names": values["feature_names"].tolist()}, destination.with_suffix(".pt"))
    return metadata


def histogram_metrics(histogram, prediction):
    """Exact annotated-point counts, including mixed superpoints and unknowns."""
    histogram = np.asarray(histogram, dtype=np.float64)
    matrix = np.stack([histogram[prediction == c, :3].sum(axis=0)
                       for c in range(3)], axis=1)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - matrix.diagonal()
    iou = np.divide(matrix.diagonal(), union, out=np.full(3, np.nan), where=union > 0)
    return {
        "confusion": matrix.astype(np.int64).tolist(),
        "known_points": int(matrix.sum()),
        "errors": int(matrix.sum() - matrix.trace()),
        "miou": float(np.nanmean(iou)),
        "iou": {k: float(v) if np.isfinite(v) else None for k, v in zip(CLASS_NAMES, iou)},
        "cavern_recall": float(matrix[0, 0] / max(matrix[0].sum(), 1)),
        "cavern_precision": float(matrix[0, 0] / max(matrix[:, 0].sum(), 1)),
    }


def evaluate(graph, checkpoint):
    x = sanitize_inference_features(graph["X"], graph["feature_names"],
                                    graph["metadata"]["voxel_size"])
    probabilities = portable_predict(x, graph["edge_u"], graph["edge_v"],
                                      graph["sp_pos"], checkpoint, graph["feature_names"])
    raw = probabilities.argmax(axis=1)
    filtered = filter_cavern_manifold(raw, graph["edge_u"], graph["edge_v"],
                                      graph["sp_pos"], graph["sp_counts"], probabilities)
    return {"raw": histogram_metrics(graph["histogram"], raw),
            "deployed": histogram_metrics(graph["histogram"], filtered)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("output/spt_finetuned"))
    parser.add_argument("--cache-dir", type=Path, default=Path("output/spt_graph_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/spt_programmed"))
    parser.add_argument("--neighbor-scale", type=float, default=0.775)
    args = parser.parse_args()
    graphs = {}
    interiors = []
    for path in sorted(args.cache_dir.glob("*.npz")):
        graph = load_graph(path)
        role = graph["metadata"].get("role")
        if role == "entrance":
            name = graph["metadata"]["cave_system"]
            if name in graphs:
                raise ValueError(f"Ambiguous cache: multiple entrance graphs for {name}")
            graphs[name] = (path, graph)
        elif role == "interior":
            interiors.append((path, graph))
    report = {"protocol": "Validation-guided edit; base models exclude their test cave. "
              "Edit strength was selected on these same validation folds, so this is not an independent test.",
              "neighbor_scale": args.neighbor_scale, "cases": {}}
    for name in FORESTED:
        filename = f"spt_graphsage_loco_{name}.npz"
        source, target = args.source_dir / filename, args.output_dir / filename
        with np.load(source, allow_pickle=False) as a:
            groups = json.loads(str(a["metadata"]))["training_groups"]
        if cave_system_id(name) in {cave_system_id(g) for g in groups}:
            raise ValueError(f"Cave leakage in source checkpoint: {name}")
        path, graph = graphs[name]
        edit_weights(source, target, args.neighbor_scale)
        report["cases"][name] = {"cache": str(path),
            "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "baseline": evaluate(graph, source), "edited": evaluate(graph, target)}
        print(name, json.dumps(report["cases"][name]), flush=True)
    source = args.source_dir / "spt_graphsage.npz"
    target = args.output_dir / "spt_graphsage.npz"
    edit_weights(source, target, args.neighbor_scale)
    report["interiors_in_sample"] = {}
    for path, graph in interiors:
        name = graph["metadata"]["cave_system"]
        report["interiors_in_sample"][name] = {
            "baseline": evaluate(graph, source), "edited": evaluate(graph, target)}
    report["summary"] = {}
    for stage in ("raw", "deployed"):
        summary = {}
        for version in ("baseline", "edited"):
            cases = [c[version][stage] for c in report["cases"].values()]
            summary[version] = {"macro_cave_miou": float(np.mean([c["miou"] for c in cases])),
                                "errors": sum(c["errors"] for c in cases),
                                "known_points": sum(c["known_points"] for c in cases)}
        summary["error_reduction_fraction"] = 1-summary["edited"]["errors"]/summary["baseline"]["errors"]
        report["summary"][stage] = summary
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
