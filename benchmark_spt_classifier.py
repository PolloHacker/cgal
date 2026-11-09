#!/usr/bin/env python3
"""Check a learned head on an unseen pure cave and synthetic entrance scenes."""
import argparse
import json
from pathlib import Path

import numpy as np

from spt_filter import load_spt_kernels, read_ply, run_spt_pipeline
from spt_learning import cave_system_id, class_histograms, portable_predict, semantic_metrics
from train_spt_classifier import (cached_graph, generate_cutmix_superpoint_graph,
                                  pipeline_fingerprint, source_fingerprint)


def histogram_metrics(histogram, node_prediction):
    truth, prediction = [], []
    for c in range(3):
        counts = histogram[:, c].astype(np.int64)
        truth.append(np.full(int(counts.sum()), c, dtype=np.int32))
        prediction.append(np.repeat(node_prediction, counts))
    return semantic_metrics(np.concatenate(truth), np.concatenate(prediction))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cave", type=Path, default=Path("input/BaumeDeLongeaigue.ply"))
    parser.add_argument("--cave-system")
    parser.add_argument("--voxel-size", type=float, default=.04)
    parser.add_argument("--seed", type=int, default=997)
    parser.add_argument("--cache-dir", type=Path, default=Path("output/spt_graph_cache/held_out"))
    parser.add_argument("--output", type=Path, default=Path("output/spt_updated/synthetic_validation.json"))
    args = parser.parse_args()
    group = cave_system_id(args.cave_system or args.cave.stem)
    with np.load(args.model, allow_pickle=False) as model:
        if group in json.loads(str(model["metadata"]))["training_groups"]:
            raise ValueError("The synthetic validation cave was used for training")
    root = Path(__file__).resolve().parent
    lib = load_spt_kernels(root/"build/libspt_kernels.so")
    xyz, normals, _ = read_ply(args.cave)
    reports = []
    for index, scenario in enumerate(("pure", "mouth", "skylight")):
        metadata = {"pipeline": pipeline_fingerprint(), "source": source_fingerprint(args.cave),
                    "cave_system": group, "split": "held_out_synthetic", "scenario": scenario,
                    "voxel_size": args.voxel_size, "seed": args.seed+index}
        def produce():
            if scenario != "pure":
                return generate_cutmix_superpoint_graph(lib, xyz, [], args.voxel_size, args.seed+index, scenario)
            result = run_spt_pipeline(lib, xyz, normals, voxel_size=args.voxel_size, pure_interior=True)
            histogram = class_histograms(np.zeros(len(xyz), dtype=np.int32),
                                         result["superpoint_p1"][result["orig_to_p0"]], result["num_p1"])
            return result, histogram
        graph = cached_graph(args.cache_dir, metadata, produce)
        probabilities = portable_predict(graph["X"], graph["edge_u"], graph["edge_v"], graph["sp_pos"],
                                          args.model, graph["feature_names"])
        report = {"scenario": scenario, "cave_system": group, "seed": args.seed+index,
                  "model": histogram_metrics(graph["histogram"], probabilities.argmax(axis=1)),
                  "partition_oracle": histogram_metrics(graph["histogram"], graph["histogram"][:, :3].argmax(axis=1))}
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"evaluation": "held-out pure interior and procedural CutMix; not real entrance accuracy",
                                      "cases": reports}, indent=2)+"\n")


if __name__ == "__main__":
    main()
