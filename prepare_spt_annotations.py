#!/usr/bin/env python3
"""Combine manually separated CloudCompare crops into semantic ground truth."""
import argparse
from pathlib import Path

import numpy as np

from spt_filter import read_ply, write_ply_binary


def combine_annotations(parts):
    coordinates, labels = [], []
    for label, paths in parts.items():
        for path in paths:
            xyz, _, _ = read_ply(Path(path))
            coordinates.append(xyz)
            labels.append(np.full(len(xyz), label, dtype=np.int32))
    if not coordinates:
        raise ValueError("Supply at least one manually separated crop")
    xyz, truth = np.vstack(coordinates), np.concatenate(labels)
    unique, inverse = np.unique(xyz, axis=0, return_inverse=True)
    low = np.full(len(unique), 3, dtype=np.int32)
    high = np.full(len(unique), -100, dtype=np.int32)
    known = truth != -100
    np.minimum.at(low, inverse[known], truth[known])
    np.maximum.at(high, inverse[known], truth[known])
    if np.any((high >= 0) & (high != low)):
        raise ValueError("The same coordinates occur in different semantic classes; correct overlapping segments")
    return unique, high


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cavern", nargs="*", default=[])
    parser.add_argument("--vegetation", nargs="*", default=[])
    parser.add_argument("--exterior-ground", nargs="*", default=[])
    parser.add_argument("--unknown", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    xyz, truth = combine_annotations({0: args.cavern, 1: args.vegetation,
                                      2: args.exterior_ground, -100: args.unknown})
    write_ply_binary(args.output, xyz, None, {"scalar_ground_truth": truth})
    values, counts = np.unique(truth, return_counts=True)
    print(f"Saved {args.output}: {dict(zip(values.tolist(), counts.tolist()))}")


if __name__ == "__main__":
    main()
