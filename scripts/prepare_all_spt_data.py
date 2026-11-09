#!/usr/bin/env python3
"""
1. Extract 50+ pre-labeled external tree + ground donor chunks from FOR-instance
   (data/trees/train_val_data/*.ply) into annotations/canopy/*.ply.
2. Combine the 9 manually segmented cave annotations in annotations/{cave,vegetation,unknown}
   into annotations/combined/<cave>_annotated.ply with exact scalar_ground_truth (0, 1, 2, -100).
"""

from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spt_filter import read_ply, write_ply_binary


def extract_for_instance_canopy_chunks(out_dir: Path, target_count: int = 50):
    """Extract clean labeled tree + local ground patches from data/trees/train_val_data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir = ROOT / "data/trees/train_val_data"
    plot_files = sorted(
        p for p in train_dir.glob("*_annotated_train.ply")
        if p.stat().st_size < 60_000_000
    )

    records = []
    chunk_idx = 0
    for plot_path in plot_files:
        if chunk_idx >= target_count:
            break
        xyz, _, extra = read_ply(plot_path)
        if "semantic_seg" not in extra or "treeID" not in extra:
            continue
        sem = extra["semantic_seg"].astype(np.int32)
        tid = extra["treeID"].astype(np.int32)

        # Find well-sampled individual trees (or pairs of adjacent trees)
        unique_tids, counts = np.unique(tid[tid > 0], return_counts=True)
        good_tids = unique_tids[(counts >= 4000) & (counts <= 65000)]
        for t_val in good_tids[:4]:
            if chunk_idx >= target_count:
                break
            tree_mask = (tid == t_val) & (sem >= 2)
            if tree_mask.sum() < 3000:
                continue
            tree_xyz = xyz[tree_mask]
            center_xy = np.median(tree_xyz[:, :2], axis=0)
            radius_xy = max(2.5, float(np.percentile(np.linalg.norm(tree_xyz[:, :2] - center_xy, axis=1), 95)))

            # Include nearby trees & ground within radius_xy * 1.15 for realistic canopy/trunk/ground chunks
            dist_xy = np.linalg.norm(xyz[:, :2] - center_xy, axis=1)
            local_mask = dist_xy <= min(radius_xy * 1.15, 7.5)
            local_xyz = xyz[local_mask].copy()
            local_sem = sem[local_mask]

            # Subsample to ~15,000 - 35,000 points if very dense
            if len(local_xyz) > 35000:
                rng = np.random.default_rng(100 + chunk_idx)
                sel = rng.choice(len(local_xyz), size=30000, replace=False)
                local_xyz = local_xyz[sel]
                local_sem = local_sem[sel]

            # Center XY at origin and base Z at 0
            local_xyz[:, 0] -= np.median(local_xyz[:, 0])
            local_xyz[:, 1] -= np.median(local_xyz[:, 1])
            local_xyz[:, 2] -= np.percentile(local_xyz[:, 2], 2)

            # Map FOR-instance semantic_seg: 1 -> 2 (Exterior Ground), 2/3 -> 1 (Vegetation)
            gt = np.where(local_sem == 1, 2, 1).astype(np.int32)
            veg_only_xyz = local_xyz[gt == 1]
            if len(veg_only_xyz) < 2000:
                continue

            chunk_idx += 1
            out_path = out_dir / f"for_instance_tree_{chunk_idx:02d}.ply"
            # Save vegetation-only points (and ground truth scalar) so both legacy and new CutMix work
            write_ply_binary(
                out_path,
                veg_only_xyz.astype(np.float64),
                None,
                {"scalar_ground_truth": np.ones(len(veg_only_xyz), dtype=np.int32)},
            )
            records.append({
                "path": str(out_path.relative_to(ROOT)),
                "cave_system": f"for_instance_{plot_path.stem}",
            })

    print(f"[Canopy Donors] Extracted {len(records)} clean pre-labeled tree chunks -> {out_dir}")
    return records


def combine_cave_annotations(out_dir: Path):
    """Combine annotations/{cave,vegetation,unknown}/<prefix>_*.ply into annotations/combined/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cave_dir = ROOT / "annotations/cave"
    veg_dir = ROOT / "annotations/vegetation"
    unk_dir = ROOT / "annotations/unknown"

    combined_info = {}
    for cave_file in sorted(cave_dir.glob("*_cave.ply")):
        prefix = cave_file.name[: -len("_cave.ply")]
        veg_file = veg_dir / f"{prefix}_vegetation.ply"
        unk_file = unk_dir / f"{prefix}_unknown.ply"

        t0 = time.perf_counter()
        xyz_list = []
        gt_list = []

        c_xyz, _, _ = read_ply(cave_file)
        xyz_list.append(c_xyz)
        gt_list.append(np.zeros(len(c_xyz), dtype=np.int32))
        n_cav = len(c_xyz)

        n_veg = 0
        if veg_file.is_file():
            v_xyz, _, _ = read_ply(veg_file)
            xyz_list.append(v_xyz)
            gt_list.append(np.ones(len(v_xyz), dtype=np.int32))
            n_veg = len(v_xyz)

        n_unk = 0
        if unk_file.is_file():
            u_xyz, _, _ = read_ply(unk_file)
            xyz_list.append(u_xyz)
            gt_list.append(np.full(len(u_xyz), -100, dtype=np.int32))
            n_unk = len(u_xyz)

        xyz_all = np.vstack(xyz_list)
        gt_all = np.concatenate(gt_list)

        out_path = out_dir / f"{prefix}_annotated.ply"
        write_ply_binary(out_path, xyz_all, None, {"scalar_ground_truth": gt_all})
        combined_info[prefix] = {
            "path": str(out_path),
            "cavern": n_cav,
            "vegetation": n_veg,
            "unknown": n_unk,
            "total": len(xyz_all),
        }
        print(
            f"[Combined] {prefix:20s} -> {out_path.name:32s} | "
            f"Cavern={n_cav:9,d} | Veg={n_veg:7,d} | Unknown={n_unk:9,d} ({time.perf_counter()-t0:.2f}s)"
        )
    return combined_info


if __name__ == "__main__":
    extract_for_instance_canopy_chunks(ROOT / "annotations/canopy", target_count=50)
    combine_cave_annotations(ROOT / "annotations/combined")
