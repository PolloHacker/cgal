#!/usr/bin/env python3
"""
1. Extract coupled multi-class external tree + ground + building donor chunks from:
   - FOR-instance (data/trees/train_val_data/*_annotated_train.ply): 15m x 15m patches
     preserving both trees (semantic_seg in {2, 3} -> 1) and forest floor (semantic_seg == 1 -> 2).
   - DALES (data/dales/DALES-2/train/*.laz): 30-50 patches (25m x 25m / radius=15m) around
     terrain slopes, roads, buildings (Class 8), fences (Class 6), ground (Class 1), and
     tree lines (Class 2), with 3x local tangent-plane Gaussian jitter upsampling (+/- 3cm).
2. Combine the 9 manually segmented cave annotations in annotations/{cave,vegetation,unknown}
   into annotations/combined/<cave>_annotated.ply with exact scalar_ground_truth (0, 1, 2, -100).
"""

import glob
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spt_filter import read_ply, write_ply_binary


def extract_dales_exterior_patch(xyz: np.ndarray, sem_labels: np.ndarray, center_xy: np.ndarray, radius: float = 15.0):
    """Crop a multi-class exterior patch from a DALES tile and map to donor classes {1, 2}."""
    mask = np.linalg.norm(xyz[:, :2] - center_xy[None, :], axis=1) <= radius
    xyz_crop, labels_crop = xyz[mask], sem_labels[mask]

    # DALES: 2 -> 1 (Vegetation), {1 (Ground), 6 (Fences), 8 (Buildings)} -> 2 (Exterior Ground/Clutter)
    donor_class = np.full(len(labels_crop), -100, dtype=np.int32)
    donor_class[labels_crop == 2] = 1
    donor_class[np.isin(labels_crop, [1, 6, 8])] = 2

    valid = donor_class >= 1
    return xyz_crop[valid], donor_class[valid]


def upsample_local_tangent_plane(
    xyz: np.ndarray,
    donor_class: np.ndarray,
    copies: int = 3,
    jitter_std: float = 0.03,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Duplicate each DALES point `copies` times with +/- 3cm Gaussian jitter along its
    local planar neighborhood to bridge the airborne LiDAR density gap at voxel_size = 0.04.
    """
    if len(xyz) < 16 or copies <= 1:
        return xyz.copy(), donor_class.copy()
    if rng is None:
        rng = np.random.default_rng(42)

    pts = np.asarray(xyz, dtype=np.float64)
    k = min(12, len(pts))
    tree = cKDTree(pts)
    _, nn_idx = tree.query(pts, k=k)
    neighbors = pts[nn_idx]  # (N, k, 3)
    centered = neighbors - np.mean(neighbors, axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / float(max(k - 1, 1))
    _, eigvecs = np.linalg.eigh(cov)
    # eigh orders ascending: [:, :, 0] is normal, [:, :, 1] and [:, :, 2] span the tangent plane
    normal = eigvecs[:, :, 0]
    tangent_1 = eigvecs[:, :, 2]
    tangent_2 = eigvecs[:, :, 1]

    out_xyz = [pts]
    out_cls = [donor_class.astype(np.int32)]
    for _ in range(copies - 1):
        u1 = np.clip(rng.normal(0.0, jitter_std, size=(len(pts), 1)), -2.0 * jitter_std, 2.0 * jitter_std)
        u2 = np.clip(rng.normal(0.0, jitter_std, size=(len(pts), 1)), -2.0 * jitter_std, 2.0 * jitter_std)
        un = np.clip(rng.normal(0.0, jitter_std * 0.15, size=(len(pts), 1)), -0.01, 0.01)
        jittered = pts + u1 * tangent_1 + u2 * tangent_2 + un * normal
        out_xyz.append(jittered)
        out_cls.append(donor_class.astype(np.int32))

    return np.vstack(out_xyz), np.concatenate(out_cls)


def extract_for_instance_canopy_chunks(out_dir: Path, target_count: int = 50):
    """
    Extract 15m x 15m coupled tree + forest floor patches from data/trees/train_val_data:
      - semantic_seg in {2, 3} -> scalar_donor_class = 1 (Vegetation)
      - semantic_seg == 1      -> scalar_donor_class = 2 (Exterior Ground)
    """
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

            # Crop a 15m x 15m patch keeping both trees (2, 3 -> 1) and forest floor (1 -> 2)
            dx = np.abs(xyz[:, 0] - center_xy[0])
            dy = np.abs(xyz[:, 1] - center_xy[1])
            local_mask = (dx <= 7.5) & (dy <= 7.5) & np.isin(sem, [1, 2, 3])
            local_xyz = xyz[local_mask].copy()
            local_sem = sem[local_mask]

            donor_class = np.where(local_sem == 1, 2, 1).astype(np.int32)
            veg_idx = np.where(donor_class == 1)[0]
            gnd_idx = np.where(donor_class == 2)[0]
            if len(veg_idx) < 2000:
                continue

            # Balanced subsampling to ~25,000 - 35,000 points preserving both trees and forest floor
            rng = np.random.default_rng(100 + chunk_idx)
            max_veg = 24000
            max_gnd = 12000
            if len(veg_idx) > max_veg:
                veg_idx = rng.choice(veg_idx, size=max_veg, replace=False)
            if len(gnd_idx) > max_gnd:
                gnd_idx = rng.choice(gnd_idx, size=max_gnd, replace=False)
            keep_idx = np.concatenate([veg_idx, gnd_idx])
            local_xyz = local_xyz[keep_idx]
            donor_class = donor_class[keep_idx]

            # Center XY at origin and base Z at forest floor level
            local_xyz[:, 0] -= np.median(local_xyz[:, 0])
            local_xyz[:, 1] -= np.median(local_xyz[:, 1])
            if np.any(donor_class == 2):
                base_z = float(np.percentile(local_xyz[donor_class == 2, 2], 10))
            else:
                base_z = float(np.percentile(local_xyz[:, 2], 2))
            local_xyz[:, 2] -= base_z

            chunk_idx += 1
            out_path = out_dir / f"for_instance_tree_{chunk_idx:02d}.ply"
            write_ply_binary(
                out_path,
                local_xyz.astype(np.float64),
                None,
                {
                    "scalar_donor_class": donor_class,
                    "scalar_ground_truth": donor_class,
                },
            )
            records.append({
                "path": f"../annotations/canopy/{out_path.name}",
                "cave_system": f"for_instance_{plot_path.stem}",
            })

    print(f"[FOR-instance Donors] Extracted {len(records)} coupled tree+ground chunks -> {out_dir}")
    return records


def _load_dales_tiles_raw(laz_paths: list[Path]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load (stem, xyz, legacy_semantic) from DALES .laz tiles using laspy."""
    try:
        import laspy  # type: ignore
        results = []
        for p in laz_paths:
            las = laspy.read(str(p))
            xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float32)
            if hasattr(las, "legacy_semantic"):
                sem = np.asarray(las.legacy_semantic, dtype=np.int32)
            else:
                cls = np.asarray(las.classification, dtype=np.int32)
                sem = np.zeros_like(cls)
                sem[np.isin(cls, [1, 5])] = 2
                sem[cls == 0] = 1
                sem[cls == 4] = 6
                sem[np.isin(cls, [12, 13, 14])] = 8
            results.append((p.stem, xyz, sem))
        return results
    except ImportError:
        pass

    # Fallback to system python3.12 with CloudCompare flatpak site-packages containing laspy + lazrs
    flatpak_sites = glob.glob(
        "/var/lib/flatpak/app/org.cloudcompare.CloudCompare/x86_64/stable/*/files/lib/python3.12/site-packages"
    )
    if not flatpak_sites:
        raise RuntimeError("laspy is required to read DALES .laz files")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_npz = Path(tmpdir) / "dales_tiles.npz"
        helper_code = f"""
import sys, numpy as np
sys.path.insert(0, {flatpak_sites[0]!r})
import laspy
paths = {[str(p) for p in laz_paths]!r}
payload = {{}}
for idx, p in enumerate(paths):
    las = laspy.read(p)
    xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float32)
    sem = np.asarray(las.legacy_semantic, dtype=np.int32)
    payload[f"xyz_{{idx}}"] = xyz
    payload[f"sem_{{idx}}"] = sem
np.savez({str(tmp_npz)!r}, **payload)
"""
        subprocess.run(["/usr/bin/python3.12", "-c", helper_code], check=True)
        archive = np.load(tmp_npz)
        return [
            (laz_paths[idx].stem, archive[f"xyz_{idx}"], archive[f"sem_{idx}"])
            for idx in range(len(laz_paths))
        ]


def extract_dales_exterior_chunks(out_dir: Path, target_count: int = 40):
    """
    Extract 30-50 coupled 25m x 25m (radius=15m) patches from DALES (data/dales/DALES-2/train/*.laz)
    around terrain slopes, roads, buildings (8), fences (6), ground (1), and tree lines (2),
    with 3x local tangent-plane Gaussian jitter upsampling (+/- 3cm).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dales_dir = ROOT / "data/dales/DALES-2/train"
    laz_files = sorted(dales_dir.glob("*.laz"))
    if not laz_files:
        print(f"[DALES Donors] No DALES .laz files found in {dales_dir}")
        return []

    # Read 10 diverse DALES training tiles (~4 patches per tile -> 40 patches)
    selected_tiles = laz_files[:10]
    tiles_data = _load_dales_tiles_raw(selected_tiles)

    records = []
    chunk_idx = 0
    for tile_idx, (stem, xyz, sem) in enumerate(tiles_data):
        if chunk_idx >= target_count:
            break
        rng = np.random.default_rng(500 + tile_idx)

        # Candidate centers near tree lines, buildings, fences, and terrain slopes
        veg_pts = xyz[sem == 2]
        struct_pts = xyz[np.isin(sem, [6, 8])]
        gnd_pts = xyz[sem == 1]
        if len(veg_pts) < 500 or len(gnd_pts) < 500:
            continue

        # Sample candidate centers from vegetation-structure-ground transitions
        candidates_pool = struct_pts if len(struct_pts) >= 500 else veg_pts
        sample_indices = rng.choice(len(candidates_pool), size=min(40, len(candidates_pool)), replace=False)
        chosen_centers: list[np.ndarray] = []

        for s_idx in sample_indices:
            if chunk_idx >= target_count or len(chosen_centers) >= 4:
                break
            c_xy = candidates_pool[s_idx, :2]
            if any(np.linalg.norm(c_xy - prev) < 35.0 for prev in chosen_centers):
                continue

            xyz_patch, cls_patch = extract_dales_exterior_patch(xyz, sem, c_xy, radius=15.0)
            n_veg = int(np.sum(cls_patch == 1))
            n_gnd = int(np.sum(cls_patch == 2))
            if n_veg < 600 or n_gnd < 1200:
                continue

            chosen_centers.append(c_xy)

            # Cap pre-upsampled points to ~12,000 so 3x tangent-plane upsampling produces ~36,000 points
            if len(xyz_patch) > 12000:
                sel = rng.choice(len(xyz_patch), size=12000, replace=False)
                xyz_patch = xyz_patch[sel]
                cls_patch = cls_patch[sel]

            xyz_up, cls_up = upsample_local_tangent_plane(
                xyz_patch, cls_patch, copies=3, jitter_std=0.03, rng=rng
            )

            # Center XY at origin and base Z at ground level (Class 2)
            xyz_up[:, 0] -= np.median(xyz_up[:, 0])
            xyz_up[:, 1] -= np.median(xyz_up[:, 1])
            gnd_z = xyz_up[cls_up == 2, 2]
            base_z = float(np.percentile(gnd_z, 10)) if len(gnd_z) > 0 else float(np.percentile(xyz_up[:, 2], 5))
            xyz_up[:, 2] -= base_z

            chunk_idx += 1
            out_path = out_dir / f"dales_exterior_{chunk_idx:02d}.ply"
            write_ply_binary(
                out_path,
                xyz_up.astype(np.float64),
                None,
                {
                    "scalar_donor_class": cls_up.astype(np.int32),
                    "scalar_ground_truth": cls_up.astype(np.int32),
                },
            )
            records.append({
                "path": f"../annotations/canopy/{out_path.name}",
                "cave_system": f"dales_{stem}",
            })

    print(f"[DALES Donors] Extracted {len(records)} 3x tangent-upsampled ground+structure+tree chunks -> {out_dir}")
    return records


def update_manifest_canopy_chunks(for_records: list[dict], dales_records: list[dict]):
    """Update docs/spt-finetune-manifest.json so canopy_chunks includes coupled FOR-instance + DALES donors."""
    manifest_path = ROOT / "docs/spt-finetune-manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text())
    # Interleave DALES and FOR-instance chunks so sequential or random draws get both
    combined = []
    max_len = max(len(for_records), len(dales_records))
    for i in range(max_len):
        if i < len(for_records):
            combined.append({
                "path": for_records[i]["path"],
                "cave_system": "for_instance",
                "source": "for_instance",
            })
        if i < len(dales_records):
            combined.append({
                "path": dales_records[i]["path"],
                "cave_system": "dales",
                "source": "dales",
            })
    manifest["canopy_chunks"] = combined
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[Manifest] Updated {manifest_path} with {len(combined)} coupled FOR-instance + DALES donor chunks")


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
            # Pure interior scans without any vegetation split (e.g. interior_lapinha, S11D_0061)
            # are 100% subterranean cavern (0). Entrance scans keep -100 on unknown transition zones.
            if not veg_file.is_file():
                gt_list.append(np.zeros(len(u_xyz), dtype=np.int32))
                n_cav += len(u_xyz)
            else:
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
    for_recs = extract_for_instance_canopy_chunks(ROOT / "annotations/canopy", target_count=50)
    dales_recs = extract_dales_exterior_chunks(ROOT / "annotations/canopy", target_count=40)
    update_manifest_canopy_chunks(for_recs, dales_recs)
    combine_cave_annotations(ROOT / "annotations/combined")
