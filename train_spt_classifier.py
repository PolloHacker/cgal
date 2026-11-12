#!/usr/bin/env python3
"""Train a two-layer PyTorch GraphSAGE from masked seeds and raw-point 3D CutMix.

Portable .npz inference uses NumPy/SciPy. Real validation requires a manifest with
explicit cave-system groups and a separate scalar_ground_truth annotation field.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from scipy.spatial import cKDTree

from spt_filter import (inspect_ply_header, load_spt_kernels, read_ply,
                        run_spt_pipeline, run_spt_pipeline_from_p0, stream_subsample_ply)
from spt_learning import (SCHEMA, cave_system_id, class_histograms, deform_canopy_chunk,
                          load_graph, save_graph, supervision_histogram)


def load_donor_chunk(path):
    """Load a coupled donor PLY returning (xyz, scalar_donor_class) with classes in {1, 2}."""
    xyz, _, fields = read_ply(Path(path))
    if "scalar_donor_class" in fields:
        cls = fields["scalar_donor_class"].astype(np.int32)
    elif "scalar_ground_truth" in fields:
        cls = fields["scalar_ground_truth"].astype(np.int32)
    else:
        cls = np.ones(len(xyz), dtype=np.int32)
    cls = np.where(cls == 2, 2, 1).astype(np.int32)
    return xyz, cls


def _unpack_donor(entry):
    if isinstance(entry, tuple) and len(entry) == 2:
        return entry[0], entry[1]
    arr = np.asarray(entry, dtype=np.float64)
    return arr, np.ones(len(arr), dtype=np.int32)


def generate_cutmix_superpoint_graph(lib, cave_xyz_f64, canopy_chunks_xyz,
                                    voxel_size=0.04, seed=42, scenario="mouth"):
    """Open a raw cave mouth/skylight/exterior_surroundings, graft coupled donors, then rerun geometry."""
    rng = np.random.default_rng(seed)
    center = cave_xyz_f64.mean(axis=0)
    local = cave_xyz_f64-center
    _, axes = np.linalg.eigh(np.cov(local[:, :2].T))
    forward = axes[:, -1] * rng.choice([-1, 1])
    across = np.array([-forward[1], forward[0]])
    longitudinal = local[:, :2] @ forward
    lateral = local[:, :2] @ across
    if scenario == "mouth":
        mouth = np.percentile(longitudinal, rng.uniform(70, 88))
        kept = longitudinal <= mouth
        near = (longitudinal > mouth-3) & kept
        origin = center[:2]+mouth*forward
    elif scenario == "exterior_surroundings":
        mouth = np.percentile(longitudinal, rng.uniform(62, 82))
        kept = longitudinal <= mouth
        near = (longitudinal > mouth-3) & kept
        if near.sum() < 20:
            near = kept
        origin = center[:2]+mouth*forward
    elif scenario == "skylight":
        mouth = np.percentile(longitudinal, rng.uniform(30, 70))
        origin = center[:2]+mouth*forward
        horizontal = np.linalg.norm(cave_xyz_f64[:, :2]-origin, axis=1)
        near = horizontal < 3.0
        if near.sum() < 20:
            return generate_cutmix_superpoint_graph(lib, cave_xyz_f64, canopy_chunks_xyz,
                                                    voxel_size, seed, "mouth")
        roof_cut = np.percentile(cave_xyz_f64[near, 2], 60)
        kept = ~(near & (cave_xyz_f64[:, 2] > roof_cut))
    else:
        raise ValueError("CutMix scenario must be mouth, skylight, or exterior_surroundings")
    ground_z = float(np.percentile(cave_xyz_f64[near, 2], 15))
    roof_z = float(np.percentile(cave_xyz_f64[near, 2], 85))
    cave = cave_xyz_f64[kept]
    width = max(2.0, min(12.0, float(np.percentile(lateral, 90)-np.percentile(lateral, 10))))
    grade = rng.uniform(0.03, 0.16)
    if scenario == "mouth":
        centers = [(rng.uniform(-width*.40, width*.40), f) for f in (1.5, 3.0, 5.0, 7.5, 10.5)]
    elif scenario == "exterior_surroundings":
        # Single continuous 25m x 30m DALES/FOR-instance ground + building + tree patch extending outward
        centers = [(0.0, 13.5), (rng.uniform(-width*.35, width*.35), 3.0), (rng.uniform(-width*.35, width*.35), 6.5)]
    else:
        centers = [(rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)) for _ in range(3)]

    donors = [_unpack_donor(c) for c in canopy_chunks_xyz] if canopy_chunks_xyz else []
    ground_rich_indices = [
        idx for idx, (pts, cls) in enumerate(donors)
        if int(np.sum(cls == 2)) >= 2500 and (np.ptp(pts[:, 0]) >= 20.0 or np.ptp(pts[:, 1]) >= 20.0)
    ]
    if not ground_rich_indices:
        ground_rich_indices = [idx for idx, (_, cls) in enumerate(donors) if int(np.sum(cls == 2)) >= 1000]

    donor_pts_list = []
    donor_cls_list = []
    for index, (a, f) in enumerate(centers):
        xy = origin+a*across+f*forward
        base_z = (ground_z+grade*max(f, 0)) if scenario in ("mouth", "exterior_surroundings") else roof_z
        if donors:
            if scenario == "exterior_surroundings" and index == 0 and ground_rich_indices:
                # Primary 25m x 30m coupled ground + building + tree patch (no overlapping ground slabs)
                donor_idx = int(rng.choice(ground_rich_indices))
                raw_chunk, raw_cls = donors[donor_idx]
                chunk, chunk_cls = deform_canopy_chunk(raw_chunk, rng, chunk_labels=raw_cls, tilt_deg=4.0)
                gnd_mask = chunk_cls == 2
                ref_z = float(np.percentile(chunk[gnd_mask, 2], 10)) if np.any(gnd_mask) else float(np.percentile(chunk[:, 2], 2))
                chunk += np.r_[xy, ground_z - ref_z]
                chunk_long = (chunk[:, :2] - center[:2]) @ forward
                keep_ext = chunk_long >= mouth - 0.25
                chunk = chunk[keep_ext]
                chunk_cls = chunk_cls[keep_ext]
            else:
                # Tree canopy/trunk donors (keep Class 1 vegetation only so ground slabs never stack in the air)
                donor_idx = int(rng.integers(len(donors)))
                raw_chunk, raw_cls = donors[donor_idx]
                veg_only = raw_chunk[raw_cls == 1]
                if len(veg_only) < 100:
                    veg_only = raw_chunk
                chunk = deform_canopy_chunk(veg_only, rng, tilt_deg=12.0)
                chunk += np.r_[xy, base_z - np.percentile(chunk[:, 2], 2)]
                if scenario in ("mouth", "exterior_surroundings"):
                    chunk_long = (chunk[:, :2] - center[:2]) @ forward
                    chunk = chunk[chunk_long >= mouth - 0.25]
                else:
                    chunk = chunk[chunk[:, 2] >= roof_cut - 0.25]
                chunk_cls = np.ones(len(chunk), dtype=np.int32)
            if len(chunk):
                donor_pts_list.append(chunk)
                donor_cls_list.append(chunk_cls)
        else:
            theta = rng.uniform(0, 2*np.pi, 1400)
            radius = rng.uniform(0.16, 0.28)+rng.normal(0, 0.012, len(theta))
            trunk = np.column_stack([xy[0]+radius*np.cos(theta), xy[1]+radius*np.sin(theta),
                                      rng.uniform(base_z, base_z+5, len(theta))])
            fol = np.r_[xy, base_z+0.8]+rng.normal(0, .55, (1200, 3))
            donor_pts_list += [trunk, fol]
            donor_cls_list += [np.ones(len(trunk), dtype=np.int32), np.ones(len(fol), dtype=np.int32)]
            for _ in range(8):
                crown = np.r_[xy+rng.uniform(-1.8, 1.8, 2), base_z+rng.uniform(4, 8)]
                c_pts = crown+rng.normal(0, .65, (500, 3))
                donor_pts_list.append(c_pts)
                donor_cls_list.append(np.ones(len(c_pts), dtype=np.int32))
    grafted_xyz = np.vstack(donor_pts_list)
    grafted_cls = np.concatenate(donor_cls_list).astype(np.int32)
    xyz = np.vstack([cave, grafted_xyz])
    truth = np.concatenate([np.zeros(len(cave), dtype=np.int32), grafted_cls])
    result = run_spt_pipeline(lib, xyz, None, voxel_size=voxel_size)
    # Every raw source point contributes, including different labels sharing a P0 voxel.
    mapping = result["superpoint_p1"][result["orig_to_p0"]]
    histogram = class_histograms(truth, mapping, result["num_p1"])
    return result, histogram


def extract_pseudo_labeled_graph(lib, path, voxel_size=0.04, is_pure_interior=False,
                                 source_path=None, label_field="scalar_ground_truth",
                                 already_p0=False):
    path = Path(path)
    source = Path(source_path) if source_path else path
    fmt, count, props, _ = inspect_ply_header(path)
    prop_names = {name for name, _ in props}
    has_ground_truth = label_field in prop_names
    if not is_pure_interior and not has_ground_truth:
        raise ValueError(
            f"Heuristic seed generation is disabled; entrance training scan {path} "
            f"must contain explicit '{label_field}' annotations"
        )
    if source.resolve() != path.resolve():
        ann_xyz, _, fields = read_ply(path)
        truth = fields[label_field].astype(np.int32)
        s_fmt, s_count, _, _ = inspect_ply_header(source)
        if s_fmt == "binary_little_endian" and s_count > 15_000_000:
            n, centroid, xyz, normals = stream_subsample_ply(lib, source, voxel_size)
            result = run_spt_pipeline_from_p0(lib, n, centroid, xyz, normals,
                                              voxel_size=voxel_size, pure_interior=False)
        else:
            xyz, normals, _ = read_ply(source)
            result = run_spt_pipeline(lib, xyz, normals, voxel_size=voxel_size, pure_interior=False)
        tree = cKDTree(result["p0_xyz_f64"])
        dist, mapping_p0 = tree.query(ann_xyz, k=1, workers=-1)
        tol = np.sqrt(3) * voxel_size * 1.05
        if np.any(dist > tol):
            raise ValueError(f"Annotation coordinates in {path} exceed {tol:.4f} m tolerance from {source}")
        histogram = supervision_histogram(result, is_pure_interior=False,
                                          point_labels=truth,
                                          point_mapping=result["superpoint_p1"][mapping_p0])
        return result, histogram

    if fmt == "binary_little_endian" and count > 15_000_000 and not has_ground_truth:
        n, centroid, xyz, normals = stream_subsample_ply(lib, path, voxel_size)
        result = run_spt_pipeline_from_p0(lib, n, centroid, xyz, normals,
                                          voxel_size=voxel_size, pure_interior=is_pure_interior)
        histogram = supervision_histogram(result, is_pure_interior=is_pure_interior)
    elif already_p0:
        xyz, normals, fields = read_ply(path)
        centroid = xyz.mean(axis=0, keepdims=True)
        p0_xyz = (xyz - centroid).astype(np.float32)
        result = run_spt_pipeline_from_p0(lib, len(xyz), centroid, p0_xyz, normals,
                                          voxel_size=voxel_size,
                                          pure_interior=is_pure_interior and not has_ground_truth)
        if has_ground_truth:
            truth = fields[label_field].astype(np.int32)
            histogram = supervision_histogram(result, is_pure_interior=False,
                                              point_labels=truth,
                                              point_mapping=result["superpoint_p1"])
        else:
            histogram = supervision_histogram(result, is_pure_interior=is_pure_interior)
    else:
        xyz, normals, fields = read_ply(path)
        result = run_spt_pipeline(lib, xyz, normals, voxel_size=voxel_size,
                                  pure_interior=is_pure_interior and not has_ground_truth)
        if has_ground_truth:
            truth = fields[label_field].astype(np.int32)
            histogram = supervision_histogram(result, is_pure_interior=False, point_labels=truth)
        else:
            histogram = supervision_histogram(result, is_pure_interior=is_pure_interior)
    return result, histogram


def validate_splits(manifest):
    train = manifest.get("training", [])
    canopy = manifest.get("canopy_chunks", [])
    validation = manifest.get("validation", [])
    for record in train+canopy+validation:
        if not record.get("path") or not record.get("cave_system"):
            raise ValueError("Every manifest entry needs path and cave_system")
    train_groups = {cave_system_id(r["cave_system"]) for r in train+canopy}
    validation_groups = {cave_system_id(r["cave_system"]) for r in validation}
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError(f"Training/canopy sources overlap validation cave systems: {sorted(overlap)}")
    for record in train:
        if record.get("role") not in ("interior", "entrance"):
            raise ValueError("Training role must be interior or entrance")
    resolved_train = {str(Path(r["path"]).resolve()) for r in train+canopy}
    if any(str(Path(r[key]).resolve()) in resolved_train for r in validation for key in ("path", "source") if key in r):
        raise ValueError("The same file appears in training and validation")
    return manifest


def source_fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def pipeline_fingerprint():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("spt_filter.py", "spt_partition.py", "spt_learning.py",
                 "src/spt_filter/spt_kernels.cpp", "build/libspt_kernels.so", "build/spt_cut_pursuit_build.json"):
        digest.update((root/name).read_bytes())
    return digest.hexdigest()


def cached_graph(cache_dir, metadata, producer):
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(cache_dir)/f"{key}.npz"
    if not path.exists():
        result, histogram = producer()
        save_graph(path, result, histogram, metadata)
    else:
        print(f"[Cache] {path}", flush=True)
    return load_graph(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--interior-caves", nargs="*", default=["input/rupt.ply", "input/BaumeDeLongeaigue.ply", "input/diaclase.ply"])
    parser.add_argument("--entrance-caves", nargs="*", default=[])
    parser.add_argument("--canopy-chunks", nargs="*", default=[])
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--cutmix-per-cave", type=int, default=3)
    parser.add_argument("--cache-dir", type=Path, default=Path("output/spt_graph_cache"))
    parser.add_argument("--output-model", type=Path, default=Path("output/spt_graphsage.npz"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.cutmix_per_cave < 1 or args.voxel_size <= 0:
        parser.error("epochs, cutmix-per-cave, and voxel-size must be positive")
    if args.output_model.suffix != ".npz":
        parser.error("output-model must have a .npz suffix; a .pt checkpoint is saved alongside it")
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        # Manifest-relative paths make an annotation folder portable.
        for key in ("training", "canopy_chunks", "validation"):
            for record in manifest.get(key, []):
                for field in ("path", "source"):
                    if field in record:
                        record[field] = str((args.manifest.parent/record[field]).resolve())
    else:
        if args.canopy_chunks:
            parser.error("Real canopy chunks require --manifest with explicit cave_system identities")
        manifest = {
            "training": [{"path": p, "cave_system": Path(p).stem, "role": role}
                         for role, paths in (("interior", args.interior_caves), ("entrance", args.entrance_caves)) for p in paths],
            "canopy_chunks": [{"path": p, "cave_system": Path(p).stem} for p in args.canopy_chunks],
            "validation": [],
        }
    validate_splits(manifest)
    for record in manifest["training"]+manifest.get("canopy_chunks", [])+manifest.get("validation", []):
        for key in ("path", "source"):
            if key in record and not Path(record[key]).is_file():
                raise FileNotFoundError(record[key])
    for record in manifest.get("validation", []):
        label = record.get("label_field", "scalar_ground_truth")
        names = {name for name, _ in inspect_ply_header(Path(record["path"]))[2]}
        if label in ("scalar_predicted_class", "scalar_cave_zone", "scalar_pseudo_label") or label not in names:
            raise ValueError(f"Validation requires a separate {label} annotation field in {record['path']}")
    import torch
    from spt_torch import train_graphs
    # Set this before geometry too: importing PyTorch can change the shared
    # OpenMP thread pool and otherwise slow the native WNNC solver substantially.
    torch.set_num_threads(8)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable to this Python process")
    root = Path(__file__).resolve().parent
    lib = load_spt_kernels(root/"build/libspt_kernels.so")
    canopy_records = manifest.get("canopy_chunks", [])
    canopy = [load_donor_chunk(record["path"]) for record in canopy_records]
    pipeline_id = pipeline_fingerprint()
    cutmix_scenarios = ("mouth", "skylight", "exterior_surroundings")
    graphs = []
    for case_index, case in enumerate(manifest["training"]):
        path = Path(case["path"])
        source_path = Path(case["source"]) if "source" in case else None
        label_field = case.get("label_field", "scalar_ground_truth")
        already_p0 = bool(case.get("already_p0", False))
        spacing = float(case.get("voxel_size", args.voxel_size))
        metadata = {"schema": SCHEMA, "pipeline": pipeline_id, "source": source_fingerprint(path),
                    "cave_system": case["cave_system"], "role": case["role"], "voxel_size": spacing,
                    "already_p0": already_p0}
        if source_path is not None:
            metadata["full_source"] = source_fingerprint(source_path)
        print(f"[Seeds] {path.name} ({case['role']})", flush=True)
        graphs.append(cached_graph(args.cache_dir, metadata,
                      lambda p=path, sp=source_path, s=spacing, r=case["role"], lf=label_field, ap0=already_p0:
                      extract_pseudo_labeled_graph(lib, p, s, r == "interior", sp, lf, ap0)))
        if case["role"] == "interior":
            xyz = None
            for i in range(args.cutmix_per_cave):
                scenario = cutmix_scenarios[i % len(cutmix_scenarios)]
                seed = args.seed+case_index*100+i
                mix_metadata = {**metadata, "role": "cutmix", "scenario": scenario, "seed": seed,
                                "canopy_sources": [source_fingerprint(r["path"]) for r in canopy_records]}
                def produce(p=path, s=spacing, sd=seed, sc=scenario, lf=label_field):
                    nonlocal xyz
                    if xyz is None:
                        pts, _, flds = read_ply(p)
                        xyz = pts[flds[lf] == 0] if lf in flds else pts
                    return generate_cutmix_superpoint_graph(lib, xyz, canopy, s, sd, sc)
                print(f"[CutMix] {path.name}, {scenario}, seed={seed}", flush=True)
                graphs.append(cached_graph(args.cache_dir, mix_metadata, produce))
    started = time.perf_counter()
    model = train_graphs(graphs, epochs=args.epochs, device=device, seed=args.seed)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    training_groups = sorted({cave_system_id(r["cave_system"]) for r in manifest["training"]+canopy_records})
    metadata = {"training_groups": training_groups, "pipeline": pipeline_id, "voxel_size": args.voxel_size,
                "epochs": args.epochs, "seed": args.seed, "device": device,
                "graphs": [g["metadata"] for g in graphs]}
    model.export(args.output_model, graphs[0]["feature_names"], metadata)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "metadata": metadata,
        "feature_names": graphs[0]["feature_names"].tolist(),
        "global_feat_mean": model.global_feat_mean.detach().cpu().numpy(),
        "global_feat_std": model.global_feat_std.detach().cpu().numpy(),
    }, args.output_model.with_suffix(".pt"))
    print(f"[Saved] {args.output_model}, training {time.perf_counter()-started:.2f}s on {device}")
    if manifest.get("validation"):
        from evaluate_spt_classifier import evaluate_manifest
        report = evaluate_manifest(lib, manifest["validation"], args.output_model, args.voxel_size)
        report_path = args.output_model.with_suffix(".validation.json")
        report_path.write_text(json.dumps(report, indent=2)+"\n")
        print(f"[Held-out validation] {report_path}")
    else:
        print("[Validation] No real held-out labels supplied; training loss is not an accuracy estimate.")


if __name__ == "__main__":
    main()
