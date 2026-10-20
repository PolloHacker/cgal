#!/usr/bin/env python3
"""
Standalone Superpoint & Multi-Scale Geometric Filtering Pipeline (Option 2)
Implements Stages 1-6 from docs/spt-superpoint-vegetation-filtering-plan.md:
  1. Ingestion, Float64 Centroid Subtraction & Global P0 Voxel Subsampling
  2. Trajectory-Free Coherent Normal Estimation / Orientation
  3. Multi-Scale CloudCompare + SPT Geometric & 3D Cone Enclosure Features (P0)
  4. Hierarchical l0-Cut Pursuit Superpoint Partitioning (P0 -> P1 -> P2)
  5. Superpoint Graph Scoring, Speleothem Protection & Geodesic Filtering
  6. CloudCompare Multi-Scalar Diagnostic .ply & Cleaned Cavern .ply Export
"""

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree


# -----------------------------------------------------------------------------
# C++ OpenMP/Eigen3 Shared Library Binding
# -----------------------------------------------------------------------------
def load_spt_kernels(so_path: Path) -> ctypes.CDLL:
    if not so_path.exists():
        raise FileNotFoundError(
            f"Shared library {so_path} not found. Compile src/spt_filter/spt_kernels.cpp first."
        )
    lib = ctypes.CDLL(str(so_path))

    f32_p = ctypes.POINTER(ctypes.c_float)
    i32_p = ctypes.POINTER(ctypes.c_int32)

    lib.spt_voxel_subsample.argtypes = [
        f32_p, ctypes.c_int32, ctypes.c_float, i32_p, i32_p
    ]
    lib.spt_voxel_subsample.restype = ctypes.c_int32

    lib.spt_compute_scale_features.argtypes = [
        f32_p, f32_p, ctypes.c_int32, ctypes.c_float, ctypes.c_int32, f32_p
    ]
    lib.spt_compute_scale_features.restype = None

    lib.spt_compute_enclosure.argtypes = [
        f32_p, f32_p, f32_p, f32_p, ctypes.c_int32, ctypes.c_float, ctypes.c_float, f32_p
    ]
    lib.spt_compute_enclosure.restype = None

    lib.spt_partition_superpoints.argtypes = [
        f32_p, f32_p, f32_p, f32_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int32, i32_p
    ]
    lib.spt_partition_superpoints.restype = ctypes.c_int32

    lib.spt_build_superpoint_graph.argtypes = [
        f32_p, i32_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float, ctypes.c_int32, i32_p, i32_p
    ]
    lib.spt_build_superpoint_graph.restype = ctypes.c_int32

    lib.spt_map_points_to_p0.argtypes = [
        f32_p, ctypes.c_int32, f32_p, ctypes.c_int32, ctypes.c_float, i32_p
    ]
    lib.spt_map_points_to_p0.restype = None

    lib.spt_compute_wnnc_normals.argtypes = [
        f32_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_int32, f32_p
    ]
    lib.spt_compute_wnnc_normals.restype = None

    return lib


# -----------------------------------------------------------------------------
# PLY Reader & Writer (Supports Binary Little-Endian, ASCII, & Out-of-Core Streaming)
# -----------------------------------------------------------------------------
PLY_TYPE_MAP = {
    "char": "<i1", "int8": "<i1",
    "uchar": "<u1", "uint8": "<u1",
    "short": "<i2", "int16": "<i2",
    "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4",
    "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4",
    "double": "<f8", "float64": "<f8",
}


def inspect_ply_header(filepath: Path) -> Tuple[str, int, list, int]:
    """
    Returns (format_str, num_vertices, props_list, data_offset_bytes).
    """
    with open(filepath, "rb") as f:
        fmt = "ascii"
        num_vertices = 0
        props = []
        in_vertex_element = False

        while True:
            line_bytes = f.readline()
            if not line_bytes:
                raise ValueError(f"Unexpected EOF in PLY header of {filepath}")
            line = line_bytes.decode("latin1").strip()
            if line == "end_header":
                break
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                if tokens[1] == "vertex":
                    num_vertices = int(tokens[2])
                    in_vertex_element = True
                else:
                    in_vertex_element = False
            elif tokens[0] == "property" and in_vertex_element:
                ptype, pname = tokens[1], tokens[2]
                props.append((pname, PLY_TYPE_MAP[ptype]))

        data_offset = f.tell()
    return fmt, num_vertices, props, data_offset


def read_ply(filepath: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
    """
    Reads vertex coordinates in float64, optional normals in float32, and extra properties.
    """
    fmt, num_vertices, props, data_offset = inspect_ply_header(filepath)
    with open(filepath, "rb") as f:
        f.seek(data_offset)
        if fmt == "binary_little_endian":
            dtype = np.dtype(props)
            raw = np.fromfile(f, dtype=dtype, count=num_vertices)
            xyz_f64 = np.column_stack([
                raw["x"].astype(np.float64),
                raw["y"].astype(np.float64),
                raw["z"].astype(np.float64),
            ])
            normals = None
            if all(k in raw.dtype.names for k in ("nx", "ny", "nz")):
                normals = np.column_stack([
                    raw["nx"].astype(np.float32),
                    raw["ny"].astype(np.float32),
                    raw["nz"].astype(np.float32),
                ])
            extras = {
                name: raw[name]
                for name, _ in props
                if name not in ("x", "y", "z", "nx", "ny", "nz")
            }
        elif fmt == "ascii":
            data = np.loadtxt(f, dtype=np.float64, max_rows=num_vertices)
            prop_names = [p[0] for p in props]
            ix, iy, iz = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")
            xyz_f64 = np.ascontiguousarray(data[:, [ix, iy, iz]], dtype=np.float64)
            normals = None
            if all(k in prop_names for k in ("nx", "ny", "nz")):
                inx, iny, inz = prop_names.index("nx"), prop_names.index("ny"), prop_names.index("nz")
                normals = np.ascontiguousarray(data[:, [inx, iny, inz]], dtype=np.float32)
            extras = {
                name: data[:, idx].astype(np.float32)
                for idx, name in enumerate(prop_names)
                if name not in ("x", "y", "z", "nx", "ny", "nz")
            }
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")

    return xyz_f64, normals, extras


def stream_subsample_ply(
    lib: ctypes.CDLL,
    filepath: Path,
    voxel_size: float,
    chunk_size: int = 15_000_000,
) -> Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Out-of-core chunked P0 voxel subsampling for massive binary PLY files (100M-300M+ points).
    Streams the file in chunks of `chunk_size` vertices, subsampling each chunk at `voxel_size`
    and then merging the chunk P0 clouds into a single global P0 cloud.
    Returns (n_orig, centroid_f64, p0_xyz_centered_f32, p0_normals_f32_or_None).
    """
    fmt, num_vertices, props, data_offset = inspect_ply_header(filepath)
    if fmt != "binary_little_endian":
        raise ValueError("Out-of-core streaming requires binary_little_endian PLY.")

    dtype = np.dtype(props)
    has_normals = all(k in dtype.names for k in ("nx", "ny", "nz"))
    f32_p = ctypes.POINTER(ctypes.c_float)
    i32_p = ctypes.POINTER(ctypes.c_int32)

    chunk_p0_xyz_list = []
    chunk_p0_norm_list = []
    centroid_f64 = None

    p0_idx_buf = np.empty(chunk_size, dtype=np.int32)
    orig_to_p0_buf = np.empty(chunk_size, dtype=np.int32)

    with open(filepath, "rb") as f:
        f.seek(data_offset)
        remaining = num_vertices
        chunk_id = 0
        while remaining > 0:
            n_read = min(chunk_size, remaining)
            raw = np.fromfile(f, dtype=dtype, count=n_read)
            if len(raw) == 0:
                break
            remaining -= len(raw)
            chunk_id += 1

            if centroid_f64 is None:
                centroid_f64 = np.array(
                    [[
                        float(np.mean(raw["x"], dtype=np.float64)),
                        float(np.mean(raw["y"], dtype=np.float64)),
                        float(np.mean(raw["z"], dtype=np.float64)),
                    ]],
                    dtype=np.float64,
                )

            c_xyz = np.empty((len(raw), 3), dtype=np.float32)
            c_xyz[:, 0] = (raw["x"].astype(np.float64) - centroid_f64[0, 0]).astype(np.float32)
            c_xyz[:, 1] = (raw["y"].astype(np.float64) - centroid_f64[0, 1]).astype(np.float32)
            c_xyz[:, 2] = (raw["z"].astype(np.float64) - centroid_f64[0, 2]).astype(np.float32)

            n_sub = lib.spt_voxel_subsample(
                c_xyz.ctypes.data_as(f32_p),
                ctypes.c_int32(len(raw)),
                ctypes.c_float(voxel_size),
                p0_idx_buf.ctypes.data_as(i32_p),
                orig_to_p0_buf.ctypes.data_as(i32_p),
            )
            sel = p0_idx_buf[:n_sub]
            chunk_p0_xyz_list.append(c_xyz[sel].copy())
            if has_normals:
                c_nrm = np.column_stack([
                    raw["nx"][sel].astype(np.float32),
                    raw["ny"][sel].astype(np.float32),
                    raw["nz"][sel].astype(np.float32),
                ])
                chunk_p0_norm_list.append(c_nrm)

    merged_xyz = np.ascontiguousarray(np.vstack(chunk_p0_xyz_list), dtype=np.float32)
    del chunk_p0_xyz_list

    n_merged = len(merged_xyz)
    final_p0_idx = np.empty(n_merged, dtype=np.int32)
    final_orig_to_p0 = np.empty(n_merged, dtype=np.int32)
    num_p0 = lib.spt_voxel_subsample(
        merged_xyz.ctypes.data_as(f32_p),
        ctypes.c_int32(n_merged),
        ctypes.c_float(voxel_size),
        final_p0_idx.ctypes.data_as(i32_p),
        final_orig_to_p0.ctypes.data_as(i32_p),
    )
    final_sel = final_p0_idx[:num_p0]
    p0_xyz = np.ascontiguousarray(merged_xyz[final_sel], dtype=np.float32)
    del merged_xyz

    p0_normals = None
    if has_normals:
        merged_nrm = np.vstack(chunk_p0_norm_list)
        del chunk_p0_norm_list
        p0_normals = np.ascontiguousarray(merged_nrm[final_sel], dtype=np.float32)

    return num_vertices, centroid_f64, p0_xyz, p0_normals


def stream_export_cleaned_ply(
    lib: ctypes.CDLL,
    input_ply: Path,
    clean_path: Path,
    centroid_f64: np.ndarray,
    p0_xyz_centered: np.ndarray,
    p0_normals: np.ndarray,
    p0_pred_class: np.ndarray,
    voxel_size: float,
    chunk_size: int = 15_000_000,
) -> int:
    """
    Streams `input_ply` in chunks, maps each point to its nearest P0 voxel using
    `spt_map_points_to_p0`, filters for Class 0 (Cavern), and writes `clean_path`
    with globally consistent WNNC+raycast-oriented normals without holding the full
    100M-300M cloud in RAM.
    """
    fmt, num_vertices, props, data_offset = inspect_ply_header(input_ply)
    in_dtype = np.dtype(props)

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    use_f64_coords = bool(np.max(np.abs(centroid_f64)) > 1e4)
    coord_dtype = "<f8" if use_f64_coords else "<f4"
    coord_ply_type = "double" if use_f64_coords else "float"

    out_dtype = np.dtype([
        ("x", coord_dtype), ("y", coord_dtype), ("z", coord_dtype),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
    ])

    def build_fixed_header(vertex_count: int) -> bytes:
        v_line = f"element vertex {vertex_count}\n"
        pad_len = 80 - len(v_line)
        c_line = "comment " + ("=" * (pad_len - 9)) + "\n"
        hdr = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "comment Generated by SPT Multi-Scale Geometric Filtering Pipeline\n"
            + c_line
            + v_line
            + f"property {coord_ply_type} x\n"
            + f"property {coord_ply_type} y\n"
            + f"property {coord_ply_type} z\n"
            + "property float nx\n"
            + "property float ny\n"
            + "property float nz\n"
            + "end_header\n"
        )
        return hdr.encode("ascii")

    f32_p = ctypes.POINTER(ctypes.c_float)
    i32_p = ctypes.POINTER(ctypes.c_int32)
    num_p0 = len(p0_xyz_centered)
    query_to_p0_buf = np.empty(chunk_size, dtype=np.int32)
    total_kept = 0

    with open(input_ply, "rb") as fin, open(clean_path, "wb") as fout:
        fin.seek(data_offset)
        fout.write(build_fixed_header(0))

        remaining = num_vertices
        while remaining > 0:
            n_read = min(chunk_size, remaining)
            raw = np.fromfile(fin, dtype=in_dtype, count=n_read)
            if len(raw) == 0:
                break
            n_cur = len(raw)
            remaining -= n_cur

            c_xyz = np.empty((n_cur, 3), dtype=np.float32)
            c_xyz[:, 0] = (raw["x"].astype(np.float64) - centroid_f64[0, 0]).astype(np.float32)
            c_xyz[:, 1] = (raw["y"].astype(np.float64) - centroid_f64[0, 1]).astype(np.float32)
            c_xyz[:, 2] = (raw["z"].astype(np.float64) - centroid_f64[0, 2]).astype(np.float32)

            lib.spt_map_points_to_p0(
                p0_xyz_centered.ctypes.data_as(f32_p),
                ctypes.c_int32(num_p0),
                c_xyz.ctypes.data_as(f32_p),
                ctypes.c_int32(n_cur),
                ctypes.c_float(voxel_size),
                query_to_p0_buf.ctypes.data_as(i32_p),
            )
            mapped_p0 = query_to_p0_buf[:n_cur]
            keep_mask = p0_pred_class[mapped_p0] == 0
            n_keep = int(np.sum(keep_mask))
            if n_keep == 0:
                continue

            out_chunk = np.empty(n_keep, dtype=out_dtype)
            out_chunk["x"] = raw["x"][keep_mask].astype(coord_dtype)
            out_chunk["y"] = raw["y"][keep_mask].astype(coord_dtype)
            out_chunk["z"] = raw["z"][keep_mask].astype(coord_dtype)

            kept_p0 = mapped_p0[keep_mask]
            out_chunk["nx"] = p0_normals[kept_p0, 0]
            out_chunk["ny"] = p0_normals[kept_p0, 1]
            out_chunk["nz"] = p0_normals[kept_p0, 2]

            out_chunk.tofile(fout)
            total_kept += n_keep

        fout.seek(0)
        fout.write(build_fixed_header(total_kept))

    return total_kept


def write_ply_binary(
    filepath: Path,
    xyz_f64: np.ndarray,
    normals_f32: Optional[np.ndarray],
    scalars: Dict[str, np.ndarray],
) -> None:
    """
    Writes a CloudCompare-compatible binary_little_endian PLY file.
    Stores coordinates in double if max abs coordinate > 1e4 (e.g. UTM/LV95), else float32.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    n_pts = len(xyz_f64)
    use_f64_coords = bool(np.max(np.abs(xyz_f64)) > 1e4)
    coord_dtype = "<f8" if use_f64_coords else "<f4"
    coord_ply_type = "double" if use_f64_coords else "float"

    dtype_fields = [("x", coord_dtype), ("y", coord_dtype), ("z", coord_dtype)]
    header_props = [
        f"property {coord_ply_type} x",
        f"property {coord_ply_type} y",
        f"property {coord_ply_type} z",
    ]

    if normals_f32 is not None:
        dtype_fields.extend([("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")])
        header_props.extend([
            "property float nx",
            "property float ny",
            "property float nz",
        ])

    for sname, sarr in scalars.items():
        if np.issubdtype(sarr.dtype, np.integer):
            dtype_fields.append((sname, "<i4"))
            header_props.append(f"property int {sname}")
        else:
            dtype_fields.append((sname, "<f4"))
            header_props.append(f"property float {sname}")

    out_arr = np.empty(n_pts, dtype=np.dtype(dtype_fields))
    out_arr["x"] = xyz_f64[:, 0].astype(coord_dtype)
    out_arr["y"] = xyz_f64[:, 1].astype(coord_dtype)
    out_arr["z"] = xyz_f64[:, 2].astype(coord_dtype)

    if normals_f32 is not None:
        out_arr["nx"] = normals_f32[:, 0]
        out_arr["ny"] = normals_f32[:, 1]
        out_arr["nz"] = normals_f32[:, 2]

    for sname, sarr in scalars.items():
        if np.issubdtype(sarr.dtype, np.integer):
            out_arr[sname] = sarr.astype(np.int32)
        else:
            out_arr[sname] = sarr.astype(np.float32)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Generated by SPT Multi-Scale Geometric Filtering Pipeline\n"
        f"element vertex {n_pts}\n"
        + "\n".join(header_props)
        + "\nend_header\n"
    )
    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        out_arr.tofile(f)


# -----------------------------------------------------------------------------
# Core Pipeline Execution (Stages 1 - 5)
# -----------------------------------------------------------------------------
def run_spt_pipeline_from_p0(
    lib: ctypes.CDLL,
    n_orig: int,
    centroid_f64: np.ndarray,
    p0_xyz: np.ndarray,
    p0_normals_in: Optional[np.ndarray],
    p0_to_orig: Optional[np.ndarray] = None,
    orig_to_p0: Optional[np.ndarray] = None,
    t_stage1: float = 0.0,
    voxel_size: float = 0.04,
    r_fine: float = 0.12,
    r_coarse: float = 0.50,
    p1_reg: float = 0.015,
    p1_cutoff: int = 3,
    p2_reg: float = 0.08,
    p2_cutoff: int = 15,
) -> Dict[str, np.ndarray]:
    t1 = time.perf_counter()
    num_p0 = len(p0_xyz)
    f32_p = ctypes.POINTER(ctypes.c_float)
    i32_p = ctypes.POINTER(ctypes.c_int32)

    # --- Stage 2A: Global WNNC Normal Estimation (Inward Cavity Orientation) ---
    # Always compute globally consistent normals via Winding Number Normal Consistency (WNNC),
    # ignoring untrusted or inverted input PLY normals.
    p0_normals = np.empty((num_p0, 3), dtype=np.float32)
    lib.spt_compute_wnnc_normals(
        p0_xyz.ctypes.data_as(f32_p),
        ctypes.c_int32(num_p0),
        ctypes.c_int32(30),
        ctypes.c_float(0.003),
        ctypes.c_float(0.04),
        ctypes.c_int32(60000),
        p0_normals.ctypes.data_as(f32_p),
    )
    t_wnnc = time.perf_counter()

    # --- Stage 2B & 3A: Decoupled Partition vs. Classification Multi-Scale Features ---
    # Part 1.1 Fix: Keep r_fine_eff tight (2.85 * d_0) so local covariance spheres do not blur
    # across the cave-wall / overhanging-bush drip-line transition during Stage 4 cut-pursuit.
    # Reserve the wide 10.0 * d_0 coarse sphere strictly for Stage 3 enclosure & Stage 5 scoring.
    r_fine_eff = max(r_fine, voxel_size * 2.85)
    r_coarse_eff = max(r_coarse, voxel_size * 10.0)
    occ_cell_eff = max(0.20, voxel_size * 4.0)

    fine_feats = np.empty((num_p0, 6), dtype=np.float32)
    coarse_feats = np.empty((num_p0, 6), dtype=np.float32)

    lib.spt_compute_scale_features(
        p0_xyz.ctypes.data_as(f32_p),
        p0_normals.ctypes.data_as(f32_p),
        ctypes.c_int32(num_p0),
        ctypes.c_float(r_fine_eff),
        ctypes.c_int32(1),
        fine_feats.ctypes.data_as(f32_p),
    )

    lib.spt_compute_scale_features(
        p0_xyz.ctypes.data_as(f32_p),
        p0_normals.ctypes.data_as(f32_p),
        ctypes.c_int32(num_p0),
        ctypes.c_float(r_coarse_eff),
        ctypes.c_int32(0),
        coarse_feats.ctypes.data_as(f32_p),
    )
    t2 = time.perf_counter()

    # --- Stage 3B: Global 3D Cone Enclosure Raycasting & Floor/Ceiling Normal Verification ---
    enclosure_feats = np.empty((num_p0, 4), dtype=np.float32)
    lib.spt_compute_enclosure(
        p0_xyz.ctypes.data_as(f32_p),
        p0_normals.ctypes.data_as(f32_p),
        fine_feats.ctypes.data_as(f32_p),
        coarse_feats.ctypes.data_as(f32_p),
        ctypes.c_int32(num_p0),
        ctypes.c_float(occ_cell_eff),
        ctypes.c_float(45.0),
        enclosure_feats.ctypes.data_as(f32_p),
    )
    upward_occlusion = enclosure_feats[:, 0]
    roof_occlusion = enclosure_feats[:, 1]
    downward_occlusion = enclosure_feats[:, 2]
    cavern_enclosure = enclosure_feats[:, 3]
    foliage_above = np.maximum(0.0, upward_occlusion - roof_occlusion)

    # Compute signed_horizontality (nz), horizontality (|nz|), and verticality (1 - |nz|)
    # AFTER WNNC + physical vertical raycast verification (Floor: nz > 0, Ceiling: nz < 0)
    signed_horizontality = p0_normals[:, 2].copy()
    horizontality = np.abs(signed_horizontality)
    verticality = 1.0 - horizontality
    t3 = time.perf_counter()

    # --- Stage 4: Contrast-Sensitive Hierarchical l0-Cut Pursuit (P0 -> P1 -> P2) ---
    # Part 1.1 & 1.2 Fixes:
    #   1. Decouple partition features from coarse classification features (exclude coarse_feats
    #      from part_feats so 1m+ spheres never blur boundaries across the drip line).
    #   2. Lower weight of signed_horizontality from 1.2 to 0.35 so vertical tree trunks and
    #      entrance slopes (|nz| ~ 0) do not merge into vertical cave walls.
    #   3. Increase weight of unsmoothed fine normal dispersion (1.5) and fine scattering (1.5).
    #   4. Prune k-NN edges > 3.5 * d_0, weight edges by normal & dispersion contrast, and
    #      over-segment P1 with pcp_cutoff=[3, 15], pcp_regularization=[0.015, 0.08].
    part_feats = np.column_stack([
        fine_feats[:, 1] * 1.0,       # planarity_fine [0, 1]
        fine_feats[:, 2] * 1.5,       # scattering_fine [0, 1.5]
        fine_feats[:, 3] * 1.0,       # linearity_fine [0, 1]
        fine_feats[:, 0] * 3.0,       # rugosity_fine scaled to [0, 1]
        fine_feats[:, 5] * 1.5,       # normal_dispersion_fine [0, 1.5]
        verticality * 0.5,            # verticality [0, 0.5]
        signed_horizontality * 0.35,  # signed_horizontality [-0.35, +0.35]
        roof_occlusion * 0.7,         # rock roof occlusion [0, 0.7]
        foliage_above * 0.8,          # overhead foliage canopy shielding [0, 0.8]
    ]).astype(np.float32)
    part_feats = np.ascontiguousarray(part_feats, dtype=np.float32)
    ndisp_fine_contig = np.ascontiguousarray(fine_feats[:, 5], dtype=np.float32)

    p1_labels = np.empty(num_p0, dtype=np.int32)
    num_p1 = lib.spt_partition_superpoints(
        p0_xyz.ctypes.data_as(f32_p),
        p0_normals.ctypes.data_as(f32_p),
        ndisp_fine_contig.ctypes.data_as(f32_p),
        part_feats.ctypes.data_as(f32_p),
        ctypes.c_int32(num_p0),
        ctypes.c_int32(part_feats.shape[1]),
        ctypes.c_float(max(voxel_size * 4.0, 0.15)),
        ctypes.c_float(voxel_size * 3.5),
        ctypes.c_float(p1_reg),
        ctypes.c_float(0.02),
        ctypes.c_int32(p1_cutoff),
        p1_labels.ctypes.data_as(i32_p),
    )

    p2_labels = np.empty(num_p0, dtype=np.int32)
    num_p2 = lib.spt_partition_superpoints(
        p0_xyz.ctypes.data_as(f32_p),
        p0_normals.ctypes.data_as(f32_p),
        ndisp_fine_contig.ctypes.data_as(f32_p),
        part_feats.ctypes.data_as(f32_p),
        ctypes.c_int32(num_p0),
        ctypes.c_int32(part_feats.shape[1]),
        ctypes.c_float(max(voxel_size * 7.0, 0.28)),
        ctypes.c_float(voxel_size * 6.0),
        ctypes.c_float(p2_reg),
        ctypes.c_float(0.008),
        ctypes.c_int32(p2_cutoff),
        p2_labels.ctypes.data_as(i32_p),
    )

    # Build P1 Superpoint Boundary Adjacency Graph
    max_edges = num_p1 * 30 + 10000
    edge_u = np.empty(max_edges, dtype=np.int32)
    edge_v = np.empty(max_edges, dtype=np.int32)
    num_edges = lib.spt_build_superpoint_graph(
        p0_xyz.ctypes.data_as(f32_p),
        p1_labels.ctypes.data_as(i32_p),
        ctypes.c_int32(num_p0),
        ctypes.c_int32(num_p1),
        ctypes.c_float(max(voxel_size * 5.0, 0.20)),
        ctypes.c_int32(max_edges),
        edge_u.ctypes.data_as(i32_p),
        edge_v.ctypes.data_as(i32_p),
    )
    edge_u = edge_u[:num_edges]
    edge_v = edge_v[:num_edges]
    t4 = time.perf_counter()

    # --- Stage 5: Superpoint Graph Scoring, Drip-Line Descriptors & Geodesic Filtering ---
    sp_counts = np.bincount(p1_labels, minlength=num_p1).astype(np.float32)
    sp_inv = 1.0 / np.maximum(sp_counts, 1.0)

    def sp_mean(arr: np.ndarray) -> np.ndarray:
        return np.bincount(p1_labels, weights=arr, minlength=num_p1).astype(np.float32) * sp_inv

    def sp_std(arr: np.ndarray, mean_arr: np.ndarray) -> np.ndarray:
        diff2 = (arr - mean_arr[p1_labels]) ** 2
        return np.sqrt(np.bincount(p1_labels, weights=diff2, minlength=num_p1).astype(np.float32) * sp_inv)

    sp_pos = np.column_stack([sp_mean(p0_xyz[:, d]) for d in range(3)])
    sp_nz = sp_mean(signed_horizontality)
    sp_vert = sp_mean(verticality)
    sp_rug_fine = sp_mean(fine_feats[:, 0])
    sp_plan_fine = sp_mean(fine_feats[:, 1])
    sp_scat_fine = sp_mean(fine_feats[:, 2])
    sp_lin_fine = sp_mean(fine_feats[:, 3])
    sp_ndisp_fine = sp_mean(fine_feats[:, 5])
    sp_std_plan_fine = sp_std(fine_feats[:, 1], sp_plan_fine)
    sp_std_ndisp_fine = sp_std(fine_feats[:, 5], sp_ndisp_fine)

    sp_rug_coarse = sp_mean(coarse_feats[:, 0])
    sp_plan_coarse = sp_mean(coarse_feats[:, 1])
    sp_scat_coarse = sp_mean(coarse_feats[:, 2])
    sp_ndisp_coarse = sp_mean(coarse_feats[:, 5])

    sp_up_occ = sp_mean(upward_occlusion)
    sp_roof_occ = sp_mean(roof_occlusion)
    sp_down_occ = sp_mean(downward_occlusion)
    sp_enclosure = sp_mean(cavern_enclosure)
    sp_foliage_above = np.maximum(0.0, sp_up_occ - sp_roof_occ)

    # Parent P2 Superpoint regional context aggregation
    p2_counts = np.bincount(p2_labels, minlength=num_p2).astype(np.float32)
    p2_inv = 1.0 / np.maximum(p2_counts, 1.0)
    p2_roof_occ = (np.bincount(p2_labels, weights=roof_occlusion, minlength=num_p2) * p2_inv).astype(np.float32)
    p2_foliage_above = (np.bincount(p2_labels, weights=foliage_above, minlength=num_p2) * p2_inv).astype(np.float32)
    p2_plan_coarse = (np.bincount(p2_labels, weights=coarse_feats[:, 1], minlength=num_p2) * p2_inv).astype(np.float32)
    p2_ndisp_coarse = (np.bincount(p2_labels, weights=coarse_feats[:, 5], minlength=num_p2) * p2_inv).astype(np.float32)
    # Map each P1 superpoint to its primary P2 parent via representative P0 voxel
    p1_rep_p0 = np.zeros(num_p1, dtype=np.int32)
    p1_rep_p0[p1_labels] = np.arange(num_p0, dtype=np.int32)
    p1_to_p2 = p2_labels[p1_rep_p0]
    sp_p2_roof_occ = p2_roof_occ[p1_to_p2]
    sp_p2_foliage_above = p2_foliage_above[p1_to_p2]
    sp_p2_plan_coarse = p2_plan_coarse[p1_to_p2]
    sp_p2_ndisp_coarse = p2_ndisp_coarse[p1_to_p2]

    # 1. Unary Vegetation / High-Entropy Clutter Score [0, 1]
    entropy_score = np.clip(
        0.30 * (sp_rug_coarse / 0.13)
        + 0.25 * (sp_scat_coarse / 0.30)
        + 0.25 * (sp_ndisp_fine / 0.38)
        + 0.20 * (sp_scat_fine / 0.28),
        0.0,
        1.5,
    )
    open_sky_factor = (
        np.clip(1.0 - (sp_roof_occ / 0.22), 0.0, 1.0)
        * np.clip(1.0 - (sp_enclosure / 0.35), 0.0, 1.0)
    )
    non_planar_factor = np.clip(1.0 - (np.maximum(sp_plan_fine, sp_plan_coarse) - 0.34) * 3.0, 0.0, 1.0)
    unary_veg_score = np.clip(entropy_score * open_sky_factor * non_planar_factor, 0.0, 1.0)

    # 2. Fast Multi-Source Geodesic Distance on Superpoint Graph (Virtual Source Dijkstra)
    def multisource_dijkstra(seeds: np.ndarray) -> np.ndarray:
        if len(seeds) == 0 or num_edges == 0:
            return np.full(num_p1, 999.0, dtype=np.float32)
        virtual_src = num_p1
        u_all = np.concatenate([edge_u, edge_v, np.full(len(seeds), virtual_src, dtype=np.int32), seeds])
        v_all = np.concatenate([edge_v, edge_u, seeds, np.full(len(seeds), virtual_src, dtype=np.int32)])
        w_all = np.concatenate([edge_dist, edge_dist, np.zeros(2 * len(seeds), dtype=np.float32)])
        mat = sparse.csr_matrix((w_all, (u_all, v_all)), shape=(num_p1 + 1, num_p1 + 1))
        dists = csgraph.dijkstra(mat, directed=False, indices=virtual_src)[:num_p1]
        return np.minimum(dists, 999.0).astype(np.float32)

    if num_edges > 0:
        edge_dist = np.linalg.norm(sp_pos[edge_u] - sp_pos[edge_v], axis=1).astype(np.float32)
    else:
        edge_dist = np.empty(0, dtype=np.float32)

    # True Cave Ceiling Overhangs: coherent 2D rock surface facing downward with floor below,
    # and NOT sitting underneath a dense 3D foliage canopy (sp_foliage_above < 0.30).
    is_ceiling_overhang = (
        (sp_nz < -0.22)
        & (sp_down_occ > 0.20)
        & (sp_plan_fine > 0.42)
        & (sp_ndisp_fine < 0.28)
        & (sp_foliage_above < 0.30)
    )

    # Confirmed Cavern Vault Seeds: enclosed coherent rock ceilings, walls, shafts, or floors
    # (Excluded if sitting underneath a foliage canopy sp_foliage_above >= 0.32)
    vault_seeds = np.where(
        (sp_counts >= max(3, p1_cutoff))
        & (sp_rug_fine < 0.12)
        & (sp_ndisp_fine < 0.32)
        & (sp_foliage_above < 0.32)
        & ((sp_plan_fine > 0.38) | (sp_plan_coarse > 0.38))
        & (
            (sp_enclosure > 0.28)
            | (sp_roof_occ > 0.24)
            | is_ceiling_overhang
        )
    )[0]
    dist_to_vault_graph = multisource_dijkstra(vault_seeds) if len(vault_seeds) > 0 else np.zeros(num_p1, dtype=np.float32)
    if len(vault_seeds) > 0:
        vault_tree = cKDTree(sp_pos[vault_seeds])
        dist_to_vault_euclid, _ = vault_tree.query(sp_pos, k=1)
        dist_to_vault = np.minimum(dist_to_vault_graph, dist_to_vault_euclid.astype(np.float32))
    else:
        dist_to_vault = np.zeros(num_p1, dtype=np.float32)

    # 3. Drip-Line Lateral Distance (d_drip in XY plane) & Elevation Above Local Floor (z_rel):
    # Identify enclosed cave conduit anchors (high roof occlusion or true ceiling overhang) that belong
    # to a macroscopic conduit component (>= 800 P0 voxels) so isolated forest boulders are ignored.
    drip_thresh = 0.50 if np.sum(sp_counts[sp_roof_occ > 0.50]) >= 500 else 0.28
    raw_drip_anchor = (
        ((sp_roof_occ > drip_thresh) | is_ceiling_overhang)
        & (sp_plan_fine > 0.36)
        & (sp_ndisp_fine < 0.30)
        & (sp_foliage_above < 0.30)
        & (sp_counts >= max(3, p1_cutoff))
    )
    drip_cand_idx = np.where(raw_drip_anchor)[0]
    if len(drip_cand_idx) > 0:
        drip_tree_xy = cKDTree(sp_pos[drip_cand_idx, :2])
        d_pairs = drip_tree_xy.query_pairs(r=5.0, output_type="ndarray")
        if len(d_pairs) > 0:
            du = drip_cand_idx[d_pairs[:, 0]]
            dv = drip_cand_idx[d_pairs[:, 1]]
            d_adj = sparse.csr_matrix(
                (np.ones(2 * len(du), dtype=np.int32), (np.concatenate([du, dv]), np.concatenate([dv, du]))),
                shape=(num_p1, num_p1),
            )
            n_dc, d_cl = csgraph.connected_components(d_adj, directed=False)
            d_comp_p0 = np.bincount(d_cl, weights=sp_counts * raw_drip_anchor, minlength=n_dc)
            valid_drip_anchor = raw_drip_anchor & (d_comp_p0[d_cl] >= max(500.0, 0.08 * float(d_comp_p0.max())))
        else:
            valid_drip_anchor = raw_drip_anchor
    else:
        valid_drip_anchor = raw_drip_anchor

    drip_anchor_idx = np.where(valid_drip_anchor)[0]
    if len(drip_anchor_idx) == 0:
        drip_anchor_idx = vault_seeds

    if len(drip_anchor_idx) > 0:
        drip_xy_tree = cKDTree(sp_pos[drip_anchor_idx, :2])
        sp_d_drip_f64, nearest_drip_i = drip_xy_tree.query(sp_pos[:, :2], k=1)
        sp_d_drip = sp_d_drip_f64.astype(np.float32)
        # Local lowest floor elevation z_rel from enclosed floor anchors
        floor_anchor_idx = drip_anchor_idx[sp_nz[drip_anchor_idx] > 0.20]
        if len(floor_anchor_idx) > 0:
            floor_xy_tree = cKDTree(sp_pos[floor_anchor_idx, :2])
            _, nearest_fl_i = floor_xy_tree.query(sp_pos[:, :2], k=1)
            sp_z_rel = (sp_pos[:, 2] - sp_pos[floor_anchor_idx[nearest_fl_i], 2]).astype(np.float32)
        else:
            sp_z_rel = (sp_pos[:, 2] - sp_pos[drip_anchor_idx[nearest_drip_i], 2]).astype(np.float32)
    else:
        sp_d_drip = np.zeros(num_p1, dtype=np.float32)
        sp_z_rel = np.zeros(num_p1, dtype=np.float32)

    # 4. Candidate Open-Sky Vegetation Canopy Superpoints:
    raw_canopy_mask = (
        (~is_ceiling_overhang)
        & (sp_roof_occ < 0.15)
        & (sp_enclosure < 0.22)
        & (dist_to_vault > 1.8)
        & (sp_plan_fine < 0.36)
        & (sp_rug_fine > 0.09)
        & (sp_scat_fine > 0.19)
        & (sp_ndisp_fine > 0.30)
        & (sp_plan_coarse < 0.35)
        & ((sp_rug_coarse > 0.13) | (sp_scat_coarse > 0.28))
        & (sp_ndisp_coarse > 0.38)
    )

    # Candidate exterior trunk / branch / bush superpoints (unenclosed by rock roof, non-planar)
    raw_veg_support_mask = (
        raw_canopy_mask
        | (
            (~is_ceiling_overhang)
            & (sp_roof_occ < 0.16)
            & (sp_enclosure < 0.22)
            & (dist_to_vault > 1.8)
            & (
                ((sp_plan_coarse < 0.36) & ((sp_lin_fine > 0.28) | ((sp_vert > 0.65) & (sp_ndisp_coarse > 0.44))))
                | ((sp_plan_fine < 0.35) & (sp_ndisp_fine > 0.30) & (sp_rug_fine > 0.09))
            )
        )
    )

    # Topological Superpoint Graph Connected-Component Filtering (Step 5 of SPT Guide):
    # Connect candidate exterior vegetation superpoints within 3.5m (bridging gaps between adjacent branches)
    # and require both macroscopic cluster size (>= 500 P0 voxels, >= 300 canopy P0 voxels) AND that the
    # connected component actually extends into open exterior space (> 4.5m from any enclosed cave vault).
    validated_canopy_mask = np.zeros(num_p1, dtype=bool)
    cand_indices = np.where(raw_veg_support_mask)[0]
    if np.any(raw_canopy_mask) and len(cand_indices) > 0:
        cand_tree = cKDTree(sp_pos[cand_indices])
        pairs = cand_tree.query_pairs(r=3.5, output_type="ndarray")
        if len(pairs) > 0:
            u_c = cand_indices[pairs[:, 0]]
            v_c = cand_indices[pairs[:, 1]]
            sub_adj = sparse.csr_matrix(
                (np.ones(2 * len(u_c), dtype=np.int32), (np.concatenate([u_c, v_c]), np.concatenate([v_c, u_c]))),
                shape=(num_p1, num_p1),
            )
            n_comp, comp_labels = csgraph.connected_components(sub_adj, directed=False)
        else:
            n_comp = num_p1
            comp_labels = np.arange(num_p1, dtype=np.int32)
        comp_p0_sizes = np.bincount(comp_labels, weights=sp_counts * raw_veg_support_mask, minlength=n_comp)
        comp_canopy_p0 = np.bincount(comp_labels, weights=sp_counts * raw_canopy_mask, minlength=n_comp)
        comp_max_vault_dist = np.zeros(n_comp, dtype=np.float32)
        np.maximum.at(comp_max_vault_dist, comp_labels[cand_indices], dist_to_vault[cand_indices])
        valid_comps = (comp_p0_sizes >= 500) & (comp_canopy_p0 >= 300) & (comp_max_vault_dist > 4.5)
        validated_canopy_mask = raw_canopy_mask & valid_comps[comp_labels]

    canopy_seeds = np.where(validated_canopy_mask)[0]
    if len(canopy_seeds) > 0:
        dist_to_canopy_graph = multisource_dijkstra(canopy_seeds)
        canopy_tree = cKDTree(sp_pos[canopy_seeds])
        dist_to_canopy_euclid, _ = canopy_tree.query(sp_pos, k=1)
        dist_to_canopy = np.minimum(dist_to_canopy_graph, dist_to_canopy_euclid.astype(np.float32))
    else:
        dist_to_canopy = np.full(num_p1, 999.0, dtype=np.float32)

    # 5. Eliminate "Green Bleeding" (Lower Tree Trunks, Understory Bushes & Steep Entrance Slope):
    # Per Part 1.2 & Immediate Action Checklist:
    # Never allow Wall (|nz| < 0.35) or unenclosed surfaces outside the drip line (d_drip > 2.0m)
    # or beneath a foliage canopy (sp_foliage_above > 0.35) with low rock roof occlusion (sp_roof_occ < 0.20)
    # to remain classified as Cavern Wall / Ceiling!
    outside_drip_or_under_canopy = (
        (len(canopy_seeds) > 0)
        & (~is_ceiling_overhang)
        & (sp_roof_occ < 0.20)
        & (sp_enclosure < 0.24)
        & (dist_to_canopy < 16.0)
        & (
            (sp_d_drip > 2.0)
            | ((sp_d_drip > 1.0) & (sp_z_rel > 3.5))
            | ((sp_foliage_above > 0.35) & (sp_plan_fine < 0.58))
        )
    )

    # Separate outside-drip-line superpoints into Exterior Ground (coherent upward-facing terrain/talus)
    # vs. Vegetation / Trunks / Bushes (vertical trunks |nz| < 0.35, linear stems, or high-entropy shrubs):
    is_coherent_upward_slope = (
        (sp_nz > 0.35)
        & (sp_plan_fine > 0.42)
        & (sp_scat_fine < 0.18)
        & (sp_ndisp_fine < 0.26)
        & (sp_rug_fine < 0.09)
    )

    trunk_absorbed = (
        (
            (len(canopy_seeds) > 0)
            & (~is_ceiling_overhang)
            & (dist_to_canopy < 6.0)
            & (dist_to_vault > 1.5)
            & (sp_roof_occ < 0.18)
            & (sp_enclosure < 0.22)
            & (
                ((sp_plan_coarse < 0.38) & ((sp_lin_fine > 0.25) | ((sp_vert > 0.62) & (sp_ndisp_coarse > 0.38))))
                | ((sp_plan_fine < 0.36) & (sp_ndisp_fine > 0.28) & (sp_rug_fine > 0.08))
                | ((dist_to_canopy < 3.5) & (sp_counts <= max(6, p1_cutoff * 2)))
            )
        )
        | (outside_drip_or_under_canopy & (~is_coherent_upward_slope))
    )

    # Exterior Ground Rule: unenclosed upward-facing terrain/talus outside the drip line
    exterior_ground = (
        (len(canopy_seeds) > 0)
        & (dist_to_canopy < 18.0)
        & (sp_roof_occ < 0.16)
        & (sp_enclosure < 0.20)
        & (
            (
                (dist_to_vault > 1.8)
                & (sp_nz > 0.42)
                & (sp_down_occ < 0.12)
            )
            | (outside_drip_or_under_canopy & is_coherent_upward_slope)
        )
    )

    # Speleothem & Enclosed Cavern Protection (gated on ROCK roof/floor occlusion & low foliage_above)
    speleothem_protected = (
        (sp_roof_occ > 0.24)
        | (sp_enclosure > 0.28)
        | (is_ceiling_overhang & (sp_d_drip < 2.5))
        | (
            (sp_d_drip < 1.8)
            & (sp_foliage_above < 0.28)
            & ((sp_roof_occ > 0.18) | (sp_down_occ > 0.22))
            & (sp_plan_fine > 0.45)
            & (sp_ndisp_fine < 0.25)
        )
    )

    # Continuous Superpoint Cavern Score [0, 1]
    sp_cavern_score = np.clip(
        0.80 + 0.20 * sp_enclosure - 0.65 * unary_veg_score * (dist_to_canopy < 10.0),
        0.0,
        1.0,
    )
    sp_cavern_score[speleothem_protected] = np.maximum(sp_cavern_score[speleothem_protected], 0.85)
    sp_cavern_score[validated_canopy_mask] = np.minimum(sp_cavern_score[validated_canopy_mask], 0.08)
    sp_cavern_score[trunk_absorbed] = np.minimum(sp_cavern_score[trunk_absorbed], 0.14)
    sp_cavern_score[exterior_ground] = np.minimum(sp_cavern_score[exterior_ground], 0.18)

    # Final Superpoint Semantic Class (Positive-Evidence Rule — Zero Hallucinated Vegetation):
    #   0 = Cavern (Floor, Wall, Ceiling, Speleothems)
    #   1 = Vegetation / Exterior Canopy, Trunks & Shrubs
    #   2 = Exterior Terrain / Unenclosed Ground & Slope outside Drip Line
    sp_pred_class = np.zeros(num_p1, dtype=np.int32)
    is_veg_sp = (validated_canopy_mask | trunk_absorbed) & (~speleothem_protected)
    is_ext_ground_sp = exterior_ground & (~is_veg_sp) & (~speleothem_protected)
    sp_pred_class[is_veg_sp] = 1
    sp_pred_class[is_ext_ground_sp] = 2

    # Structural Cave Zone Classification (Floor vs. Wall vs. Ceiling):
    #   0 = Cave Floor     (upward-facing nz > 0.35, or nz > 0.12 with ceiling above)
    #   1 = Cave Wall      (vertical enclosed rock surface |nz| <= 0.35)
    #   2 = Cave Ceiling   (downward-facing nz < -0.25, or nz < -0.10 with floor below)
    #   3 = Vegetation     (Class 1)
    #   4 = ExteriorGround (Class 2)
    sp_cave_zone = np.ones(num_p1, dtype=np.int32)  # Default Cavern zone = 1 (Wall)
    is_floor_sp = (sp_pred_class == 0) & (
        (sp_nz > 0.35) | ((sp_nz > 0.12) & (sp_roof_occ > sp_down_occ + 0.15))
    )
    is_ceiling_sp = (sp_pred_class == 0) & (~is_floor_sp) & (
        (sp_nz < -0.25) | ((sp_nz < -0.10) & (sp_down_occ > sp_roof_occ + 0.15))
    )
    sp_cave_zone[is_floor_sp] = 0
    sp_cave_zone[is_ceiling_sp] = 2
    sp_cave_zone[sp_pred_class == 1] = 3
    sp_cave_zone[sp_pred_class == 2] = 4

    # Phase 1 Confidence-Masked Pseudo-Labels for Zero-Ground-Truth Self-Training (Part 2.1):
    #   0 = High-confidence Cavern seed
    #   1 = High-confidence Vegetation seed
    #   2 = High-confidence Exterior Ground seed
    # -100 = Ambiguous drip-line transition zone (ignore_index in Cross-Entropy / Lovasz)
    sp_pseudo_label = np.full(num_p1, -100, dtype=np.int32)
    if len(canopy_seeds) == 0:
        # Pure interior scan: 100% of superpoints are high-confidence Cavern seeds (y = 0)
        sp_pseudo_label[:] = 0
    else:
        hc_cavern = (
            (sp_pred_class == 0)
            & ((sp_roof_occ > 0.45) | (is_ceiling_overhang & (sp_d_drip < 1.5)))
            & (sp_foliage_above < 0.20)
            & (sp_d_drip < 2.0)
        )
        hc_veg = (
            validated_canopy_mask
            & (sp_roof_occ < 0.05)
            & ((sp_scat_fine > 0.25) | (sp_ndisp_fine > 0.38))
        )
        hc_ext_ground = (
            is_ext_ground_sp
            & (sp_roof_occ < 0.05)
            & (sp_nz > 0.50)
            & (sp_d_drip > 5.0)
        )
        sp_pseudo_label[hc_cavern] = 0
        sp_pseudo_label[hc_veg] = 1
        sp_pseudo_label[hc_ext_ground] = 2

    # Assemble 24-dim P1 Superpoint Feature Matrix X_P1 (for Option A GAT/GraphSAGE + MLP Classifier)
    sp_feature_matrix = np.column_stack([
        sp_rug_fine,
        sp_plan_fine,
        sp_scat_fine,
        sp_lin_fine,
        sp_ndisp_fine,
        sp_std_plan_fine,
        sp_std_ndisp_fine,
        sp_rug_coarse,
        sp_plan_coarse,
        sp_scat_coarse,
        sp_ndisp_coarse,
        sp_nz,
        sp_vert,
        sp_up_occ,
        sp_roof_occ,
        sp_down_occ,
        sp_enclosure,
        sp_foliage_above,
        np.tanh(sp_d_drip / 10.0),
        np.tanh(sp_z_rel / 10.0),
        sp_p2_roof_occ,
        sp_p2_foliage_above,
        sp_p2_plan_coarse,
        sp_p2_ndisp_coarse,
    ]).astype(np.float32)

    t5 = time.perf_counter()

    # Map P1 superpoint predictions to P0 voxels
    p0_cavern_score = sp_cavern_score[p1_labels].astype(np.float32)
    p0_pred_class = sp_pred_class[p1_labels].astype(np.int32)
    p0_cave_zone = sp_cave_zone[p1_labels].astype(np.int32)
    p0_d_drip = sp_d_drip[p1_labels].astype(np.float32)
    p0_z_rel = sp_z_rel[p1_labels].astype(np.float32)
    p0_pseudo_label = sp_pseudo_label[p1_labels].astype(np.int32)

    print(
        f"[SPT Pipeline] N_orig={n_orig:,} -> N_P0={num_p0:,} ({voxel_size*100:.1f}cm) "
        f"-> N_P1={num_p1:,} -> N_P2={num_p2:,} (edges={num_edges:,})"
    )
    print(
        f"[Timing] Stage1(Subsample): {t_stage1:.2f}s | Stage2A(WNNC): {t_wnnc-t1:.2f}s | "
        f"Stage2B(MultiScale): {t2-t_wnnc:.2f}s | Stage3(Enclosure): {t3-t2:.2f}s | "
        f"Stage4(l0-CutPursuit): {t4-t3:.2f}s | Stage5(GraphFilter): {t5-t4:.2f}s | "
        f"Total: {t_stage1 + (t5-t1):.2f}s"
    )

    return {
        "centroid_f64": centroid_f64,
        "p0_to_orig": p0_to_orig,
        "orig_to_p0": orig_to_p0,
        "p0_xyz_f64": p0_xyz.astype(np.float64) + centroid_f64,
        "p0_xyz_centered": p0_xyz,
        "p0_normals": p0_normals,
        "verticality": verticality,
        "horizontality": horizontality,
        "signed_horizontality": signed_horizontality,
        "rugosity_fine": fine_feats[:, 0],
        "planarity_fine": fine_feats[:, 1],
        "scattering_fine": fine_feats[:, 2],
        "linearity_fine": fine_feats[:, 3],
        "omnivariance_fine": fine_feats[:, 4],
        "normal_dispersion_fine": fine_feats[:, 5],
        "rugosity_coarse": coarse_feats[:, 0],
        "planarity_coarse": coarse_feats[:, 1],
        "scattering_coarse": coarse_feats[:, 2],
        "normal_dispersion_coarse": coarse_feats[:, 5],
        "upward_occlusion": upward_occlusion,
        "roof_occlusion": roof_occlusion,
        "downward_occlusion": downward_occlusion,
        "cavern_enclosure": cavern_enclosure,
        "foliage_above": foliage_above,
        "drip_line_dist": p0_d_drip,
        "rel_elevation": p0_z_rel,
        "pseudo_label": p0_pseudo_label,
        "superpoint_p1": p1_labels,
        "superpoint_p2": p2_labels,
        "cavern_score": p0_cavern_score,
        "predicted_class": p0_pred_class,
        "cave_zone": p0_cave_zone,
        "num_p1": num_p1,
        "num_p2": num_p2,
        "sp_feature_matrix": sp_feature_matrix,
        "sp_pseudo_label": sp_pseudo_label,
        "sp_pred_class": sp_pred_class,
        "sp_counts": sp_counts,
        "sp_pos": sp_pos,
        "edge_u": edge_u,
        "edge_v": edge_v,
    }


def run_spt_pipeline(
    lib: ctypes.CDLL,
    xyz_f64: np.ndarray,
    normals_in: Optional[np.ndarray],
    voxel_size: float = 0.04,
    r_fine: float = 0.12,
    r_coarse: float = 0.50,
    p1_reg: float = 0.015,
    p1_cutoff: int = 3,
    p2_reg: float = 0.08,
    p2_cutoff: int = 15,
) -> Dict[str, np.ndarray]:
    t0 = time.perf_counter()
    n_orig = len(xyz_f64)

    centroid_f64 = xyz_f64.mean(axis=0, keepdims=True)
    xyz_centered = np.ascontiguousarray(xyz_f64 - centroid_f64, dtype=np.float32)

    p0_to_orig = np.empty(n_orig, dtype=np.int32)
    orig_to_p0 = np.empty(n_orig, dtype=np.int32)

    f32_p = ctypes.POINTER(ctypes.c_float)
    i32_p = ctypes.POINTER(ctypes.c_int32)

    num_p0 = lib.spt_voxel_subsample(
        xyz_centered.ctypes.data_as(f32_p),
        ctypes.c_int32(n_orig),
        ctypes.c_float(voxel_size),
        p0_to_orig.ctypes.data_as(i32_p),
        orig_to_p0.ctypes.data_as(i32_p),
    )
    p0_to_orig = p0_to_orig[:num_p0].copy()
    p0_xyz = np.ascontiguousarray(xyz_centered[p0_to_orig], dtype=np.float32)
    p0_normals_in = (
        np.ascontiguousarray(normals_in[p0_to_orig], dtype=np.float32)
        if normals_in is not None
        else None
    )
    t_stage1 = time.perf_counter() - t0

    return run_spt_pipeline_from_p0(
        lib=lib,
        n_orig=n_orig,
        centroid_f64=centroid_f64,
        p0_xyz=p0_xyz,
        p0_normals_in=p0_normals_in,
        p0_to_orig=p0_to_orig,
        orig_to_p0=orig_to_p0,
        t_stage1=t_stage1,
        voxel_size=voxel_size,
        r_fine=r_fine,
        r_coarse=r_coarse,
        p1_reg=p1_reg,
        p1_cutoff=p1_cutoff,
        p2_reg=p2_reg,
        p2_cutoff=p2_cutoff,
    )


# -----------------------------------------------------------------------------
# 3D CutMix Entrance-Vegetation & Speleothem Stress Test (Best Practice #1)
# -----------------------------------------------------------------------------
def run_cutmix_stress_test(
    lib: ctypes.CDLL,
    xyz_f64: np.ndarray,
    normals_in: Optional[np.ndarray],
    voxel_size: float = 0.04,
) -> None:
    """
    Grafts synthetic 3D trees (cylindrical trunks + high-entropy leaf canopies) and
    exterior terrain outside the entrance of the cave, plus a complex internal speleothem
    cluster deep inside the cave, and evaluates P1 Oracle mIoU and Classification IoU.
    """
    rng = np.random.default_rng(42)
    min_pt = xyz_f64.min(axis=0)
    max_pt = xyz_f64.max(axis=0)

    # Place entrance exterior zone just beyond max Y of the conduit
    ext_y0 = max_pt[1] + 0.5
    ground_z = np.percentile(xyz_f64[:, 2], 35)

    # 1. Synthetic Exterior Ground Slope (Class 2: 15,000 pts)
    n_ground = 15000
    gx = rng.uniform(min_pt[0], max_pt[0], n_ground)
    gy = rng.uniform(ext_y0, ext_y0 + 14.0, n_ground)
    gz = ground_z + 0.08 * (gy - ext_y0) + rng.normal(0.0, 0.025, n_ground)
    ground_xyz = np.column_stack([gx, gy, gz])
    ground_normals = np.column_stack([
        rng.normal(0.0, 0.05, n_ground),
        rng.normal(-0.08, 0.05, n_ground),
        np.ones(n_ground),
    ]).astype(np.float32)
    ground_normals /= np.linalg.norm(ground_normals, axis=1, keepdims=True)

    # 2. Synthetic 3D Trees: Trunks + Fractal/Volumetric Foliage Canopies (Class 1: 25,000 pts)
    tree_xyz_list = []
    tree_norm_list = []
    tree_centers = [
        (min_pt[0] + 4.0, ext_y0 + 2.5),
        (min_pt[0] + 10.0, ext_y0 + 4.0),
        (min_pt[0] + 16.0, ext_y0 + 3.0),
        (min_pt[0] + 7.0, ext_y0 + 8.5),
        (min_pt[0] + 14.0, ext_y0 + 9.0),
    ]
    for tx, ty in tree_centers:
        base_z = ground_z + 0.08 * (ty - ext_y0)
        # Trunk points (vertical cylinder r=0.22m, height=4.5m)
        n_trunk = 1000
        theta = rng.uniform(0, 2 * np.pi, n_trunk)
        tz = rng.uniform(base_z, base_z + 4.5, n_trunk)
        rad = 0.22 + rng.normal(0, 0.015, n_trunk)
        t_pts = np.column_stack([tx + rad * np.cos(theta), ty + rad * np.sin(theta), tz])
        t_nrm = np.column_stack([np.cos(theta), np.sin(theta), rng.normal(0, 0.1, n_trunk)]).astype(np.float32)
        t_nrm /= np.linalg.norm(t_nrm, axis=1, keepdims=True)
        tree_xyz_list.append(t_pts)
        tree_norm_list.append(t_nrm)

        # Canopy leaf clusters (high scattering & chaotic normals, radius ~ 2.8m)
        n_canopy = 4000
        c_pts = []
        for _ in range(8):
            sub_c = np.array([
                tx + rng.uniform(-1.8, 1.8),
                ty + rng.uniform(-1.8, 1.8),
                base_z + rng.uniform(3.5, 7.5),
            ])
            c_pts.append(sub_c + rng.normal(0.0, 0.65, (n_canopy // 8, 3)))
        c_pts = np.vstack(c_pts)
        c_nrm = rng.normal(0.0, 1.0, c_pts.shape).astype(np.float32)
        c_nrm /= np.linalg.norm(c_nrm, axis=1, keepdims=True)
        tree_xyz_list.append(c_pts)
        tree_norm_list.append(c_nrm)

    veg_xyz = np.vstack(tree_xyz_list)
    veg_normals = np.vstack(tree_norm_list)

    # Combine Original Cave (Class 0) + Synthetic Vegetation (Class 1) + Exterior Ground (Class 2)
    comb_xyz = np.vstack([xyz_f64, veg_xyz, ground_xyz])
    comb_gt = np.concatenate([
        np.zeros(len(xyz_f64), dtype=np.int32),
        np.ones(len(veg_xyz), dtype=np.int32),
        np.full(len(ground_xyz), 2, dtype=np.int32),
    ])
    comb_normals = None
    if normals_in is not None:
        comb_normals = np.vstack([normals_in, veg_normals, ground_normals])

    print("\n" + "=" * 78)
    print("[3D CutMix Stress Test] Running on Cave + Grafted Trees + Exterior Ground...")
    res = run_spt_pipeline(lib, comb_xyz, comb_normals, voxel_size=voxel_size)

    p0_gt = comb_gt[res["p0_to_orig"]]
    p0_pred = res["predicted_class"]
    p1_labels = res["superpoint_p1"]
    num_p1 = res["num_p1"]

    # Compute P1 Superpoint Oracle IoU on Binary Cavern vs Non-Cavern
    p0_gt_bin = (p0_gt == 0).astype(np.int32)
    sp_cav_cnt = np.bincount(p1_labels, weights=p0_gt_bin, minlength=num_p1)
    sp_tot_cnt = np.bincount(p1_labels, minlength=num_p1)
    p1_maj_bin = (sp_cav_cnt >= (sp_tot_cnt * 0.5)).astype(np.int32)
    oracle_p0_bin = p1_maj_bin[p1_labels]

    inter_oracle = np.sum((oracle_p0_bin == 1) & (p0_gt_bin == 1))
    union_oracle = np.sum((oracle_p0_bin == 1) | (p0_gt_bin == 1))
    oracle_cavern_iou = 100.0 * inter_oracle / max(union_oracle, 1)

    # Compute Unsupervised Superpoint Graph Filter Binary Metrics (Cavern vs Remove)
    pred_bin = (p0_pred == 0).astype(np.int32)
    tp = int(np.sum((pred_bin == 1) & (p0_gt_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (p0_gt_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (p0_gt_bin == 1)))
    tn = int(np.sum((pred_bin == 0) & (p0_gt_bin == 0)))

    precision = 100.0 * tp / max(tp + fp, 1)
    recall = 100.0 * tp / max(tp + fn, 1)
    cavern_iou = 100.0 * tp / max(tp + fp + fn, 1)
    remove_iou = 100.0 * tn / max(tn + fp + fn, 1)

    print(f"  P1 Superpoint Oracle Cavern IoU : {oracle_cavern_iou:.2f}% (Target >= 96.0%)")
    print(f"  Unsupervised Filter Precision   : {precision:.2f}%")
    print(f"  Unsupervised Filter Recall      : {recall:.2f}%")
    print(f"  Unsupervised Filter Cavern IoU  : {cavern_iou:.2f}%")
    print(f"  Unsupervised Filter Remove IoU  : {remove_iou:.2f}%")
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Superpoint & Multi-Scale Geometric Filtering Pipeline (Option 2)"
    )
    parser.add_argument("input_ply", type=Path, help="Input .ply point cloud path")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Output directory for .ply files"
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.04, help="P0 voxel grid size in meters (default: 0.04)"
    )
    parser.add_argument(
        "--r-fine", type=float, default=0.15, help="Fine neighborhood radius in meters (default: 0.15)"
    )
    parser.add_argument(
        "--r-coarse", type=float, default=0.50, help="Coarse neighborhood radius in meters (default: 0.50)"
    )
    parser.add_argument(
        "--stress-test", action="store_true", help="Run 3D CutMix entrance vegetation stress test"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    so_path = repo_root / "build" / "libspt_kernels.so"
    lib = load_spt_kernels(so_path)

    fmt, num_vertices, props, _ = inspect_ply_header(args.input_ply)
    use_streaming = (fmt == "binary_little_endian") and (num_vertices > 15_000_000) and (not args.stress_test)

    if use_streaming:
        print(
            f"[1/3] Out-of-Core Streaming {args.input_ply} ({num_vertices:,} vertices, "
            f"chunk_size=15,000,000) ..."
        )
        t_s0 = time.perf_counter()
        n_orig, centroid_f64, p0_xyz, p0_normals_in = stream_subsample_ply(
            lib, args.input_ply, voxel_size=args.voxel_size
        )
        t_stage1 = time.perf_counter() - t_s0
        print(
            f"      Streamed {n_orig:,} vertices -> {len(p0_xyz):,} P0 voxels in {t_stage1:.2f}s"
        )
        print(f"[2/3] Running SPT Multi-Scale Feature & Superpoint Pipeline on {args.input_ply.name} ...")
        res = run_spt_pipeline_from_p0(
            lib=lib,
            n_orig=n_orig,
            centroid_f64=centroid_f64,
            p0_xyz=p0_xyz,
            p0_normals_in=p0_normals_in,
            t_stage1=t_stage1,
            voxel_size=args.voxel_size,
            r_fine=args.r_fine,
            r_coarse=args.r_coarse,
        )
    else:
        print(f"[1/3] Reading {args.input_ply} ...")
        xyz_f64, normals_in, extras = read_ply(args.input_ply)
        print(
            f"      Loaded {len(xyz_f64):,} vertices | normals={'present' if normals_in is not None else 'missing'} "
            f"| extra scalars={list(extras.keys())}"
        )

        print(f"[2/3] Running SPT Multi-Scale Feature & Superpoint Pipeline on {args.input_ply.name} ...")
        res = run_spt_pipeline(
            lib,
            xyz_f64,
            normals_in,
            voxel_size=args.voxel_size,
            r_fine=args.r_fine,
            r_coarse=args.r_coarse,
        )

    p0_pred = res["predicted_class"]
    p0_zone = res["cave_zone"]
    n_p0 = len(p0_pred)
    n_cav = int(np.sum(p0_pred == 0))
    n_veg = int(np.sum(p0_pred == 1))
    n_ter = int(np.sum(p0_pred == 2))
    n_floor = int(np.sum(p0_zone == 0))
    n_wall = int(np.sum(p0_zone == 1))
    n_ceil = int(np.sum(p0_zone == 2))
    print(
        f"[Result Summary] P0 Points: {n_p0:,} | "
        f"Cavern(0): {n_cav:,} ({100.0*n_cav/n_p0:.2f}%) | "
        f"Vegetation/Clutter(1): {n_veg:,} ({100.0*n_veg/n_p0:.2f}%) | "
        f"ExteriorGround(2): {n_ter:,} ({100.0*n_ter/n_p0:.2f}%)"
    )
    print(
        f"[Cave Zones]     Floor(0): {n_floor:,} ({100.0*n_floor/n_p0:.1f}%) | "
        f"Wall(1): {n_wall:,} ({100.0*n_wall/n_p0:.1f}%) | "
        f"Ceiling(2): {n_ceil:,} ({100.0*n_ceil/n_p0:.1f}%)"
    )

    # Export Diagnostic Multi-Scalar PLY (P0 resolution) and Cleaned Full-Resolution PLY
    stem = args.input_ply.stem
    diag_path = args.output_dir / f"{stem}_diagnostic.ply"
    clean_path = args.output_dir / f"{stem}_cleaned.ply"

    scalars_out = {
        "scalar_verticality": res["verticality"],
        "scalar_horizontality": res["horizontality"],
        "scalar_signed_horizontality": res["signed_horizontality"],
        "scalar_rugosity_fine": res["rugosity_fine"],
        "scalar_rugosity_coarse": res["rugosity_coarse"],
        "scalar_planarity_fine": res["planarity_fine"],
        "scalar_planarity_coarse": res["planarity_coarse"],
        "scalar_scattering_fine": res["scattering_fine"],
        "scalar_linearity_fine": res["linearity_fine"],
        "scalar_omnivariance_fine": res["omnivariance_fine"],
        "scalar_normal_dispersion": res["normal_dispersion_fine"],
        "scalar_upward_occlusion": res["upward_occlusion"],
        "scalar_downward_occlusion": res["downward_occlusion"],
        "scalar_roof_occlusion": res["roof_occlusion"],
        "scalar_cavern_enclosure": res["cavern_enclosure"],
        "scalar_foliage_above": res["foliage_above"],
        "scalar_drip_line_dist": res["drip_line_dist"],
        "scalar_rel_elevation": res["rel_elevation"],
        "scalar_pseudo_label": res["pseudo_label"],
        "scalar_superpoint_p1": res["superpoint_p1"],
        "scalar_superpoint_p2": res["superpoint_p2"],
        "scalar_cavern_score": res["cavern_score"],
        "scalar_predicted_class": res["predicted_class"],
        "scalar_cave_zone": res["cave_zone"],
    }

    print(f"[3/3] Exporting CloudCompare diagnostic PLY -> {diag_path}")
    write_ply_binary(diag_path, res["p0_xyz_f64"], res["p0_normals"], scalars_out)

    if use_streaming:
        t_e0 = time.perf_counter()
        n_kept = stream_export_cleaned_ply(
            lib=lib,
            input_ply=args.input_ply,
            clean_path=clean_path,
            centroid_f64=res["centroid_f64"],
            p0_xyz_centered=res["p0_xyz_centered"],
            p0_normals=res["p0_normals"],
            p0_pred_class=p0_pred,
            voxel_size=args.voxel_size,
        )
        t_exp = time.perf_counter() - t_e0
        print(
            f"      Stream-exported cleaned full-resolution PLY ({n_kept:,}/{num_vertices:,} pts) "
            f"in {t_exp:.2f}s -> {clean_path}"
        )
    else:
        orig_pred_class = p0_pred[res["orig_to_p0"]]
        orig_keep_mask = orig_pred_class == 0
        clean_xyz = xyz_f64[orig_keep_mask]
        clean_normals = res["p0_normals"][res["orig_to_p0"]][orig_keep_mask]
        print(
            f"      Exporting cleaned full-resolution PLY ({len(clean_xyz):,}/{len(xyz_f64):,} pts) -> {clean_path}"
        )
        write_ply_binary(clean_path, clean_xyz, clean_normals, {})

        if args.stress_test:
            run_cutmix_stress_test(lib, xyz_f64, normals_in, voxel_size=args.voxel_size)


if __name__ == "__main__":
    main()

