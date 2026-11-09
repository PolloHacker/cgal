"""Native boundary graph and genuinely nested upstream l0 cut-pursuit partitions."""
import ctypes
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@lru_cache(maxsize=1)
def load_cut_pursuit():
    build = Path(__file__).resolve().parent / "build"
    wrapper = build / "cp_d0_dist.py"
    if not wrapper.exists():
        raise RuntimeError("Build the cut-pursuit solver with: "
                           ".venv/bin/python scripts/build_spt_cut_pursuit.py")
    sys.path.insert(0, str(build))
    spec = importlib.util.spec_from_file_location("spt_upstream_cp", wrapper)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise RuntimeError("Rebuild cut-pursuit for this Python/NumPy environment: "
                           "python scripts/build_spt_cut_pursuit.py") from exc
    # Refuse older 16 bit builds before passing a million-point graph to native code.
    labels, _ = module.cp_d0_dist(
        1, np.zeros((1, 1), dtype=np.float32, order="F"),
        np.zeros(2, dtype=np.uint32), np.empty(0, dtype=np.uint32),
        verbose=False, max_num_threads=1)
    if labels.dtype.itemsize < 4:
        raise RuntimeError("Cut-pursuit needs a 32 bit component-ID build")
    return module.cp_d0_dist


def build_partition_graph(lib, xyz, normals, dispersion, voxel_size):
    """Undirected union of radius-pruned k=10 edges; only isolated rows get 1-NN."""
    n = len(xyz)
    neighbors = np.full((n, 10), -1, dtype=np.int32)
    weights = np.zeros((n, 10), dtype=np.float32)
    f32 = ctypes.POINTER(ctypes.c_float)
    i32 = ctypes.POINTER(ctypes.c_int32)
    lib.spt_build_partition_graph(
        xyz.ctypes.data_as(f32), normals.ctypes.data_as(f32),
        np.ascontiguousarray(dispersion).ctypes.data_as(f32), n, 3.5 * voxel_size,
        neighbors.ctypes.data_as(i32), weights.ctypes.data_as(f32))
    isolated = np.flatnonzero(neighbors[:, 0] < 0)
    if len(isolated) and n > 1:
        distances, indices = cKDTree(xyz).query(xyz[isolated], k=2, workers=-1)
        # P0 voxel representatives are distinct, so the first neighbor is self.
        j = indices[:, 1]
        neighbors[isolated, 0] = j
        alignment = np.clip(np.abs(np.sum(normals[isolated] * normals[j], axis=1)), 0, 1)
        jump = dispersion[isolated] - dispersion[j]
        weights[isolated, 0] = (np.exp(-(1-alignment)/0.35**2 - jump**2/0.20**2)
                                * voxel_size / (distances[:, 1] + voxel_size*0.25))
    valid = neighbors >= 0
    rows = np.repeat(np.arange(n, dtype=np.int32), valid.sum(axis=1))
    graph = sparse.csr_matrix((weights[valid], (rows, neighbors[valid])), shape=(n, n))
    # A k-NN edge may exist in only one direction. Count its cut penalty once.
    graph = sparse.triu(graph.maximum(graph.T), k=1, format="csr")
    graph.eliminate_zeros()
    return graph


def cut_partition(features, xyz, graph, regularization, cutoff, spatial_weight, counts=None):
    if len(xyz) <= 1:
        return np.zeros(len(xyz), dtype=np.int32)
    if graph.nnz == 0:
        return np.arange(len(xyz), dtype=np.int32)
    values = np.asfortranarray(np.column_stack([xyz-xyz.mean(axis=0), features]).T, dtype=np.float32)
    coordinate_weights = np.ones(values.shape[0], dtype=np.float32)
    coordinate_weights[:3] = spatial_weight
    labels, _ = load_cut_pursuit()(
        values.shape[0], values, graph.indptr.astype(np.uint32), graph.indices.astype(np.uint32),
        edge_weights=np.ascontiguousarray(graph.data * regularization, dtype=np.float32),
        vert_weights=None if counts is None else counts.astype(np.float32),
        coor_weights=coordinate_weights, min_comp_weight=float(cutoff),
        cp_it_max=10, cp_dif_tol=1e-3, split_damp_ratio=0.7,
        max_num_threads=int(os.environ.get("OMP_NUM_THREADS", min(os.cpu_count() or 1, 8))),
        verbose=False, balance_parallel_split=False)
    # Equal minimizer values can share an upstream component ID even across
    # disconnected graph regions. Splitting those ties changes no energy and
    # ensures superpoints cannot silently bridge the gaps we just pruned.
    edges = graph.tocoo()
    same = labels[edges.row] == labels[edges.col]
    interior = sparse.csr_matrix((np.ones(int(same.sum()), dtype=np.int8),
                                  (edges.row[same], edges.col[same])), shape=graph.shape)
    return connected_components(interior, directed=False)[1].astype(np.int32)


def partition_hierarchy(lib, xyz, normals, dispersion, features, voxel_size,
                        p1_reg=0.015, p1_cutoff=3, p2_reg=0.08, p2_cutoff=15):
    graph = build_partition_graph(lib, xyz, normals, dispersion, voxel_size)
    p1 = cut_partition(features, xyz, graph, p1_reg, p1_cutoff, 0.02)
    n1 = int(p1.max()) + 1
    counts = np.bincount(p1, minlength=n1).astype(np.float32)

    def mean(a):
        return np.column_stack([np.bincount(p1, weights=a[:, d], minlength=n1) / counts
                                for d in range(a.shape[1])]).astype(np.float32)

    coo = graph.tocoo()
    u, v = p1[coo.row], p1[coo.col]
    boundary = u != v
    small, large = np.minimum(u[boundary], v[boundary]), np.maximum(u[boundary], v[boundary])
    reduced = sparse.csr_matrix((coo.data[boundary], (small, large)), shape=(n1, n1))
    reduced.sum_duplicates()
    parent = cut_partition(mean(features), mean(xyz), reduced,
                           p2_reg, p2_cutoff, 0.008, counts)
    edges = reduced.tocoo()
    return p1, parent[p1], parent, edges.row.astype(np.int32), edges.col.astype(np.int32)
