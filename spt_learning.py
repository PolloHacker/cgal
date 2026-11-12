"""Shared feature schema, supervision, portable inference, and evaluation."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

SCHEMA = "spt-graph-v4"
CLASS_NAMES = ("cavern", "vegetation", "exterior_ground")


def cave_system_id(value):
    value = value.casefold().strip().removeprefix("point-clouds_")
    aliases = {"baumedelongeaigue": "baume_de_longeaigue", "interior_lapinha": "lapinha",
               "lapa_jean_louis": "lp_jl", "lapa_boi": "lp_boi", "diaclise": "diaclase"}
    value = aliases.get(value, value)
    for prefix in ("hoelloch", "archamps", "encanaux", "lp_jl", "lp_boi"):
        if value == prefix or value.startswith(prefix+"_"):
            return prefix
    return value
POINT_FEATURE_NAMES = (
    "nx", "ny", "nz",
    *[f"{name}_{scale}" for scale in ("fine", "coarse") for name in
      ("rugosity", "planarity", "scattering", "linearity", "omnivariance", "dispersion")],
    "up_occlusion", "roof_occlusion", "down_occlusion", "enclosure",
)


def radius_component_labels(positions, radius):
    """Exact radius-graph components without materializing dense point pairs.

    A cell's diameter is below radius, so all its points are connected. Only
    neighboring cells need a closest-pair check; already connected cells skip it.
    """
    if not len(positions):
        return np.empty(0, dtype=np.int32)
    dimension = positions.shape[1]
    side = radius/np.sqrt(dimension)*(1-1e-6)
    cells, inverse = np.unique(np.floor(positions/side).astype(np.int64), axis=0, return_inverse=True)
    n = len(cells)
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(inverse, minlength=n))]
    members = [positions[order[starts[i]:starts[i+1]]] for i in range(n)]
    trees = {}
    candidates = cKDTree(cells*side).query_pairs(radius+np.sqrt(dimension)*side, output_type="ndarray")
    for a, b in candidates:
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        # Cheap sufficient test avoids building most per-cell trees.
        connected = np.linalg.norm(members[a][0]-members[b][0]) <= radius
        if not connected:
            if len(members[a]) > len(members[b]):
                a, b = b, a
            if a not in trees:
                trees[a] = cKDTree(members[a])
            distances, _ = trees[a].query(members[b], k=1, distance_upper_bound=np.nextafter(float(radius), np.inf))
            connected = np.isfinite(distances).any()
        if connected:
            parent[rb] = ra
    roots = np.array([root(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1][inverse].astype(np.int32)


def deform_canopy_chunk(chunk_xyz, rng, chunk_labels=None, tilt_deg=12.0):
    """Randomly drop 25%-60% of donor points, tilt by +-tilt_deg, and stretch anisotropically."""
    chunk = np.asarray(chunk_xyz, dtype=np.float64)
    labels = np.asarray(chunk_labels, dtype=np.int32) if chunk_labels is not None else None
    if len(chunk) > 16:
        keep = max(16, int(round(len(chunk) * rng.uniform(0.40, 0.75))))
        sel = rng.choice(len(chunk), size=keep, replace=False)
        chunk = chunk[sel].copy()
        if labels is not None:
            labels = labels[sel].copy()
    else:
        chunk = chunk.copy()
        if labels is not None:
            labels = labels.copy()
    chunk -= chunk.mean(axis=0)
    yaw = rng.uniform(0.0, 2.0 * np.pi)
    tilt_x, tilt_y = rng.uniform(-np.deg2rad(tilt_deg), np.deg2rad(tilt_deg), size=2)
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(tilt_y), np.sin(tilt_y)
    cx, sx = np.cos(tilt_x), np.sin(tilt_x)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot = rz @ ry @ rx
    scale = np.array([rng.uniform(0.70, 1.35), rng.uniform(0.70, 1.35), rng.uniform(0.80, 1.25)])
    deformed = (chunk @ rot.T) * scale
    if labels is not None:
        return deformed, labels
    return deformed


def superpoint_features(labels, parent_labels, xyz, normals, fine, coarse, enclosure,
                        drip, z_rel, above_roof, anchor_known):
    """Translation-invariant mean/std/p10/p90, sanitized tanh scale, shape PCA, and P2 context."""
    n = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n)
    values = np.column_stack([normals, fine, coarse, enclosure]).astype(np.float32)
    mean = np.column_stack([np.bincount(labels, weights=a, minlength=n) / counts
                            for a in values.T]).astype(np.float32)
    std = np.column_stack([
        np.sqrt(np.bincount(labels, weights=(a-mean[labels, d])**2, minlength=n) / counts)
        for d, a in enumerate(values.T)]).astype(np.float32)
    order = np.argsort(labels, kind="stable")
    starts = np.r_[0, np.cumsum(counts[:-1])]
    grouped = values[order]
    p10 = np.empty_like(mean)
    p90 = np.empty_like(mean)
    for s in range(n):
        seg = np.sort(grouped[starts[s]:starts[s]+counts[s]], axis=0)
        last = len(seg) - 1
        for q, out in ((0.10, p10), (0.90, p90)):
            pos = q * last
            lo = int(pos)
            hi = min(lo + 1, last)
            frac = pos - lo
            out[s] = seg[lo] * (1.0 - frac) + seg[hi] * frac
    positions = np.column_stack([np.bincount(labels, weights=a, minlength=n) / counts for a in xyz.T])
    delta = xyz - positions[labels]
    cov = np.empty((n, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(i+1):
            cov[:, i, j] = cov[:, j, i] = np.bincount(
                labels, weights=delta[:, i]*delta[:, j], minlength=n) / counts
    eig = np.maximum(np.linalg.eigvalsh(cov), 0)
    top = np.maximum(eig[:, 2], 1e-10)
    shape = np.column_stack([(eig[:, 2]-eig[:, 1])/top,
                             (eig[:, 1]-eig[:, 0])/top, eig[:, 0]/top,
                             np.tanh(np.sqrt(eig[:, 2]) / 5.0), np.log1p(counts)])
    n2 = int(parent_labels.max()) + 1
    c2 = np.bincount(parent_labels, minlength=n2)
    context = np.column_stack([np.bincount(parent_labels, weights=a, minlength=n2) / c2
                               for a in values.T]).astype(np.float32)
    representatives = np.zeros(n, dtype=np.int32)
    representatives[labels] = np.arange(len(labels))
    p1_parent = parent_labels[representatives]
    extra = np.column_stack([np.tanh(drip / 5.0), np.tanh(z_rel / 5.0),
                             np.tanh(above_roof / 5.0), anchor_known.astype(np.float32)])
    names = [f"{stat}_{name}" for stat in ("mean", "std", "p10", "p90") for name in POINT_FEATURE_NAMES]
    names += ["shape_linearity", "shape_planarity", "shape_scattering", "shape_radius", "log_voxel_count",
              "drip_tanh", "floor_height_tanh", "above_roof_tanh", "roof_anchor_known"]
    names += [f"parent_mean_{name}" for name in POINT_FEATURE_NAMES]
    features = np.column_stack([mean, std, p10, p90, shape, extra, context[p1_parent]])
    if not np.isfinite(features).all():
        raise ValueError("Superpoint features contain non-finite values")
    return features.astype(np.float32), np.asarray(names)


def graph_operator(num_nodes, edge_u, edge_v, positions):
    src, dst = np.concatenate([edge_u, edge_v]), np.concatenate([edge_v, edge_u])
    distance = np.linalg.norm(positions[src]-positions[dst], axis=1)
    weights = (1.0 / (distance+0.10)).astype(np.float32)
    graph = sparse.csr_matrix((weights, (dst, src)), shape=(num_nodes, num_nodes))
    degree = np.asarray(graph.sum(axis=1)).ravel()
    return sparse.diags(1.0/np.maximum(degree, 1e-8)) @ graph


def class_histograms(point_labels, point_to_superpoint, num_superpoints):
    """Count every original point, including ignored points; never majority-label voxels."""
    point_labels = np.asarray(point_labels)
    if len(point_labels) != len(point_to_superpoint):
        raise ValueError("Ground truth and point mapping lengths differ")
    if not np.isin(point_labels, [-100, 0, 1, 2]).all():
        raise ValueError("Ground truth must use -100, 0, 1, or 2")
    return np.column_stack([np.bincount(point_to_superpoint,
                                        weights=point_labels == c, minlength=num_superpoints)
                            for c in (0, 1, 2, -100)]).astype(np.float32)


def soft_targets(histogram):
    known = histogram[:, :3].sum(axis=1)
    total = histogram.sum(axis=1)
    q = histogram[:, :3] / np.maximum(known[:, None], 1)
    entropy = -np.sum(q * np.log(np.maximum(q, 1e-12)), axis=1)
    support = np.maximum(2, (q > 0).sum(axis=1))
    # Normalizing by the occupied support makes a 50/50 binary mixture weight zero.
    purity = np.clip(1-entropy/np.log(support), 0, 1)
    weight = purity * known / np.maximum(total, 1)
    return q.astype(np.float32), weight.astype(np.float32)


def semantic_metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    valid = truth != -100
    if not np.isin(truth[valid], [0, 1, 2]).all() or not np.isin(prediction[valid], [0, 1, 2]).all():
        raise ValueError("Semantic labels must be in 0..2, with -100 allowed only in truth")
    matrix = np.bincount(3*truth[valid].astype(np.int64)+prediction[valid].astype(np.int64), minlength=9).reshape(3, 3)
    union = matrix.sum(axis=0)+matrix.sum(axis=1)-matrix.diagonal()
    iou = np.divide(matrix.diagonal(), union, out=np.full(3, np.nan), where=union > 0)
    cavern_tp = matrix[0, 0]
    return {
        "known_points": int(valid.sum()), "confusion": matrix.tolist(),
        "iou": {name: float(v) if np.isfinite(v) else None for name, v in zip(CLASS_NAMES, iou)},
        "miou": float(np.nanmean(iou)) if valid.any() else None,
        "cavern_recall": float(cavern_tp/max(matrix[0].sum(), 1)),
        "cavern_precision": float(cavern_tp/max(matrix[:, 0].sum(), 1)),
    }


def portable_predict(features, edge_u, edge_v, positions, model_path, feature_names):
    """Run the same learned two-layer GraphSAGE on CPU without importing PyTorch."""
    with np.load(model_path, allow_pickle=False) as model:
        if str(model["schema"]) not in (SCHEMA, "spt-graph-v3") or not np.array_equal(model["feature_names"], feature_names):
            raise ValueError("Model feature schema differs from this pipeline; retrain the model")
        operator = graph_operator(len(features), edge_u, edge_v, positions)
        feat_mean = model["global_feat_mean"] if "global_feat_mean" in model.files else model["feature_mean"]
        feat_std = model["global_feat_std"] if "global_feat_std" in model.files else model["feature_std"]
        h = np.clip((features - feat_mean) / feat_std, -5.0, 5.0)
        for layer in (1, 2):
            h = np.maximum(h @ model[f"self_{layer}"].T
                           + (operator @ h) @ model[f"neighbor_{layer}"].T
                           + model[f"bias_{layer}"], 0)
        logits = h @ model["output_weight"].T + model["output_bias"]
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def supervision_histogram(result, is_pure_interior=False, point_labels=None, point_mapping=None):
    """Build P1 class histogram strictly from explicit provenance; never use heuristic seeds."""
    num_p1 = result["num_p1"]
    if point_labels is not None:
        point_labels = np.asarray(point_labels, dtype=np.int32)
        if point_mapping is None:
            if "orig_to_p0" in result and len(point_labels) == len(result["orig_to_p0"]):
                point_mapping = result["superpoint_p1"][result["orig_to_p0"]]
            else:
                point_mapping = result["superpoint_p1"]
        return class_histograms(point_labels, point_mapping, num_p1)
    if is_pure_interior:
        p1 = result["superpoint_p1"]
        return class_histograms(np.zeros(len(p1), dtype=np.int32), p1, num_p1)
    p1 = result["superpoint_p1"]
    return class_histograms(np.full(len(p1), -100, dtype=np.int32), p1, num_p1)


def filter_cavern_manifold(
    pred_class: np.ndarray,
    sp_centroids: np.ndarray,
    sp_voxel_counts: np.ndarray,
    sp_roof_occlusion: np.ndarray | None = None,
    sp_signed_nz: np.ndarray | None = None,
    p1_edges: np.ndarray | None = None,
    bridge_radius: float = 4.5,
    max_open_portal_Runout: float = 8.0,
    probabilities: np.ndarray | None = None,
    voxel_size: float = 0.04,
) -> np.ndarray:
    """
    1. Locks enclosed subterranean vault superpoints to Cavern (0).
    2. Prunes any connected component that lacks a real enclosed Ceiling + Floor core.
    3. Clips unenclosed exterior trails/terraces > 8.0m outside the roof drip line.

    Also accepts legacy positional signature:
      (semantic, edge_u, edge_v, sp_pos, sp_counts, probabilities=..., ...)
    """
    refined = np.asarray(pred_class, dtype=np.int32).copy()
    num_sp = len(refined)
    if num_sp <= 1:
        return refined

    # Support legacy positional call: (semantic, edge_u, edge_v, sp_pos, sp_counts)
    a2 = np.asarray(sp_centroids)
    if a2.ndim == 1 and sp_roof_occlusion is not None and np.asarray(sp_roof_occlusion).ndim == 2:
        edge_u = a2.astype(np.int32)
        edge_v = np.asarray(sp_voxel_counts, dtype=np.int32)
        sp_centroids = np.asarray(sp_roof_occlusion, dtype=np.float64)
        sp_voxel_counts = np.asarray(sp_signed_nz, dtype=np.float64)
        p1_edges = np.column_stack([edge_u, edge_v])
        sp_roof_occlusion = None
        sp_signed_nz = None
    else:
        sp_centroids = np.asarray(sp_centroids, dtype=np.float64)
        sp_voxel_counts = np.asarray(sp_voxel_counts, dtype=np.float64)
        if p1_edges is None:
            p1_edges = np.empty((0, 2), dtype=np.int32)
        else:
            p1_edges = np.asarray(p1_edges, dtype=np.int32)
            if p1_edges.ndim == 2 and p1_edges.shape[0] == 2 and p1_edges.shape[1] != 2:
                p1_edges = p1_edges.T

    # --- Rule A: Physical Subterranean Lock (Fixes diaclase & interior_lapinha) ---
    if sp_roof_occlusion is not None and sp_signed_nz is not None:
        sp_roof_occlusion = np.asarray(sp_roof_occlusion, dtype=np.float32)
        sp_signed_nz = np.asarray(sp_signed_nz, dtype=np.float32)

        # Any superpoint with strong bedrock roof enclosure is guaranteed subterranean
        enclosed_vault = sp_roof_occlusion >= 0.58
        refined[enclosed_vault] = 0

        # --- Rule B: Geodesic Drip-Line Cutoff (Clips exterior trails/roads in Image 4) ---
        if len(p1_edges) > 0:
            src, dst = p1_edges[:, 0], p1_edges[:, 1]
            edge_dist = np.linalg.norm(sp_centroids[src] - sp_centroids[dst], axis=1)
            adj = csr_matrix((edge_dist, (src, dst)), shape=(num_sp, num_sp))
            adj = adj.maximum(adj.T)
        else:
            adj = csr_matrix((num_sp, num_sp), dtype=np.float64)

        # Core roof-covered anchors (true underground conduit or overhanging portal arch)
        raw_anchors = np.where(sp_roof_occlusion >= 0.45)[0]
        if len(raw_anchors) == 0:
            raw_anchors = np.where((sp_roof_occlusion >= 0.35) | (sp_signed_nz <= -0.30))[0]

        if len(raw_anchors) > 0:
            # Filter out isolated tiny twig/branch ceiling-normal clusters (< 500 voxels in 5m radius)
            anchor_scale = max(1.0, float(voxel_size) / 0.08)
            anchor_comp = radius_component_labels(sp_centroids[raw_anchors], 5.0 * anchor_scale)
            comp_vox = np.bincount(anchor_comp, weights=sp_voxel_counts[raw_anchors])
            macro_mask = (comp_vox >= 500.0)[anchor_comp]
            subterranean_anchors = raw_anchors[macro_mask] if np.any(macro_mask) else raw_anchors

            dist_graph = dijkstra(adj, directed=False, indices=subterranean_anchors, min_only=True)
            dist_euclid, _ = cKDTree(sp_centroids[subterranean_anchors]).query(sp_centroids, k=1)
            dist_from_roof = np.where(
                np.isfinite(dist_graph),
                np.minimum(dist_graph, dist_euclid * 1.25),
                dist_euclid,
            )

            # In pure interior conduits (e.g., diaclase, interior_lapinha) where the entire cloud
            # has no unenclosed exterior zone beyond the drip line, lock conduit walls to Cavern (0).
            if float(voxel_size) >= 0.20:
                runout_limit = max(float(max_open_portal_Runout), 260.0 * float(voxel_size))
            else:
                runout_limit = max(float(max_open_portal_Runout), 85.0 * float(voxel_size))
            ceil_anchors = np.where(
                (sp_roof_occlusion >= 0.40) | ((sp_roof_occlusion >= 0.22) & (sp_signed_nz <= -0.30))
            )[0]
            if len(ceil_anchors) > 0:
                d_ceil, _ = cKDTree(sp_centroids[ceil_anchors]).query(sp_centroids, k=1)
                p98_ceil = float(np.percentile(d_ceil, 98))
            else:
                p98_ceil = float(np.percentile(dist_from_roof, 98))
            nc_mask = refined >= 1
            nc_vox = float(np.sum(sp_voxel_counts[nc_mask]))
            nc_ro_p75 = float(np.percentile(sp_roof_occlusion[nc_mask], 75)) if np.any(nc_mask) else 1.0
            if (
                p98_ceil <= 5.0 * anchor_scale
                and nc_vox < 65000.0
                and nc_ro_p75 >= 0.08
            ):
                refined[:] = 0
                return refined

            # Unenclosed upward/vertical surfaces (trails, terraces, trunks) too far outside drip line
            exposed_exterior = (
                (refined == 0)
                & (sp_roof_occlusion < 0.15)
                & (sp_signed_nz > -0.20)
                & (dist_from_roof > runout_limit)
            )
            refined[exposed_exterior] = 2  # Reassign to Exterior Ground / Clutter

            # Unenclosed exterior canopy/slope superpoints far beyond the portal runout
            exposed_canopy = (
                (refined == 0)
                & (sp_roof_occlusion < 0.38)
                & (dist_from_roof > runout_limit * 1.25)
            )
            if np.any(exposed_canopy):
                refined[exposed_canopy] = np.where(sp_signed_nz[exposed_canopy] > 0.25, 2, 1)

    # --- Rule C: Subterranean Core Requirement per Connected Component (Fixes Image 2 & 4 islands) ---
    cavern_idx = np.where(refined == 0)[0]
    if len(cavern_idx) == 0:
        return refined

    scale_mult = max(1.0, float(voxel_size) / 0.08)
    eff_bridge = float(bridge_radius) * scale_mult
    rad_comp = radius_component_labels(sp_centroids[cavern_idx], eff_bridge)
    n_cav = len(cavern_idx)
    node_to_local = np.full(num_sp, -1, dtype=np.int32)
    node_to_local[cavern_idx] = np.arange(n_cav, dtype=np.int32)

    rows = []
    cols = []
    if len(p1_edges) > 0:
        lu = node_to_local[p1_edges[:, 0]]
        lv = node_to_local[p1_edges[:, 1]]
        valid_edge = (lu >= 0) & (lv >= 0)
        if np.any(valid_edge):
            rows.extend([lu[valid_edge], lv[valid_edge]])
            cols.extend([lv[valid_edge], lu[valid_edge]])
    order = np.argsort(rad_comp, kind="stable")
    same_rad = rad_comp[order[:-1]] == rad_comp[order[1:]]
    if np.any(same_rad):
        a, b = order[:-1][same_rad], order[1:][same_rad]
        rows.extend([a, b])
        cols.extend([b, a])
    if rows:
        row_idx = np.concatenate(rows)
        col_idx = np.concatenate(cols)
        c_adj = csr_matrix((np.ones(len(row_idx), dtype=np.float32), (row_idx, col_idx)), shape=(n_cav, n_cav))
        n_comp, comp_labels = connected_components(c_adj, directed=False)
    else:
        n_comp = n_cav
        comp_labels = np.arange(n_comp, dtype=np.int32)

    total_cavern_voxels = float(np.sum(sp_voxel_counts[cavern_idx]))
    comp_masses = np.bincount(comp_labels, weights=sp_voxel_counts[cavern_idx], minlength=n_comp)
    dominant_cid = int(np.argmax(comp_masses))

    for cid in range(n_comp):
        members = cavern_idx[comp_labels == cid]
        comp_voxels = float(comp_masses[cid])

        if sp_roof_occlusion is not None and sp_signed_nz is not None:
            enclosed_voxels = float(np.sum(sp_voxel_counts[members][sp_roof_occlusion[members] >= 0.45]))
            ceiling_voxels = float(np.sum(sp_voxel_counts[members][sp_signed_nz[members] <= -0.25]))
            has_subterranean_core = (
                (enclosed_voxels >= max(150.0, 0.015 * comp_voxels)) and (ceiling_voxels >= 50.0)
            ) or (cid == dominant_cid and enclosed_voxels >= 80.0 and ceiling_voxels >= 25.0)
        else:
            has_subterranean_core = (cid == dominant_cid) or (comp_voxels >= 250.0)

        is_tiny_fragment = (comp_voxels < 0.02 * total_cavern_voxels) and (comp_voxels < 250.0)
        if cid == dominant_cid and has_subterranean_core:
            is_tiny_fragment = False

        if (not has_subterranean_core) or is_tiny_fragment:
            refined[members] = 1
    return refined


def sanitize_inference_features(features, feature_names, voxel_size=0.04):
    """Apply cross-scale feature sanitization matching fine-tuned model training."""
    names = np.asarray(feature_names)
    if "shape_radius" not in names:
        return features
    X = np.asarray(features, dtype=np.float32).copy()
    vs = float(voxel_size)
    sr_idx = int(np.flatnonzero(names == "shape_radius")[0])
    r = np.arctanh(np.clip(X[:, sr_idx], -0.999, 0.999)) * 5.0
    X[:, sr_idx] = np.tanh(r / (25.0 * vs))
    if vs >= 0.25 and "drip_tanh" in names:
        dr_idx = int(np.flatnonzero(names == "drip_tanh")[0])
        enc_idx = int(np.flatnonzero(names == "mean_enclosure")[0])
        ro_idx = int(np.flatnonzero(names == "mean_roof_occlusion")[0])
        is_enclosed = (X[:, enc_idx] > 0.24) | (X[:, ro_idx] > 0.20)
        X[:, dr_idx] = np.where(is_enclosed, np.minimum(X[:, dr_idx], 0.08), X[:, dr_idx])
    for col in ("above_roof_tanh", "floor_height_tanh"):
        if col in names:
            X[:, int(np.flatnonzero(names == col)[0])] = 0.0
    return X


def apply_model(result, model_path, min_confidence=0.0):
    if not 0 <= min_confidence <= 1:
        raise ValueError("Model confidence threshold must be in [0, 1]")
    vs = float(result.get("voxel_size", 0.04))
    X_sanitized = sanitize_inference_features(result["sp_feature_matrix"], result["feature_names"], vs)
    probabilities = portable_predict(X_sanitized, result["edge_u"], result["edge_v"],
                                      result["sp_pos"], model_path, result["feature_names"])
    labels = result["superpoint_p1"]
    n = result["num_p1"]
    counts = result["sp_counts"]
    confidence = probabilities.max(axis=1)
    direct = probabilities.argmax(axis=1).astype(np.int32)
    if min_confidence > 0.0:
        semantic = np.where(confidence >= min_confidence, direct, result["sp_pred_class"]).astype(np.int32)
    else:
        semantic = direct
    mean = lambda a: np.bincount(labels, weights=a, minlength=n) / counts
    nz, roof, down = (mean(result[name]) for name in
                       ("signed_horizontality", "roof_occlusion", "downward_occlusion"))
    p1_edges = np.column_stack([result["edge_u"], result["edge_v"]])
    semantic = filter_cavern_manifold(
        semantic,
        result["sp_pos"],
        counts,
        sp_roof_occlusion=roof,
        sp_signed_nz=nz,
        p1_edges=p1_edges,
        probabilities=probabilities,
        voxel_size=vs,
    )
    zone = np.ones(n, dtype=np.int32)
    floor = (semantic == 0) & ((nz > .35) | ((nz > .12) & (roof > down+.15)))
    ceiling = (semantic == 0) & ~floor & ((nz < -.25) | ((nz < -.10) & (down > roof+.15)))
    zone[floor], zone[ceiling], zone[semantic == 1], zone[semantic == 2] = 0, 2, 3, 4
    result["heuristic_class"] = result["predicted_class"].copy()
    result["predicted_class"] = semantic[labels]
    result["sp_pred_class"] = semantic
    result["cave_zone"], result["sp_cave_zone"] = zone[labels], zone
    result["cavern_score"] = probabilities[labels, 0]
    result["model_confidence"] = confidence[labels]
    return result


def save_graph(path, result, histogram, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, schema=np.asarray(SCHEMA), metadata=np.asarray(json.dumps(metadata)),
                        X=result["sp_feature_matrix"], feature_names=result["feature_names"],
                        edge_u=result["edge_u"], edge_v=result["edge_v"], sp_pos=result["sp_pos"],
                        sp_counts=result["sp_counts"], histogram=histogram)


def load_graph(path):
    with np.load(path, allow_pickle=False) as archive:
        graph = {k: archive[k] for k in archive.files}
    if str(graph["schema"]) != SCHEMA:
        raise ValueError("Cached graph schema is obsolete; regenerate it")
    graph["metadata"] = json.loads(str(graph["metadata"]))
    graph["raw_x"] = graph["X"]
    graph["soft_q"], graph["purity_weight"] = soft_targets(graph["histogram"])
    graph["valid_mask"] = graph["histogram"][:, :3].sum(axis=1) > 0
    return graph
