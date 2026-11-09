"""Shared feature schema, supervision, portable inference, and evaluation."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

SCHEMA = "spt-graph-v3"
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


def deform_canopy_chunk(chunk_xyz, rng):
    """Randomly drop 25%-60% of tree points, tilt by +-12 deg, and stretch anisotropically."""
    chunk = np.asarray(chunk_xyz, dtype=np.float64)
    if len(chunk) > 16:
        keep = max(16, int(round(len(chunk) * rng.uniform(0.40, 0.75))))
        chunk = chunk[rng.choice(len(chunk), size=keep, replace=False)].copy()
    else:
        chunk = chunk.copy()
    chunk -= chunk.mean(axis=0)
    yaw = rng.uniform(0.0, 2.0 * np.pi)
    tilt_x, tilt_y = rng.uniform(-np.deg2rad(12.0), np.deg2rad(12.0), size=2)
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(tilt_y), np.sin(tilt_y)
    cx, sx = np.cos(tilt_x), np.sin(tilt_x)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot = rz @ ry @ rx
    scale = np.array([rng.uniform(0.70, 1.35), rng.uniform(0.70, 1.35), rng.uniform(0.80, 1.25)])
    return (chunk @ rot.T) * scale


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
        if str(model["schema"]) != SCHEMA or not np.array_equal(model["feature_names"], feature_names):
            raise ValueError("Model feature schema differs from this pipeline; retrain the model")
        operator = graph_operator(len(features), edge_u, edge_v, positions)
        h = np.clip((features-model["feature_mean"]) / model["feature_std"], -5.0, 5.0)
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


def filter_cavern_manifold(semantic, edge_u, edge_v, sp_pos, sp_counts, probabilities=None, bridge_radius=4.5):
    """Keep the main connected 3D cavern manifold and reclassify tiny detached canopy islands."""
    semantic = np.asarray(semantic, dtype=np.int32).copy()
    cav_nodes = np.flatnonzero(semantic == 0)
    if len(cav_nodes) <= 1:
        return semantic
    rad_comp = radius_component_labels(sp_pos[cav_nodes], bridge_radius)
    n_cav = len(cav_nodes)
    node_to_local = np.full(len(semantic), -1, dtype=np.int32)
    node_to_local[cav_nodes] = np.arange(n_cav, dtype=np.int32)
    lu, lv = node_to_local[edge_u], node_to_local[edge_v]
    valid_edge = (lu >= 0) & (lv >= 0)
    rows = [lu[valid_edge], lv[valid_edge]]
    cols = [lv[valid_edge], lu[valid_edge]]
    order = np.argsort(rad_comp, kind="stable")
    same_rad = rad_comp[order[:-1]] == rad_comp[order[1:]]
    if np.any(same_rad):
        a, b = order[:-1][same_rad], order[1:][same_rad]
        rows.extend([a, b])
        cols.extend([b, a])
    row_idx = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    col_idx = np.concatenate(cols) if cols else np.empty(0, dtype=np.int32)
    adj = sparse.csr_matrix((np.ones(len(row_idx), dtype=np.float32), (row_idx, col_idx)), shape=(n_cav, n_cav))
    n_comp, comp = connected_components(adj, directed=False)
    if n_comp <= 1:
        return semantic
    comp_mass = np.bincount(comp, weights=sp_counts[cav_nodes], minlength=n_comp)
    main_mass = float(comp_mass.max())
    keep_comp = (comp_mass >= 0.02 * main_mass) | (comp_mass >= 250)
    keep_comp[int(comp_mass.argmax())] = True
    isolated_local = np.flatnonzero(~keep_comp[comp])
    if len(isolated_local):
        isolated_nodes = cav_nodes[isolated_local]
        if probabilities is not None:
            semantic[isolated_nodes] = probabilities[isolated_nodes, 1:].argmax(axis=1).astype(np.int32) + 1
        else:
            semantic[isolated_nodes] = 1
    return semantic


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
    semantic = filter_cavern_manifold(semantic, result["edge_u"], result["edge_v"],
                                      result["sp_pos"], counts, probabilities=probabilities)
    mean = lambda a: np.bincount(labels, weights=a, minlength=n) / counts
    nz, roof, down = (mean(result[name]) for name in
                       ("signed_horizontality", "roof_occlusion", "downward_occlusion"))
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
    graph["soft_q"], graph["purity_weight"] = soft_targets(graph["histogram"])
    graph["valid_mask"] = graph["histogram"][:, :3].sum(axis=1) > 0
    return graph
