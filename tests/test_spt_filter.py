"""Boundary, masking, hierarchy, streaming, and learned-inference regressions."""
import ctypes
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from spt_filter import (load_spt_kernels, read_ply, run_spt_pipeline, stream_export_cleaned_ply,
                        stream_subsample_ply, write_ply_binary)
from spt_learning import (class_histograms, deform_canopy_chunk, filter_cavern_manifold,
                          portable_predict, radius_component_labels, semantic_metrics,
                          soft_targets, superpoint_features, supervision_histogram)
from spt_partition import build_partition_graph, partition_hierarchy
from train_spt_classifier import extract_pseudo_labeled_graph, validate_splits

ROOT = Path(__file__).resolve().parents[1]


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = load_spt_kernels(ROOT/"build/libspt_kernels.so")

    def test_dense_neighborhood_uses_actual_nearest_ten(self):
        xyz = np.random.default_rng(2).uniform(0, .08, (100, 3)).astype(np.float32)
        normals = np.tile([0, 0, 1], (len(xyz), 1)).astype(np.float32)
        dispersion = np.zeros(len(xyz), dtype=np.float32)
        neighbors = np.empty((len(xyz), 10), dtype=np.int32)
        weights = np.empty_like(neighbors, dtype=np.float32)
        f = ctypes.POINTER(ctypes.c_float)
        i = ctypes.POINTER(ctypes.c_int32)
        self.lib.spt_build_partition_graph(xyz.ctypes.data_as(f), normals.ctypes.data_as(f),
                                          dispersion.ctypes.data_as(f), len(xyz), .14,
                                          neighbors.ctypes.data_as(i), weights.ctypes.data_as(f))
        expected = cKDTree(xyz).query(xyz, k=11)[1][:, 1:]
        np.testing.assert_array_equal(neighbors, expected)

    def test_air_gap_and_isolated_neighbor_exception(self):
        xyz = np.zeros((13, 3), dtype=np.float32)
        xyz[:6, 0] = np.arange(6)*.01
        xyz[6:12, 0] = 2+np.arange(6)*.01
        xyz[12, 0] = 4
        normals = np.tile([0, 0, 1], (len(xyz), 1)).astype(np.float32)
        graph = build_partition_graph(self.lib, xyz, normals, np.zeros(13, dtype=np.float32), .04).tocoo()
        long = np.linalg.norm(xyz[graph.row]-xyz[graph.col], axis=1) > .14
        self.assertTrue(np.any(long))
        self.assertTrue(np.all((graph.row[long] == 12) | (graph.col[long] == 12)))
        self.assertFalse(np.any((graph.row < 6) & ((graph.col >= 6) & (graph.col < 12))))

    def test_normal_contrast_preserves_weak_boundary_affinity(self):
        xyz = np.array([[0, 0, 0], [.04, 0, 0], [.08, 0, 0]], dtype=np.float32)
        normals = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
        graph = build_partition_graph(self.lib, xyz, normals, np.zeros(3, dtype=np.float32), .04)
        self.assertLess(graph[0, 2], graph[0, 1]*.001)

    def test_partition_is_nested_and_respects_large_feature_jump(self):
        xyz = np.zeros((40, 3), dtype=np.float32)
        xyz[:, 0] = np.arange(40)*.02
        normals = np.tile([0, 0, 1], (40, 1)).astype(np.float32)
        features = np.zeros((40, 2), dtype=np.float32)
        features[20:] = 20
        p1, p2, parent, _, _ = partition_hierarchy(self.lib, xyz, normals, np.zeros(40, dtype=np.float32), features, .04)
        np.testing.assert_array_equal(p2, parent[p1])
        self.assertFalse(set(p1[:20]) & set(p1[20:]))

    def test_disconnected_identical_surfaces_keep_distinct_components(self):
        xyz = np.zeros((20, 3), dtype=np.float32)
        xyz[:10, 0] = np.arange(10)*.01
        xyz[10:, 0] = 5+np.arange(10)*.01
        normals = np.tile([0, 0, 1], (20, 1)).astype(np.float32)
        p1, p2, _, _, _ = partition_hierarchy(self.lib, xyz, normals,
                                              np.zeros(20, dtype=np.float32), np.zeros((20, 2), dtype=np.float32), .04)
        self.assertFalse(set(p1[:10]) & set(p1[10:]))
        self.assertFalse(set(p2[:10]) & set(p2[10:]))

    def test_streaming_grid_is_global_and_full_resolution_mapping_is_exact(self):
        rng = np.random.default_rng(3)
        xyz = rng.uniform(0, 2, (200, 3))+[2_530_000, 1_190_000, 0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"input.ply"
            write_ply_binary(path, xyz, None, {})
            a = stream_subsample_ply(self.lib, path, .2, chunk_size=13)
            b = stream_subsample_ply(self.lib, path, .2, chunk_size=71)
            np.testing.assert_allclose(a[1], xyz.mean(axis=0, keepdims=True), rtol=0, atol=1e-8)
            np.testing.assert_allclose(a[1], b[1], rtol=0, atol=1e-8)
            cells_a, cells_b = np.floor(a[2]/.2).astype(int), np.floor(b[2]/.2).astype(int)
            self.assertEqual(set(map(tuple, cells_a)), set(map(tuple, cells_b)))
            semantic = (cells_a[:, 0] > 0).astype(np.int32)
            normals = np.tile([0, 0, 1], (len(a[2]), 1)).astype(np.float32)
            kept = stream_export_cleaned_ply(self.lib, path, Path(directory)/"clean.ply", a[1], a[2], normals,
                                             semantic, .2, chunk_size=19)
            original_cells = np.floor((xyz-a[1]).astype(np.float32)/.2).astype(int)
            self.assertEqual(kept, int(np.sum(original_cells[:, 0] <= 0)))
            cleaned, clean_normals, _ = read_ply(Path(directory)/"clean.ply")
            self.assertEqual(len(cleaned), kept)
            np.testing.assert_allclose(clean_normals[:, 2], 1)

    def test_open_plane_is_not_implicitly_pure_interior(self):
        x, y = np.meshgrid(np.linspace(0, 2, 20), np.linspace(0, 2, 20))
        xyz = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
        result = run_spt_pipeline(self.lib, xyz, None, voxel_size=.08)
        self.assertTrue(np.all(result["sp_pseudo_label"] == -100))
        np.testing.assert_array_equal(result["superpoint_p2"], result["p1_to_p2"][result["superpoint_p1"]])


class SupervisionTests(unittest.TestCase):
    def test_histogram_preserves_mixed_voxel_sources_and_unknown_mass(self):
        histogram = class_histograms(np.array([0, 0, 1, 1, 2, -100]), np.array([0, 0, 0, 0, 1, 1]), 2)
        np.testing.assert_array_equal(histogram, [[2, 2, 0, 0], [0, 0, 1, 1]])
        targets, purity = soft_targets(histogram)
        np.testing.assert_allclose(targets[0], [.5, .5, 0])
        np.testing.assert_allclose(purity, [0, .5], atol=1e-7)

    def test_radius_components_match_brute_force_in_2d_and_3d(self):
        for dimension in (2, 3):
            points = np.random.default_rng(dimension).uniform(-5, 5, (500, dimension))
            pairs = cKDTree(points).query_pairs(.8, output_type="ndarray")
            graph = sparse.csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
            _, expected = connected_components(graph, directed=False)
            actual = radius_component_labels(points, .8)
            np.testing.assert_array_equal(actual[:, None] == actual, expected[:, None] == expected)

    def test_canopy_source_cannot_leak_into_held_out_cave(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_splits({"training": [{"path": "rupt.ply", "cave_system": "rupt", "role": "interior"}],
                             "canopy_chunks": [{"path": "canopy.ply", "cave_system": "Lp_JL"}],
                             "validation": [{"path": "crop.ply", "cave_system": "lp_jl"}]})

    def test_semantic_metrics_ignore_unknown_and_use_three_class_schema(self):
        metrics = semantic_metrics(np.array([0, 0, 1, 2, -100]), np.array([0, 1, 1, 2, 0]))
        self.assertEqual(metrics["known_points"], 4)
        self.assertEqual(metrics["cavern_recall"], .5)
        with self.assertRaises(ValueError):
            semantic_metrics(np.array([4]), np.array([0]))

    def test_ignored_and_fully_impure_nodes_have_zero_direct_loss_gradient(self):
        import torch
        from spt_torch import masked_purity_loss
        logits = torch.zeros((3, 3), requires_grad=True)
        q = torch.tensor([[1., 0, 0], [0, 0, 0], [.5, .5, 0]])
        loss = masked_purity_loss(logits, q, torch.tensor([1., 0, 0]), torch.ones(3))
        loss.backward()
        self.assertGreater(float(logits.grad[0].abs().sum()), 0)
        self.assertEqual(float(logits.grad[1:].abs().sum()), 0)

    def test_partially_unknown_node_does_not_receive_hard_lovasz_target(self):
        import torch
        from spt_torch import masked_purity_loss
        logits = torch.zeros((1, 3), requires_grad=True)
        arguments = (logits, torch.tensor([[1., 0, 0]]), torch.tensor([.1]), torch.ones(1))
        self.assertEqual(float(masked_purity_loss(*arguments, lovasz_weight=1).detach()),
                         float(masked_purity_loss(*arguments, lovasz_weight=0).detach()))

    def test_manual_annotation_merge_rejects_conflicting_duplicates(self):
        from prepare_spt_annotations import combine_annotations
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"segment.ply"
            write_ply_binary(p, np.array([[1., 2., 3.]]), None, {})
            with self.assertRaisesRegex(ValueError, "different semantic classes"):
                combine_annotations({0: [p], 1: [p]})
            xyz, truth = combine_annotations({0: [p], -100: [p]})
            np.testing.assert_array_equal(truth, [0])
            self.assertEqual(len(xyz), 1)

    def test_torch_and_portable_graphsage_predictions_match(self):
        import torch
        from spt_torch import SuperpointGraphSAGE, torch_graph
        graph = {"X": np.random.default_rng(1).normal(size=(4, 5)).astype(np.float32),
                 "edge_u": np.array([0, 1, 2], dtype=np.int32), "edge_v": np.array([1, 2, 3], dtype=np.int32),
                 "sp_pos": np.arange(12, dtype=np.float32).reshape(4, 3)}
        names = np.array([f"f{i}" for i in range(5)])
        model = SuperpointGraphSAGE(np.zeros(5), np.ones(5), hidden_dim=8).eval()
        with torch.no_grad():
            expected = model(torch.from_numpy(graph["X"]), torch_graph(graph, "cpu")).softmax(dim=1).numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.npz"
            model.export(path, names, {"training_groups": []})
            actual = portable_predict(graph["X"], graph["edge_u"], graph["edge_v"], graph["sp_pos"], path, names)
            np.testing.assert_allclose(actual, expected, atol=2e-6)
            with self.assertRaisesRegex(ValueError, "schema"):
                portable_predict(graph["X"], graph["edge_u"], graph["edge_v"], graph["sp_pos"], path, names[::-1])

    def test_heuristic_seeds_are_disabled_and_entrance_requires_ground_truth(self):
        result = {"num_p1": 2, "superpoint_p1": np.array([0, 0, 1, 1], dtype=np.int32)}
        hist = supervision_histogram(result, is_pure_interior=False)
        np.testing.assert_array_equal(hist, [[0, 0, 0, 2], [0, 0, 0, 2]])
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"unannotated_entrance.ply"
            write_ply_binary(p, np.zeros((20, 3)), None, {})
            with self.assertRaisesRegex(ValueError, "Heuristic seed generation is disabled"):
                extract_pseudo_labeled_graph(GeometryTests.lib, p, voxel_size=0.04, is_pure_interior=False)

    def test_noise_robust_kl_clamps_outlier_superpoints(self):
        import torch
        from spt_torch import masked_purity_loss
        logits = torch.tensor([[100.0, -100.0, -100.0]], requires_grad=True)
        q = torch.tensor([[0.0, 1.0, 0.0]])
        loss = masked_purity_loss(logits, q, torch.tensor([1.0]), torch.ones(1), lovasz_weight=0.0)
        self.assertAlmostEqual(float(loss.detach()), 2.0, places=5)
        loss.backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_p10_p90_and_tanh_feature_sanitization_and_cutmix_deformation(self):
        rng = np.random.default_rng(7)
        chunk = rng.normal(size=(200, 3))
        deformed = deform_canopy_chunk(chunk, rng)
        self.assertTrue(80 <= len(deformed) <= 150)
        labels = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        parent = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        xyz = rng.uniform(-50, 50, size=(5, 3)).astype(np.float32)
        normals = np.tile([0.0, 0.0, 1.0], (5, 1)).astype(np.float32)
        fine = np.zeros((5, 6), dtype=np.float32)
        coarse = np.zeros((5, 6), dtype=np.float32)
        enclosure = np.zeros((5, 4), dtype=np.float32)
        drip = np.array([500.0, -500.0], dtype=np.float32)
        z_rel = np.array([500.0, -500.0], dtype=np.float32)
        above = np.array([500.0, -500.0], dtype=np.float32)
        known = np.array([True, False])
        feats, names = superpoint_features(labels, parent, xyz, normals, fine, coarse, enclosure,
                                           drip, z_rel, above, known)
        self.assertEqual(feats.shape, (2, 104))
        self.assertTrue(any(n.startswith("p10_") for n in names))
        self.assertTrue(any(n.startswith("p90_") for n in names))
        self.assertFalse(any(n.startswith("min_") or n.startswith("max_") for n in names))
        for bounded_col in ("shape_radius", "drip_tanh", "floor_height_tanh", "above_roof_tanh"):
            idx = int(np.flatnonzero(names == bounded_col)[0])
            self.assertTrue(np.all(np.abs(feats[:, idx]) <= 1.0))

    def test_filter_cavern_manifold_removes_isolated_floating_canopy_island(self):
        semantic = np.array([0, 0, 0, 1], dtype=np.int32)
        edge_u = np.array([0, 1], dtype=np.int32)
        edge_v = np.array([1, 0], dtype=np.int32)
        sp_pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [25.0, 25.0, 25.0], [26.0, 25.0, 25.0]], dtype=np.float32)
        sp_counts = np.array([500, 500, 10, 100], dtype=np.int32)
        probs = np.array([[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.55, 0.40, 0.05], [0.1, 0.8, 0.1]], dtype=np.float32)
        filtered = filter_cavern_manifold(semantic, edge_u, edge_v, sp_pos, sp_counts, probabilities=probs)
        np.testing.assert_array_equal(filtered, [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
