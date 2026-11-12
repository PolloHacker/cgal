"""Weight-edit invariants and portable/PyTorch agreement."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from spt_learning import portable_predict
from spt_torch import SuperpointGraphSAGE, torch_graph
from spt_weight_space import edit_weights, histogram_metrics


class WeightSpaceTests(unittest.TestCase):
    def test_edit_changes_only_target_tensor_and_formats_agree(self):
        torch.manual_seed(13)
        names = np.array(["a", "b", "c", "d"])
        model = SuperpointGraphSAGE(np.zeros(4), np.ones(4), hidden_dim=6)
        graph = {"X": np.random.default_rng(13).normal(size=(7, 4)).astype(np.float32),
                 "edge_u": np.arange(6), "edge_v": np.arange(1, 7),
                 "sp_pos": np.arange(21).reshape(7, 3).astype(np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/"source.npz", Path(directory)/"edited.npz"
            model.export(source, names, {"training_groups": ["training_cave"]})
            original = source.read_bytes()
            edit_weights(source, target, .775)
            self.assertEqual(source.read_bytes(), original)
            with np.load(source) as a, np.load(target) as b:
                for key in a.files:
                    if key == "neighbor_2":
                        np.testing.assert_array_equal(b[key], a[key]*np.float32(.775))
                    elif key != "metadata":
                        np.testing.assert_array_equal(b[key], a[key])
                self.assertEqual(json.loads(str(b["metadata"]))["training_groups"], ["training_cave"])
            portable = portable_predict(graph["X"], graph["edge_u"], graph["edge_v"],
                                         graph["sp_pos"], target, names)
            checkpoint = torch.load(target.with_suffix(".pt"), weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            with torch.no_grad():
                expected = model(torch.from_numpy(graph["X"]), torch_graph(graph, "cpu")).softmax(1).numpy()
            np.testing.assert_allclose(portable, expected, atol=2e-7, rtol=2e-6)
            with self.assertRaises(ValueError):
                edit_weights(target, Path(directory)/"twice.npz", .775)
            with self.assertRaises(ValueError):
                edit_weights(source, source, .775)
            for scale in [0, -1, float("nan"), float("inf"), 3]:
                with self.assertRaises(ValueError):
                    edit_weights(source, target, scale)

    def test_metrics_count_mixed_nodes_and_ignore_unknown_points(self):
        histogram = np.array([[8, 2, 0, 50], [1, 9, 0, 0], [0, 0, 0, 100]])
        metrics = histogram_metrics(histogram, np.array([0, 1, 2]))
        self.assertEqual(metrics["known_points"], 20)
        self.assertEqual(metrics["errors"], 3)
        self.assertEqual(metrics["confusion"], [[8, 1, 0], [2, 9, 0], [0, 0, 0]])
        self.assertIsNone(metrics["iou"]["exterior_ground"])


if __name__ == "__main__":
    unittest.main()
