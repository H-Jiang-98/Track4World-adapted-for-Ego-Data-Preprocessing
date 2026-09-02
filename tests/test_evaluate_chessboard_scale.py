from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_chessboard_scale import (
    _edge_metrics,
    bilinear_sample_points,
    build_source_point_map,
    evaluate_chessboard_scale,
    fit_similarity_scale,
)


class ChessboardScaleHelperTests(unittest.TestCase):
    def test_bilinear_sampling_uses_four_valid_neighbors(self) -> None:
        height, width = 3, 4
        yy, xx = np.mgrid[:height, :width]
        point_map = np.stack([xx, yy, xx + yy], axis=-1).astype(np.float64)
        valid_map = np.ones((height, width), dtype=bool)

        points, valid = bilinear_sample_points(
            point_map,
            valid_map,
            np.array([[0.25, 0.75], [2.0, 1.0]], dtype=np.float64),
        )

        np.testing.assert_allclose(points[0], [0.25, 0.75, 1.0])
        np.testing.assert_allclose(points[1], [2.0, 1.0, 3.0])
        np.testing.assert_array_equal(valid, [True, True])

        valid_map[0, 1] = False
        points, valid = bilinear_sample_points(
            point_map, valid_map, np.array([[0.25, 0.75]], dtype=np.float64)
        )
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(points[0]).all())

    def test_source_point_map_scatter_respects_validity(self) -> None:
        point_map, valid_map = build_source_point_map(
            np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32),
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
            np.array([True, False, True], dtype=bool),
            (2, 2),
        )
        np.testing.assert_array_equal(point_map[0, 0], [1, 2, 3])
        self.assertFalse(valid_map[0, 1])
        self.assertTrue(valid_map[1, 0])

    def test_similarity_fit_recovers_known_scale(self) -> None:
        rows, cols = 4, 5
        yy, xx = np.mgrid[:rows, :cols]
        reference = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3)
        angle = 0.37
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        expected_scale = 1.25
        translation = np.array([0.4, -0.3, 2.0])
        observed = expected_scale * reference @ rotation + translation

        result = fit_similarity_scale(reference, observed)

        self.assertAlmostEqual(result["scale_ratio"], expected_scale, places=10)
        self.assertAlmostEqual(result["rmse_coordinate_units"], 0.0, places=10)
        np.testing.assert_allclose(result["translation"], translation, atol=1e-10)

    def test_edge_metrics_report_true_scale_ratio(self) -> None:
        rows, cols = 3, 4
        yy, xx = np.mgrid[:rows, :cols]
        points = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3)
        result = _edge_metrics(points * 1.1, rows, cols, 1.0)
        self.assertAlmostEqual(result["horizontal"]["scale_ratio"]["median"], 1.1)
        self.assertAlmostEqual(result["vertical"]["absolute_error_percent"]["median"], 10.0)


class ChessboardScaleInputTests(unittest.TestCase):
    def test_non_metric_flows_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flow_path = root / "flows.npz"
            sources = np.array([0], dtype=np.int32)
            np.savez_compressed(
                flow_path,
                schema_version=np.asarray("track4world.ego_flows.v1"),
                coordinate_system=np.asarray("source_camera"),
                pixel_convention=np.asarray("unpadded_uv_integer"),
                source_frame_index=sources,
                target_frame_index=np.array([[0, 1]], dtype=np.int32),
                query_offsets=np.array([0, 0], dtype=np.int64),
                query_uv=np.empty((0, 2), dtype=np.int32),
                track_xyz=np.empty((2, 0, 3), dtype=np.float32),
                track_valid=np.empty((2, 0), dtype=bool),
                confidence=np.empty((2, 0, 2), dtype=np.float32),
                image_size_hw=np.array([2, 2], dtype=np.int32),
                horizon_length=np.asarray(2, dtype=np.int32),
                metric_scale_enabled=np.asarray(False, dtype=bool),
            )
            with self.assertRaisesRegex(ValueError, "metric_scale_enabled"):
                evaluate_chessboard_scale(flow_path, square_size_m=0.045)


if __name__ == "__main__":
    unittest.main()
