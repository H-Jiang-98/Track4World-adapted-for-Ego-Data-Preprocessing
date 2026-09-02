from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_3dff_static_background import (
    compute_horizon_drift_metrics,
    compute_horizon_drift_distributions,
    compute_drift_metrics,
    evaluate_3dff_static_background,
    load_flows_npz,
    select_static_queries,
)


class StaticQuerySelectionTests(unittest.TestCase):
    def test_uv_rows_select_the_corresponding_static_pixels(self) -> None:
        static_mask = np.array(
            [
                [True, False, True],
                [False, True, False],
            ],
            dtype=bool,
        )
        uv = np.array([[0, 0], [1, 0], [1, 1], [2, 1]], dtype=np.int32)

        selected = select_static_queries(uv, static_mask)

        np.testing.assert_array_equal(selected, [True, False, True, False])

    def test_out_of_bounds_uv_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            select_static_queries(
                np.array([[2, 0]], dtype=np.int32),
                np.ones((2, 2), dtype=bool),
            )


class DriftMetricTests(unittest.TestCase):
    def test_max_drift_is_computed_per_point_before_spatial_aggregation(self) -> None:
        # Point 0 peaks in the middle and recovers. Point 1 peaks at the end.
        errors = np.array(
            [
                [1.0, 2.0],
                [5.0, 3.0],
                [1.0, 4.0],
            ],
            dtype=np.float32,
        )
        valid = np.ones_like(errors, dtype=bool)

        result = compute_drift_metrics(
            errors,
            valid,
            scene_scale=10.0,
            frame_indices=[1, 2, 3],
            robust_temporal_percentile=50.0,
            relative_thresholds_percent=[20.0],
            absolute_thresholds=[2.0],
        )

        raw = result["raw_coordinate_units"]
        normalized = result["percent_of_frame0_static_median_depth"]
        self.assertAlmostEqual(raw["static_max_drift50"], 4.5)
        self.assertAlmostEqual(raw["static_max_drift90"], 4.9, places=6)
        self.assertAlmostEqual(raw["static_max_drift_mean"], 4.5)
        self.assertAlmostEqual(normalized["static_max_drift50"], 45.0)
        # Final-frame median is 2.5, proving the metric retained the earlier peak.
        self.assertNotAlmostEqual(raw["static_max_drift50"], 2.5)

    def test_time_aggregation_balances_frames_and_respects_validity(self) -> None:
        errors = np.array(
            [
                [1.0, 100.0, 100.0],
                [3.0, 5.0, 9.0],
            ],
            dtype=np.float32,
        )
        valid = np.array(
            [
                [True, False, False],
                [True, True, True],
            ],
            dtype=bool,
        )

        result = compute_drift_metrics(
            errors,
            valid,
            scene_scale=10.0,
            frame_indices=[1, 2],
            robust_temporal_percentile=95.0,
            relative_thresholds_percent=[50.0],
            absolute_thresholds=[],
        )

        raw = result["raw_coordinate_units"]
        # Per-frame means are 1 and 17/3, and frames receive equal weight.
        self.assertAlmostEqual(raw["static_epe3d"], (1.0 + 17.0 / 3.0) / 2.0)
        self.assertAlmostEqual(result["coverage"]["mean_per_frame"], 2.0 / 3.0)
        accuracy = result["accuracy"]["relative_thresholds"][0]
        # At threshold 5: frame accuracies are 1 and 2/3.
        self.assertAlmostEqual(accuracy["time_balanced_accuracy"], 5.0 / 6.0)


class HorizonMetricTests(unittest.TestCase):
    def test_only_requested_metrics_use_per_query_temporal_aggregation(self) -> None:
        tracks = np.zeros((4, 3, 3), dtype=np.float32)
        tracks[..., 2] = 10.0
        tracks[1:, 0, 0] = [1.0, 5.0, 3.0]
        tracks[1:, 1, 0] = [2.0, 4.0, 6.0]
        tracks[1:, 2, 0] = [7.0, 8.0, 9.0]
        valid = np.ones((4, 3), dtype=np.bool_)
        confidence = np.ones((4, 3, 2), dtype=np.float32)
        # Product 0.04 is below the default 0.1 threshold at every target.
        confidence[1:, 2] = 0.2

        metrics = compute_horizon_drift_metrics(
            tracks,
            valid,
            confidence,
            np.ones(3, dtype=bool),
            scene_scale=10.0,
            robust_temporal_percentile=50.0,
        )

        self.assertEqual(
            set(metrics),
            {"max_drift", "robust_max_drift", "coverage"},
        )
        # Per-point max values are 5 and 6, normalized by depth 10.
        self.assertAlmostEqual(metrics["max_drift"]["median"], 55.0)
        # Per-point temporal medians are 3 and 4.
        self.assertAlmostEqual(metrics["robust_max_drift"]["median"], 35.0)
        self.assertAlmostEqual(metrics["coverage"]["mean_per_frame"], 200.0 / 3.0)
        self.assertEqual(metrics["coverage"]["eligible_query_count"], 2)

        _, max_samples, robust_samples = compute_horizon_drift_distributions(
            tracks,
            valid,
            confidence,
            np.ones(3, dtype=bool),
            scene_scale=10.0,
            robust_temporal_percentile=50.0,
        )
        np.testing.assert_allclose(max_samples, [50.0, 60.0])
        np.testing.assert_allclose(robust_samples, [30.0, 40.0])

    def test_npz_evaluation_uses_source_masks_and_frame0_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_dir = root / "3d_ff_ego_output"
            mask_dir = root / "final_dyn_mask"
            flow_dir.mkdir()
            mask_dir.mkdir()

            sources = np.array([0, 2], dtype=np.int32)
            horizon = 3
            targets = sources[:, None] + np.arange(horizon, dtype=np.int32)[None]
            offsets = np.array([0, 2, 4], dtype=np.int64)
            query_uv = np.array(
                [[0, 0], [1, 0], [0, 0], [1, 0]],
                dtype=np.int32,
            )
            tracks = np.zeros((horizon, 4, 3), dtype=np.float32)
            tracks[0, :, 2] = [10.0, 20.0, 100.0, 100.0]
            tracks[1:, :, 2] = tracks[0, :, 2]
            tracks[1, :, 0] = [0.5, 1.0, 1.0, 4.0]
            tracks[2, :, 0] = [1.0, 2.0, 3.0, 8.0]
            payload = {
                "schema_version": np.asarray("track4world.ego_flows.v1"),
                "coordinate_system": np.asarray("source_camera"),
                "source_frame_index": sources,
                "target_frame_index": targets,
                "query_offsets": offsets,
                "query_uv": query_uv,
                "track_xyz": tracks,
                "track_valid": np.ones((horizon, 4), dtype=np.bool_),
                "confidence": np.ones((horizon, 4, 2), dtype=np.float32),
                "image_size_hw": np.array([1, 2], dtype=np.int32),
                "horizon_length": np.asarray(horizon, dtype=np.int32),
            }
            np.savez_compressed(flow_dir / "flows.npz", **payload)
            Image.fromarray(np.zeros((1, 2), dtype=np.uint8)).save(
                mask_dir / "mask_0000.png"
            )
            # The second source keeps only its query at (0, 0).
            Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(
                mask_dir / "mask_0002.png"
            )
            plot_path = flow_dir / "metrics.png"

            report = evaluate_3dff_static_background(
                flow_dir / "flows.npz",
                mask_dir,
                static_erosion_iterations=0,
                robust_temporal_percentile=50.0,
                plot_output=plot_path,
            )

            self.assertEqual(len(report["horizons"]), 2)
            self.assertAlmostEqual(
                report["selection"]["frame0_static_median_depth_coordinate_units"],
                15.0,
            )
            second = report["horizons"][1]
            self.assertEqual(second["source_frame_index"], 2)
            self.assertEqual(second["static_query_count"], 1)
            # Source-2 max drift is 3, but normalization stays at frame-0 scale 15.
            self.assertAlmostEqual(second["max_drift"]["median"], 20.0)
            self.assertEqual(
                set(second["metrics"]),
                {"max_drift", "robust_max_drift", "coverage"},
            )
            self.assertNotIn("accuracy", report)
            self.assertNotIn("static_epe3d", str(report))
            self.assertTrue(plot_path.is_file())
            self.assertGreater(plot_path.stat().st_size, 0)

            loaded = load_flows_npz(flow_dir / "flows.npz")
            np.testing.assert_array_equal(loaded["query_offsets"], offsets)

    def test_missing_source_mask_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            flow_dir = root / "3d_ff_ego_output"
            mask_dir = root / "final_dyn_mask"
            flow_dir.mkdir()
            mask_dir.mkdir()
            sources = np.array([0, 1], dtype=np.int32)
            payload = {
                "source_frame_index": sources,
                "target_frame_index": sources[:, None]
                + np.arange(2, dtype=np.int32)[None],
                "query_offsets": np.array([0, 1, 2], dtype=np.int64),
                "query_uv": np.array([[0, 0], [0, 0]], dtype=np.int32),
                "track_xyz": np.ones((2, 2, 3), dtype=np.float32),
                "track_valid": np.ones((2, 2), dtype=np.bool_),
                "confidence": np.ones((2, 2, 2), dtype=np.float32),
                "image_size_hw": np.array([1, 1], dtype=np.int32),
            }
            np.savez_compressed(flow_dir / "flows.npz", **payload)
            Image.fromarray(np.zeros((1, 1), dtype=np.uint8)).save(
                mask_dir / "mask_0000.png"
            )

            with self.assertRaisesRegex(FileNotFoundError, "mask_0001"):
                evaluate_3dff_static_background(
                    flow_dir / "flows.npz",
                    mask_dir,
                    static_erosion_iterations=0,
                )


if __name__ == "__main__":
    unittest.main()
