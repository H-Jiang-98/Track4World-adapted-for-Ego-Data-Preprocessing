from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from visualization.vis_3d_ff_ego import (
    FLOW_SCHEMA_VERSION,
    TrailAccumulator,
    build_full_track_segments,
    build_point_cloud,
    build_source_cache,
    build_track_heads,
    build_trail_segments,
    build_trajectory_colors,
    load_flows,
    resolve_downsample,
    resolve_trajectory_downsample,
    rotation_matrix_to_wxyz,
    scene_view_for_cache,
    source_world_tracks,
)


def make_payload() -> dict[str, np.ndarray]:
    horizon = 3
    sources = np.array([0, 1], dtype=np.int32)
    query_offsets = np.array([0, 2, 3], dtype=np.int64)
    query_count = 3
    xyz = np.zeros((horizon, query_count, 3), dtype=np.float32)
    for timestep in range(horizon):
        xyz[timestep] = np.array(
            [[1 + timestep, 0, 1], [2 + timestep, 0, 1], [0, 1 + timestep, 2]],
            dtype=np.float32,
        )
    valid = np.ones((horizon, query_count), dtype=np.bool_)
    valid[2, 1] = False
    xyz[2, 1] = np.nan
    confidence = np.full((horizon, query_count, 2), 0.8, dtype=np.float32)
    confidence[:, 1] = 0.2
    confidence[1, 0] = [0.4, 0.6]
    c2w = np.eye(4, dtype=np.float32)[None].repeat(4, axis=0)
    c2w[1, :3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    c2w[1, :3, 3] = [10.0, 20.0, 30.0]
    return {
        "schema_version": np.asarray(FLOW_SCHEMA_VERSION, dtype="<U40"),
        "coordinate_system": np.asarray("source_camera", dtype="<U24"),
        "pixel_convention": np.asarray("unpadded_uv_integer", dtype="<U32"),
        "source_frame_index": sources,
        "target_frame_index": sources[:, None] + np.arange(horizon, dtype=np.int32),
        "query_offsets": query_offsets,
        "query_uv": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32),
        "query_rgb": np.array(
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8
        ),
        "track_xyz": xyz,
        "track_valid": valid,
        "confidence": confidence,
        "c2w": c2w,
        "horizon_length": np.asarray(horizon, dtype=np.int32),
        "effective_frame_count": np.asarray(4, dtype=np.int32),
        "image_size_hw": np.asarray([2, 2], dtype=np.int32),
    }


class EgoViewerDataTests(unittest.TestCase):
    def test_loads_ragged_npz_without_pickle(self) -> None:
        payload = make_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.npz"
            np.savez(path, **payload)
            data = load_flows(path)
        self.assertEqual(data.source_count, 2)
        self.assertEqual(data.query_count, 3)
        self.assertEqual(data.query_rgb.dtype, np.uint8)
        np.testing.assert_array_equal(data.query_offsets, [0, 2, 3])

    def test_missing_query_rgb_is_rejected(self) -> None:
        payload = make_payload()
        payload.pop("query_rgb")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.npz"
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "query_rgb"):
                load_flows(path)

    def test_source_tracks_are_transformed_by_selected_source_pose(self) -> None:
        payload = make_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.npz"
            np.savez(path, **payload)
            data = load_flows(path)
        xyz_world, rgb = source_world_tracks(data, 1)
        # Source ordinal 1 owns global query column 2 and has a 90-degree Z
        # rotation plus translation [10,20,30].
        np.testing.assert_allclose(xyz_world[0, 0], [9.0, 20.0, 32.0])
        np.testing.assert_array_equal(rgb, [[70, 80, 90]])

    def test_point_and_trail_filters_use_validity_and_mean_confidence(self) -> None:
        payload = make_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.npz"
            np.savez(path, **payload)
            data = load_flows(path)
        points, colors = build_point_cloud(data, 0, 1, 0.5, 1)
        self.assertEqual(points.shape, (1, 3))
        np.testing.assert_array_equal(colors, [[10, 20, 30]])
        segments, segment_colors = build_trail_segments(data, 0, 2, 0.5, 1, 2)
        self.assertEqual(segments.shape, (2, 2, 3))
        np.testing.assert_array_equal(
            segment_colors,
            [
                [[10, 20, 30], [10, 20, 30]],
                [[10, 20, 30], [10, 20, 30]],
            ],
        )

    def test_trajectory_palette_is_bright_and_deterministic(self) -> None:
        uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)
        colors_a = build_trajectory_colors(uv, source_index=2)
        colors_b = build_trajectory_colors(uv, source_index=2)
        self.assertEqual(colors_a.shape, (4, 3))
        self.assertEqual(colors_a.dtype, np.uint8)
        np.testing.assert_array_equal(colors_a, colors_b)
        self.assertTrue(np.all(colors_a.max(axis=1) >= 240))

    def test_directional_segment_colors_have_dim_tail_and_bright_head(self) -> None:
        payload = make_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.npz"
            np.savez(path, **payload)
            data = load_flows(path)
        cache = build_source_cache(data, 0)
        segments, colors = build_trail_segments(
            data,
            0,
            1,
            0.0,
            1,
            1,
            source_cache=cache,
            trajectory_colors=cache.trajectory_colors,
            directional_colors=True,
        )
        self.assertEqual(segments.shape[0], 2)
        tail_luma = colors[:, 0] @ np.array([0.2126, 0.7152, 0.0722])
        head_luma = colors[:, 1] @ np.array([0.2126, 0.7152, 0.0722])
        self.assertTrue(np.all(tail_luma < head_luma))

    def test_track_heads_use_the_same_sampled_trajectory_colors(self) -> None:
        payload = make_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.npz"
            np.savez(path, **payload)
            data = load_flows(path)
        cache = build_source_cache(data, 0)
        points, colors = build_track_heads(
            data,
            0,
            1,
            0.0,
            1,
            source_cache=cache,
            trajectory_colors=cache.trajectory_colors,
        )
        self.assertEqual(points.shape, (2, 3))
        np.testing.assert_array_equal(colors, cache.trajectory_colors)

    def test_full_track_segments_cover_the_entire_horizon(self) -> None:
        payload = make_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.npz"
            np.savez(path, **payload)
            data = load_flows(path)
        cache = build_source_cache(data, 0)
        segments, colors = build_full_track_segments(
            data,
            0,
            0.0,
            1,
            source_cache=cache,
            trajectory_colors=cache.trajectory_colors,
            directional_colors=True,
        )
        # The second query is invalid at the final timestep, so three of the
        # four possible query/transition pairs survive.
        self.assertEqual(segments.shape, (3, 2, 3))
        self.assertEqual(colors.shape, (3, 2, 3))

    def test_scene_view_is_finite_and_targets_track_volume(self) -> None:
        payload = make_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.npz"
            np.savez(path, **payload)
            data = load_flows(path)
        position, look_at = scene_view_for_cache(build_source_cache(data, 0))
        self.assertTrue(np.isfinite(position).all())
        self.assertTrue(np.isfinite(look_at).all())
        self.assertGreater(np.linalg.norm(np.asarray(position) - look_at), 0.1)

    def test_source_cache_precomputes_derived_arrays_without_mutating_input(
        self,
    ) -> None:
        payload = make_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.npz"
            np.savez(path, **payload)
            data = load_flows(path)
        valid_before = data.track_valid.copy()
        cache = build_source_cache(data, 0)
        self.assertEqual(cache.world_xyz.shape, (3, 2, 3))
        self.assertEqual(cache.confidence_mean.shape, (3, 2))
        self.assertEqual(cache.trajectory_colors.shape, (2, 3))
        self.assertEqual(cache.trajectory_colors.dtype, np.uint8)
        np.testing.assert_allclose(cache.confidence_mean[1], [0.5, 0.2])
        np.testing.assert_array_equal(data.track_valid, valid_before)
        build_point_cloud(data, 0, 1, 0.5, 1, source_cache=cache)
        np.testing.assert_array_equal(data.track_valid, valid_before)

    def test_resolve_downsample_honors_point_budget(self) -> None:
        self.assertEqual(resolve_downsample(100, 1, 50), 2)
        self.assertEqual(resolve_downsample(100, 4, 50), 4)
        self.assertEqual(resolve_downsample(0, 1, 1), 1)

    def test_resolve_trajectory_downsample_honors_horizon_segment_budget(self) -> None:
        # 100 queries × 9 transitions must be sampled at least every 4 rows to
        # remain under a 250-segment full-horizon budget.
        self.assertEqual(resolve_trajectory_downsample(100, 1, 10, 250), 4)
        # A larger explicit step is never silently reduced.
        self.assertEqual(resolve_trajectory_downsample(100, 7, 10, 250), 7)
        self.assertEqual(resolve_trajectory_downsample(0, 1, 10, 1), 1)

    def test_trail_accumulator_appends_sequential_steps_and_rebuilds_on_jump(
        self,
    ) -> None:
        payload = make_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.npz"
            np.savez(path, **payload)
            data = load_flows(path)
        cache = build_source_cache(data, 0)
        accumulator = TrailAccumulator()
        accumulator.update(cache, 0, 0.5, 1, 2)
        self.assertEqual(accumulator.rebuild_count, 1)
        self.assertEqual(accumulator.append_count, 0)
        accumulator.update(cache, 1, 0.5, 1, 2)
        accumulator.update(cache, 2, 0.5, 1, 2)
        self.assertEqual(accumulator.rebuild_count, 1)
        self.assertEqual(accumulator.append_count, 2)
        incremental_segments, incremental_colors = accumulator.update(
            cache, 2, 0.5, 1, 2
        )
        full_segments, full_colors = build_trail_segments(
            data, 0, 2, 0.5, 1, 2, source_cache=cache
        )
        np.testing.assert_array_equal(incremental_segments, full_segments)
        np.testing.assert_array_equal(incremental_colors, full_colors)
        self.assertEqual(accumulator.append_count, 2)
        accumulator.update(cache, 0, 0.5, 1, 2)
        self.assertEqual(accumulator.rebuild_count, 2)

    def test_rotation_matrix_conversion_returns_wxyz(self) -> None:
        identity = rotation_matrix_to_wxyz(np.eye(3))
        np.testing.assert_allclose(identity, [1.0, 0.0, 0.0, 0.0])
        rotation_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(
            rotation_matrix_to_wxyz(rotation_z),
            [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        )


if __name__ == "__main__":
    unittest.main()
