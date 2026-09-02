from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import demo_3dff_ego as demo
from track4world.nets.model_3dff_ego import (
    SequenceCache,
    Track4World3DFFEgo,
)


def make_cache(
    *,
    frame_count: int = 4,
    height: int = 4,
    width: int = 5,
    pad: tuple[int, int, int, int] | None = None,
    model_window_length: int = 2,
    metric_scale: float = 2.0,
) -> SequenceCache:
    if pad is None:
        pad_height = (-height) % 64
        pad_width = (-width) % 64
        pad = (
            pad_width // 2,
            pad_width - pad_width // 2,
            pad_height // 2,
            pad_height - pad_height // 2,
        )
    left, right, top, bottom = pad
    padded_height = height + top + bottom
    padded_width = width + left + right
    feature_time_base = torch.arange(frame_count, dtype=torch.float32).reshape(
        1, frame_count, 1, 1, 1
    )
    feature_time = feature_time_base.expand(
        1, frame_count, 128, padded_height // 8, padded_width // 8
    ).clone()
    context_time = feature_time_base.expand(
        1, frame_count, 128, padded_height // 8, padded_width // 8
    ).clone()
    feature3d_time = feature_time_base.expand(
        1, frame_count, 256, padded_height // 8, padded_width // 8
    ).clone()
    points = torch.zeros(frame_count, 3, padded_height, padded_width)
    yy, xx = torch.meshgrid(
        torch.arange(padded_height, dtype=torch.float32),
        torch.arange(padded_width, dtype=torch.float32),
        indexing="ij",
    )
    for frame_index in range(frame_count):
        points[frame_index, 0] = xx - left
        points[frame_index, 1] = yy - top
        points[frame_index, 2] = 1.0 + frame_index
    masks = torch.ones(frame_count, 1, padded_height, padded_width)
    poses = torch.eye(4).repeat(frame_count, 1, 1)
    poses[:, 0, 3] = torch.arange(frame_count, dtype=torch.float32)
    intrinsic = torch.tensor(
        [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    shared_intrinsics = intrinsic.reshape(1, 1, 3, 3).repeat(
        1, frame_count, 1, 1
    )
    da3_k = torch.tensor(
        [[10.0, 0.0, 2.0], [0.0, 11.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).reshape(1, 1, 3, 3).repeat(1, frame_count, 1, 1)
    cache = SequenceCache(
        t_eff=frame_count,
        batch_size=1,
        image_height=height,
        image_width=width,
        padded_height=padded_height,
        padded_width=padded_width,
        pad=pad,
        image_dtype=torch.float32,
        device=torch.device("cpu"),
        model_window_length=model_window_length,
        mask_threshold=0.5,
        metric_scale=torch.tensor(metric_scale),
        metric_scale_enabled=False,
        raw_da3_intrinsics_px=da3_k,
        shared_da3_intrinsics_px=da3_k.clone(),
        shared_intrinsics=shared_intrinsics,
        ego_output_px=torch.tensor(
            [[10.0, 0.0, 2.0], [0.0, 11.0, 2.0], [0.0, 0.0, 1.0]]
        ),
        fmaps=feature_time.clone(),
        ctxfeats=context_time,
        fmaps3d_detail=feature3d_time,
        pms=feature_time_base.expand(
            1, frame_count, 3, padded_height // 8, padded_width // 8
        ).clone(),
        points=points,
        masks=masks,
        world_points=points.clone(),
        camera_poses=poses,
    )
    cache.validate()
    return cache


class _ObservationStub:
    def __init__(self, frame_count: int) -> None:
        k = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, frame_count, 1, 1)
        k[..., 0, 0] = 32.0
        k[..., 1, 1] = 33.0
        k[..., 0, 2] = 16.0
        k[..., 1, 2] = 16.0
        self.raw = k

    def observed_intrinsics(self):
        return self.raw, self.raw.clone(), 64, 64


class SyntheticCacheModel(Track4World3DFFEgo):
    """A CPU interface double that never constructs a real backbone."""

    def __init__(self, frame_count: int, window_length: int = 2) -> None:
        torch.nn.Module.__init__(self)
        self.register_parameter("_test_parameter", torch.nn.Parameter(torch.zeros(())))
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        )
        self.seqlen = window_length
        self.flow_dim = 128
        self.flow3d_dim = 256
        self.mask_threshold = 0.5
        self.use_metric_scale = False
        self.feature_calls = 0
        self.tracking_head_calls = 0
        self.captured_infer_kwargs = None
        self._observation_stub = _ObservationStub(frame_count)

    def _install_ego_adapter(self):
        return self._observation_stub

    def forward_window_unified(self, *args, **kwargs):
        self.tracking_head_calls += 1
        raise AssertionError("encode_sequence must not invoke the tracking head")

    def get_fmaps(self, images_, batch_size, frame_count, sw, is_training):
        self.feature_calls += 1
        self._metric_scale = torch.tensor(3.0, device=images_.device)
        height, width = images_.shape[-2:]
        h8, w8 = height // 8, width // 8
        time = torch.arange(frame_count, device=images_.device, dtype=images_.dtype)
        feature_base = time.reshape(frame_count, 1, 1, 1)
        feature = feature_base.expand(
            frame_count, 128, h8, w8
        )
        context = feature.clone()
        feature3d = feature_base.expand(frame_count, 256, h8, w8)
        points = torch.ones(frame_count, 3, height, width, device=images_.device)
        masks = torch.ones(frame_count, 1, height, width, device=images_.device)
        poses = torch.eye(4, device=images_.device).repeat(frame_count, 1, 1)
        intrinsic = torch.tensor(
            [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
            device=images_.device,
        ).reshape(1, 1, 3, 3).repeat(1, frame_count, 1, 1)
        return (
            feature,
            context,
            feature3d,
            feature_base.expand(frame_count, 3, h8, w8),
            points,
            masks,
            points.clone(),
            poses,
            intrinsic,
        )

    def infer(self, images, **kwargs):
        self.tracking_head_calls += 1
        self.captured_infer_kwargs = {"images": images, **kwargs}
        family = {"metric_scale": self._metric_scale}
        return [family, family.copy()], kwargs["eval_dict"]


class CliAndPlanningTests(unittest.TestCase):
    def test_h_and_s_are_required(self) -> None:
        parser = demo.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--H", "16"])
        args = parser.parse_args(["--H", "16", "--S", "8"])
        self.assertEqual((args.horizon_length, args.window_stride), (16, 8))

    def test_checkpoint_window_length_uses_both_embeddings(self) -> None:
        state = {
            "time_emb": torch.zeros(1, 16, 128),
            "module.time_emb3d": torch.zeros(1, 16, 256),
        }
        self.assertEqual(demo.checkpoint_window_length(state), 16)
        self.assertEqual(
            demo.checkpoint_window_length(
                {
                    "time_emb": torch.zeros(1, 2, 128),
                    "time_emb3d": torch.zeros(1, 2, 256),
                }
            ),
            2,
        )

    def test_checkpoint_window_length_rejects_mismatch_odd_and_missing(self) -> None:
        with self.assertRaisesRegex(ValueError, "disagree"):
            demo.checkpoint_window_length(
                {
                    "time_emb": torch.zeros(1, 16, 128),
                    "time_emb3d": torch.zeros(1, 8, 256),
                }
            )
        with self.assertRaisesRegex(ValueError, "even"):
            demo.checkpoint_window_length(
                {
                    "time_emb": torch.zeros(1, 3, 128),
                    "time_emb3d": torch.zeros(1, 3, 256),
                }
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            demo.checkpoint_window_length({"time_emb": torch.zeros(1, 2, 1)})

    def test_plan_computes_sources_n_and_t_eff(self) -> None:
        plan = demo.plan_temporal_windows(35, 32, 2, 16)
        self.assertEqual(plan.window_count, 2)
        np.testing.assert_array_equal(plan.source_frame_indices, [0, 2])
        self.assertEqual(plan.effective_frame_count, 34)
        self.assertEqual(plan.windows[1].frame_indices, list(range(2, 34)))
        self.assertEqual(plan.windows[1].target_frame_indices[0], 2)

    def test_exact_horizon_keeps_truncated_start_equal_to_t_eff(self) -> None:
        plan = demo.plan_temporal_windows(32, 32, 32, 16)
        self.assertEqual(plan.effective_frame_count, 32)
        self.assertEqual(plan.input_frame_count, 32)

    def test_w_two_requires_normal_legacy_horizon_but_remains_supported(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 3"):
            demo.plan_temporal_windows(4, 2, 1, 2)
        plan = demo.plan_temporal_windows(4, 4, 1, 2)
        self.assertEqual(plan.model_window_length, 2)
        self.assertEqual(plan.effective_frame_count, 4)

    def test_all_temporal_boundaries_are_rejected(self) -> None:
        cases = [
            ((15, 16, 1, 16), "smaller"),
            ((128, 16, 1, 16), "T <= 128"), 
            ((32, 24, 1, 16), "divisible"),
            ((32, 16, 0, 16), "positive"),
            ((32, 16, 17, 16), "must not exceed"),
            ((32, 16, 1, 3), "even"),
        ]
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, message):
                    demo.plan_temporal_windows(*arguments)

    def test_rgb_and_masks_are_truncated_to_t_eff(self) -> None:
        plan = demo.plan_temporal_windows(35, 32, 2, 16)
        rgb = torch.arange(35).reshape(1, 35, 1, 1, 1)
        masks = torch.arange(35).reshape(35, 1, 1)
        rgb_eff, masks_eff = demo.truncate_effective_inputs(rgb, masks, plan)
        self.assertEqual(rgb_eff.shape[1], 34)
        self.assertEqual(masks_eff.shape[0], 34)
        self.assertEqual(int(rgb_eff[0, -1]), 33)


class CacheInterfaceTests(unittest.TestCase):
    def test_every_cache_axis_is_sliced_on_the_correct_time_axis(self) -> None:
        cache = make_cache(frame_count=4)
        window = cache.window_eval_dict(1, 3)
        self.assertEqual(tuple(window["fmaps"].shape[:2]), (1, 2))
        self.assertEqual(window["fmaps"][0, 0, 0, 0, 0].item(), 1)
        self.assertEqual(window["fmaps"][0, 1, 0, 0, 0].item(), 2)
        torch.testing.assert_close(window["points"], cache.points[1:3])
        torch.testing.assert_close(window["masks"], cache.masks[1:3])
        torch.testing.assert_close(window["world_points"], cache.world_points[1:3])
        torch.testing.assert_close(window["camera_poses"], cache.camera_poses[1:3])
        torch.testing.assert_close(
            window["intrinsics"], cache.shared_intrinsics[:, 1:3]
        )

    def test_source_becomes_window_local_anchor_on_every_axis(self) -> None:
        cache = make_cache(frame_count=4)
        window = cache.window_eval_dict(2, 4)
        self.assertEqual(window["fmaps"][0, 0, 0, 0, 0].item(), 2)
        torch.testing.assert_close(window["points"][0], cache.points[2])
        torch.testing.assert_close(window["camera_poses"][0], cache.camera_poses[2])

    def test_encode_calls_get_fmaps_once_and_tracking_head_zero_times(self) -> None:
        model = SyntheticCacheModel(frame_count=4)
        images = torch.zeros(1, 4, 3, 16, 20)
        cache = model.encode_sequence(images)
        self.assertEqual(model.feature_calls, 1)
        self.assertEqual(model.tracking_head_calls, 0)
        self.assertEqual(cache.t_eff, 4)
        self.assertEqual(cache.batch_size, 1)
        self.assertEqual((cache.image_height, cache.image_width), (16, 20))
        self.assertEqual(cache.pad, (22, 22, 24, 24))
        self.assertEqual(cache.image_dtype, torch.float32)
        self.assertEqual(cache.device, torch.device("cpu"))
        self.assertEqual(cache.metric_scale.item(), 3.0)
        torch.testing.assert_close(
            cache.ego_output_px,
            torch.tensor(
                [[32.0, 0.0, 9.5], [0.0, 32.0, 7.5], [0.0, 0.0, 1.0]]
            ),
        )

    def test_cached_tracking_passes_no_outer_stride_and_restores_scale(self) -> None:
        model = SyntheticCacheModel(frame_count=4, window_length=2)
        images = torch.zeros(1, 4, 3, 16, 20)
        cache = model.encode_sequence(images)
        model._metric_scale = torch.tensor(999.0)
        output = model.track_cached_window(cache, images, 0, 4, iters=3)
        self.assertEqual(len(output), 2)
        self.assertEqual(model.feature_calls, 1)
        self.assertEqual(model.tracking_head_calls, 1)
        self.assertIsNone(model.captured_infer_kwargs["stride"])
        self.assertEqual(model.captured_infer_kwargs["window_len"], 2)
        self.assertEqual(model.captured_infer_kwargs["iters"], 3)
        self.assertEqual(model._metric_scale.item(), 3.0)
        self.assertEqual(
            tuple(model.captured_infer_kwargs["eval_dict"]["fmaps"].shape[:2]),
            (1, 4),
        )

    def test_cache_rejects_dtype_device_or_shape_mismatch(self) -> None:
        model = SyntheticCacheModel(frame_count=4)
        images = torch.zeros(1, 4, 3, 16, 20)
        cache = model.encode_sequence(images)
        with self.assertRaisesRegex(ValueError, "dtype/device"):
            model.track_cached_window(cache, images.half(), 0, 4, 1)
        with self.assertRaisesRegex(ValueError, "do not match"):
            model.track_cached_window(cache, images[:, :3], 0, 2, 1)


class GeometryAndValidityTests(unittest.TestCase):
    def test_pose_transform_is_p_source_inverse_p_target(self) -> None:
        source = np.eye(4, dtype=np.float32)
        source[0, 3] = 4.0
        target = np.eye(4, dtype=np.float32)
        target[0, 3] = 7.0
        points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        transformed = demo.transform_target_camera_to_source_camera(
            points, source, target
        )
        np.testing.assert_allclose(transformed, [[4.0, 2.0, 3.0]])

    def test_j0_uses_source_geometry_and_target_validity_precedes_sampling(self) -> None:
        cache = make_cache(frame_count=3, height=4, width=4)
        yy, xx = np.meshgrid(
            np.arange(4, dtype=np.float32),
            np.arange(4, dtype=np.float32),
            indexing="ij",
        )
        dense = np.zeros((1, 3, 4, 4, 3), dtype=np.float32)
        for local in range(3):
            dense[0, local, ..., 0] = xx
            dense[0, local, ..., 1] = yy
            dense[0, local, ..., 2] = 2.0
        dense[0, 0] = 99.0  # raw Tracking Head j=0 must not become NPZ j=0
        dense[0, 1, 1, 1, 2] = -1.0
        conf = np.zeros((1, 3, 2, 4, 4), dtype=np.float32)
        conf[:, :, 0] = 0.25
        conf[:, :, 1] = 0.75
        poses = torch.eye(4).repeat(3, 1, 1)
        poses[1, 0, 3] = 1.0
        poses[2, 0, 3] = 2.0
        output = [
            {
                "camera_poses": poses,
                # Deliberately false target masks: target validity must not use them.
                "mask": torch.zeros(1, 3, 4, 4, dtype=torch.bool),
            },
            {
                "flow_3d": torch.from_numpy(dense),
                "visconf_maps_e": torch.from_numpy(conf),
            },
        ]
        tracks = demo.build_source_tracks(
            cache,
            output,
            source=0,
            end=3,
            depth_edge_rtol=0.0,
            metric_scale_enabled=False,
            source_rgb=np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4),
        )
        lookup = {tuple(uv): index for index, uv in enumerate(tracks.query_uv.tolist())}
        corner = lookup[(0, 0)]
        np.testing.assert_array_equal(tracks.query_rgb[corner], [0, 16, 32])
        np.testing.assert_allclose(tracks.track_xyz[corner, 0], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(tracks.track_xyz[corner, 1], [1.0, 0.0, 2.0])
        self.assertTrue(tracks.track_valid[corner, 1])
        center = lookup[(1, 1)]
        np.testing.assert_array_equal(tracks.query_rgb[center], [5, 21, 37])
        self.assertFalse(tracks.track_valid[center, 1])
        self.assertTrue(np.isnan(tracks.track_xyz[center, 1]).all())
        np.testing.assert_allclose(tracks.confidence[center, 1], [0.25, 0.75])
        self.assertTrue(tracks.track_valid[:, 0].all())

    def test_metric_scale_applies_to_xyz_and_pose_translation_but_raw_is_saved(self) -> None:
        cache = make_cache(frame_count=4, metric_scale=3.0)
        points, _ = demo.source_geometry_from_cache(
            cache, 0, metric_scale_enabled=True
        )
        self.assertEqual(points[0, 0, 2], 3.0)
        poses = demo.scaled_c2w_from_cache(cache, metric_scale_enabled=True)
        self.assertEqual(poses[1, 0, 3], 3.0)
        self.assertEqual(cache.metric_scale.item(), 3.0)


class SerializationAndPreprocessingTests(unittest.TestCase):
    def _payload_fixture(self):
        plan = demo.plan_temporal_windows(5, 4, 1, 2)
        cache = make_cache(frame_count=5, model_window_length=2)
        first_xyz = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
        first = demo.SourceTracks(
            query_uv=np.array([[0, 0], [1, 1]], dtype=np.int32),
            query_rgb=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
            track_xyz=first_xyz,
            track_valid=np.ones((2, 4), dtype=np.bool_),
            confidence=np.full((2, 4, 2), 0.5, dtype=np.float32),
        )
        second = demo.SourceTracks(
            query_uv=np.empty((0, 2), dtype=np.int32),
            query_rgb=np.empty((0, 3), dtype=np.uint8),
            track_xyz=np.empty((0, 4, 3), dtype=np.float32),
            track_valid=np.empty((0, 4), dtype=np.bool_),
            confidence=np.empty((0, 4, 2), dtype=np.float32),
        )
        return demo.build_flows_payload(
            plan, cache, [first, second], metric_scale_enabled=False
        )

    def test_npz_schema_is_ragged_and_pickle_free(self) -> None:
        payload = self._payload_fixture()
        np.testing.assert_array_equal(payload["query_offsets"], [0, 2, 2])
        self.assertEqual(payload["track_xyz"].shape, (4, 2, 3))
        self.assertEqual(payload["track_valid"].shape, (4, 2))
        self.assertEqual(payload["confidence"].shape, (4, 2, 2))
        self.assertEqual(payload["query_rgb"].shape, (2, 3))
        self.assertEqual(payload["query_rgb"].dtype, np.uint8)
        np.testing.assert_array_equal(
            payload["query_rgb"], [[10, 20, 30], [40, 50, 60]]
        )
        np.testing.assert_array_equal(
            payload["track_xyz"][:, 0],
            np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)[0],
        )
        np.testing.assert_array_equal(
            payload["track_xyz"][:, 1],
            np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)[1],
        )
        np.testing.assert_array_equal(payload["source_frame_index"], [0, 1])
        np.testing.assert_array_equal(
            payload["target_frame_index"],
            [[0, 1, 2, 3], [1, 2, 3, 4]],
        )
        self.assertEqual(payload["last_complete_source_frame_index"].dtype, np.int32)
        self.assertEqual(int(payload["truncated_frame_start"]), 5)
        self.assertEqual(int(payload["effective_frame_count"]), 5)
        self.assertEqual(payload["schema_version"].dtype.kind, "U")
        self.assertTrue(all(not value.dtype.hasobject for value in payload.values()))

    def test_only_one_atomic_flows_npz_remains(self) -> None:
        payload = self._payload_fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / demo.OUTPUT_DIRECTORY_NAME
            output.mkdir()
            (output / "stale.npy").write_bytes(b"old")
            demo.save_flows_npz(output, payload)
            self.assertEqual([path.name for path in output.iterdir()], ["flows.npz"])
            with np.load(output / "flows.npz", allow_pickle=False) as loaded:
                self.assertEqual(set(loaded.files), set(payload))
                demo.validate_flows_payload(
                    {name: loaded[name] for name in loaded.files}
                )

    def test_intrinsics_are_scaled_anisotropically_to_original_pixels(self) -> None:
        resized_k = np.array(
            [[100.0, 1.5, 50.0], [2.0, 200.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        original_k = demo.scale_intrinsics_to_original(
            resized_k,
            original_height=720,
            original_width=1920,
            resized_height=288,
            resized_width=512,
        )
        np.testing.assert_allclose(
            original_k,
            [[375.0, 5.625, 187.5], [5.0, 500.0, 100.0], [0.0, 0.0, 1.0]],
        )

    def test_camera_intrinsics_xml_contains_original_pixel_matrix(self) -> None:
        resized_k = np.array(
            [[100.0, 0.0, 50.0], [0.0, 200.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / demo.INTRINSICS_FILENAME
            demo.save_camera_intrinsics_xml(
                path,
                resized_k,
                original_height=720,
                original_width=1920,
                resized_height=288,
                resized_width=512,
            )
            root = ET.parse(path).getroot()
            self.assertEqual(root.tag, "opencv_storage")
            self.assertEqual(root.findtext("image_width"), "1920")
            self.assertEqual(root.findtext("image_height"), "720")
            self.assertEqual(root.findtext("pixel_units"), "original_image_pixels")
            self.assertAlmostEqual(float(root.findtext("fx")), 375.0)
            self.assertAlmostEqual(float(root.findtext("fy")), 500.0)
            self.assertAlmostEqual(float(root.findtext("cx")), 187.5)
            self.assertAlmostEqual(float(root.findtext("cy")), 100.0)
            matrix_node = root.find("camera_matrix")
            self.assertIsNotNone(matrix_node)
            matrix_values = np.fromstring(matrix_node.findtext("data"), sep=" ")
            np.testing.assert_allclose(
                matrix_values.reshape(3, 3),
                [[375.0, 0.0, 187.5], [0.0, 500.0, 100.0], [0.0, 0.0, 1.0]],
            )

    def test_preprocessing_replacement_removes_old_tail_frames(self) -> None:
        rgb = torch.zeros(1, 3, 3, 2, 2)
        masks = torch.zeros(3, 2, 2)

        def fake_save(rgb_tensor, dynamic_masks, base, fps):
            rgb_dir = base / "final_rgb"
            mask_dir = base / "final_dyn_mask"
            rgb_dir.mkdir()
            mask_dir.mkdir()
            (rgb_dir / "rgb.mp4").write_bytes(b"video")
            for index in range(rgb_tensor.shape[1]):
                (rgb_dir / f"frame_{index:04d}.png").write_bytes(b"rgb")
                (mask_dir / f"mask_{index:04d}.png").write_bytes(b"mask")

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            old_rgb = base / "final_rgb"
            old_mask = base / "final_dyn_mask"
            old_rgb.mkdir()
            old_mask.mkdir()
            (old_rgb / "frame_9999.png").write_bytes(b"tail")
            (old_mask / "mask_9999.png").write_bytes(b"tail")
            with mock.patch.object(demo, "save_preprocessed_inputs", fake_save):
                demo.save_truncated_preprocessed_inputs(rgb, masks, base, 15.0)
            self.assertFalse((old_rgb / "frame_9999.png").exists())
            self.assertFalse((old_mask / "mask_9999.png").exists())
            self.assertEqual(len(list(old_rgb.glob("frame_*.png"))), 3)
            self.assertEqual(len(list(old_mask.glob("mask_*.png"))), 3)


if __name__ == "__main__":
    unittest.main()
