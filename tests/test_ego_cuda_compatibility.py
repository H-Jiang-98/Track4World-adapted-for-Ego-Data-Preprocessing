"""Real-checkpoint numerical compatibility test for the ego-centric path.

This suite is intentionally not represented as CPU compatibility: without
CUDA it is explicitly skipped.  The pure CPU tests cover only cache slicing,
coordinates, validity, and serialization contracts.
"""

from __future__ import annotations

import argparse
import gc
import json
import unittest
from pathlib import Path

import numpy as np
import torch

import utils3d
import demo_3dff_ego as demo
from _demo_3dff_common import build_flow_mask


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints/track4world_da3.pth"
CONFIG = ROOT / "track4world/config/eval/v1.json"


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        ckpt_init=str(CHECKPOINT),
        coordinate="world_depthanythingv3",
        use_original_backbone=False,
        metric_scale=True,
    )


def _load_model(device: torch.device):
    with CONFIG.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    return demo.load_ego_model(_args(), config, device)


@unittest.skipUnless(
    torch.cuda.is_available() and CHECKPOINT.is_file(),
    "real ego-centric numerical compatibility requires CUDA and track4world_da3.pth",
)
class EgoRealCheckpointCompatibilityTests(unittest.TestCase):
    def test_full_cache_matches_legacy_infer_at_two_model_windows(self) -> None:
        device = torch.device("cuda:0")
        state = torch.load(
            str(CHECKPOINT),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        model_window = demo.checkpoint_window_length(state)
        del state
        frame_count = 2 * model_window
        self.assertLess(frame_count, 128)
        # CorrBlock builds five levels and downsamples after every level. The
        # 1/8-resolution flow grid must therefore start at 32x32 or larger.
        image_size = 256
        generator = torch.Generator(device="cpu").manual_seed(20260831)
        images_cpu = torch.randint(
            0,
            256,
            (1, frame_count, 3, image_size, image_size),
            generator=generator,
            dtype=torch.int32,
        ).to(torch.float16)
        images = images_cpu.to(device)
        inference_iters = 2

        legacy = _load_model(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            legacy_output, _ = legacy.infer(
                images,
                iters=inference_iters,
                sw=None,
                is_training=False,
                tracking3d=True,
                force_projection=True,
            )
        legacy_geometry, legacy_motion = legacy_output
        legacy_raw_k, _, legacy_ego, _ = legacy.observed_da3_intrinsics_px(
            output_height=image_size,
            output_width=image_size,
        )
        legacy_saved = {
            "points": legacy_geometry["points"].detach().cpu(),
            "flow_3d": legacy_motion["flow_3d"].detach().cpu(),
            "confidence": legacy_motion["visconf_maps_e"].detach().cpu(),
            "c2w": legacy_geometry["camera_poses"].detach().cpu(),
            "metric_scale": legacy_geometry["metric_scale"].detach().cpu(),
            "raw_k": legacy_raw_k.detach().cpu(),
            "ego_K": legacy_ego.detach().cpu(),
            "mask": legacy_geometry["mask"].detach().cpu(),
        }
        del legacy_output, legacy_geometry, legacy_motion, legacy
        gc.collect()
        torch.cuda.empty_cache()

        cached = _load_model(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            sequence_cache = cached.encode_sequence(images)
            cached_output = cached.track_cached_window(
                sequence_cache,
                images,
                0,
                frame_count,
                inference_iters,
            )
        cached_geometry, cached_motion = cached_output

        cached_raw_parameters = torch.stack(
            [
                sequence_cache.raw_da3_intrinsics_px[..., 0, 0],
                sequence_cache.raw_da3_intrinsics_px[..., 1, 1],
                sequence_cache.raw_da3_intrinsics_px[..., 0, 2],
                sequence_cache.raw_da3_intrinsics_px[..., 1, 2],
            ],
            dim=-1,
        ).cpu()
        torch.testing.assert_close(
            cached_raw_parameters,
            legacy_saved["raw_k"],
            rtol=1e-5,
            atol=1e-5,
        )
        ego_parameters = legacy_saved["ego_K"][0, 0]
        expected_ego = torch.zeros(3, 3, dtype=ego_parameters.dtype)
        expected_ego[0, 0] = ego_parameters[0]
        expected_ego[1, 1] = ego_parameters[1]
        expected_ego[0, 2] = ego_parameters[2]
        expected_ego[1, 2] = ego_parameters[3]
        expected_ego[2, 2] = 1.0
        torch.testing.assert_close(
            sequence_cache.ego_output_px.cpu(),
            expected_ego,
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            sequence_cache.metric_scale.cpu(),
            legacy_saved["metric_scale"],
            rtol=1e-5,
            atol=1e-5,
        )
        # Model-layer comparison includes raw Tracking Head j=0 by comparing
        # the complete dense tensor before NPZ's deliberate j=0 replacement.
        torch.testing.assert_close(
            cached_motion["flow_3d"].cpu(),
            legacy_saved["flow_3d"],
            rtol=2e-3,
            atol=2e-3,
        )
        torch.testing.assert_close(
            cached_motion["visconf_maps_e"].cpu(),
            legacy_saved["confidence"],
            rtol=2e-3,
            atol=2e-3,
        )
        torch.testing.assert_close(
            cached_geometry["points"].cpu(),
            legacy_saved["points"],
            rtol=2e-3,
            atol=2e-3,
        )
        torch.testing.assert_close(
            cached_geometry["camera_poses"].cpu(),
            legacy_saved["c2w"],
            rtol=2e-3,
            atol=2e-3,
        )

        legacy_results = {
            "rgbs": images_cpu[0],
            "points": legacy_saved["points"][0],
            "traj_3d": legacy_saved["flow_3d"][0],
            "visconf": legacy_saved["confidence"][0],
            "masks": legacy_saved["mask"][0],
            "camera_poses": legacy_saved["c2w"],
        }
        common_mask = build_flow_mask(legacy_results, depth_edge_rtol=0.04)
        pixel_uv = utils3d.numpy.image_pixel(width=image_size, height=image_size)
        _, common_uv = utils3d.numpy.image_mesh(pixel_uv, mask=common_mask, tri=True)
        common_uv = np.asarray(common_uv, dtype=np.int32).reshape(-1, 2)
        new_tracks = demo.build_source_tracks(
            sequence_cache,
            cached_output,
            source=0,
            end=frame_count,
            depth_edge_rtol=0.04,
            metric_scale_enabled=True,
            source_rgb=images_cpu[0, 0],
        )
        new_lookup = {tuple(uv): row for row, uv in enumerate(new_tracks.query_uv.tolist())}
        new_rows = np.asarray([new_lookup[tuple(uv)] for uv in common_uv], dtype=np.int64)
        u, v = common_uv[:, 0], common_uv[:, 1]
        legacy_flow = legacy_saved["flow_3d"][0].numpy()
        legacy_poses = legacy_saved["c2w"].numpy()
        legacy_confidence = (
            legacy_saved["confidence"][0].numpy().transpose(0, 2, 3, 1)
        )
        np.testing.assert_allclose(
            new_tracks.confidence[new_rows],
            legacy_confidence[:, v, u].transpose(1, 0, 2),
            rtol=2e-3,
            atol=2e-3,
        )
        for target in range(1, frame_count):
            expected_xyz = demo.transform_target_camera_to_source_camera(
                legacy_flow[target, v, u],
                legacy_poses[0],
                legacy_poses[target],
            )
            np.testing.assert_allclose(
                new_tracks.track_xyz[new_rows, target],
                expected_xyz,
                rtol=2e-3,
                atol=2e-3,
            )

        del cached_output, cached_geometry, cached_motion, sequence_cache, cached
        gc.collect()
        torch.cuda.empty_cache()
