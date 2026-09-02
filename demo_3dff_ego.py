"""Full-sequence cached ego-centric 3D first-frame tracking.

DA3 and flow-feature extraction run once over every effective input frame. A
source stride selects horizon anchors; each horizon then invokes the legacy
tracking head with a window-local slice of the full-sequence cache.  The 3D
outputs are ``3d_ff_ego_output/flows.npz`` and the sibling
``camera_intrinsics.xml`` calibration file.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import utils3d
from _demo_3dff_common import (
    _finite_depth_mask,
    _publish_output_directory,
    build_parser as build_common_parser,
    validate_args,
)
from _demo_3dff_support import (
    CHECKPOINT_URLS,
    DEFAULT_CHECKPOINTS,
    copy_input_video,
    prepare_inputs,
    save_preprocessed_inputs,
)
from track4world.nets.model_3dff_ego import (
    SequenceCache,
    Track4World3DFFEgo,
)


logger = logging.getLogger(__name__)

FLOWS_FILENAME = "flows.npz"
INTRINSICS_FILENAME = "camera_intrinsics.xml"
# Public descriptive alias retained for callers that spell out the artifact
# type in their constants.
CAMERA_INTRINSICS_FILENAME = INTRINSICS_FILENAME
OUTPUT_DIRECTORY_NAME = "3d_ff_ego_output"
MAX_EGO_FRAMES = 127
FLOW_SCHEMA_VERSION = "track4world.ego_flows.v1"


@dataclass(frozen=True)
class TemporalWindow:
    """One complete horizon on the global effective-frame time axis."""

    index: int
    source_frame_index: int
    end_frame_index_exclusive: int

    @property
    def frame_indices(self) -> list[int]:
        return list(range(self.source_frame_index, self.end_frame_index_exclusive))

    @property
    def target_frame_indices(self) -> list[int]:
        # j=0 is intentionally the source itself.
        return self.frame_indices


@dataclass(frozen=True)
class TemporalPlan:
    """Validated source plan and the maximal frame prefix it consumes."""

    input_frame_count: int
    horizon_length: int
    source_stride: int
    model_window_length: int
    effective_frame_count: int
    windows: tuple[TemporalWindow, ...]

    @property
    def source_frame_indices(self) -> np.ndarray:
        return np.asarray(
            [window.source_frame_index for window in self.windows],
            dtype=np.int32,
        )

    @property
    def window_count(self) -> int:
        return len(self.windows)


@dataclass(frozen=True)
class SourceTracks:
    """Ragged per-source rows before concatenation into the NPZ payload."""

    query_uv: np.ndarray
    query_rgb: np.ndarray
    track_xyz: np.ndarray
    track_valid: np.ndarray
    confidence: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = build_common_parser()
    parser.description = "Track4World ego-centric tracking with one full-sequence cache"
    for action in parser._actions:
        if action.dest == "save_base_dir":
            action.default = "results/cat_ego"
            action.help = (
                "Output root; flows are written to 3d_ff_ego_output/flows.npz "
                "and the average camera intrinsics to camera_intrinsics.xml."
            )
    parser.add_argument(
        "--intrinsics_xml",
        "--intrinsics-xml",
        dest="intrinsics_xml",
        default=None,
        help=(
            "Path for the average camera-intrinsics XML. Defaults to "
            "<save_base_dir>/camera_intrinsics.xml."
        ),
    )
    parser.add_argument(
        "--H",
        "--horizon_length",
        "--horizon-length",
        dest="horizon_length",
        type=int,
        required=True,
        help="Required horizon length including its source; must be a multiple of W.",
    )
    parser.add_argument(
        "--S",
        "--window_stride",
        "--window-stride",
        dest="window_stride",
        type=int,
        required=True,
        help="Required stride between source frames; must satisfy 1 <= S <= H.",
    )
    return parser


def checkpoint_window_length(state_dict: Mapping[str, Any]) -> int:
    """Read W from both checkpoint temporal embeddings and validate agreement."""
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint must be a mapping of parameter names to tensors.")

    def embedding_length(name: str) -> int:
        matches = [
            value
            for key, value in state_dict.items()
            if key == name or key.endswith(f".{name}")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Checkpoint must contain exactly one {name!r} tensor, found "
                f"{len(matches)}."
            )
        tensor = matches[0]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError(
                f"Checkpoint {name} must have shape (1,W,C), got "
                f"{getattr(tensor, 'shape', None)}."
            )
        if tensor.shape[0] != 1 or tensor.shape[1] <= 0:
            raise ValueError(
                f"Checkpoint {name} must have shape (1,W,C), got {tuple(tensor.shape)}."
            )
        return int(tensor.shape[1])

    window_2d = embedding_length("time_emb")
    window_3d = embedding_length("time_emb3d")
    if window_2d != window_3d:
        raise ValueError(
            "Checkpoint time_emb and time_emb3d disagree on W: "
            f"{window_2d} vs {window_3d}."
        )
    if window_2d < 2 or window_2d % 2:
        raise ValueError(
            f"Checkpoint W must be even and at least 2, got W={window_2d}."
        )
    return window_2d


def validate_temporal_arg_values(horizon_length: int, window_stride: int) -> None:
    """Validate CLI values that do not depend on input T or checkpoint W."""
    if horizon_length < 3:
        raise ValueError(
            "--H/--horizon_length must be at least 3 for the legacy horizon "
            f"output contract, got {horizon_length}."
        )
    if window_stride < 1:
        raise ValueError(f"--S/--window_stride must be positive, got {window_stride}.")
    if window_stride > horizon_length:
        raise ValueError(
            "--S/--window_stride must not exceed --H/--horizon_length, got "
            f"S={window_stride}, H={horizon_length}."
        )


def plan_temporal_windows(
    frame_count: int,
    horizon_length: int,
    window_stride: int,
    model_window_length: int,
) -> TemporalPlan:
    """Return complete sources and T_eff for required H/S and checkpoint W."""
    validate_temporal_arg_values(horizon_length, window_stride)
    if frame_count > 128:
        raise ValueError(
            "Ego-centric input requires T <= 128; the T > 128 boundary is "
            f"unsupported, got T={frame_count}."
        )
    if frame_count < horizon_length:
        raise ValueError(
            f"Input frame count T={frame_count} is smaller than H={horizon_length}."
        )
    if model_window_length < 2 or model_window_length % 2:
        raise ValueError(
            f"Runtime checkpoint W must be even and at least 2, got {model_window_length}."
        )
    if horizon_length % model_window_length:
        raise ValueError(
            f"--H must be divisible by checkpoint W={model_window_length}, got "
            f"H={horizon_length}."
        )

    window_count = (frame_count - horizon_length) // window_stride + 1
    sources = [index * window_stride for index in range(window_count)]
    effective_frame_count = sources[-1] + horizon_length
    windows = tuple(
        TemporalWindow(
            index=index,
            source_frame_index=source,
            end_frame_index_exclusive=source + horizon_length,
        )
        for index, source in enumerate(sources)
    )
    return TemporalPlan(
        input_frame_count=frame_count,
        horizon_length=horizon_length,
        source_stride=window_stride,
        model_window_length=model_window_length,
        effective_frame_count=effective_frame_count,
        windows=windows,
    )


def _load_checkpoint_state(args: argparse.Namespace) -> Mapping[str, Any]:
    checkpoint_path = (
        Path(args.ckpt_init) if args.ckpt_init else DEFAULT_CHECKPOINTS[args.coordinate]
    )
    if checkpoint_path.is_file():
        logger.info("Loading checkpoint from %s.", checkpoint_path)
        try:
            return torch.load(
                str(checkpoint_path),
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except TypeError:
            return torch.load(str(checkpoint_path), map_location="cpu")

    logger.warning(
        "Checkpoint %s does not exist; downloading the matching pretrained model.",
        checkpoint_path,
    )
    return torch.hub.load_state_dict_from_url(
        CHECKPOINT_URLS[args.coordinate],
        map_location="cpu",
        check_hash=False,
    )


def load_ego_model(
    args: argparse.Namespace,
    config: dict[str, Any],
    device: torch.device,
) -> Track4World3DFFEgo:
    """Construct the model with W obtained solely from checkpoint buffers."""
    state_dict = _load_checkpoint_state(args)
    model_window_length = checkpoint_window_length(state_dict)
    logger.info("Initializing ego-centric model with checkpoint W=%d.", model_window_length)
    model = Track4World3DFFEgo(
        **config["model"],
        seqlen=model_window_length,
        use_3d=True,
        use_model="depthanythingv3",
    )
    model.load_pretrained_with_remap(state_dict)
    if model.seqlen != model_window_length:
        raise RuntimeError(
            f"Constructed model seqlen={model.seqlen}, expected W={model_window_length}."
        )
    if args.use_original_backbone:
        model.switch_to_original_backbone()
    model.to(device)
    model.requires_grad_(False)
    model.eval()
    model.use_metric_scale = bool(args.metric_scale)
    logger.info(
        "Metric-scale application is %s; raw DA3 scale is always saved.",
        "enabled" if model.use_metric_scale else "disabled",
    )
    return model


def _publish_directory_set(staged_to_final: Mapping[Path, Path]) -> None:
    """Transactionally replace a small set of sibling directories."""
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for final in staged_to_final.values():
            if final.exists() and not final.is_dir():
                raise NotADirectoryError(f"Expected output directory: {final}")
            if final.exists():
                backup = final.with_name(f".{final.name}.backup-{uuid.uuid4().hex}")
                final.replace(backup)
                backups[final] = backup
        for staged, final in staged_to_final.items():
            staged.replace(final)
            published.append(final)
    except Exception:
        for final in reversed(published):
            if final.exists():
                shutil.rmtree(final, ignore_errors=True)
        for final, backup in backups.items():
            if backup.exists() and not final.exists():
                backup.replace(final)
        raise
    for backup in backups.values():
        shutil.rmtree(backup, ignore_errors=True)


def save_truncated_preprocessed_inputs(
    rgb_tensor: torch.Tensor,
    dynamic_masks: torch.Tensor,
    save_base_dir: Path,
    fps: float,
) -> None:
    """Save exactly ``[0,T_eff)`` and replace both old directories as a unit."""
    if rgb_tensor.ndim != 5 or rgb_tensor.shape[0] != 1:
        raise ValueError(f"Expected RGB shape (1,T,3,H,W), got {rgb_tensor.shape}.")
    if dynamic_masks.shape[0] != rgb_tensor.shape[1]:
        raise ValueError(
            "RGB/mask frame counts differ: "
            f"{rgb_tensor.shape[1]} vs {dynamic_masks.shape[0]}."
        )
    save_base_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".preprocessed-tmp-", dir=save_base_dir)
    )
    try:
        save_preprocessed_inputs(rgb_tensor, dynamic_masks, staging, fps)
        _publish_directory_set(
            {
                staging / "final_rgb": save_base_dir / "final_rgb",
                staging / "final_dyn_mask": save_base_dir / "final_dyn_mask",
            }
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def truncate_effective_inputs(
    rgb_tensor: torch.Tensor,
    dynamic_masks: torch.Tensor,
    plan: TemporalPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous RGB/mask prefixes covering exactly ``T_eff`` frames."""
    if rgb_tensor.ndim != 5 or rgb_tensor.shape[0] != 1:
        raise ValueError(f"Expected RGB shape (1,T,3,H,W), got {rgb_tensor.shape}.")
    if rgb_tensor.shape[1] != plan.input_frame_count:
        raise ValueError(
            f"RGB T={rgb_tensor.shape[1]} differs from planned "
            f"T={plan.input_frame_count}."
        )
    if dynamic_masks.shape[0] != plan.input_frame_count:
        raise ValueError(
            f"Mask T={dynamic_masks.shape[0]} differs from planned "
            f"T={plan.input_frame_count}."
        )
    return (
        rgb_tensor[:, : plan.effective_frame_count].contiguous(),
        dynamic_masks[: plan.effective_frame_count].contiguous(),
    )


def _validate_camera_poses(camera_poses: np.ndarray) -> None:
    poses = np.asarray(camera_poses)
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"Expected C2W shape (T,4,4), got {poses.shape}.")
    if not np.isfinite(poses).all():
        raise ValueError("C2W contains NaN or infinity.")
    bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=poses.dtype)
    if not np.allclose(poses[:, 3], bottom, atol=1e-5, rtol=0.0):
        raise ValueError("C2W matrices do not have homogeneous bottom rows.")
    try:
        np.linalg.inv(poses.astype(np.float64))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Every C2W matrix must be invertible.") from exc


def transform_target_camera_to_source_camera(
    points: np.ndarray,
    source_c2w: np.ndarray,
    target_c2w: np.ndarray,
) -> np.ndarray:
    """Apply ``P_source^-1 P_target`` to target-camera XYZ rows."""
    vertices = np.asarray(points, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected point rows with shape (N,3), got {vertices.shape}.")
    transform = np.linalg.inv(np.asarray(source_c2w, dtype=np.float64)) @ np.asarray(
        target_c2w, dtype=np.float64
    )
    transformed = vertices @ transform[:3, :3].T + transform[:3, 3]
    return transformed.astype(np.float32, copy=False)


def _query_uv_from_source_mask(source_mask: np.ndarray) -> np.ndarray:
    height, width = source_mask.shape
    pixel_uv = utils3d.numpy.image_pixel(width=width, height=height)
    _, query_uv = utils3d.numpy.image_mesh(pixel_uv, mask=source_mask, tri=True)
    query_uv = np.asarray(query_uv, dtype=np.int32).reshape(-1, 2)
    return query_uv


def source_geometry_from_cache(
    cache: SequenceCache,
    source: int,
    *,
    metric_scale_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unpadded DA3 source XYZ/mask in the active scale state."""
    cache.validate()
    if not 0 <= source < cache.t_eff:
        raise ValueError(f"Source {source} is outside T_eff={cache.t_eff}.")
    points = cache.unpad(cache.points[source]).permute(1, 2, 0).float()
    if metric_scale_enabled:
        points = points * cache.metric_scale.float()
    masks = cache.unpad(cache.masks[source])[0] > cache.mask_threshold
    return points.cpu().numpy(), masks.cpu().numpy()


def scaled_c2w_from_cache(
    cache: SequenceCache,
    *,
    metric_scale_enabled: bool,
) -> np.ndarray:
    """Return all effective C2W poses in the same scale as saved XYZ."""
    poses = cache.camera_poses.reshape(cache.t_eff, 4, 4).float().clone()
    if metric_scale_enabled:
        poses[:, :3, 3] *= cache.metric_scale.float()
    poses_np = poses.cpu().numpy().astype(np.float32, copy=False)
    _validate_camera_poses(poses_np)
    return poses_np


def build_source_tracks(
    cache: SequenceCache,
    tracking_output: list[dict[str, torch.Tensor]],
    *,
    source: int,
    end: int,
    depth_edge_rtol: float,
    metric_scale_enabled: bool,
    source_rgb: torch.Tensor | np.ndarray,
) -> SourceTracks:
    """Convert one dense legacy horizon into ragged, validity-aware tracks.

    ``source_rgb`` is the already resized RGB frame used by the model.  Keeping
    the sampled color beside the query rows makes the resulting NPZ fully
    self-contained for visualization.
    """
    if len(tracking_output) != 2:
        raise ValueError("Legacy infer output must contain geometry and motion families.")
    geometry, motion = tracking_output
    horizon = end - source
    source_points, source_model_mask = source_geometry_from_cache(
        cache,
        source,
        metric_scale_enabled=metric_scale_enabled,
    )
    source_valid = _finite_depth_mask(source_points, depth_edge_rtol)
    query_uv = _query_uv_from_source_mask(source_model_mask & source_valid)
    query_count = query_uv.shape[0]

    source_rgb_array = (
        source_rgb.detach().cpu().numpy()
        if isinstance(source_rgb, torch.Tensor)
        else np.asarray(source_rgb)
    )
    if source_rgb_array.ndim != 3:
        raise ValueError(
            "source_rgb must have shape (3,H,W) or (H,W,3), got "
            f"{source_rgb_array.shape}."
        )
    if source_rgb_array.shape == (
        3,
        cache.image_height,
        cache.image_width,
    ):
        source_rgb_array = source_rgb_array.transpose(1, 2, 0)
    elif source_rgb_array.shape != (
        cache.image_height,
        cache.image_width,
        3,
    ):
        raise ValueError(
            "source_rgb dimensions do not match the encoded image grid: "
            f"got {source_rgb_array.shape}, expected "
            f"(3,{cache.image_height},{cache.image_width}) or "
            f"({cache.image_height},{cache.image_width},3)."
        )
    if not np.isfinite(source_rgb_array).all():
        raise ValueError("source_rgb must contain finite values.")
    source_rgb_array = np.clip(np.rint(source_rgb_array), 0, 255).astype(
        np.uint8, copy=False
    )
    u_rgb, v_rgb = query_uv[:, 0], query_uv[:, 1]
    query_rgb = source_rgb_array[v_rgb, u_rgb]
    if query_rgb.shape != (query_count, 3):
        raise RuntimeError(
            f"Sampled source RGB has shape {query_rgb.shape}; expected "
            f"({query_count},3)."
        )

    dense_tracks = motion["flow_3d"]
    confidence_maps = motion["visconf_maps_e"]
    expected_track_shape = (
        1,
        horizon,
        cache.image_height,
        cache.image_width,
        3,
    )
    expected_conf_shape = (
        1,
        horizon,
        2,
        cache.image_height,
        cache.image_width,
    )
    if tuple(dense_tracks.shape) != expected_track_shape:
        raise ValueError(
            f"Unexpected dense tracking shape {tuple(dense_tracks.shape)}; "
            f"expected {expected_track_shape}."
        )
    if tuple(confidence_maps.shape) != expected_conf_shape:
        raise ValueError(
            f"Unexpected confidence shape {tuple(confidence_maps.shape)}; "
            f"expected {expected_conf_shape}."
        )

    poses = geometry["camera_poses"].detach().float().cpu().numpy()
    if tuple(poses.shape) != (horizon, 4, 4):
        raise ValueError(f"Unexpected horizon C2W shape: {poses.shape}.")
    _validate_camera_poses(poses)
    confidence_dense = (
        confidence_maps[0].detach().float().cpu().numpy().transpose(0, 2, 3, 1)
    )
    dense_tracks_np = dense_tracks[0].detach().float().cpu().numpy()

    track_xyz = np.full((query_count, horizon, 3), np.nan, dtype=np.float32)
    track_valid = np.zeros((query_count, horizon), dtype=np.bool_)
    confidence = np.empty((query_count, horizon, 2), dtype=np.float32)
    u = query_uv[:, 0]
    v = query_uv[:, 1]

    # NPZ j=0 is authoritative DA3 source geometry, not the tracking head's
    # raw flow_000 prediction. Its confidence still comes from that head.
    track_xyz[:, 0] = source_points[v, u]
    track_valid[:, 0] = True
    confidence[:, 0] = confidence_dense[0, v, u]

    for local_target in range(1, horizon):
        target_map = dense_tracks_np[local_target]
        dense_valid = _finite_depth_mask(target_map, depth_edge_rtol)
        sampled_valid = dense_valid[v, u]
        sampled_points = target_map[v, u]
        if sampled_valid.any():
            track_xyz[sampled_valid, local_target] = (
                transform_target_camera_to_source_camera(
                    sampled_points[sampled_valid],
                    poses[0],
                    poses[local_target],
                )
            )
        track_valid[:, local_target] = sampled_valid
        # Confidence is retained even when geometry is invalid.
        confidence[:, local_target] = confidence_dense[local_target, v, u]

    return SourceTracks(
        query_uv=query_uv,
        query_rgb=query_rgb,
        track_xyz=track_xyz,
        track_valid=track_valid,
        confidence=confidence,
    )


def build_flows_payload(
    plan: TemporalPlan,
    cache: SequenceCache,
    source_tracks: list[SourceTracks],
    *,
    metric_scale_enabled: bool,
) -> dict[str, np.ndarray]:
    """Build and validate the non-pickled ragged ``flows.npz`` schema."""
    if metric_scale_enabled != cache.metric_scale_enabled:
        raise ValueError("Payload metric-scale state differs from SequenceCache.")
    if len(source_tracks) != plan.window_count:
        raise ValueError(
            f"Expected {plan.window_count} completed sources, got {len(source_tracks)}."
        )
    offsets = np.zeros(plan.window_count + 1, dtype=np.int64)
    for index, tracks in enumerate(source_tracks):
        offsets[index + 1] = offsets[index] + tracks.query_uv.shape[0]
    query_count = int(offsets[-1])

    query_uv = np.concatenate([tracks.query_uv for tracks in source_tracks], axis=0)
    query_rgb = np.concatenate([tracks.query_rgb for tracks in source_tracks], axis=0)
    # Time is the leading axis. query_offsets therefore delimit source-owned
    # column intervals [offset_i, offset_{i+1}) without Qmax padding.
    track_xyz = np.concatenate(
        [tracks.track_xyz.transpose(1, 0, 2) for tracks in source_tracks],
        axis=1,
    )
    track_valid = np.concatenate(
        [tracks.track_valid.T for tracks in source_tracks],
        axis=1,
    )
    confidence = np.concatenate(
        [tracks.confidence.transpose(1, 0, 2) for tracks in source_tracks],
        axis=1,
    )

    source_frame_index = plan.source_frame_indices
    target_frame_index = source_frame_index[:, None] + np.arange(
        plan.horizon_length, dtype=np.int32
    )[None]
    c2w = scaled_c2w_from_cache(
        cache,
        metric_scale_enabled=metric_scale_enabled,
    )
    payload = {
        "schema_version": np.asarray(FLOW_SCHEMA_VERSION, dtype="<U40"),
        "coordinate_system": np.asarray("source_camera", dtype="<U24"),
        "pixel_convention": np.asarray("unpadded_uv_integer", dtype="<U32"),
        "source_frame_index": source_frame_index.astype(np.int32, copy=False),
        "target_frame_index": target_frame_index.astype(np.int32, copy=False),
        "query_offsets": offsets,
        "query_uv": query_uv.astype(np.int32, copy=False),
        "query_rgb": query_rgb.astype(np.uint8, copy=False),
        "track_xyz": track_xyz.astype(np.float32, copy=False),
        "track_valid": track_valid.astype(np.bool_, copy=False),
        "confidence": confidence.astype(np.float32, copy=False),
        "ego_K": cache.ego_output_px.detach().float().cpu().numpy(),
        "metric_scale": np.asarray(
            cache.metric_scale.detach().float().cpu().item(), dtype=np.float32
        ),
        "metric_scale_enabled": np.asarray(metric_scale_enabled, dtype=np.bool_),
        "c2w": c2w,
        "input_frame_count": np.asarray(plan.input_frame_count, dtype=np.int32),
        "effective_frame_count": np.asarray(
            plan.effective_frame_count, dtype=np.int32
        ),
        "truncated_frame_start": np.asarray(
            plan.effective_frame_count, dtype=np.int32
        ),
        "horizon_length": np.asarray(plan.horizon_length, dtype=np.int32),
        "source_stride": np.asarray(plan.source_stride, dtype=np.int32),
        "model_window_length": np.asarray(
            plan.model_window_length, dtype=np.int32
        ),
        "image_size_hw": np.asarray(
            [cache.image_height, cache.image_width], dtype=np.int32
        ),
        "last_complete_source_frame_index": np.asarray(
            source_frame_index[-1], dtype=np.int32
        ),
    }
    validate_flows_payload(payload)
    if query_count != payload["track_xyz"].shape[1]:
        raise AssertionError("Ragged query count changed while building payload.")
    return payload


def validate_flows_payload(payload: Mapping[str, np.ndarray]) -> None:
    """Validate shapes, dtypes, ragged offsets, and j=0 conventions."""
    required = {
        "schema_version",
        "coordinate_system",
        "pixel_convention",
        "source_frame_index",
        "target_frame_index",
        "query_offsets",
        "query_uv",
        "query_rgb",
        "track_xyz",
        "track_valid",
        "confidence",
        "ego_K",
        "metric_scale",
        "metric_scale_enabled",
        "c2w",
        "input_frame_count",
        "effective_frame_count",
        "truncated_frame_start",
        "horizon_length",
        "source_stride",
        "model_window_length",
        "image_size_hw",
        "last_complete_source_frame_index",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"flows payload is missing fields: {sorted(missing)}.")
    for name, value in payload.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"flows field {name} is not a NumPy array.")
        if value.dtype.hasobject:
            raise TypeError(f"flows field {name} requires pickle/object storage.")

    sources = payload["source_frame_index"]
    targets = payload["target_frame_index"]
    offsets = payload["query_offsets"]
    query_uv = payload["query_uv"]
    query_rgb = payload["query_rgb"]
    tracks = payload["track_xyz"]
    valid = payload["track_valid"]
    confidence = payload["confidence"]
    horizon = int(payload["horizon_length"])
    source_count = sources.shape[0]
    if sources.dtype != np.int32 or sources.ndim != 1 or source_count < 1:
        raise ValueError("source_frame_index must be a non-empty int32 vector.")
    if targets.shape != (source_count, horizon) or targets.dtype != np.int32:
        raise ValueError("target_frame_index must have shape (M,H) and dtype int32.")
    if not np.array_equal(
        targets,
        sources[:, None] + np.arange(horizon, dtype=np.int32)[None],
    ):
        raise ValueError("target_frame_index[i,j] must equal source_frame_index[i]+j.")
    effective_frame_count = int(payload["effective_frame_count"])
    if np.any(targets < 0) or np.any(targets >= effective_frame_count):
        raise ValueError("target_frame_index contains an out-of-range frame.")
    if offsets.shape != (source_count + 1,) or offsets.dtype != np.int64:
        raise ValueError("query_offsets must be an int64 vector of shape (M+1,).")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("query_offsets must start at zero and be non-decreasing.")
    query_count = int(offsets[-1])
    if query_uv.shape != (query_count, 2) or query_uv.dtype != np.int32:
        raise ValueError("query_uv must have shape (Q,2) and dtype int32.")
    if query_rgb.shape != (query_count, 3) or query_rgb.dtype != np.uint8:
        raise ValueError("query_rgb must have shape (Q,3) and dtype uint8.")
    if tracks.shape != (horizon, query_count, 3) or tracks.dtype != np.float32:
        raise ValueError("track_xyz must have shape (H,Q,3) and dtype float32.")
    if valid.shape != (horizon, query_count) or valid.dtype != np.bool_:
        raise ValueError("track_valid must have shape (H,Q) and dtype bool.")
    if confidence.shape != (horizon, query_count, 2) or confidence.dtype != np.float32:
        raise ValueError("confidence must have shape (H,Q,2) and dtype float32.")
    if query_count and not valid[0].all():
        raise ValueError("Every selected query must have track_valid=True at j=0.")
    if np.any(np.isfinite(tracks[~valid])):
        raise ValueError("Invalid track_xyz entries must be NaN.")
    if not np.isfinite(tracks[valid]).all():
        raise ValueError("Valid track_xyz entries must be finite.")
    if payload["ego_K"].shape != (3, 3):
        raise ValueError("ego_K must have shape (3,3).")
    if payload["c2w"].shape != (effective_frame_count, 4, 4):
        raise ValueError("c2w must cover exactly T_eff frames.")
    if int(payload["truncated_frame_start"]) != int(
        payload["effective_frame_count"]
    ):
        raise ValueError("truncated_frame_start must always equal T_eff.")
    if payload["last_complete_source_frame_index"].dtype != np.int32:
        raise ValueError("last_complete_source_frame_index must be int32.")
    if int(payload["last_complete_source_frame_index"]) != int(sources[-1]):
        raise ValueError("last_complete_source_frame_index does not match the last source.")


def _intrinsics_to_numpy(intrinsics_px: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return a finite 3x3 intrinsic matrix in high-precision NumPy form."""
    if isinstance(intrinsics_px, torch.Tensor):
        intrinsics_array = intrinsics_px.detach().cpu().numpy()
    else:
        intrinsics_array = np.asarray(intrinsics_px)
    if intrinsics_array.shape != (3, 3):
        raise ValueError(
            "Camera intrinsics must have shape (3,3), got "
            f"{intrinsics_array.shape}."
        )
    if np.iscomplexobj(intrinsics_array):
        raise ValueError("Camera intrinsics must contain real values.")
    intrinsics_array = np.asarray(intrinsics_array, dtype=np.float64)
    if not np.isfinite(intrinsics_array).all():
        raise ValueError("Camera intrinsics contain NaN or infinity.")
    if intrinsics_array[0, 0] <= 0 or intrinsics_array[1, 1] <= 0:
        raise ValueError("Camera intrinsics must have positive focal lengths.")
    return intrinsics_array


def _positive_image_dimension(name: str, value: int) -> int:
    """Validate and normalize one image dimension for calibration metadata."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    try:
        normalized = int(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.") from exc
    if normalized <= 0 or normalized != value:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return normalized


def scale_intrinsics_to_original(
    intrinsics_px: np.ndarray | torch.Tensor,
    original_height: int,
    original_width: int,
    resized_height: int,
    resized_width: int,
) -> np.ndarray:
    """Convert model-grid K to the original image's pixel coordinate system.

    ``SequenceCache.ego_output_px`` is expressed on the unpadded, resized
    image grid.  Because the resize can be independently rounded to a 64-pixel
    multiple in each direction, focal lengths and principal-point coordinates
    use separate width/height scale factors:

    ``sx = original_width / resized_width`` and
    ``sy = original_height / resized_height``.

    The matrix follows the same integer-pixel convention as ``ego_K`` (the
    model's half-pixel correction has already been applied), so the conversion
    is the usual anisotropic left scaling ``diag(sx, sy, 1) @ K``.
    """
    original_height = _positive_image_dimension("original_height", original_height)
    original_width = _positive_image_dimension("original_width", original_width)
    resized_height = _positive_image_dimension("resized_height", resized_height)
    resized_width = _positive_image_dimension("resized_width", resized_width)
    intrinsics = _intrinsics_to_numpy(intrinsics_px)

    scale_x = original_width / resized_width
    scale_y = original_height / resized_height
    scaled = intrinsics.copy()
    scaled[0, :] *= scale_x
    scaled[1, :] *= scale_y
    return scaled


def _format_xml_float(value: float) -> str:
    """Format a calibration value without losing useful double precision."""
    formatted = format(float(value), ".17g")
    # OpenCV's XML reader distinguishes integer-looking tokens from floating
    # point values even when the matrix ``dt`` is ``d``.  Keep a decimal point
    # on integral values so ``FileStorage.getNode(...).mat()`` round-trips the
    # matrix as float64 rather than interpreting the token as raw integer data.
    if "." not in formatted and "e" not in formatted.lower():
        formatted += ".0"
    return formatted


def save_camera_intrinsics_xml(
    path: Path,
    intrinsics_px: np.ndarray | torch.Tensor,
    *,
    original_height: int,
    original_width: int,
    resized_height: int,
    resized_width: int,
) -> Path:
    """Atomically write the temporal-mean camera K as an OpenCV-style XML.

    ``intrinsics_px`` must be the average K on the resized model grid (the
    value stored in the NPZ ``ego_K`` field).  The matrix written to
    ``camera_matrix`` and to the scalar ``fx/fy/cx/cy`` fields is converted to
    original-image pixel units before serialization.  The XML uses the
    conventional ``opencv_storage`` root so it can be read by OpenCV's
    ``FileStorage`` as well as ordinary XML parsers.
    """
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(
            f"Camera-intrinsics XML path is a directory: {output_path}"
        )
    original_height = _positive_image_dimension("original_height", original_height)
    original_width = _positive_image_dimension("original_width", original_width)
    resized_height = _positive_image_dimension("resized_height", resized_height)
    resized_width = _positive_image_dimension("resized_width", resized_width)
    intrinsics_original_px = scale_intrinsics_to_original(
        intrinsics_px,
        original_height,
        original_width,
        resized_height,
        resized_width,
    )

    scale_x = original_width / resized_width
    scale_y = original_height / resized_height
    root = ET.Element("opencv_storage")
    ET.SubElement(root, "schema_version").text = (
        "track4world.camera_intrinsics.v1"
    )
    ET.SubElement(root, "calibration_type").text = "temporal_mean"
    ET.SubElement(root, "pixel_units").text = "original_image_pixels"
    ET.SubElement(root, "pixel_convention").text = "unpadded_uv_integer"

    # Keep both the concise OpenCV-style names and explicit names so that the
    # file remains self-describing when consumed without OpenCV.
    ET.SubElement(root, "image_width").text = str(original_width)
    ET.SubElement(root, "image_height").text = str(original_height)
    ET.SubElement(root, "original_image_width").text = str(original_width)
    ET.SubElement(root, "original_image_height").text = str(original_height)
    ET.SubElement(root, "resized_width").text = str(resized_width)
    ET.SubElement(root, "resized_height").text = str(resized_height)
    ET.SubElement(root, "scale_x").text = _format_xml_float(scale_x)
    ET.SubElement(root, "scale_y").text = _format_xml_float(scale_y)

    fx, fy = intrinsics_original_px[0, 0], intrinsics_original_px[1, 1]
    cx, cy = intrinsics_original_px[0, 2], intrinsics_original_px[1, 2]
    for name, value in (("fx", fx), ("fy", fy), ("cx", cx), ("cy", cy)):
        ET.SubElement(root, name).text = _format_xml_float(value)

    matrix = ET.SubElement(root, "camera_matrix", {"type_id": "opencv-matrix"})
    ET.SubElement(matrix, "rows").text = "3"
    ET.SubElement(matrix, "cols").text = "3"
    ET.SubElement(matrix, "dt").text = "d"
    ET.SubElement(matrix, "data").text = " ".join(
        _format_xml_float(value) for value in intrinsics_original_px.reshape(-1)
    )

    tree = ET.ElementTree(root)
    # ElementTree.indent is available in the supported Python versions; retain
    # a small fallback for environments that import this script with older
    # stdlib versions.
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.tmp-",
            dir=output_path.parent,
            delete=False,
        ) as stream:
            staging_path = Path(stream.name)
            tree.write(stream, encoding="utf-8", xml_declaration=True)
            stream.flush()
        staging_path.replace(output_path)
    finally:
        if staging_path is not None and staging_path.exists():
            try:
                staging_path.unlink()
            except OSError:
                # The destination has already been replaced in the successful
                # case; a best-effort cleanup must not hide a serialization
                # error in the exceptional case.
                pass

    return output_path


# A descriptive alias is useful to callers that prefer a verb-style name.
write_camera_intrinsics_xml = save_camera_intrinsics_xml


def save_flows_npz(output_dir: Path, payload: Mapping[str, np.ndarray]) -> None:
    """Write one verified NPZ and atomically replace the prior 3D output."""
    validate_flows_payload(payload)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        np.savez_compressed(staging / FLOWS_FILENAME, **payload)
        with np.load(staging / FLOWS_FILENAME, allow_pickle=False) as loaded:
            round_trip = {name: loaded[name] for name in loaded.files}
        validate_flows_payload(round_trip)
        if set(round_trip) != set(payload):
            raise RuntimeError("flows.npz field set changed during serialization.")
        _publish_output_directory(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    validate_temporal_arg_values(args.horizon_length, args.window_stride)
    if args.coordinate != "world_depthanythingv3":
        raise ValueError("The ego-centric demo requires --coordinate world_depthanythingv3.")
    if not torch.cuda.is_available():
        raise RuntimeError("Track4World ego-centric inference requires a CUDA-capable GPU.")

    video_path = Path(args.mp4_path)
    rgb_dir = Path(args.rgb_dir) if args.rgb_dir is not None else None
    config_path = Path(args.config_path)
    if rgb_dir is not None and not rgb_dir.is_dir():
        raise NotADirectoryError(f"RGB image directory not found: {rgb_dir}")
    if rgb_dir is None and not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    save_base_dir = Path(args.save_base_dir)
    save_base_dir.mkdir(parents=True, exist_ok=True)
    if rgb_dir is None:
        copy_input_video(video_path, save_base_dir)
    else:
        logger.info("Using RGB image sequence from %s.", rgb_dir)

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    (
        rgb_tensor,
        dynamic_masks,
        fps,
        original_image_size,
    ) = prepare_inputs(args, return_original_size=True)
    original_height, original_width = original_image_size
    device = torch.device("cuda:0")
    model = load_ego_model(args, config, device)
    plan = plan_temporal_windows(
        int(rgb_tensor.shape[1]),
        args.horizon_length,
        args.window_stride,
        model.seqlen,
    )
    t_eff = plan.effective_frame_count
    rgb_tensor, dynamic_masks = truncate_effective_inputs(
        rgb_tensor,
        dynamic_masks,
        plan,
    )
    save_truncated_preprocessed_inputs(rgb_tensor, dynamic_masks, save_base_dir, fps)

    rgb_gpu = rgb_tensor.to(device=device, dtype=torch.float16, non_blocking=True)
    source_results: list[SourceTracks] = []
    start_time = time.perf_counter()
    intrinsics_xml_arg = getattr(args, "intrinsics_xml", None)
    intrinsics_xml_path = (
        Path(intrinsics_xml_arg)
        if intrinsics_xml_arg
        else save_base_dir / INTRINSICS_FILENAME
    )
    try:
        logger.info(
            "Encoding T_eff=%d real frames once; W=%d, H=%d, S=%d, sources=%d.",
            t_eff,
            model.seqlen,
            plan.horizon_length,
            plan.source_stride,
            plan.window_count,
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16
        ):
            cache = model.encode_sequence(rgb_gpu)
            for window in plan.windows:
                logger.info(
                    "Running legacy tracking head for source %d (%d/%d).",
                    window.source_frame_index,
                    window.index + 1,
                    plan.window_count,
                )
                output = model.track_cached_window(
                    cache,
                    rgb_gpu,
                    window.source_frame_index,
                    window.end_frame_index_exclusive,
                    args.inference_iters,
                )
                source_results.append(
                    build_source_tracks(
                        cache,
                        output,
                        source=window.source_frame_index,
                        end=window.end_frame_index_exclusive,
                        depth_edge_rtol=args.depth_edge_rtol,
                        metric_scale_enabled=model.use_metric_scale,
                        source_rgb=rgb_tensor[0, window.source_frame_index],
                    )
                )
                del output
                torch.cuda.empty_cache()

        payload = build_flows_payload(
            plan,
            cache,
            source_results,
            metric_scale_enabled=model.use_metric_scale,
        )
        save_flows_npz(save_base_dir / OUTPUT_DIRECTORY_NAME, payload)
        # ``ego_output_px`` is the temporal mean K used by every source and
        # target.  Serialize that same matrix separately after converting it
        # from the resized model grid to the original image grid.
        save_camera_intrinsics_xml(
            intrinsics_xml_path,
            cache.ego_output_px,
            original_height=original_height,
            original_width=original_width,
            resized_height=cache.image_height,
            resized_width=cache.image_width,
        )
        logger.info(
            "Saved temporal-mean camera intrinsics in original pixel units "
            "for image (%d,%d) to %s.",
            original_height,
            original_width,
            intrinsics_xml_path,
        )
    finally:
        del model
        torch.cuda.empty_cache()

    logger.info(
        "Ego-centric cache run completed in %.2f seconds; wrote %s and %s.",
        time.perf_counter() - start_time,
        save_base_dir / OUTPUT_DIRECTORY_NAME / FLOWS_FILENAME,
        intrinsics_xml_path,
    )


if __name__ == "__main__":
    main()
