"""Interactive visualization for the ego-centric 3D-FF NPZ output.

The ego-centric demo stores every source horizon in one ragged ``flows.npz``.
Tracks are expressed in the source-camera coordinate system; this viewer
transforms the selected horizon to the common world frame using that source's
``c2w`` pose.  Point colors come from ``query_rgb`` and trajectory colors are
kept separate so paths remain easy to distinguish from the point cloud.

The renderer intentionally imports :mod:`viser` only from :func:`main`.  This
keeps the schema and geometry helpers usable in headless environments (and in
unit tests) where the optional interactive dependency is not installed.
"""

from __future__ import annotations

import argparse
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


FLOW_SCHEMA_VERSION = "track4world.ego_flows.v1"
DEFAULT_NPZ_PATH = "results/cat_ego/3d_ff_ego_output/flows.npz"
DEFAULT_MAX_RENDER_POINTS = 12000
DEFAULT_MAX_TRAJECTORY_SEGMENTS = 30000
DEFAULT_TRAJECTORY_DOWNSAMPLE = 50
DEFAULT_TRAIL_LENGTH = 5
DEFAULT_MAX_SEGMENT_DISPLACEMENT = 3.0
DEFAULT_LINE_WIDTH = 3.0
DEFAULT_POINT_SIZE = 0.01


@dataclass(frozen=True)
class FlowData:
    """Validated arrays needed by the renderer."""

    source_frame_index: np.ndarray
    target_frame_index: np.ndarray
    query_offsets: np.ndarray
    query_uv: np.ndarray
    query_rgb: np.ndarray
    track_xyz: np.ndarray
    track_valid: np.ndarray
    confidence: np.ndarray
    c2w: np.ndarray
    horizon_length: int
    effective_frame_count: int
    image_size_hw: tuple[int, int]

    @property
    def source_count(self) -> int:
        return int(self.source_frame_index.shape[0])

    @property
    def query_count(self) -> int:
        return int(self.query_uv.shape[0])


@dataclass
class SourceCache:
    """Derived arrays for one source, materialized lazily."""

    source_index: int
    source_frame: int
    world_xyz: np.ndarray
    query_rgb: np.ndarray
    confidence_mean: np.ndarray
    valid: np.ndarray
    trajectory_colors: np.ndarray | None = None

    @property
    def query_count(self) -> int:
        return int(self.world_xyz.shape[1])


def _hsv_to_rgb_uint8(
    hues: np.ndarray,
    *,
    saturation: float = 0.9,
    value: float = 1.0,
) -> np.ndarray:
    """Convert a vector of HSV hues to bright uint8 RGB colors."""

    hue_array = np.asarray(hues, dtype=np.float64).reshape(-1)
    if not (0.0 <= saturation <= 1.0 and 0.0 <= value <= 1.0):
        raise ValueError("HSV saturation and value must be within [0,1].")
    if hue_array.size == 0:
        return np.empty((0, 3), dtype=np.uint8)

    h6 = (hue_array % 1.0) * 6.0
    sector = np.floor(h6).astype(np.int64) % 6
    fraction = h6 - np.floor(h6)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))

    rgb = np.empty((hue_array.size, 3), dtype=np.float64)
    rgb[:, 0] = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4],
        [value, q, p, p, t],
        default=value,
    )
    rgb[:, 1] = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4],
        [t, value, value, q, p],
        default=p,
    )
    rgb[:, 2] = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4],
        [p, p, t, value, value],
        default=q,
    )
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def build_trajectory_colors(query_uv: np.ndarray, source_index: int) -> np.ndarray:
    """Build a deterministic high-contrast color for every query track."""

    uv = np.asarray(query_uv)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"query_uv must have shape (Q,2), got {uv.shape}.")
    if source_index < 0:
        raise ValueError(f"source_index must be non-negative, got {source_index}.")
    count = int(uv.shape[0])
    if count == 0:
        return np.empty((0, 3), dtype=np.uint8)

    uv_float = uv.astype(np.float64, copy=False)
    uv_min = uv_float.min(axis=0)
    uv_span = np.ptp(uv_float, axis=0)
    uv_norm = (uv_float - uv_min) / np.maximum(uv_span, 1.0)
    spatial_phase = (0.73 * uv_norm[:, 0] + 0.27 * uv_norm[:, 1]) % 1.0
    indices = np.arange(count, dtype=np.float64)
    golden_phase = (indices * 0.6180339887498949) % 1.0
    hues = (0.55 * spatial_phase + 0.45 * golden_phase + 0.137 * source_index) % 1.0
    return _hsv_to_rgb_uint8(hues, saturation=0.92, value=1.0)


def _scalar_text(value: np.ndarray, name: str) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar string.")
        if value.dtype.kind not in "US":
            raise ValueError(f"{name} must use a Unicode/string dtype.")
        return str(value.item())
    if isinstance(value, str):
        return value
    raise ValueError(f"{name} must be a scalar string.")


def _scalar_int(payload: dict[str, np.ndarray], name: str) -> int:
    value = payload[name]
    if value.ndim != 0 or value.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer scalar.")
    return int(value)


def _validate_c2w(c2w: np.ndarray, frame_count: int) -> None:
    if c2w.shape != (frame_count, 4, 4):
        raise ValueError(f"c2w must have shape ({frame_count},4,4), got {c2w.shape}.")
    if c2w.dtype.kind != "f" or not np.isfinite(c2w).all():
        raise ValueError("c2w must contain finite floating-point values.")
    bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(c2w[:, 3, :], bottom, atol=1e-5, rtol=0.0):
        raise ValueError("c2w matrices must have homogeneous bottom rows.")
    try:
        np.linalg.inv(c2w.astype(np.float64, copy=False))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Every c2w matrix must be invertible.") from exc


def load_flows(path: str | Path) -> FlowData:
    """Load and validate an ego-centric ``flows.npz`` without pickle."""

    flows_path = Path(path)
    if not flows_path.is_file():
        raise FileNotFoundError(f"flows.npz not found: {flows_path}")

    with np.load(flows_path, allow_pickle=False) as loaded:
        payload = {name: loaded[name] for name in loaded.files}

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
        "c2w",
        "horizon_length",
        "effective_frame_count",
        "image_size_hw",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"flows.npz is missing fields: {sorted(missing)}.")
    if any(array.dtype.hasobject for array in payload.values()):
        raise ValueError("flows.npz must not contain object arrays.")

    if _scalar_text(payload["schema_version"], "schema_version") != FLOW_SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version; expected {FLOW_SCHEMA_VERSION!r}.")
    if _scalar_text(payload["coordinate_system"], "coordinate_system") != "source_camera":
        raise ValueError("The viewer requires coordinate_system='source_camera'.")
    if _scalar_text(payload["pixel_convention"], "pixel_convention") != "unpadded_uv_integer":
        raise ValueError("The viewer requires pixel_convention='unpadded_uv_integer'.")

    sources = payload["source_frame_index"]
    targets = payload["target_frame_index"]
    offsets = payload["query_offsets"]
    query_uv = payload["query_uv"]
    query_rgb = payload["query_rgb"]
    track_xyz = payload["track_xyz"]
    track_valid = payload["track_valid"]
    confidence = payload["confidence"]

    if sources.ndim != 1 or sources.dtype != np.int32 or sources.size == 0:
        raise ValueError("source_frame_index must be a non-empty int32 vector.")
    source_count = int(sources.size)
    horizon = _scalar_int(payload, "horizon_length")
    effective_frame_count = _scalar_int(payload, "effective_frame_count")
    if horizon < 1 or effective_frame_count < 1:
        raise ValueError("horizon_length and effective_frame_count must be positive.")

    if targets.shape != (source_count, horizon) or targets.dtype != np.int32:
        raise ValueError("target_frame_index must have shape (M,H) and dtype int32.")
    expected_targets = sources[:, None] + np.arange(horizon, dtype=np.int32)[None, :]
    if not np.array_equal(targets, expected_targets):
        raise ValueError("target_frame_index[i,j] must equal source_frame_index[i]+j.")
    if np.any(targets < 0) or np.any(targets >= effective_frame_count):
        raise ValueError("target_frame_index contains an out-of-range frame.")

    if offsets.shape != (source_count + 1,) or offsets.dtype != np.int64:
        raise ValueError("query_offsets must have shape (M+1,) and dtype int64.")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("query_offsets must start at zero and be non-decreasing.")
    query_count = int(offsets[-1])

    if query_uv.shape != (query_count, 2) or query_uv.dtype != np.int32:
        raise ValueError("query_uv must have shape (Q,2) and dtype int32.")
    if query_rgb.shape != (query_count, 3) or query_rgb.dtype != np.uint8:
        raise ValueError("query_rgb must have shape (Q,3) and dtype uint8.")
    if track_xyz.shape != (horizon, query_count, 3) or track_xyz.dtype != np.float32:
        raise ValueError("track_xyz must have shape (H,Q,3) and dtype float32.")
    if track_valid.shape != (horizon, query_count) or track_valid.dtype != np.bool_:
        raise ValueError("track_valid must have shape (H,Q) and dtype bool.")
    if confidence.shape != (horizon, query_count, 2) or confidence.dtype != np.float32:
        raise ValueError("confidence must have shape (H,Q,2) and dtype float32.")
    if query_count and not track_valid[0].all():
        raise ValueError("Every selected query must be valid at timestep zero.")
    if np.any(np.isfinite(track_xyz[~track_valid])):
        raise ValueError("Invalid track_xyz entries must be NaN.")
    if not np.isfinite(track_xyz[track_valid]).all():
        raise ValueError("Valid track_xyz entries must be finite.")

    image_size = payload["image_size_hw"]
    if image_size.shape != (2,) or image_size.dtype != np.int32:
        raise ValueError("image_size_hw must have shape (2,) and dtype int32.")
    if np.any(image_size <= 0):
        raise ValueError("image_size_hw must be positive.")
    if query_count:
        if np.any(query_uv[:, 0] < 0) or np.any(query_uv[:, 0] >= image_size[1]):
            raise ValueError("query_uv contains a u coordinate outside image_size_hw.")
        if np.any(query_uv[:, 1] < 0) or np.any(query_uv[:, 1] >= image_size[0]):
            raise ValueError("query_uv contains a v coordinate outside image_size_hw.")

    c2w = payload["c2w"]
    _validate_c2w(c2w, effective_frame_count)
    if np.any(sources < 0) or np.any(sources >= effective_frame_count):
        raise ValueError("source_frame_index contains an out-of-range frame.")

    return FlowData(
        source_frame_index=sources,
        target_frame_index=targets,
        query_offsets=offsets,
        query_uv=query_uv,
        query_rgb=query_rgb,
        track_xyz=track_xyz,
        track_valid=track_valid,
        confidence=confidence,
        c2w=c2w.astype(np.float32, copy=False),
        horizon_length=horizon,
        effective_frame_count=effective_frame_count,
        image_size_hw=(int(image_size[0]), int(image_size[1])),
    )


def source_query_slice(data: FlowData, source_index: int) -> slice:
    """Return the ragged query-column slice owned by a source ordinal."""

    if not (0 <= source_index < data.source_count):
        raise IndexError(
            f"source ordinal {source_index} is outside [0,{data.source_count})."
        )
    return slice(
        int(data.query_offsets[source_index]),
        int(data.query_offsets[source_index + 1]),
    )


def build_source_cache(data: FlowData, source_index: int) -> SourceCache:
    """Materialize invariant display data for one source."""

    columns = source_query_slice(data, source_index)
    source_frame = int(data.source_frame_index[source_index])
    xyz_source = data.track_xyz[:, columns]
    pose = data.c2w[source_frame]
    world_xyz = np.empty(xyz_source.shape, dtype=np.float32)
    np.einsum("ij,tqj->tqi", pose[:3, :3], xyz_source, out=world_xyz)
    world_xyz += pose[:3, 3]
    query_rgb = data.query_rgb[columns]
    confidence_mean = data.confidence[:, columns].mean(axis=-1)
    valid = data.track_valid[:, columns].copy()
    valid &= np.isfinite(world_xyz).all(axis=-1)
    valid &= np.isfinite(confidence_mean)
    trajectory_colors = build_trajectory_colors(data.query_uv[columns], source_index)
    return SourceCache(
        source_index=source_index,
        source_frame=source_frame,
        world_xyz=world_xyz,
        query_rgb=query_rgb,
        confidence_mean=confidence_mean.astype(np.float32, copy=False),
        valid=valid,
        trajectory_colors=trajectory_colors,
    )


def source_world_tracks(data: FlowData, source_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return selected source tracks in world coordinates and source RGB."""

    cache = build_source_cache(data, source_index)
    return cache.world_xyz, cache.query_rgb


def _sample_indices(cache: SourceCache, downsample: int) -> np.ndarray:
    indices = np.arange(cache.query_count, dtype=np.int64)
    return indices[:: max(1, int(downsample))]


def resolve_downsample(
    query_count: int,
    requested_step: int,
    max_render_points: int = DEFAULT_MAX_RENDER_POINTS,
) -> int:
    """Choose a step honoring the user request and point-count budget."""

    if query_count < 0:
        raise ValueError(f"query_count must be non-negative, got {query_count}.")
    if requested_step < 1:
        raise ValueError(f"requested_step must be positive, got {requested_step}.")
    if max_render_points < 1:
        raise ValueError(f"max_render_points must be positive, got {max_render_points}.")
    automatic_step = max(1, (query_count + max_render_points - 1) // max_render_points)
    return max(int(requested_step), automatic_step)


def resolve_trajectory_downsample(
    query_count: int,
    requested_step: int,
    horizon_length: int,
    max_segments: int,
) -> int:
    """Bound full-horizon line count while honoring the requested step."""

    if query_count < 0:
        raise ValueError(f"query_count must be non-negative, got {query_count}.")
    if requested_step < 1:
        raise ValueError(f"requested_step must be positive, got {requested_step}.")
    if horizon_length < 1:
        raise ValueError(f"horizon_length must be positive, got {horizon_length}.")
    if max_segments < 1:
        raise ValueError(f"max_segments must be positive, got {max_segments}.")
    transitions = max(1, horizon_length - 1)
    automatic_step = max(
        1, (query_count * transitions + max_segments - 1) // max_segments
    )
    return max(int(requested_step), automatic_step)


def _confidence_mask_from_cache(
    cache: SourceCache,
    timestep: int,
    confidence_threshold: float,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = _sample_indices(cache, downsample)
    valid = cache.valid[timestep, indices].copy()
    valid &= cache.confidence_mean[timestep, indices] >= confidence_threshold
    return indices, valid


def build_point_cloud(
    data: FlowData,
    source_index: int,
    timestep: int,
    confidence_threshold: float,
    downsample: int,
    source_cache: SourceCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the filtered world-space point cloud for one GUI state."""

    if not (0 <= timestep < data.horizon_length):
        raise IndexError(f"timestep {timestep} is outside the horizon.")
    cache = source_cache if source_cache is not None else build_source_cache(data, source_index)
    if cache.source_index != source_index:
        raise ValueError("source_cache belongs to a different source.")
    indices, valid = _confidence_mask_from_cache(
        cache, timestep, confidence_threshold, downsample
    )
    return cache.world_xyz[timestep, indices][valid], cache.query_rgb[indices][valid]


def build_track_heads(
    data: FlowData,
    source_index: int,
    timestep: int,
    confidence_threshold: float,
    downsample: int,
    source_cache: SourceCache | None = None,
    trajectory_colors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return colored current endpoints for the sampled trajectories.

    The endpoint overlay is separate from the RGB point cloud, making the
    moving end of each path easy to identify when the cloud is opaque.
    """

    if not (0 <= timestep < data.horizon_length):
        raise IndexError(f"timestep {timestep} is outside the horizon.")
    cache = source_cache if source_cache is not None else build_source_cache(data, source_index)
    if cache.source_index != source_index:
        raise ValueError("source_cache belongs to a different source.")
    indices, valid = _confidence_mask_from_cache(
        cache, timestep, confidence_threshold, downsample
    )
    base_colors = trajectory_colors
    if base_colors is None:
        base_colors = cache.trajectory_colors
    if base_colors is None:
        base_colors = build_trajectory_colors(
            data.query_uv[source_query_slice(data, source_index)],
            source_index=source_index,
        )
    base_colors = np.asarray(base_colors)
    if base_colors.shape != cache.query_rgb.shape or base_colors.dtype != np.uint8:
        raise ValueError(
            "trajectory_colors must have the same shape as query_rgb and dtype uint8."
        )
    selected = indices[valid]
    return cache.world_xyz[timestep, selected], base_colors[selected]


def _build_segment_for_step(
    cache: SourceCache,
    current: int,
    confidence_threshold: float,
    downsample: int,
    trajectory_colors: np.ndarray | None = None,
    directional_colors: bool = False,
    max_segment_displacement: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build only the adjacent segment for one newly reached timestep."""

    if current <= 0 or current >= cache.world_xyz.shape[0]:
        raise IndexError(f"segment timestep {current} is outside the horizon.")
    indices = _sample_indices(cache, downsample)
    valid = cache.valid[current - 1, indices] & cache.valid[current, indices]
    valid &= cache.confidence_mean[current - 1, indices] >= confidence_threshold
    valid &= cache.confidence_mean[current, indices] >= confidence_threshold
    if max_segment_displacement is not None:
        if max_segment_displacement <= 0:
            raise ValueError("max_segment_displacement must be positive when set.")
        displacement = np.linalg.norm(
            cache.world_xyz[current, indices] - cache.world_xyz[current - 1, indices],
            axis=-1,
        )
        valid &= displacement <= max_segment_displacement
    if not valid.any():
        return np.empty((0, 2, 3), dtype=np.float32), np.empty((0, 2, 3), dtype=np.uint8)

    local_indices = indices[valid]
    segments = np.stack(
        [cache.world_xyz[current - 1, local_indices], cache.world_xyz[current, local_indices]],
        axis=1,
    ).astype(np.float32, copy=False)
    if trajectory_colors is None:
        base_colors = cache.query_rgb
    else:
        base_colors = np.asarray(trajectory_colors)
        if base_colors.shape != cache.query_rgb.shape or base_colors.dtype != np.uint8:
            raise ValueError(
                "trajectory_colors must have the same shape as query_rgb and dtype uint8."
            )
    selected_colors = base_colors[local_indices]
    if directional_colors:
        tail_colors = np.clip(selected_colors.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
        colors = np.stack([tail_colors, selected_colors], axis=1)
    else:
        colors = np.repeat(selected_colors[:, None, :], 2, axis=1)
    return segments, colors


def build_trail_segments(
    data: FlowData,
    source_index: int,
    timestep: int,
    confidence_threshold: float,
    downsample: int,
    trail_length: int,
    source_cache: SourceCache | None = None,
    trajectory_colors: np.ndarray | None = None,
    directional_colors: bool = False,
    max_segment_displacement: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build adjacent world-space line segments ending at ``timestep``."""

    if timestep <= 0:
        return np.empty((0, 2, 3), dtype=np.float32), np.empty((0, 2, 3), dtype=np.uint8)
    if timestep >= data.horizon_length:
        raise IndexError(f"timestep {timestep} is outside the horizon.")
    cache = source_cache if source_cache is not None else build_source_cache(data, source_index)
    if cache.source_index != source_index:
        raise ValueError("source_cache belongs to a different source.")

    segments: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    first = max(1, timestep - max(1, int(trail_length)) + 1)
    for current in range(first, timestep + 1):
        step_segments, step_colors = _build_segment_for_step(
            cache,
            current,
            confidence_threshold,
            downsample,
            trajectory_colors=trajectory_colors,
            directional_colors=directional_colors,
            max_segment_displacement=max_segment_displacement,
        )
        segments.append(step_segments)
        colors.append(step_colors)
    if not segments:
        return np.empty((0, 2, 3), dtype=np.float32), np.empty((0, 2, 3), dtype=np.uint8)
    return np.concatenate(segments, axis=0).astype(np.float32, copy=False), np.concatenate(colors, axis=0)


def build_full_track_segments(
    data: FlowData,
    source_index: int,
    confidence_threshold: float,
    downsample: int,
    source_cache: SourceCache | None = None,
    trajectory_colors: np.ndarray | None = None,
    directional_colors: bool = False,
    max_segment_displacement: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build all adjacent segments in a source horizon."""

    return build_trail_segments(
        data,
        source_index,
        data.horizon_length - 1,
        confidence_threshold,
        downsample,
        data.horizon_length,
        source_cache=source_cache,
        trajectory_colors=trajectory_colors,
        directional_colors=directional_colors,
        max_segment_displacement=max_segment_displacement,
    )


@dataclass
class TrailAccumulator:
    """Incrementally maintain the visible trail for sequential playback."""

    _cache: SourceCache | None = None
    _timestep: int | None = None
    _confidence_threshold: float | None = None
    _downsample: int | None = None
    _trail_length: int | None = None
    _trajectory_colors: np.ndarray | None = None
    _directional_colors: bool = False
    _max_segment_displacement: float | None = None
    _segments: deque[np.ndarray] = field(default_factory=deque)
    _colors: deque[np.ndarray] = field(default_factory=deque)
    _flat_segments: np.ndarray | None = None
    _flat_colors: np.ndarray | None = None
    rebuild_count: int = 0
    append_count: int = 0

    @staticmethod
    def _empty() -> tuple[np.ndarray, np.ndarray]:
        return np.empty((0, 2, 3), dtype=np.float32), np.empty((0, 2, 3), dtype=np.uint8)

    def _invalidate_flat(self) -> None:
        self._flat_segments = None
        self._flat_colors = None

    def _append(
        self,
        cache: SourceCache,
        current: int,
        confidence_threshold: float,
        downsample: int,
        trajectory_colors: np.ndarray | None,
        directional_colors: bool,
        max_segment_displacement: float | None,
    ) -> None:
        segments, colors = _build_segment_for_step(
            cache,
            current,
            confidence_threshold,
            downsample,
            trajectory_colors=trajectory_colors,
            directional_colors=directional_colors,
            max_segment_displacement=max_segment_displacement,
        )
        self._segments.append(segments)
        self._colors.append(colors)
        self.append_count += 1
        self._invalidate_flat()

    def _flatten(self) -> tuple[np.ndarray, np.ndarray]:
        if self._flat_segments is not None and self._flat_colors is not None:
            return self._flat_segments, self._flat_colors
        nonempty_segments = [item for item in self._segments if item.shape[0]]
        nonempty_colors = [item for item in self._colors if item.shape[0]]
        if not nonempty_segments:
            self._flat_segments, self._flat_colors = self._empty()
        else:
            self._flat_segments = np.concatenate(nonempty_segments, axis=0)
            self._flat_colors = np.concatenate(nonempty_colors, axis=0)
        return self._flat_segments, self._flat_colors

    def update(
        self,
        cache: SourceCache,
        timestep: int,
        confidence_threshold: float,
        downsample: int,
        trail_length: int,
        trajectory_colors: np.ndarray | None = None,
        directional_colors: bool = False,
        max_segment_displacement: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update incrementally when possible, rebuilding on discontinuity."""

        horizon = cache.world_xyz.shape[0]
        if not (0 <= timestep < horizon):
            raise IndexError(f"timestep {timestep} is outside the horizon.")
        threshold = float(confidence_threshold)
        step = max(1, int(downsample))
        length = max(1, int(trail_length))
        jump_limit = None if max_segment_displacement is None else float(max_segment_displacement)
        if jump_limit is not None and jump_limit <= 0:
            raise ValueError("max_segment_displacement must be positive when set.")

        same_configuration = (
            self._cache is cache
            and self._confidence_threshold == threshold
            and self._downsample == step
            and self._trail_length == length
            and self._trajectory_colors is trajectory_colors
            and self._directional_colors == bool(directional_colors)
            and self._max_segment_displacement == jump_limit
        )
        sequential = same_configuration and self._timestep is not None and timestep == self._timestep + 1
        unchanged = same_configuration and timestep == self._timestep
        if unchanged:
            return self._flatten()

        if not sequential:
            self._cache = cache
            self._confidence_threshold = threshold
            self._downsample = step
            self._trail_length = length
            self._trajectory_colors = trajectory_colors
            self._directional_colors = bool(directional_colors)
            self._max_segment_displacement = jump_limit
            self._segments = deque(maxlen=length)
            self._colors = deque(maxlen=length)
            self._timestep = None
            self._invalidate_flat()
            self.rebuild_count += 1
            first = max(1, timestep - length + 1)
            for current in range(first, timestep + 1):
                self._append(
                    cache,
                    current,
                    threshold,
                    step,
                    trajectory_colors,
                    bool(directional_colors),
                    jump_limit,
                )
        else:
            self._append(
                cache,
                timestep,
                threshold,
                step,
                trajectory_colors,
                bool(directional_colors),
                jump_limit,
            )
        self._timestep = timestep
        return self._flatten()


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper rotation matrix to Viser's ``(w,x,y,z)`` order."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got {matrix.shape}.")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            [0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s]
        )
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            s = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1e-12))
            quat = np.array([(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s])
        elif axis == 1:
            s = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1e-12))
            quat = np.array([(matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s])
        else:
            s = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1e-12))
            quat = np.array([(matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s])
    quat /= np.linalg.norm(quat)
    return tuple(float(value) for value in quat)


def _camera_path_segments(c2w: np.ndarray) -> np.ndarray:
    centers = c2w[:, :3, 3]
    if centers.shape[0] < 2:
        return np.empty((0, 2, 3), dtype=np.float32)
    return np.stack([centers[:-1], centers[1:]], axis=1).astype(np.float32, copy=False)


def _sample_cache_points(cache: SourceCache, max_points: int = 50000) -> np.ndarray:
    """Collect a bounded, deterministic sample without flattening all tracks."""

    if max_points < 1:
        raise ValueError(f"max_points must be positive, got {max_points}.")
    horizon = cache.world_xyz.shape[0]
    per_timestep = max(1, int(np.ceil(max_points / max(horizon, 1))))
    samples: list[np.ndarray] = []
    for timestep in range(horizon):
        valid = cache.valid[timestep] & np.isfinite(cache.world_xyz[timestep]).all(axis=-1)
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            continue
        if indices.size > per_timestep:
            indices = indices[np.linspace(0, indices.size - 1, per_timestep, dtype=np.int64)]
        samples.append(cache.world_xyz[timestep, indices])
    if not samples:
        return np.empty((0, 3), dtype=np.float32)
    result = np.concatenate(samples, axis=0)
    if result.shape[0] > max_points:
        result = result[np.linspace(0, result.shape[0] - 1, max_points, dtype=np.int64)]
    return result


def scene_bounds_for_cache(cache: SourceCache) -> tuple[np.ndarray, np.ndarray]:
    """Return robust lower/upper world bounds for a source cache."""

    points = _sample_cache_points(cache)
    if points.size == 0:
        return np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64)
    lower, upper = np.percentile(points, [1.0, 99.0], axis=0)
    return lower.astype(np.float64), upper.astype(np.float64)


def scene_view_for_cache(cache: SourceCache) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return a robust camera ``(position, look_at)`` for one source."""

    lower, upper = scene_bounds_for_cache(cache)
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        center = np.zeros(3, dtype=np.float64)
        radius = 1.0
    else:
        center = (lower + upper) * 0.5
        radius = max(float(np.linalg.norm(upper - lower)) * 0.5, 0.1)
    offset = np.asarray([1.8, 1.35, 1.8], dtype=np.float64) * radius
    position = center + offset
    return tuple(float(value) for value in position), tuple(float(value) for value in center)


def _unit_direction(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Normalize a direction, falling back when a pose contains a degenerate axis."""

    direction = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1e-8:
        direction = np.asarray(fallback, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
    return direction / max(norm, 1e-8)


def source_camera_view_for_cache(
    data: FlowData,
    cache: SourceCache,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Return a view aligned with the selected source camera.

    ``track_xyz`` is produced by unprojecting depth into the conventional
    camera basis (x right, y down, z forward).  For a ``c2w`` pose this means
    the world-space viewing ray is ``R[:, 2]`` and the image-up direction is
    ``-R[:, 1]``.  Viser's orbit controls are then initialized with the same
    ray and up vector, so the first view is actually the source video view and
    subsequent mouse orbiting starts from a meaningful pivot.
    """

    source_frame = int(cache.source_frame)
    if not 0 <= source_frame < data.c2w.shape[0]:
        raise ValueError(
            f"source frame {source_frame} is outside c2w range [0,{data.c2w.shape[0]})."
        )
    pose = np.asarray(data.c2w[source_frame], dtype=np.float64)
    rotation = pose[:3, :3]
    origin = pose[:3, 3]
    forward = _unit_direction(rotation[:, 2], np.array([0.0, 0.0, 1.0]))
    image_up = _unit_direction(-rotation[:, 1], np.array([0.0, 1.0, 0.0]))
    # Remove any numerical component parallel to the forward ray.  This keeps
    # Viser's camera roll stable even when input poses are only approximately
    # orthonormal.
    image_up = image_up - forward * float(np.dot(image_up, forward))
    image_up = _unit_direction(image_up, np.array([0.0, 1.0, 0.0]))

    lower, upper = scene_bounds_for_cache(cache)
    if np.isfinite(lower).all() and np.isfinite(upper).all():
        center = (lower + upper) * 0.5
        radius = max(float(np.linalg.norm(upper - lower)) * 0.5, 0.1)
    else:
        center = origin + forward
        radius = 1.0

    # Keep the target on the source optical axis.  This is intentional: a
    # target at the percentile-box center can introduce a large yaw when a
    # track is asymmetric, which is exactly the hard-to-correct starting view
    # reported for the screen-facing video.
    axial_distance = float(np.dot(center - origin, forward))
    if not np.isfinite(axial_distance) or axial_distance <= 0.0:
        axial_distance = float(np.linalg.norm(center - origin))
    # The 2.5× margin keeps the 50°-ish default Viser field of view from
    # cropping a wide screen when the source camera is used as the viewpoint.
    distance = max(axial_distance, 2.5 * radius, 0.1)
    # A small backward offset avoids placing the interactive camera exactly
    # inside the source-camera frame while preserving its optical direction.
    position = origin - forward * (0.05 * radius)
    look_at = origin + forward * distance
    return (
        tuple(float(value) for value in position),
        tuple(float(value) for value in look_at),
        tuple(float(value) for value in image_up),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ego-centric Track4World 3D-FF flows.npz")
    parser.add_argument("--npz_path", "--flows_path", "--flows-path", default=DEFAULT_NPZ_PATH)
    parser.add_argument("--share", action="store_true", help="Request a Viser share URL.")
    parser.add_argument("--initial_source", type=int, default=0, help="Zero-based source ordinal in source_frame_index.")
    parser.add_argument("--confidence_threshold", type=float, default=0.1, help="Minimum mean of the two confidence channels.")
    parser.add_argument("--point_downsample", type=int, default=1, help="Keep every Nth query row before rendering.")
    parser.add_argument(
        "--trajectory_downsample",
        type=int,
        default=DEFAULT_TRAJECTORY_DOWNSAMPLE,
        help="Keep every Nth query row for trajectory lines (legacy default: 50).",
    )
    parser.add_argument(
        "--max_trajectory_segments",
        type=int,
        default=DEFAULT_MAX_TRAJECTORY_SEGMENTS,
        help="Adaptive full-horizon trajectory segment budget; the manual line sampling step is also honored.",
    )
    parser.add_argument("--max_render_points", type=int, default=DEFAULT_MAX_RENDER_POINTS, help="Adaptive per-source point budget (the manual step is also honored).")
    args = parser.parse_args()
    data = load_flows(args.npz_path)
    if not (0 <= args.initial_source < data.source_count):
        raise ValueError(f"--initial_source must be in [0,{data.source_count}), got {args.initial_source}.")
    if not (0.0 <= args.confidence_threshold <= 1.0):
        raise ValueError("--confidence_threshold must be within [0,1].")
    if args.point_downsample < 1:
        raise ValueError("--point_downsample must be positive.")
    if args.trajectory_downsample < 1:
        raise ValueError("--trajectory_downsample must be positive.")
    if args.max_trajectory_segments < 1:
        raise ValueError("--max_trajectory_segments must be positive.")
    if args.max_render_points < 1:
        raise ValueError("--max_render_points must be positive.")

    try:
        import viser
    except ImportError as exc:
        raise RuntimeError("The interactive viewer requires viser; install requirements.txt first.") from exc

    server = viser.ViserServer()
    if args.share:
        server.request_share_url()

    source_options = tuple(str(int(frame)) for frame in data.source_frame_index)
    source_option_to_ordinal = {option: ordinal for ordinal, option in enumerate(source_options)}
    with server.gui.add_folder("Ego 3D-FF"):
        gui_source = server.gui.add_dropdown("Source frame", source_options, initial_value=source_options[args.initial_source])
        gui_timestep = server.gui.add_slider("Horizon timestep", min=0, max=data.horizon_length - 1, step=1, initial_value=0)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=24)
        gui_threshold = server.gui.add_slider("Confidence threshold", min=0.0, max=1.0, step=0.01, initial_value=float(args.confidence_threshold))
        source_query_counts = np.diff(data.query_offsets)
        max_source_query_count = int(source_query_counts.max()) if source_query_counts.size else 0
        max_point_downsample = max(1, min(100, max(1, max_source_query_count)))
        max_trajectory_downsample = max_point_downsample
        gui_downsample = server.gui.add_slider("Query downsample", min=1, max=max_point_downsample, step=1, initial_value=min(max_point_downsample, args.point_downsample))
        gui_trajectory_downsample = server.gui.add_slider("Trajectory downsample", min=1, max=max_trajectory_downsample, step=1, initial_value=min(max_trajectory_downsample, args.trajectory_downsample))
        max_points_min = 100
        max_points_max = max(max_points_min, min(100000, max(data.query_count, args.max_render_points)))
        gui_max_render_points = server.gui.add_slider("Max render points", min=max_points_min, max=max_points_max, step=100, initial_value=min(max_points_max, max(max_points_min, args.max_render_points)))
        max_trajectory_segments_max = max(5000, min(500000, max(DEFAULT_MAX_TRAJECTORY_SEGMENTS, data.query_count)))
        gui_max_trajectory_segments = server.gui.add_slider("Max trajectory segments", min=1000, max=max_trajectory_segments_max, step=1000, initial_value=min(max_trajectory_segments_max, max(1000, args.max_trajectory_segments)))
        gui_trail = server.gui.add_slider("Trail length", min=1, max=max(1, data.horizon_length), step=1, initial_value=min(DEFAULT_TRAIL_LENGTH, max(1, data.horizon_length)))
        gui_point_size = server.gui.add_slider("Point size", min=0.001, max=0.2, step=0.001, initial_value=DEFAULT_POINT_SIZE)
        gui_line_width = server.gui.add_slider("Line width", min=0.5, max=10.0, step=0.1, initial_value=DEFAULT_LINE_WIDTH)
        gui_show_points = server.gui.add_checkbox("Show points", True)
        gui_show_tracks = server.gui.add_checkbox("Show tracks", True)
        gui_show_track_heads = server.gui.add_checkbox("Show track heads", True)
        gui_full_tracks = server.gui.add_checkbox("Full trajectory (all horizon)", False)
        gui_show_cameras = server.gui.add_checkbox("Show cameras", True)
        gui_sampling_status = (
            server.gui.add_markdown("")
            if hasattr(server.gui, "add_markdown")
            else None
        )
        gui_reset_camera = server.gui.add_button("Frame scene")
        gui_source_camera_view = server.gui.add_button("View source camera")

    source_cache_lru: OrderedDict[int, SourceCache] = OrderedDict()

    def get_source_cache(source_index: int) -> SourceCache:
        cached = source_cache_lru.pop(source_index, None)
        if cached is None:
            cached = build_source_cache(data, source_index)
        source_cache_lru[source_index] = cached
        while len(source_cache_lru) > 2:
            source_cache_lru.popitem(last=False)
        return cached

    def frame_clients(
        cache: SourceCache,
        *,
        source_camera: bool = False,
        client=None,
    ) -> None:
        if source_camera:
            camera_position, camera_look_at, camera_up = source_camera_view_for_cache(data, cache)
        else:
            camera_position, camera_look_at = scene_view_for_cache(cache)
            camera_up = (0.0, 1.0, 0.0)
        clients = (client,) if client is not None else tuple(server.get_clients().values())
        for connected_client in clients:
            connected_client.camera.position = camera_position
            connected_client.camera.look_at = camera_look_at
            connected_client.camera.up_direction = camera_up

    @server.on_client_connect
    def _on_connect(_client) -> None:
        selected = source_option_to_ordinal[str(gui_source.value)]
        frame_clients(get_source_cache(selected), source_camera=True, client=_client)

    @gui_reset_camera.on_click
    def _on_reset(_event) -> None:
        gui_timestep.value = 0
        update_scene()
        selected = source_option_to_ordinal[str(gui_source.value)]
        frame_clients(get_source_cache(selected))

    @gui_source_camera_view.on_click
    def _on_source_camera_view(_event) -> None:
        selected = source_option_to_ordinal[str(gui_source.value)]
        frame_clients(get_source_cache(selected), source_camera=True)

    initial_cache = get_source_cache(args.initial_source)
    initial_downsample = resolve_downsample(initial_cache.query_count, int(gui_downsample.value), int(gui_max_render_points.value))
    initial_points, initial_colors = build_point_cloud(data, args.initial_source, 0, float(gui_threshold.value), initial_downsample, source_cache=initial_cache)
    point_node = server.scene.add_point_cloud(name="/ego/source_points", points=initial_points, colors=initial_colors, point_size=gui_point_size.value, point_shape="rounded", visible=True)
    line_node = server.scene.add_line_segments(name="/ego/tracks", points=np.empty((0, 2, 3), dtype=np.float32), colors=np.empty((0, 2, 3), dtype=np.uint8), line_width=gui_line_width.value, visible=True)
    camera_path = _camera_path_segments(data.c2w)
    camera_path_node = server.scene.add_line_segments(name="/ego/camera_path", points=camera_path, colors=np.full((camera_path.shape[0], 2, 3), 180, dtype=np.uint8), line_width=1.0, visible=True)
    lower_bound, upper_bound = scene_bounds_for_cache(initial_cache)
    extent = float(np.linalg.norm(upper_bound - lower_bound))
    frame_size = max(extent * 0.05, 0.05)
    source_frame_node = server.scene.add_frame(name="/ego/source_camera", position=tuple(data.c2w[initial_cache.source_frame, :3, 3]), wxyz=rotation_matrix_to_wxyz(data.c2w[initial_cache.source_frame, :3, :3]), axes_length=frame_size, axes_radius=frame_size * 0.08, visible=True)
    initial_trajectory_downsample = resolve_trajectory_downsample(initial_cache.query_count, int(gui_trajectory_downsample.value), data.horizon_length, int(gui_max_trajectory_segments.value))
    initial_head_points, initial_head_colors = build_track_heads(data, args.initial_source, 0, float(gui_threshold.value), initial_trajectory_downsample, source_cache=initial_cache, trajectory_colors=initial_cache.trajectory_colors)
    track_head_node = server.scene.add_point_cloud(name="/ego/track_heads", points=initial_head_points, colors=initial_head_colors, point_size=max(1.8 * float(gui_point_size.value), 0.014), point_shape="circle", visible=True)
    trail_accumulator = TrailAccumulator()
    full_segments: tuple[np.ndarray, np.ndarray] = (np.empty((0, 2, 3), dtype=np.float32), np.empty((0, 2, 3), dtype=np.uint8))
    full_segments_key = None
    last_line_assignment_key = None
    last_point_key = (id(initial_cache), 0, float(gui_threshold.value), initial_downsample)
    last_head_key = None
    last_sampling_status = None

    def update_scene() -> None:
        nonlocal full_segments, full_segments_key, last_line_assignment_key, last_point_key, last_head_key, last_sampling_status
        ordinal = source_option_to_ordinal[str(gui_source.value)]
        timestep = int(gui_timestep.value)
        cache = get_source_cache(ordinal)
        threshold = float(gui_threshold.value)
        downsample = resolve_downsample(cache.query_count, int(gui_downsample.value), int(gui_max_render_points.value))
        trajectory_downsample = resolve_trajectory_downsample(cache.query_count, int(gui_trajectory_downsample.value), data.horizon_length, int(gui_max_trajectory_segments.value))
        visible_history = (
            data.horizon_length - 1
            if gui_full_tracks.value
            else min(timestep, max(1, int(gui_trail.value)))
        )
        sampled_track_count = (cache.query_count + trajectory_downsample - 1) // trajectory_downsample
        estimated_segments = sampled_track_count * max(0, visible_history)
        sampling_status = (
            f"轨迹采样：每 {trajectory_downsample} 个 query，约 "
            f"{sampled_track_count} 条 track、{estimated_segments} 条线段"
        )
        if sampling_status != last_sampling_status:
            if gui_sampling_status is not None:
                try:
                    gui_sampling_status.content = sampling_status
                except AttributeError:
                    # A lightweight test/dummy GUI may expose add_markdown
                    # without returning a mutable handle.
                    pass
            last_sampling_status = sampling_status
        point_key = (id(cache), timestep, threshold, downsample)
        if point_key != last_point_key:
            points, colors = build_point_cloud(data, ordinal, timestep, threshold, downsample, source_cache=cache)
            point_node.points = points
            point_node.colors = colors
            last_point_key = point_key
        point_node.point_size = float(gui_point_size.value)
        point_node.visible = bool(gui_show_points.value)
        head_key = (id(cache), timestep, threshold, trajectory_downsample)
        if head_key != last_head_key:
            head_points, head_colors = build_track_heads(data, ordinal, timestep, threshold, trajectory_downsample, source_cache=cache, trajectory_colors=cache.trajectory_colors)
            track_head_node.points = head_points
            track_head_node.colors = head_colors
            last_head_key = head_key
        track_head_node.point_size = max(1.8 * float(gui_point_size.value), 0.014)
        track_head_node.visible = bool(gui_show_track_heads.value)
        line_node.line_width = float(gui_line_width.value)
        line_node.visible = bool(gui_show_tracks.value)
        if gui_show_tracks.value:
            if gui_full_tracks.value:
                current_full_key = (id(cache), threshold, trajectory_downsample, id(cache.trajectory_colors))
                if current_full_key != full_segments_key:
                    full_segments = build_full_track_segments(data, ordinal, threshold, trajectory_downsample, source_cache=cache, trajectory_colors=cache.trajectory_colors, directional_colors=True, max_segment_displacement=DEFAULT_MAX_SEGMENT_DISPLACEMENT)
                    full_segments_key = current_full_key
                segments, segment_colors = full_segments
                line_assignment_key = ("full", current_full_key)
            else:
                segments, segment_colors = trail_accumulator.update(cache, timestep, threshold, trajectory_downsample, int(gui_trail.value), cache.trajectory_colors, True, DEFAULT_MAX_SEGMENT_DISPLACEMENT)
                line_assignment_key = ("trail", id(cache), timestep, threshold, trajectory_downsample, int(gui_trail.value))
            if line_assignment_key != last_line_assignment_key:
                line_node.points = segments
                line_node.colors = segment_colors
                last_line_assignment_key = line_assignment_key
        else:
            last_line_assignment_key = None
        camera_path_node.visible = bool(gui_show_cameras.value)
        source_frame_node.visible = bool(gui_show_cameras.value)
        source_frame_node.position = tuple(data.c2w[cache.source_frame, :3, 3])
        source_frame_node.wxyz = rotation_matrix_to_wxyz(data.c2w[cache.source_frame, :3, :3])

    @gui_source.on_update
    def _on_source_update(_event) -> None:
        selected = source_option_to_ordinal[str(gui_source.value)]
        get_source_cache(selected)
        update_scene()
        frame_clients(get_source_cache(selected), source_camera=True)

    for control in (gui_timestep, gui_threshold, gui_downsample, gui_trajectory_downsample, gui_max_trajectory_segments, gui_max_render_points, gui_trail, gui_point_size, gui_line_width, gui_show_points, gui_show_tracks, gui_show_track_heads, gui_full_tracks, gui_show_cameras):
        @control.on_update
        def _on_control_update(_event) -> None:
            update_scene()

    update_scene()
    # Also frame any client that was already attached before the scene nodes
    # finished loading (the connect callback handles later clients).
    frame_clients(initial_cache, source_camera=True)
    while True:
        if gui_playing.value:
            gui_timestep.value = (int(gui_timestep.value) + 1) % data.horizon_length
        time.sleep(1.0 / max(float(gui_fps.value), 1.0))


if __name__ == "__main__":
    main()
