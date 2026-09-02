"""Evaluate ego-centric 3D-FF static-background consistency per horizon.

``demo_3dff_ego.py`` writes all source windows to one ragged ``flows.npz``
file. ``track_xyz`` has a local horizon time axis and ``query_offsets``
identifies the query columns owned by each source. This evaluator applies the
temporal consistency rules independently to every source and reports three metrics:
MaxDrift, RobustMaxDrift, and Coverage.

Drift values are normalized by the median depth of static queries in global
frame 0 and are reported as percentages. The metric measures temporal
consistency rather than absolute 3D reconstruction accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


_FLOW_SCHEMA_VERSION = "track4world.ego_flows.v1"
FLOW_SCHEMA_VERSION = _FLOW_SCHEMA_VERSION
DEFAULT_FLOWS_PATH = Path("results/cat_ego/3d_ff_ego_output/flows.npz")
_FLOW_FILE_RE = re.compile(r"flow_(\d+)\.ply")
_VIS_FILE_RE = re.compile(r"vis_(\d+)\.npy")


def _parse_float_list(value: str, option_name: str) -> list[float]:
    """Parse a comma-separated list of finite, non-negative floats."""
    if not value.strip():
        return []
    try:
        parsed = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{option_name} must be a comma-separated list of numbers."
        ) from exc
    if any(not math.isfinite(item) or item < 0 for item in parsed):
        raise argparse.ArgumentTypeError(
            f"{option_name} values must be finite and non-negative."
        )
    return parsed


def _discover_indexed_files(
    directory: Path,
    pattern: re.Pattern[str],
) -> dict[int, Path]:
    """Return legacy PLY/visibility files keyed by their frame index."""
    indexed: dict[int, Path] = {}
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        frame_index = int(match.group(1))
        if frame_index in indexed:
            raise ValueError(
                f"Duplicate frame index {frame_index} for {pattern.pattern} in {directory}."
            )
        indexed[frame_index] = path
    return indexed


def discover_frame_files(flow_dir: Path | str) -> tuple[list[int], list[Path], list[Path]]:
    """Discover the former PLY/NPY layout (kept for import compatibility)."""
    directory = Path(flow_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"3D-FF output directory not found: {directory}")
    flow_by_index = _discover_indexed_files(directory, _FLOW_FILE_RE)
    vis_by_index = _discover_indexed_files(directory, _VIS_FILE_RE)
    if not flow_by_index:
        raise FileNotFoundError(f"No flow_XXX.ply files found in {directory}.")
    if 0 not in flow_by_index:
        raise FileNotFoundError(f"flow_000.ply is required in {directory}.")
    flow_indices = sorted(flow_by_index)
    expected = list(range(flow_indices[-1] + 1))
    if flow_indices != expected:
        missing = sorted(set(expected) - set(flow_indices))
        raise ValueError(f"Flow frame indices are not consecutive; missing {missing}.")
    if set(vis_by_index) != set(flow_by_index):
        raise ValueError(
            "Flow/visibility frame indices differ: "
            f"missing vis={sorted(set(flow_by_index) - set(vis_by_index))}, "
            f"extra vis={sorted(set(vis_by_index) - set(flow_by_index))}."
        )
    if len(flow_indices) < 2:
        raise ValueError("At least flow_000.ply and one target flow frame are required.")
    return (
        flow_indices,
        [flow_by_index[index] for index in flow_indices],
        [vis_by_index[index] for index in flow_indices],
    )


def load_ply_vertices(path: Path | str) -> np.ndarray:
    """Load legacy PLY vertices without changing their saved row order."""
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Reading PLY files requires trimesh.") from exc
    point_cloud = trimesh.load(str(path), process=False)
    if not hasattr(point_cloud, "vertices"):
        raise ValueError(f"PLY does not contain a vertex array: {path}")
    vertices = np.asarray(point_cloud.vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected PLY vertices with shape (N, 3), got {vertices.shape}.")
    return vertices.astype(np.float32, copy=False)


def load_visibility(path: Path | str, expected_rows: int) -> np.ndarray:
    """Load a legacy row-aligned visibility vector."""
    visibility = np.load(path, allow_pickle=False).reshape(-1)
    if visibility.shape != (expected_rows,):
        raise ValueError(
            f"Expected {expected_rows} visibility rows in {path}, got {visibility.shape[0]}."
        )
    return visibility.astype(np.float32, copy=False)


def load_static_mask(
    path: Path | str,
    *,
    dynamic_threshold: int,
    erosion_iterations: int,
) -> np.ndarray:
    """Load a 0=static, high-valued=dynamic mask and erode static regions."""
    mask_path = Path(path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"Dynamic mask not found: {mask_path}")
    with Image.open(mask_path) as image:
        dynamic_values = np.asarray(image.convert("L"))
    static_mask = dynamic_values <= dynamic_threshold

    if erosion_iterations > 0:
        try:
            from scipy.ndimage import binary_erosion
        except ImportError as exc:
            raise RuntimeError("Static-mask erosion requires scipy.") from exc
        static_mask = binary_erosion(
            static_mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=erosion_iterations,
            border_value=0,
        )
    return np.asarray(static_mask, dtype=bool)


def select_static_queries(flow_pixel_uv: np.ndarray, static_mask: np.ndarray) -> np.ndarray:
    """Map one source's query UV rows to its static-background mask."""
    uv = np.asarray(flow_pixel_uv)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"Expected flow_pixel_uv shape (N, 2), got {uv.shape}.")
    if not np.issubdtype(uv.dtype, np.integer):
        if not np.isfinite(uv).all() or not np.equal(uv, np.rint(uv)).all():
            raise ValueError("flow_pixel_uv must contain finite integer pixel coordinates.")
        uv = np.rint(uv).astype(np.int64)
    else:
        uv = uv.astype(np.int64, copy=False)

    mask = np.asarray(static_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D static mask, got {mask.shape}.")
    height, width = mask.shape
    u = uv[:, 0]
    v = uv[:, 1]
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not in_bounds.all():
        bad_row = int(np.flatnonzero(~in_bounds)[0])
        raise ValueError(
            f"flow_pixel_uv row {bad_row}={uv[bad_row].tolist()} is outside "
            f"mask size (H={height}, W={width})."
        )
    flat_uv = v * width + u
    if np.unique(flat_uv).size != uv.shape[0]:
        raise ValueError("flow_pixel_uv contains duplicate query coordinates.")
    return mask[v, u]


def _distance_summary(values: np.ndarray) -> dict[str, float]:
    """Summarize a finite, non-empty one-dimensional distance array."""
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Distance summaries require a finite, non-empty vector.")
    return {
        "mean": float(np.mean(values, dtype=np.float64)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def _time_balanced_accuracy(
    errors: np.ndarray,
    valid: np.ndarray,
    threshold: float,
) -> float | None:
    """Retained for compatibility with callers of the former utility API."""
    per_frame: list[float] = []
    for frame_row in range(errors.shape[0]):
        frame_values = errors[frame_row, valid[frame_row]]
        if frame_values.size:
            per_frame.append(float(np.mean(frame_values <= threshold)))
    if not per_frame:
        return None
    return float(np.mean(per_frame, dtype=np.float64))


def compute_drift_metrics(
    errors: np.ndarray,
    valid: np.ndarray,
    *,
    scene_scale: float,
    frame_indices: Sequence[int],
    robust_temporal_percentile: float,
    relative_thresholds_percent: Sequence[float],
    absolute_thresholds: Sequence[float],
) -> dict[str, Any]:
    """Legacy matrix aggregator retained for existing direct callers.

    The NPZ evaluator does not call this function: its report deliberately
    contains only MaxDrift, RobustMaxDrift, and Coverage.
    """
    errors = np.asarray(errors, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if errors.ndim != 2 or valid.shape != errors.shape:
        raise ValueError(
            f"Expected matching 2D errors/valid arrays, got {errors.shape}/{valid.shape}."
        )
    if errors.shape[0] != len(frame_indices):
        raise ValueError("frame_indices must contain one index per error row.")
    if errors.shape[1] == 0:
        raise ValueError("No static-background query points were selected.")
    if not math.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError(f"scene_scale must be positive and finite, got {scene_scale}.")
    if not 0 <= robust_temporal_percentile <= 100:
        raise ValueError("robust_temporal_percentile must be in [0, 100].")
    if np.any(valid & ~np.isfinite(errors)):
        raise ValueError("Valid drift entries must be finite.")

    static_count = errors.shape[1]
    per_frame: list[dict[str, Any]] = []
    frame_means: list[float] = []
    frame_medians: list[float] = []
    frame_p90s: list[float] = []
    coverages: list[float] = []
    for row, frame_index in enumerate(frame_indices):
        frame_values = errors[row, valid[row]]
        coverage = float(frame_values.size / static_count)
        coverages.append(coverage)
        entry: dict[str, Any] = {
            "frame_index": int(frame_index),
            "valid_static_query_count": int(frame_values.size),
            "coverage": coverage,
            "mean_drift": None,
            "median_drift": None,
            "p90_drift": None,
            "mean_drift_percent_of_median_depth": None,
            "median_drift_percent_of_median_depth": None,
            "p90_drift_percent_of_median_depth": None,
        }
        if frame_values.size:
            summary = _distance_summary(frame_values)
            frame_means.append(summary["mean"])
            frame_medians.append(summary["median"])
            frame_p90s.append(summary["p90"])
            entry.update(
                {
                    "mean_drift": summary["mean"],
                    "median_drift": summary["median"],
                    "p90_drift": summary["p90"],
                    "mean_drift_percent_of_median_depth": 100.0 * summary["mean"] / scene_scale,
                    "median_drift_percent_of_median_depth": 100.0 * summary["median"] / scene_scale,
                    "p90_drift_percent_of_median_depth": 100.0 * summary["p90"] / scene_scale,
                }
            )
        per_frame.append(entry)

    valid_counts = valid.sum(axis=0)
    eligible_tracks = valid_counts > 0
    if not eligible_tracks.any():
        raise ValueError("No static-background query is valid in any target frame.")
    max_drift = np.where(valid, errors, -np.inf).max(axis=0)[eligible_tracks]
    temporal_errors = np.where(valid[:, eligible_tracks], errors[:, eligible_tracks], np.nan)
    robust_max_drift = np.nanpercentile(
        temporal_errors,
        robust_temporal_percentile,
        axis=0,
    )
    max_summary = _distance_summary(max_drift)
    robust_summary = _distance_summary(robust_max_drift)
    raw_metrics = {
        "static_epe3d": float(np.mean(frame_means, dtype=np.float64)),
        "static_median_auc": float(np.mean(frame_medians, dtype=np.float64)),
        "static_p90_auc": float(np.mean(frame_p90s, dtype=np.float64)),
        "static_max_drift50": max_summary["median"],
        "static_max_drift90": max_summary["p90"],
        "static_max_drift_mean": max_summary["mean"],
        "static_robust_max_drift50": robust_summary["median"],
        "static_robust_max_drift90": robust_summary["p90"],
        "static_robust_max_drift_mean": robust_summary["mean"],
    }
    normalized = {key: 100.0 * value / scene_scale for key, value in raw_metrics.items()}
    relative_accuracies = []
    for threshold_percent in relative_thresholds_percent:
        threshold = scene_scale * float(threshold_percent) / 100.0
        relative_accuracies.append(
            {
                "threshold_percent_of_median_depth": float(threshold_percent),
                "threshold_in_coordinate_units": threshold,
                "time_balanced_accuracy": _time_balanced_accuracy(errors, valid, threshold),
            }
        )
    absolute_accuracies = [
        {
            "threshold_in_coordinate_units": float(threshold),
            "time_balanced_accuracy": _time_balanced_accuracy(errors, valid, float(threshold)),
        }
        for threshold in absolute_thresholds
    ]
    target_count = errors.shape[0]
    track_coverage = valid_counts[eligible_tracks] / target_count
    return {
        "raw_coordinate_units": raw_metrics,
        "percent_of_frame0_static_median_depth": normalized,
        "accuracy": {
            "relative_thresholds": relative_accuracies,
            "absolute_thresholds": absolute_accuracies,
        },
        "coverage": {
            "mean_per_frame": float(np.mean(coverages, dtype=np.float64)),
            "minimum_frame": float(np.min(coverages)),
            "maximum_frame": float(np.max(coverages)),
            "frames_with_at_least_one_valid_query_fraction": float(np.mean(np.asarray(coverages) > 0)),
            "tracks_with_at_least_one_valid_target_count": int(eligible_tracks.sum()),
            "tracks_with_at_least_one_valid_target_fraction": float(eligible_tracks.mean()),
            "tracks_visible_in_every_target_fraction": float(np.mean(valid_counts == target_count)),
            "mean_temporal_coverage_of_eligible_tracks": float(np.mean(track_coverage, dtype=np.float64)),
        },
        "per_frame": per_frame,
    }


# ---------------------------------------------------------------------------
# ``flows.npz`` loading and validation
# ---------------------------------------------------------------------------


_NPZ_REQUIRED_FIELDS = {
    "source_frame_index",
    "target_frame_index",
    "query_offsets",
    "query_uv",
    "track_xyz",
    "track_valid",
    "confidence",
    "image_size_hw",
}


def _npz_scalar(value: Any, name: str) -> Any:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar, got shape {array.shape}.")
    return array.item()


def validate_flows_npz_payload(payload: Mapping[str, np.ndarray]) -> None:
    """Validate the fields consumed from the ego-centric NPZ schema."""
    missing = _NPZ_REQUIRED_FIELDS - set(payload)
    if missing:
        raise ValueError(f"flows.npz is missing fields: {sorted(missing)}.")
    for name, value in payload.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"flows field {name} is not a NumPy array.")
        if value.dtype.hasobject:
            raise TypeError(f"flows field {name} requires pickle/object storage.")

    if "schema_version" in payload:
        schema_version = str(_npz_scalar(payload["schema_version"], "schema_version"))
        if schema_version != _FLOW_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported flows.npz schema_version: "
                f"{schema_version!r}; expected {_FLOW_SCHEMA_VERSION!r}."
            )
    if "coordinate_system" in payload:
        coordinate_system = str(_npz_scalar(payload["coordinate_system"], "coordinate_system"))
        if coordinate_system != "source_camera":
            raise ValueError(
                "This evaluator requires source_camera coordinates, got "
                f"{coordinate_system!r}."
            )
    if "pixel_convention" in payload:
        pixel_convention = str(_npz_scalar(payload["pixel_convention"], "pixel_convention"))
        if pixel_convention != "unpadded_uv_integer":
            raise ValueError(
                "This evaluator requires pixel_convention='unpadded_uv_integer'."
            )

    sources = np.asarray(payload["source_frame_index"])
    targets = np.asarray(payload["target_frame_index"])
    offsets = np.asarray(payload["query_offsets"])
    query_uv = np.asarray(payload["query_uv"])
    tracks = np.asarray(payload["track_xyz"])
    valid = np.asarray(payload["track_valid"])
    confidence = np.asarray(payload["confidence"])
    image_size = np.asarray(payload["image_size_hw"])

    if sources.dtype != np.int32 or sources.ndim != 1 or sources.size == 0:
        raise ValueError("source_frame_index must be a non-empty int32 vector.")
    if np.any(sources < 0) or np.any(np.diff(sources) <= 0):
        raise ValueError("source_frame_index must be strictly increasing and non-negative.")
    if int(sources[0]) != 0:
        raise ValueError("source_frame_index must include global frame 0 as its first source.")

    if tracks.dtype != np.float32 or tracks.ndim != 3 or tracks.shape[-1] != 3:
        raise ValueError(f"track_xyz must have shape (H, Q, 3) and dtype float32, got {tracks.shape}/{tracks.dtype}.")
    horizon = int(
        _npz_scalar(
            payload.get("horizon_length", np.asarray(tracks.shape[0])),
            "horizon_length",
        )
    )
    if horizon < 2:
        raise ValueError(f"horizon_length must be at least 2, got {horizon}.")
    if tracks.shape[0] != horizon:
        raise ValueError("horizon_length does not agree with track_xyz.")

    source_count = int(sources.shape[0])
    if targets.dtype != np.int32 or targets.shape != (source_count, horizon):
        raise ValueError(
            "target_frame_index must have shape (source_count, horizon) and dtype int32."
        )
    expected_targets = sources[:, None] + np.arange(horizon, dtype=np.int32)[None]
    if not np.array_equal(targets, expected_targets):
        raise ValueError("target_frame_index[i,j] must equal source_frame_index[i] + j.")
    if "effective_frame_count" in payload:
        effective_frame_count = int(
            _npz_scalar(payload["effective_frame_count"], "effective_frame_count")
        )
        if effective_frame_count < 1 or np.any(targets >= effective_frame_count):
            raise ValueError("target_frame_index contains an out-of-range frame.")

    if offsets.dtype != np.int64 or offsets.shape != (source_count + 1,):
        raise ValueError("query_offsets must be an int64 vector of shape (M+1,).")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("query_offsets must start at zero and be non-decreasing.")
    query_count = int(offsets[-1])
    if query_count < 0:
        raise ValueError("query_offsets cannot contain a negative query count.")
    if query_uv.dtype != np.int32 or query_uv.shape != (query_count, 2):
        raise ValueError("query_uv must have shape (Q, 2) and dtype int32.")
    if valid.dtype != np.bool_ or valid.shape != (horizon, query_count):
        raise ValueError("track_valid must have shape (H, Q) and dtype bool.")
    if confidence.dtype != np.float32 or confidence.shape != (horizon, query_count, 2):
        raise ValueError("confidence must have shape (H, Q, 2) and dtype float32.")
    if np.any(valid & ~np.isfinite(tracks).all(axis=-1)):
        raise ValueError("Valid track_xyz entries must be finite.")

    if image_size.dtype != np.int32 or image_size.shape != (2,):
        raise ValueError("image_size_hw must have shape (2,) and dtype int32.")
    if np.any(image_size <= 0):
        raise ValueError("image_size_hw values must be positive.")

    if "metric_scale" in payload:
        metric_scale = float(_npz_scalar(payload["metric_scale"], "metric_scale"))
        if not math.isfinite(metric_scale) or metric_scale <= 0:
            raise ValueError("metric_scale must be finite and positive when present.")
    if "c2w" in payload:
        c2w = np.asarray(payload["c2w"])
        if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
            raise ValueError("c2w must have shape (T_eff, 4, 4) when present.")
        if c2w.dtype.kind != "f" or not np.isfinite(c2w).all():
            raise ValueError("c2w must contain finite floating-point values.")
        if c2w.shape[0] <= int(sources[-1]) or np.any(targets >= c2w.shape[0]):
            raise ValueError("c2w does not cover all source/target frames.")
        homogeneous_bottom = np.array([0.0, 0.0, 0.0, 1.0])
        if not np.allclose(c2w[:, 3, :], homogeneous_bottom, atol=1e-5, rtol=0.0):
            raise ValueError("c2w matrices must have homogeneous bottom rows.")
        try:
            np.linalg.inv(c2w.astype(np.float64, copy=False))
        except np.linalg.LinAlgError as exc:
            raise ValueError("Every c2w matrix must be invertible.") from exc


validate_flows_payload = validate_flows_npz_payload


def load_flows_npz(path: Path | str) -> dict[str, np.ndarray]:
    """Load and validate a flows NPZ without enabling pickle."""
    flows_path = Path(path)
    if not flows_path.is_file():
        raise FileNotFoundError(f"flows.npz not found: {flows_path}")
    try:
        with np.load(flows_path, allow_pickle=False) as loaded:
            payload = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    except ValueError as exc:
        raise ValueError(f"Could not read non-pickled flows.npz: {flows_path}") from exc
    validate_flows_npz_payload(payload)
    return payload


# Friendly aliases used by a few repository utilities.
load_flows = load_flows_npz


# ---------------------------------------------------------------------------
# Per-horizon metrics
# ---------------------------------------------------------------------------


def _percentile_summary_percent(
    values: np.ndarray,
    scene_scale: float,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "unit": "percent_of_frame0_static_median_depth",
            "sample_count": 0,
            **{
                key: None
                for key in ("p25", "p50", "median", "p75", "p90", "mean")
            },
        }
    if not np.isfinite(values).all():
        raise ValueError("Drift distribution contains a non-finite value.")
    factor = 100.0 / scene_scale
    return {
        "unit": "percent_of_frame0_static_median_depth",
        "sample_count": int(values.size),
        "p25": float(np.percentile(values, 25) * factor),
        "p50": float(np.percentile(values, 50) * factor),
        "median": float(np.percentile(values, 50) * factor),
        "p75": float(np.percentile(values, 75) * factor),
        "p90": float(np.percentile(values, 90) * factor),
        "mean": float(np.mean(values) * factor),
    }


def _empty_horizon_metrics(
    *,
    static_count: int,
    target_frame_indices: Sequence[int],
    scene_scale: float,
) -> dict[str, Any]:
    """Return JSON-safe metrics for an empty/entirely invalid horizon."""
    if not math.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError(f"scene_scale must be positive and finite, got {scene_scale}.")
    undefined = static_count == 0
    per_frame = [
        {
            "frame_index": int(frame_index),
            "valid_static_query_count": 0,
            "coverage": None if undefined else 0.0,
        }
        for frame_index in target_frame_indices
    ]
    coverage = {
        "unit": "percent",
        "static_query_count": int(static_count),
        "eligible_query_count": 0,
        "mean_per_frame": None if undefined else 0.0,
        "minimum_frame": None if undefined else 0.0,
        "maximum_frame": None if undefined else 0.0,
        "frames_with_at_least_one_valid_query_fraction": None if undefined else 0.0,
        "tracks_with_at_least_one_valid_target_count": 0,
        "tracks_with_at_least_one_valid_target_fraction": None if undefined else 0.0,
        "tracks_visible_in_every_target_fraction": None if undefined else 0.0,
        "mean_temporal_coverage_of_eligible_tracks": None,
        "per_frame": per_frame,
    }
    return {
        "max_drift": _percentile_summary_percent(np.empty(0), scene_scale),
        "robust_max_drift": _percentile_summary_percent(np.empty(0), scene_scale),
        "coverage": coverage,
    }


def _compute_horizon_drift_metrics(
    track_xyz: np.ndarray,
    track_valid: np.ndarray,
    confidence: np.ndarray,
    static_queries: np.ndarray,
    *,
    scene_scale: float,
    target_frame_indices: Sequence[int],
    visibility_threshold: float,
    ignore_visibility: bool,
    robust_temporal_percentile: float,
    return_distributions: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Compute the three requested metrics for one source horizon."""
    tracks = np.asarray(track_xyz, dtype=np.float32)
    geometry_valid = np.asarray(track_valid, dtype=bool)
    confidence = np.asarray(confidence, dtype=np.float32)
    static_queries = np.asarray(static_queries, dtype=bool).reshape(-1)
    if tracks.ndim != 3 or tracks.shape[-1] != 3:
        raise ValueError(f"Expected track_xyz shape (H, Q, 3), got {tracks.shape}.")
    horizon, query_count, _ = tracks.shape
    if horizon < 2:
        raise ValueError("track_xyz must contain a source frame and at least one target frame.")
    if geometry_valid.shape != (horizon, query_count):
        raise ValueError(
            f"Expected track_valid shape {(horizon, query_count)}, got {geometry_valid.shape}."
        )
    if confidence.shape != (horizon, query_count, 2):
        raise ValueError(
            f"Expected confidence shape {(horizon, query_count, 2)}, got {confidence.shape}."
        )
    if static_queries.shape != (query_count,):
        raise ValueError(
            f"Expected static_queries shape {(query_count,)}, got {static_queries.shape}."
        )
    if len(target_frame_indices) != horizon - 1:
        raise ValueError(
            "target_frame_indices must contain one global index for each j=1..H-1."
        )
    if not math.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError(f"scene_scale must be positive and finite, got {scene_scale}.")
    if not math.isfinite(visibility_threshold) or visibility_threshold < 0:
        raise ValueError("visibility_threshold must be finite and non-negative.")
    if not 0 <= robust_temporal_percentile <= 100:
        raise ValueError("robust_temporal_percentile must be in [0, 100].")

    static_indices = np.flatnonzero(static_queries)
    static_count = int(static_indices.size)
    if static_count == 0:
        return (
            _empty_horizon_metrics(
                static_count=0,
                target_frame_indices=target_frame_indices,
                scene_scale=scene_scale,
            ),
            np.empty(0),
            np.empty(0),
        )

    finite = np.isfinite(tracks).all(axis=-1)
    reference_geometry = geometry_valid[0] & finite[0] & (tracks[0, :, 2] > 0)
    visibility = np.prod(confidence, axis=-1, dtype=np.float32)
    if ignore_visibility:
        reference_valid = reference_geometry
        target_valid = geometry_valid[1:] & finite[1:]
    else:
        reference_valid = (
            reference_geometry
            & np.isfinite(visibility[0])
            & (visibility[0] >= visibility_threshold)
        )
        target_valid = (
            geometry_valid[1:]
            & finite[1:]
            & np.isfinite(visibility[1:])
            & (visibility[1:] >= visibility_threshold)
        )

    reference = tracks[0, static_indices].astype(np.float64, copy=False)
    errors = np.linalg.norm(
        tracks[1:, static_indices].astype(np.float64, copy=False)
        - reference[None],
        axis=-1,
    ).astype(
        np.float32, copy=False
    )
    valid = target_valid[:, static_indices] & reference_valid[static_indices]
    valid &= np.isfinite(errors)

    target_count = errors.shape[0]
    valid_counts = valid.sum(axis=0)
    eligible = valid_counts > 0
    per_frame_counts = valid.sum(axis=1)
    per_frame_coverage = per_frame_counts / static_count
    per_frame = [
        {
            "frame_index": int(frame_index),
            "valid_static_query_count": int(count),
            "coverage": float(100.0 * count / static_count),
        }
        for frame_index, count in zip(target_frame_indices, per_frame_counts)
    ]
    coverage: dict[str, Any] = {
        "unit": "percent",
        "static_query_count": static_count,
        "eligible_query_count": int(eligible.sum()),
        "mean_per_frame": float(100.0 * np.mean(per_frame_coverage)),
        "minimum_frame": float(100.0 * np.min(per_frame_coverage)),
        "maximum_frame": float(100.0 * np.max(per_frame_coverage)),
        "frames_with_at_least_one_valid_query_fraction": float(
            100.0 * np.mean(per_frame_coverage > 0)
        ),
        "tracks_with_at_least_one_valid_target_count": int(eligible.sum()),
        "tracks_with_at_least_one_valid_target_fraction": float(100.0 * np.mean(eligible)),
        "tracks_visible_in_every_target_fraction": float(
            100.0 * np.mean(valid_counts == target_count)
        ),
        "mean_temporal_coverage_of_eligible_tracks": (
            float(100.0 * np.mean(valid_counts[eligible] / target_count))
            if eligible.any() and target_count
            else None
        ),
        "per_frame": per_frame,
    }

    if not eligible.any():
        empty = _empty_horizon_metrics(
            static_count=static_count,
            target_frame_indices=target_frame_indices,
            scene_scale=scene_scale,
        )
        empty["coverage"] = coverage
        return empty, np.empty(0), np.empty(0)

    max_values = np.max(np.where(valid, errors, -np.inf), axis=0)[eligible]
    temporal_values = np.where(valid[:, eligible], errors[:, eligible], np.nan)
    robust_values = np.nanpercentile(
        temporal_values,
        robust_temporal_percentile,
        axis=0,
    )
    metrics = {
        "max_drift": _percentile_summary_percent(max_values, scene_scale),
        "robust_max_drift": _percentile_summary_percent(robust_values, scene_scale),
        "coverage": coverage,
    }
    if return_distributions:
        return metrics, max_values.astype(np.float64), robust_values.astype(np.float64)
    return metrics, np.empty(0), np.empty(0)


def compute_horizon_drift_metrics(
    track_xyz: np.ndarray,
    track_valid: np.ndarray,
    confidence: np.ndarray,
    static_queries: np.ndarray,
    *,
    scene_scale: float,
    target_frame_indices: Sequence[int] | None = None,
    visibility_threshold: float = 0.1,
    ignore_visibility: bool = False,
    robust_temporal_percentile: float = 95.0,
) -> dict[str, Any]:
    """Public single-horizon calculator for the three requested metrics."""
    tracks = np.asarray(track_xyz)
    if tracks.ndim < 1:
        raise ValueError("track_xyz must have at least one dimension.")
    horizon = int(tracks.shape[0])
    if target_frame_indices is None:
        target_frame_indices = list(range(1, horizon))
    metrics, _, _ = _compute_horizon_drift_metrics(
        track_xyz,
        track_valid,
        confidence,
        static_queries,
        scene_scale=scene_scale,
        target_frame_indices=target_frame_indices,
        visibility_threshold=visibility_threshold,
        ignore_visibility=ignore_visibility,
        robust_temporal_percentile=robust_temporal_percentile,
    )
    return metrics


compute_horizon_metrics = compute_horizon_drift_metrics


def compute_horizon_drift_distributions(
    track_xyz: np.ndarray,
    track_valid: np.ndarray,
    confidence: np.ndarray,
    static_queries: np.ndarray,
    *,
    scene_scale: float,
    target_frame_indices: Sequence[int] | None = None,
    visibility_threshold: float = 0.1,
    ignore_visibility: bool = False,
    robust_temporal_percentile: float = 95.0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Return metrics and normalized per-query Max/RobustMax samples for plotting."""
    tracks = np.asarray(track_xyz)
    if tracks.ndim < 1:
        raise ValueError("track_xyz must have at least one dimension.")
    horizon = int(tracks.shape[0])
    if target_frame_indices is None:
        target_frame_indices = list(range(1, horizon))
    metrics, max_values, robust_values = _compute_horizon_drift_metrics(
        track_xyz,
        track_valid,
        confidence,
        static_queries,
        scene_scale=scene_scale,
        target_frame_indices=target_frame_indices,
        visibility_threshold=visibility_threshold,
        ignore_visibility=ignore_visibility,
        robust_temporal_percentile=robust_temporal_percentile,
        return_distributions=True,
    )
    factor = 100.0 / scene_scale
    return metrics, max_values * factor, robust_values * factor


compute_horizon_distributions = compute_horizon_drift_distributions


def _resolve_mask_path(
    mask_dir: Path,
    source_frame_index: int,
    *,
    mask_override: Path | None = None,
) -> Path:
    """Resolve ``mask_XXXX.png`` for a source, with a frame-0 override."""
    if mask_override is not None and source_frame_index == 0:
        return mask_override
    candidates = (
        mask_dir / f"mask_{source_frame_index:04d}.png",
        mask_dir / f"mask_{source_frame_index}.png",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _aggregate_horizon_reports(horizons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate scalar summaries with equal weight per non-empty horizon."""
    summary: dict[str, Any] = {
        "horizon_count": int(len(horizons)),
        "horizons_with_max_drift": 0,
        "horizons_with_robust_max_drift": 0,
    }
    for metric_name, count_key in (
        ("max_drift", "horizons_with_max_drift"),
        ("robust_max_drift", "horizons_with_robust_max_drift"),
    ):
        values_by_key: dict[str, list[float]] = {
            key: [] for key in ("p25", "p50", "median", "p75", "p90", "mean")
        }
        for horizon in horizons:
            metric = horizon["metrics"][metric_name]
            for key in values_by_key:
                value = metric.get(key)
                if value is not None:
                    values_by_key[key].append(float(value))
        summary[metric_name] = {
            "unit": "percent_of_frame0_static_median_depth",
            "sample_count": int(
                sum(
                    int(horizon["metrics"][metric_name].get("sample_count", 0))
                    for horizon in horizons
                )
            ),
            **{
                key: (float(np.mean(values)) if values else None)
                for key, values in values_by_key.items()
            },
        }
        summary[count_key] = len(values_by_key["median"])

    coverage_keys = (
        "mean_per_frame",
        "minimum_frame",
        "maximum_frame",
        "frames_with_at_least_one_valid_query_fraction",
        "tracks_with_at_least_one_valid_target_fraction",
        "tracks_visible_in_every_target_fraction",
        "mean_temporal_coverage_of_eligible_tracks",
    )
    coverage_values: dict[str, list[float]] = {key: [] for key in coverage_keys}
    static_count = 0
    eligible_count = 0
    for horizon in horizons:
        coverage = horizon["metrics"]["coverage"]
        static_count += int(coverage.get("static_query_count", 0))
        eligible_count += int(coverage.get("eligible_query_count", 0))
        for key in coverage_keys:
            value = coverage.get(key)
            if value is not None:
                coverage_values[key].append(float(value))
    summary["coverage"] = {
        "unit": "percent",
        **{
            key: (float(np.mean(values)) if values else None)
            for key, values in coverage_values.items()
        },
    }
    summary["coverage"].update(
        {
            "static_query_count": static_count,
            "eligible_query_count": eligible_count,
        }
    )
    return summary


def _plot_distribution_values(
    entry: Mapping[str, Any],
    samples_key: str,
    metric_key: str,
) -> np.ndarray:
    """Extract finite samples from a plotting entry or a JSON metric summary.

    The evaluator keeps the potentially large per-query distributions outside
    the JSON report and passes them to this function through the
    ``*_samples_percent`` fields.  Accepting the compact report form as a
    fallback makes the public plotting helper useful on ``report["horizons"]``
    as well: a summary is represented by its available quartiles.
    """
    value: Any = entry.get(samples_key)
    if value is None:
        value = entry.get(samples_key.removesuffix("_percent"))
    if value is None:
        nested = entry.get("metrics")
        if isinstance(nested, Mapping):
            value = nested.get(metric_key)
    if isinstance(value, Mapping):
        explicit_samples = value.get("samples")
        if explicit_samples is not None:
            value = explicit_samples
        else:
            # p25/p50/p75 are the most useful compact approximation when raw
            # samples were intentionally not serialized into the JSON report.
            value = [
                value.get("p25"),
                value.get("p50", value.get("median")),
                value.get("p75"),
            ]
    if value is None:
        return np.empty(0, dtype=np.float64)
    try:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.empty(0, dtype=np.float64)
    return values[np.isfinite(values)]


def _plot_coverage_values(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the coverage form accepted by :func:`plot_horizon_metrics`."""
    value: Any = entry.get("coverage")
    if not isinstance(value, Mapping):
        nested = entry.get("metrics")
        if isinstance(nested, Mapping):
            nested_value = nested.get("coverage")
            if nested_value is not None:
                value = nested_value
    if isinstance(value, Mapping):
        return value
    if value is None:
        value = entry.get("coverage_percent")
    if value is None:
        return {}
    try:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return {}
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        "mean_per_frame": float(np.mean(values)),
        "minimum_frame": float(np.min(values)),
        "maximum_frame": float(np.max(values)),
    }


def plot_horizon_metrics(
    horizon_distributions: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    output_path: Path | str,
) -> None:
    """Draw MaxDrift/RobustMaxDrift boxplots and a Coverage panel."""
    try:
        if "MPLCONFIGDIR" not in os.environ:
            matplotlib_config = Path(tempfile.gettempdir()) / "track4world-matplotlib"
            matplotlib_config.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(matplotlib_config)
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Horizon plots require matplotlib.") from exc

    if isinstance(horizon_distributions, Mapping):
        if isinstance(horizon_distributions.get("horizons"), Sequence):
            entries = list(horizon_distributions["horizons"])
        else:
            entries = list(horizon_distributions.values())
    else:
        entries = list(horizon_distributions)
    if not entries:
        raise ValueError("At least one horizon is required for plotting.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = []
    for index, entry in enumerate(entries):
        label_value = entry.get(
            "source_frame_index",
            entry.get("horizon_index", entry.get("horizon", index)),
        )
        try:
            labels.append(str(int(label_value)))
        except (TypeError, ValueError):
            labels.append(str(label_value))
    positions = np.arange(1, len(entries) + 1, dtype=np.float64)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(max(10.0, 0.55 * len(entries)), 12.0),
        sharex=True,
    )
    try:
        plot_specs = (
            (
                "max_drift_samples_percent",
                "MaxDrift (% of frame-0 static median depth)",
                "tab:blue",
            ),
            (
                "robust_max_drift_samples_percent",
                "RobustMaxDrift (% of frame-0 static median depth)",
                "tab:orange",
            ),
        )
        for axis, (key, title, color) in zip(axes[:2], plot_specs):
            box_data: list[np.ndarray] = []
            box_positions: list[float] = []
            for position, entry in zip(positions, entries):
                values = _plot_distribution_values(
                    entry,
                    key,
                    "max_drift" if key.startswith("max_") else "robust_max_drift",
                )
                if values.size:
                    box_data.append(values)
                    box_positions.append(float(position))
            if box_data:
                artists = axis.boxplot(
                    box_data,
                    positions=box_positions,
                    widths=0.6,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                for patch in artists["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.45)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No eligible static queries",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                )
            axis.set_ylabel("percent")
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)

        coverage_axis = axes[2]
        coverage_entries = [_plot_coverage_values(entry) for entry in entries]
        mean_values = np.asarray(
            [coverage.get("mean_per_frame") for coverage in coverage_entries],
            dtype=np.float64,
        )
        minimum_values = np.asarray(
            [coverage.get("minimum_frame") for coverage in coverage_entries],
            dtype=np.float64,
        )
        maximum_values = np.asarray(
            [coverage.get("maximum_frame") for coverage in coverage_entries],
            dtype=np.float64,
        )
        valid_mean = np.isfinite(mean_values)
        if valid_mean.any():
            coverage_axis.plot(
                positions[valid_mean],
                mean_values[valid_mean],
                marker="o",
                label="mean per frame",
                color="tab:green",
            )
            valid_band = np.isfinite(minimum_values) & np.isfinite(maximum_values)
            if valid_band.any():
                coverage_axis.fill_between(
                    positions[valid_band],
                    minimum_values[valid_band],
                    maximum_values[valid_band],
                    color="tab:green",
                    alpha=0.15,
                    label="min–max per frame",
                )
            coverage_axis.legend(loc="best")
        else:
            coverage_axis.text(
                0.5,
                0.5,
                "No coverage values",
                transform=coverage_axis.transAxes,
                ha="center",
                va="center",
            )
        coverage_axis.set_ylabel("percent")
        coverage_axis.set_title("Coverage")
        coverage_axis.set_ylim(0, 100)
        coverage_axis.grid(axis="y", alpha=0.25)
        axes[-1].set_xticks(positions)
        axes[-1].set_xticklabels(labels, rotation=45, ha="right")
        axes[-1].set_xlabel("source frame (horizon)")
        fig.tight_layout()
        fig.savefig(output, dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)


plot_horizon_statistics = plot_horizon_metrics


def _frame0_scale_from_other_sources(
    payload: Mapping[str, np.ndarray],
    static_queries_by_horizon: Sequence[np.ndarray],
) -> float | None:
    """Fallback scale estimate using source poses when source 0 has no points."""
    if "c2w" not in payload:
        return None
    c2w = np.asarray(payload["c2w"], dtype=np.float64)
    sources = payload["source_frame_index"]
    offsets = payload["query_offsets"]
    tracks = payload["track_xyz"]
    valid = payload["track_valid"]
    try:
        world_to_frame0 = np.linalg.inv(c2w[0])
    except np.linalg.LinAlgError:
        return None
    depths: list[np.ndarray] = []
    for source_position, source in enumerate(sources):
        q0 = int(offsets[source_position])
        q1 = int(offsets[source_position + 1])
        if q1 <= q0:
            continue
        points = tracks[0, q0:q1].astype(np.float64, copy=False)
        geometry = (
            static_queries_by_horizon[source_position]
            & valid[0, q0:q1]
            & np.isfinite(points).all(axis=1)
            & (points[:, 2] > 0)
        )
        if not geometry.any():
            continue
        pose = c2w[int(source)]
        frame0_from_source = world_to_frame0 @ pose
        transformed = points @ frame0_from_source[:3, :3].T + frame0_from_source[:3, 3]
        geometry &= np.isfinite(transformed).all(axis=1) & (transformed[:, 2] > 0)
        if geometry.any():
            depths.append(transformed[geometry, 2])
    if not depths:
        return None
    scale = float(np.median(np.concatenate(depths)))
    return scale if math.isfinite(scale) and scale > 0 else None


def evaluate_3dff_static_background(
    flows_path: Path | str | None = None,
    mask_dir: Path | str | None = None,
    *,
    visibility_threshold: float = 0.1,
    ignore_visibility: bool = False,
    dynamic_mask_threshold: int = 127,
    static_erosion_iterations: int = 4,
    robust_temporal_percentile: float = 95.0,
    plot_output: Path | str | None = None,
    # Compatibility aliases for the previous API/CLI.
    flow_dir: Path | str | None = None,
    mask_path: Path | str | None = None,
    relative_thresholds_percent: Sequence[float] | None = None,
    absolute_thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Evaluate MaxDrift, RobustMaxDrift and Coverage for every horizon."""
    # These options belonged to the removed accuracy metrics.  Keep accepting
    # them so old callers do not fail at the call boundary, but do not compute
    # or emit those metrics in the new report.
    del relative_thresholds_percent, absolute_thresholds
    if flows_path is not None and flow_dir is not None:
        raise ValueError("Provide either flows_path or flow_dir, not both.")
    flows_path = Path(flows_path or flow_dir or DEFAULT_FLOWS_PATH)
    if flows_path.is_dir():
        flows_path = flows_path / "flows.npz"
    if not math.isfinite(visibility_threshold) or visibility_threshold < 0:
        raise ValueError("visibility_threshold must be finite and non-negative.")
    if not math.isfinite(robust_temporal_percentile) or not 0 <= robust_temporal_percentile <= 100:
        raise ValueError("robust_temporal_percentile must be in [0, 100].")
    if not 0 <= dynamic_mask_threshold <= 255:
        raise ValueError("dynamic_mask_threshold must be in [0, 255].")
    if static_erosion_iterations < 0:
        raise ValueError("static_erosion_iterations must be non-negative.")

    payload = load_flows_npz(flows_path)
    sources = payload["source_frame_index"]
    targets = payload["target_frame_index"]
    offsets = payload["query_offsets"]
    query_uv = payload["query_uv"]
    tracks = payload["track_xyz"]
    track_valid = payload["track_valid"]
    confidence = payload["confidence"]
    height, width = (int(value) for value in payload["image_size_hw"])

    explicit_mask = Path(mask_path) if mask_path is not None else None
    if mask_dir is None:
        if explicit_mask is not None and explicit_mask.is_dir():
            mask_dir = explicit_mask
            explicit_mask = None
        elif explicit_mask is not None:
            mask_dir = explicit_mask.parent
        else:
            mask_dir = flows_path.parent.parent / "final_dyn_mask"
    mask_dir = Path(mask_dir)
    if mask_dir.is_file():
        if explicit_mask is not None and explicit_mask != mask_dir:
            raise ValueError("mask_dir and mask_path refer to different files.")
        explicit_mask = mask_dir
        mask_dir = mask_dir.parent
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"Dynamic-mask directory not found: {mask_dir}")
    if explicit_mask is not None and explicit_mask.is_dir():
        raise ValueError("mask_path must be a mask file when mask_dir is provided.")

    static_queries_by_horizon: list[np.ndarray] = []
    mask_paths: list[Path] = []
    for horizon_position, source in enumerate(sources):
        source_mask_path = _resolve_mask_path(
            mask_dir,
            int(source),
            mask_override=explicit_mask,
        )
        source_mask = load_static_mask(
            source_mask_path,
            dynamic_threshold=dynamic_mask_threshold,
            erosion_iterations=static_erosion_iterations,
        )
        if source_mask.shape != (height, width):
            raise ValueError(
                f"Mask {source_mask_path} has shape {source_mask.shape}; "
                f"expected {(height, width)} from flows.npz image_size_hw."
            )
        q0 = int(offsets[horizon_position])
        q1 = int(offsets[horizon_position + 1])
        static_queries_by_horizon.append(
            select_static_queries(query_uv[q0:q1], source_mask)
        )
        mask_paths.append(source_mask_path)

    source_zero_position = int(np.flatnonzero(sources == 0)[0])
    q0 = int(offsets[source_zero_position])
    q1 = int(offsets[source_zero_position + 1])
    source_zero_points = tracks[0, q0:q1]
    source_zero_geometry = (
        track_valid[0, q0:q1]
        & np.isfinite(source_zero_points).all(axis=1)
        & (source_zero_points[:, 2] > 0)
    )
    scale_candidates = (
        static_queries_by_horizon[source_zero_position] & source_zero_geometry
    )
    scale_source = "source_0_static_queries"
    if scale_candidates.any():
        scene_scale = float(np.median(source_zero_points[scale_candidates, 2]))
    else:
        # Ragged payloads are allowed to have an empty source-0 interval. If
        # poses are available, estimate the same frame-0 scale from another
        # source after mapping its source-camera points into frame 0.
        fallback_scale = _frame0_scale_from_other_sources(
            payload,
            static_queries_by_horizon,
        )
        if fallback_scale is None:
            raise ValueError(
                "No finite positive-depth static query is available in global frame 0 "
                "for normalization."
            )
        scene_scale = fallback_scale
        scale_source = "all_source_queries_transformed_to_frame0"
    if not math.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError(f"Invalid frame-0 static median depth: {scene_scale}.")

    horizon_reports: list[dict[str, Any]] = []
    plot_entries: list[dict[str, Any]] = []
    for horizon_position, source in enumerate(sources):
        q0 = int(offsets[horizon_position])
        q1 = int(offsets[horizon_position + 1])
        target_frame_indices = [int(index) for index in targets[horizon_position, 1:]]
        metrics, max_values, robust_values = _compute_horizon_drift_metrics(
            tracks[:, q0:q1],
            track_valid[:, q0:q1],
            confidence[:, q0:q1],
            static_queries_by_horizon[horizon_position],
            scene_scale=scene_scale,
            target_frame_indices=target_frame_indices,
            visibility_threshold=visibility_threshold,
            ignore_visibility=ignore_visibility,
            robust_temporal_percentile=robust_temporal_percentile,
            return_distributions=True,
        )
        static_count = int(static_queries_by_horizon[horizon_position].sum())
        horizon_report = {
            "horizon_index": int(horizon_position),
            "source_frame_index": int(source),
            "target_frame_indices": target_frame_indices,
            "mask_path": str(mask_paths[horizon_position].resolve()),
            "static_query_count": static_count,
            "static_query_percent": (
                float(100.0 * np.mean(static_queries_by_horizon[horizon_position]))
                if q1 > q0
                else 0.0
            ),
            "metrics": metrics,
        }
        # Direct aliases are convenient for simple consumers; ``metrics`` is
        # the canonical three-metric object.
        horizon_report.update(metrics)
        horizon_reports.append(horizon_report)
        plot_entries.append(
            {
                "source_frame_index": int(source),
                "max_drift_samples_percent": max_values * 100.0 / scene_scale,
                "robust_max_drift_samples_percent": robust_values * 100.0 / scene_scale,
                "coverage": metrics["coverage"],
            }
        )

    coordinate_system = str(
        _npz_scalar(
            payload.get("coordinate_system", np.asarray("source_camera")),
            "coordinate_system",
        )
    )
    summary = _aggregate_horizon_reports(horizon_reports)
    report: dict[str, Any] = {
        "schema_version": 2,
        "metric_name": "3D-FF per-horizon static-background consistency",
        "metric_names": ["MaxDrift", "RobustMaxDrift", "Coverage"],
        "interpretation": (
            "Lower MaxDrift and RobustMaxDrift are better. Coverage is the "
            "percentage of selected static queries valid at target frames."
        ),
        "input": {
            "flows_path": str(flows_path.resolve()),
            "mask_directory": str(mask_dir.resolve()),
            "coordinate_system": coordinate_system,
            "source_frame_indices": [int(value) for value in sources],
        },
        "configuration": {
            "visibility_threshold": (
                None if ignore_visibility else float(visibility_threshold)
            ),
            "ignore_visibility": bool(ignore_visibility),
            "dynamic_mask_threshold_uint8": int(dynamic_mask_threshold),
            "static_erosion_iterations_3x3": int(static_erosion_iterations),
            "robust_temporal_percentile": float(robust_temporal_percentile),
            "normalization": "percent of global frame-0 static median depth",
            "normalization_source": scale_source,
        },
        "selection": {
            "image_height": height,
            "image_width": width,
            "all_query_count": int(query_uv.shape[0]),
            "horizon_count": int(len(sources)),
            "total_static_query_count": int(
                sum(int(values.sum()) for values in static_queries_by_horizon)
            ),
            "frame0_static_median_depth_coordinate_units": scene_scale,
        },
        "definitions": {
            "point_drift": "norm(track_xyz[j,q] - track_xyz[0,q], 2)",
            "visibility": "confidence[j,q,0] * confidence[j,q,1]",
            "max_drift": "maximum point_drift over valid target frames per query",
            "robust_max_drift": (
                f"temporal percentile {robust_temporal_percentile:g} of point_drift "
                "over valid target frames per query"
            ),
            "coverage": "valid static queries divided by selected static queries, per target frame",
        },
        "horizons": horizon_reports,
        "summary": summary,
        # Keep a compact top-level alias for consumers of the former report
        # shape; it still contains only the three requested metric families.
        "metrics": {
            "max_drift": summary["max_drift"],
            "robust_max_drift": summary["robust_max_drift"],
            "coverage": summary["coverage"],
        },
        "plot": None,
    }
    if plot_output is not None:
        plot_path = Path(plot_output)
        plot_horizon_metrics(plot_entries, plot_path)
        report["plot"] = {
            "path": str(plot_path.resolve()),
            "horizon_count": int(len(plot_entries)),
            "metrics": ["MaxDrift", "RobustMaxDrift", "Coverage"],
        }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ego-centric 3D-FF MaxDrift, RobustMaxDrift and Coverage "
            "for every horizon."
        )
    )
    parser.add_argument("--flows-path", type=Path, default=None, help="Path to flows.npz.")
    parser.add_argument(
        "--flow-dir",
        type=Path,
        default=None,
        help="Compatibility alias: directory containing flows.npz.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Directory containing source masks mask_XXXX.png.",
    )
    parser.add_argument(
        "--mask-path",
        type=Path,
        default=None,
        help="Compatibility alias for a frame-0 mask or mask directory.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path.")
    parser.add_argument("--plot-output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--visibility-threshold", type=float, default=0.1)
    parser.add_argument(
        "--ignore-visibility",
        action="store_true",
        help="Ignore confidence values when deciding whether a track is valid.",
    )
    parser.add_argument("--dynamic-mask-threshold", type=int, default=127)
    parser.add_argument("--static-erosion-iterations", type=int, default=4)
    parser.add_argument("--robust-temporal-percentile", type=float, default=95.0)
    # Parse old options for command-line compatibility; they intentionally do
    # not affect the three-metric report.
    parser.add_argument(
        "--relative-thresholds-percent",
        default="0.5,1,2",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--absolute-thresholds",
        default="",
        help=argparse.SUPPRESS,
    )
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    flows_path = getattr(args, "flows_path", None)
    flow_dir = getattr(args, "flow_dir", None)
    if flows_path is not None and flow_dir is not None:
        raise ValueError("--flows-path and --flow-dir cannot be used together.")
    visibility_threshold = getattr(args, "visibility_threshold", 0.1)
    if not math.isfinite(visibility_threshold) or visibility_threshold < 0:
        raise ValueError("--visibility-threshold must be finite and non-negative.")
    dynamic_mask_threshold = getattr(args, "dynamic_mask_threshold", 127)
    if not 0 <= dynamic_mask_threshold <= 255:
        raise ValueError("--dynamic-mask-threshold must be in [0, 255].")
    erosion_iterations = getattr(args, "static_erosion_iterations", 4)
    if erosion_iterations < 0:
        raise ValueError("--static-erosion-iterations must be non-negative.")
    robust_percentile = getattr(args, "robust_temporal_percentile", 95.0)
    if (
        not math.isfinite(robust_percentile)
        or not 0 <= robust_percentile <= 100
    ):
        raise ValueError("--robust-temporal-percentile must be in [0, 100].")


def _print_summary(
    report: Mapping[str, Any],
    output_path: Path,
    plot_path: Path | None = None,
) -> None:
    if "summary" not in report:
        # Gracefully print reports produced by the pre-NPZ evaluator when a
        # caller still uses this helper directly.
        metrics = report.get("metrics", {})
        normalized = metrics.get("percent_of_frame0_static_median_depth", {})
        coverage = metrics.get("coverage", {})
        print("3D-FF static-background evaluation")
        if "static_max_drift50" in normalized:
            print(
                "  Static-MaxDrift50: "
                f"{normalized['static_max_drift50']:.4f}% median depth"
            )
        if "mean_per_frame" in coverage:
            print(f"  mean coverage: {coverage['mean_per_frame']:.2%}")
        print(f"  report: {output_path}")
        if plot_path is not None:
            print(f"  plot: {plot_path}")
        return
    summary = report["summary"]
    max_drift = summary["max_drift"]
    robust = summary["robust_max_drift"]
    coverage = summary["coverage"]
    print("3D-FF static-background horizon evaluation")
    print(f"  horizons: {summary['horizon_count']}")
    print(
        "  MaxDrift median: "
        f"{max_drift['median']:.4f}%"
        if max_drift["median"] is not None
        else "  MaxDrift median: n/a"
    )
    print(
        "  RobustMaxDrift median: "
        f"{robust['median']:.4f}%"
        if robust["median"] is not None
        else "  RobustMaxDrift median: n/a"
    )
    print(
        "  mean coverage: "
        f"{coverage['mean_per_frame']:.2f}%"
        if coverage["mean_per_frame"] is not None
        else "  mean coverage: n/a"
    )
    print(f"  report: {output_path}")
    if plot_path is not None:
        print(f"  plot: {plot_path}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_cli_args(args)

    flows_path = args.flows_path or args.flow_dir
    if flows_path is None:
        flows_path = DEFAULT_FLOWS_PATH
    flows_path = Path(flows_path)
    if flows_path.is_dir():
        flows_path = flows_path / "flows.npz"
    output_path = args.output or flows_path.parent / "static_background_metrics.json"
    plot_path = args.plot_output or flows_path.parent / "static_background_metrics.png"

    report = evaluate_3dff_static_background(
        flows_path,
        args.mask_dir,
        visibility_threshold=args.visibility_threshold,
        ignore_visibility=args.ignore_visibility,
        dynamic_mask_threshold=args.dynamic_mask_threshold,
        static_erosion_iterations=args.static_erosion_iterations,
        robust_temporal_percentile=args.robust_temporal_percentile,
        plot_output=plot_path,
        mask_path=args.mask_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    _print_summary(report, output_path, plot_path)


if __name__ == "__main__":
    main()
