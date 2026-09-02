"""Evaluate metric scale from chessboard corners in ego-centric 3D-FF output.

The ego-centric demo stores one ragged query interval per source frame in
``flows.npz``.  The first track sample, ``track_xyz[0]``, is the authoritative
DA3 source geometry for that source; later samples are point-flow targets and
are intentionally not used here.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from evaluate_3dff_static_background import load_flows_npz


DEFAULT_PATTERN_COLS = 8
DEFAULT_PATTERN_ROWS = 11
DEFAULT_OUTPUT_NAME = "chessboard_scale_metrics.json"
DEFAULT_PLOT_NAME = "chessboard_scale_metrics.png"
DEFAULT_OVERLAY_NAME = "chessboard_corner_overlays"


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Chessboard evaluation requires opencv-python (cv2)."
        ) from exc
    return cv2


def _finite_positive(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if result < minimum or result != value:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def _scalar(payload: Mapping[str, np.ndarray], name: str, default: Any = None) -> Any:
    if name not in payload:
        return default
    value = np.asarray(payload[name])
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    return value.item()


def _resolve_frame_path(rgb_dir: Path, frame_index: int) -> Path | None:
    """Resolve the preprocessed model-grid RGB frame for one global index."""
    candidates = (
        rgb_dir / f"frame_{frame_index:04d}.png",
        rgb_dir / f"frame_{frame_index}.png",
        rgb_dir / f"{frame_index:05d}.png",
        rgb_dir / f"{frame_index:04d}.png",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_source_point_map(
    query_uv: np.ndarray,
    source_xyz: np.ndarray,
    source_valid: np.ndarray,
    image_size_hw: tuple[int, int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scatter one source's ragged query rows into an HxWx3 point map."""
    height, width = (int(value) for value in image_size_hw)
    if height <= 0 or width <= 0:
        raise ValueError("image_size_hw must contain positive dimensions.")
    uv = np.asarray(query_uv)
    xyz = np.asarray(source_xyz)
    valid = np.asarray(source_valid, dtype=bool)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"query_uv must have shape (N,2), got {uv.shape}.")
    if xyz.shape != (uv.shape[0], 3):
        raise ValueError(f"source_xyz must have shape ({uv.shape[0]},3), got {xyz.shape}.")
    if valid.shape != (uv.shape[0],):
        raise ValueError(f"source_valid must have shape ({uv.shape[0]},), got {valid.shape}.")
    if not np.issubdtype(uv.dtype, np.integer):
        raise ValueError("query_uv must contain integer pixel coordinates.")
    uv = uv.astype(np.int64, copy=False)
    in_bounds = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    if not in_bounds.all():
        bad = int(np.flatnonzero(~in_bounds)[0])
        raise ValueError(f"query_uv row {bad} is outside image_size_hw={height,width}.")

    point_map = np.full((height, width, 3), np.nan, dtype=np.float64)
    valid_map = np.zeros((height, width), dtype=bool)
    if uv.shape[0]:
        linear = uv[:, 1] * width + uv[:, 0]
        if np.unique(linear).size != linear.size:
            raise ValueError("A source query interval contains duplicate query_uv rows.")
        finite = np.isfinite(xyz).all(axis=1)
        usable = valid & finite
        point_map[uv[usable, 1], uv[usable, 0]] = xyz[usable]
        valid_map[uv[usable, 1], uv[usable, 0]] = True
    return point_map, valid_map


def bilinear_sample_points(
    point_map: np.ndarray,
    valid_map: np.ndarray,
    corners_uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a point map at subpixel corners using valid four-neighborhoods."""
    points = np.asarray(point_map)
    valid_pixels = np.asarray(valid_map, dtype=bool)
    corners = np.asarray(corners_uv, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"point_map must have shape (H,W,3), got {points.shape}.")
    if valid_pixels.shape != points.shape[:2]:
        raise ValueError("valid_map shape must match point_map spatial dimensions.")
    if corners.ndim != 2 or corners.shape[1] != 2:
        raise ValueError(f"corners_uv must have shape (N,2), got {corners.shape}.")
    if not np.isfinite(corners).all():
        raise ValueError("corners_uv must contain finite coordinates.")

    height, width = points.shape[:2]
    output = np.full((corners.shape[0], 3), np.nan, dtype=np.float64)
    sampled_valid = np.zeros(corners.shape[0], dtype=bool)
    if corners.shape[0] == 0:
        return output, sampled_valid

    x = corners[:, 0]
    y = corners[:, 1]
    inside = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    if not inside.any():
        return output, sampled_valid

    indices = np.flatnonzero(inside)
    xi = x[indices]
    yi = y[indices]
    x0 = np.floor(xi).astype(np.int64)
    y0 = np.floor(yi).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = (xi - x0)[:, None]
    dy = (yi - y0)[:, None]

    neighborhood_valid = (
        valid_pixels[y0, x0]
        & valid_pixels[y0, x1]
        & valid_pixels[y1, x0]
        & valid_pixels[y1, x1]
    )
    if neighborhood_valid.any():
        local = np.flatnonzero(neighborhood_valid)
        p00 = points[y0[local], x0[local]]
        p10 = points[y0[local], x1[local]]
        p01 = points[y1[local], x0[local]]
        p11 = points[y1[local], x1[local]]
        values = (
            (1.0 - dx[local]) * (1.0 - dy[local]) * p00
            + dx[local] * (1.0 - dy[local]) * p10
            + (1.0 - dx[local]) * dy[local] * p01
            + dx[local] * dy[local] * p11
        )
        finite = np.isfinite(values).all(axis=1)
        good_local = local[finite]
        output[indices[good_local]] = values[finite]
        sampled_valid[indices[good_local]] = True
    return output, sampled_valid


def fit_similarity_scale(
    true_points: np.ndarray,
    predicted_points: np.ndarray,
) -> dict[str, Any]:
    """Fit a proper-rotation similarity transform and return its scale."""
    reference = np.asarray(true_points, dtype=np.float64)
    observed = np.asarray(predicted_points, dtype=np.float64)
    if reference.shape != observed.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("true_points and predicted_points must both have shape (N,3).")
    if reference.shape[0] < 3 or not np.isfinite(reference).all() or not np.isfinite(observed).all():
        raise ValueError("Similarity fitting requires at least 3 finite point pairs.")

    centered_reference = reference - reference.mean(axis=0)
    centered_observed = observed - observed.mean(axis=0)
    variance = float(np.mean(np.sum(centered_reference * centered_reference, axis=1)))
    if not math.isfinite(variance) or variance <= 1e-15:
        raise ValueError("Reference points are degenerate for similarity fitting.")
    covariance = centered_reference.T @ centered_observed / reference.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Similarity fit produced an invalid positive scale: {scale}.")
    aligned = scale * centered_reference @ rotation + observed.mean(axis=0)
    residual = observed - aligned
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    translation = observed.mean(axis=0) - scale * reference.mean(axis=0) @ rotation
    return {
        "scale_ratio": scale,
        "translation": translation.tolist(),
        "rotation": rotation.tolist(),
        "rmse_coordinate_units": rmse,
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p05": None,
            "p95": None,
        }
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def _edge_metrics(predicted_points: np.ndarray, rows: int, cols: int, square_size_m: float) -> dict[str, Any]:
    grid = np.asarray(predicted_points, dtype=np.float64).reshape(rows, cols, 3)
    horizontal = np.linalg.norm(grid[:, 1:] - grid[:, :-1], axis=-1).reshape(-1)
    vertical = np.linalg.norm(grid[1:, :] - grid[:-1, :], axis=-1).reshape(-1)

    def one_direction(lengths: np.ndarray) -> dict[str, Any]:
        ratios = lengths / square_size_m
        signed = (ratios - 1.0) * 100.0
        absolute = np.abs(signed)
        return {
            "edge_count": int(lengths.size),
            "predicted_length_m": _distribution(lengths),
            "scale_ratio": _distribution(ratios),
            "signed_error_percent": _distribution(signed),
            "absolute_error_percent": _distribution(absolute),
        }

    return {
        "horizontal": one_direction(horizontal),
        "vertical": one_direction(vertical),
        "combined": one_direction(np.concatenate([horizontal, vertical])),
    }


def _write_json(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.tmp-", suffix=".json", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def plot_chessboard_scale_metrics(source_reports: list[Mapping[str, Any]], path: Path) -> Path:
    """Write a compact cross-source diagnostic plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    successful = [item for item in source_reports if item.get("status") == "ok"]
    if not successful:
        raise ValueError("Cannot plot chessboard metrics without successful sources.")
    sources = np.asarray([item["source_frame_index"] for item in successful], dtype=float)
    ratios = np.asarray([item["fit"]["scale_ratio"] for item in successful], dtype=float)
    signed = np.asarray([item["fit"]["signed_error_percent"] for item in successful], dtype=float)
    absolute = np.asarray([item["fit"]["absolute_error_percent"] for item in successful], dtype=float)
    horizontal = np.asarray([
        item["edges"]["horizontal"]["signed_error_percent"]["median"] for item in successful
    ], dtype=float)
    vertical = np.asarray([
        item["edges"]["vertical"]["signed_error_percent"]["median"] for item in successful
    ], dtype=float)
    rmse = np.asarray([item["fit"]["rmse_coordinate_units"] for item in successful], dtype=float)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(sources, ratios, "o-", label="similarity scale ratio α")
    axes[0, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("predicted / true")
    axes[0, 0].set_title("Whole-board scale")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 1].plot(sources, signed, "o-", label="signed")
    axes[0, 1].plot(sources, absolute, "s--", label="absolute")
    axes[0, 1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_ylabel("error (%)")
    axes[0, 1].set_title("Whole-board scale error")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)
    axes[1, 0].plot(sources, horizontal, "o-", label="horizontal edges")
    axes[1, 0].plot(sources, vertical, "s-", label="vertical edges")
    axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("source frame")
    axes[1, 0].set_ylabel("median signed error (%)")
    axes[1, 0].set_title("Adjacent edge lengths")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)
    axes[1, 1].plot(sources, rmse, "o-")
    axes[1, 1].set_xlabel("source frame")
    axes[1, 1].set_ylabel("RMSE (coordinate units)")
    axes[1, 1].set_title("Similarity-fit residual")
    axes[1, 1].grid(alpha=0.25)
    for axis in axes.flat:
        axis.set_xticks(sources)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _draw_overlay(
    image: np.ndarray,
    corners: np.ndarray | None,
    pattern_size: tuple[int, int],
    text: str,
    path: Path,
) -> None:
    cv2 = _require_cv2()
    canvas = image.copy()
    if corners is not None and corners.shape[0] == pattern_size[0] * pattern_size[1]:
        cv2.drawChessboardCorners(
            canvas,
            pattern_size,
            corners.astype(np.float32, copy=False).reshape(-1, 1, 2),
            True,
        )
        for index, corner in enumerate(corners):
            x, y = (int(round(value)) for value in corner)
            cv2.putText(canvas, str(index), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1] - 1, 600), 28), (0, 0, 0), -1)
    cv2.putText(canvas, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Cannot write chessboard overlay: {path}")


def _summary(values: list[float]) -> dict[str, Any]:
    return _distribution(np.asarray(values, dtype=np.float64))


def evaluate_chessboard_scale(
    flows_path: Path | str,
    *,
    rgb_dir: Path | str | None = None,
    pattern_cols: int = DEFAULT_PATTERN_COLS,
    pattern_rows: int = DEFAULT_PATTERN_ROWS,
    square_size_m: float,
    output: Path | str | None = None,
    plot_output: Path | str | None = None,
    overlay_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate source-frame metric scale using detected chessboard corners."""
    flows_path = Path(flows_path)
    pattern_cols = _positive_int(pattern_cols, "pattern_cols", minimum=2)
    pattern_rows = _positive_int(pattern_rows, "pattern_rows", minimum=2)
    square_size_m = _finite_positive(square_size_m, "square_size_m")
    payload = load_flows_npz(flows_path)
    metric_enabled = bool(_scalar(payload, "metric_scale_enabled", False))
    if not metric_enabled:
        raise ValueError(
            "flows.npz was generated without --metric_scale; metric-scale evaluation requires "
            "metric_scale_enabled=True."
        )

    image_size = tuple(int(value) for value in payload["image_size_hw"])
    if rgb_dir is None:
        rgb_dir_path = flows_path.parent.parent / "final_rgb"
    else:
        rgb_dir_path = Path(rgb_dir)
    if not rgb_dir_path.is_dir():
        raise NotADirectoryError(f"RGB directory not found: {rgb_dir_path}")

    output_path = Path(output) if output is not None else flows_path.parent / DEFAULT_OUTPUT_NAME
    plot_path = (
        Path(plot_output) if plot_output is not None
        else output_path.with_name(DEFAULT_PLOT_NAME)
    )
    overlays_path = (
        Path(overlay_dir) if overlay_dir is not None
        else output_path.parent / DEFAULT_OVERLAY_NAME
    )
    overlays_path.mkdir(parents=True, exist_ok=True)

    cv2 = _require_cv2()
    pattern_size = (pattern_cols, pattern_rows)
    flags = (
        getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0)
        | getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
        | getattr(cv2, "CALIB_CB_ACCURACY", 0)
    )
    query_uv = payload["query_uv"]
    offsets = payload["query_offsets"]
    source_indices = payload["source_frame_index"]
    tracks = payload["track_xyz"]
    track_valid = payload["track_valid"]
    true_grid = np.stack(
        np.meshgrid(
            np.arange(pattern_cols, dtype=np.float64) * square_size_m,
            np.arange(pattern_rows, dtype=np.float64) * square_size_m,
            indexing="xy",
        ),
        axis=-1,
    )
    true_points = np.concatenate(
        [true_grid.reshape(-1, 2), np.zeros((pattern_cols * pattern_rows, 1), dtype=np.float64)],
        axis=1,
    )

    source_reports: list[dict[str, Any]] = []
    for source_position, source_value in enumerate(source_indices):
        source = int(source_value)
        report: dict[str, Any] = {
            "source_frame_index": source,
            "source_position": int(source_position),
            "status": "skipped",
        }
        frame_path = _resolve_frame_path(rgb_dir_path, source)
        report["frame_path"] = str(frame_path.resolve()) if frame_path is not None else None
        corners: np.ndarray | None = None
        image = None
        try:
            if frame_path is None:
                raise FileNotFoundError(f"No RGB frame for source {source} in {rgb_dir_path}.")
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Cannot decode RGB frame: {frame_path}")
            if image.shape[:2] != image_size:
                raise ValueError(
                    f"RGB frame has shape {image.shape[:2]}; expected {image_size} from flows.npz."
                )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if not hasattr(cv2, "findChessboardCornersSB"):
                raise RuntimeError("This OpenCV build lacks findChessboardCornersSB.")
            detected, detected_corners = cv2.findChessboardCornersSB(
                gray, pattern_size, flags=flags
            )
            corners = (
                np.asarray(detected_corners, dtype=np.float64).reshape(-1, 2)
                if detected_corners is not None
                else np.empty((0, 2), dtype=np.float64)
            )
            report["detected_corner_count"] = int(corners.shape[0])
            report["detection_success"] = bool(detected and corners.shape[0] == true_points.shape[0])
            if not report["detection_success"]:
                raise ValueError(
                    f"Chessboard detection failed: expected {true_points.shape[0]} corners, "
                    f"got {corners.shape[0]}."
                )

            q0, q1 = int(offsets[source_position]), int(offsets[source_position + 1])
            point_map, valid_map = build_source_point_map(
                query_uv[q0:q1], tracks[0, q0:q1], track_valid[0, q0:q1], image_size
            )
            predicted_points, corner_valid = bilinear_sample_points(
                point_map, valid_map, corners
            )
            report["valid_3d_corner_count"] = int(corner_valid.sum())
            if not corner_valid.all():
                raise ValueError(
                    f"Only {int(corner_valid.sum())}/{true_points.shape[0]} chessboard corners "
                    "have valid source 3D samples."
                )
            # Keep the exact correspondences in the report so a scale result
            # can be audited without reopening the large NPZ point cloud.
            report["corners_uv"] = corners.tolist()
            report["corner_xyz"] = predicted_points.tolist()
            fit = fit_similarity_scale(true_points, predicted_points)
            fit["predicted_square_size_m"] = fit["scale_ratio"] * square_size_m
            fit["signed_error_percent"] = (fit["scale_ratio"] - 1.0) * 100.0
            fit["absolute_error_percent"] = abs(fit["signed_error_percent"])
            fit["rmse_in_square_units"] = fit["rmse_coordinate_units"] / square_size_m
            report["fit"] = fit
            report["edges"] = _edge_metrics(
                predicted_points, pattern_rows, pattern_cols, square_size_m
            )
            report["status"] = "ok"
        except (FileNotFoundError, OSError, RuntimeError, ValueError, cv2.error) as exc:
            report["reason"] = str(exc)
        finally:
            if image is not None:
                overlay_text = (
                    f"source {source}: {report['status']}"
                    + (f" ({report['reason']})" if report["status"] != "ok" else "")
                )
                _draw_overlay(image, corners, pattern_size, overlay_text, overlays_path / f"frame_{source:04d}.png")
        source_reports.append(report)

    successful = [item for item in source_reports if item["status"] == "ok"]
    scale_ratios = [float(item["fit"]["scale_ratio"]) for item in successful]
    signed_errors = [float(item["fit"]["signed_error_percent"]) for item in successful]
    absolute_errors = [float(item["fit"]["absolute_error_percent"]) for item in successful]
    edge_ratios = [
        float(item["edges"]["combined"]["scale_ratio"]["median"])
        for item in successful
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "metric_name": "chessboard source-geometry metric-scale accuracy",
        "status": "ok" if successful else "failed",
        "input": {
            "flows_path": str(flows_path.resolve()),
            "rgb_directory": str(rgb_dir_path.resolve()),
            "coordinate_system": str(_scalar(payload, "coordinate_system", "source_camera")),
            "pixel_convention": str(_scalar(payload, "pixel_convention", "unpadded_uv_integer")),
            "metric_scale_enabled": metric_enabled,
            "metric_scale_raw": _scalar(payload, "metric_scale", None),
            "image_size_hw": list(image_size),
        },
        "configuration": {
            "pattern_inner_corners": [pattern_cols, pattern_rows],
            "square_size_m": square_size_m,
            "corner_detector": "cv2.findChessboardCornersSB",
            "corner_sampling": "bilinear_four_neighbor_finite_only",
            "source_track_sample": "track_xyz[0, q0:q1]",
        },
        "summary": {
            "source_count": int(len(source_reports)),
            "evaluated_source_count": int(len(successful)),
            "skipped_source_count": int(len(source_reports) - len(successful)),
            "scale_ratio": _summary(scale_ratios),
            "signed_scale_error_percent": _summary(signed_errors),
            "absolute_scale_error_percent": _summary(absolute_errors),
            "combined_edge_scale_ratio_median_per_source": _summary(edge_ratios),
        },
        "sources": source_reports,
        "outputs": {
            "json": str(output_path.resolve()),
            "plot": str(plot_path.resolve()) if successful else None,
            "corner_overlay_directory": str(overlays_path.resolve()),
        },
    }
    _write_json(output_path, report)
    if successful:
        plot_chessboard_scale_metrics(source_reports, plot_path)
    else:
        raise ValueError(
            f"No source frame had a complete valid chessboard; diagnostics were written to {output_path}."
        )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate metric-scale accuracy from source-frame chessboard 3D corners."
    )
    parser.add_argument("--flows-path", type=Path, required=True, help="Path to flows.npz.")
    parser.add_argument(
        "--rgb-dir", type=Path, default=None,
        help="Model-grid RGB directory; defaults to <flows parent>/../final_rgb.",
    )
    parser.add_argument("--pattern-cols", type=int, default=DEFAULT_PATTERN_COLS,
                        help="Chessboard inner-corner columns (default: 8).")
    parser.add_argument("--pattern-rows", type=int, default=DEFAULT_PATTERN_ROWS,
                        help="Chessboard inner-corner rows (default: 11).")
    parser.add_argument("--square-size-m", type=float, required=True,
                        help="True chessboard square edge length in metres.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path.")
    parser.add_argument("--plot-output", type=Path, default=None, help="Output plot PNG path.")
    parser.add_argument("--overlay-dir", type=Path, default=None,
                        help="Directory for per-source corner overlay PNGs.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = evaluate_chessboard_scale(
        args.flows_path,
        rgb_dir=args.rgb_dir,
        pattern_cols=args.pattern_cols,
        pattern_rows=args.pattern_rows,
        square_size_m=args.square_size_m,
        output=args.output,
        plot_output=args.plot_output,
        overlay_dir=args.overlay_dir,
    )
    summary = report["summary"]
    print("Chessboard metric-scale evaluation")
    print(f"  evaluated sources: {summary['evaluated_source_count']}/{summary['source_count']}")
    print(f"  scale ratio median: {summary['scale_ratio']['median']:.6f}")
    print(
        "  absolute scale error median: "
        f"{summary['absolute_scale_error_percent']['median']:.4f}%"
    )
    print(f"  report: {report['outputs']['json']}")
    print(f"  plot: {report['outputs']['plot']}")
    print(f"  overlays: {report['outputs']['corner_overlay_directory']}")


if __name__ == "__main__":
    main()
