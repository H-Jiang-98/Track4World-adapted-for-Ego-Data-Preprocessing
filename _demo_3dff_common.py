"""Internal inference and output helpers for the final 3D-FF demo.

Input, checkpoint, and basic output helpers live in ``_demo_3dff_support.py``.
For every selected frame ``t``, this script saves:

* ``frame_t.ply``: filtered XYZ points reconstructed from frame ``t``;
* ``pixel_uv_t.npy``: integer ``(u, v)`` image coordinates aligned with the
  vertices in ``frame_t.ply``;
* ``flow_t.ply``: first-frame points tracked to frame ``t``;
* ``flow_pixel_uv.npy``: frame-0 ``(u, v)`` query coordinates shared by every
  ``flow_t.ply``;
* ``vis_t.npy``: confidence values aligned with ``flow_t.ply``.

The core correspondence guarantee is::

    frame_t.ply.vertices[i] <-> pixel_uv_t.npy[i]
    flow_t.ply.vertices[i] <-> flow_pixel_uv.npy[i] <-> vis_t.npy[i]

The UV coordinates refer to the resized image that is actually passed to the
model, not the original-resolution input image.

For world-coordinate backbones, both saved PLY families are transformed into
the frame-0 camera coordinate system. ``camera_base`` intentionally keeps its
original per-frame camera-coordinate behavior because it has no usable poses.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import torch

import utils3d
from _demo_3dff_support import (
    TensorResults,
    save_ply,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# First-frame inference
# =============================================================================


def infer_first_frame(
    rgb_tensor: torch.Tensor,
    model: torch.nn.Module,
    inference_iters: int,
    device: torch.device,
) -> TensorResults:
    """Run full-sequence first-frame 3D tracking and remove the batch axis."""
    if rgb_tensor.ndim != 5 or rgb_tensor.shape[0] != 1:
        raise ValueError(f"Expected RGB shape (1, T, 3, H, W), got {rgb_tensor.shape}.")
    # model.infer() uses a pair-only output layout when T == 2.  Requiring
    # T >= 3 keeps the normal (B, T, ...) first-frame tracking contract.
    if rgb_tensor.shape[1] < 3 or rgb_tensor.shape[2] != 3:
        raise ValueError("3D-FF expects at least three RGB frames with three channels.")

    rgb_gpu = rgb_tensor.to(device=device, dtype=torch.float16, non_blocking=True)
    torch.cuda.empty_cache()
    start_time = time.perf_counter()
    logger.info("Running first-frame 3D inference on %d frame(s).", rgb_tensor.shape[1])
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=torch.float16),
    ):
        output, _ = model.infer(
            rgb_gpu,
            iters=inference_iters,
            sw=None,
            is_training=False,
            tracking3d=True,
            force_projection=True,
        )
    logger.info(
        "First-frame 3D inference finished in %.2f seconds.",
        time.perf_counter() - start_time,
    )

    geometry, motion = output
    results = {
        "traj_3d": motion["flow_3d"][0],
        "visconf": motion["visconf_maps_e"][0],
        # RGB is used only during saving and remains on the CPU.
        "rgbs": rgb_tensor[0],
        "points": geometry["points"][0],
        "masks": geometry["mask"][0],
        "camera_poses": geometry["camera_poses"],
    }
    validate_3dff_results(results)
    return results


def validate_3dff_results(results: TensorResults) -> None:
    """Validate the tensor layouts assumed by 3D-FF post-processing."""
    frame_count, channels, height, width = results["rgbs"].shape
    expected_shapes = {
        "traj_3d": (frame_count, height, width, 3),
        "visconf": (frame_count, 2, height, width),
        "points": (frame_count, height, width, 3),
        "masks": (frame_count, height, width),
    }
    if channels != 3:
        raise ValueError(f"Expected three RGB channels, got {channels}.")
    for key, expected in expected_shapes.items():
        actual = tuple(results[key].shape)
        if actual != expected:
            raise ValueError(
                f"Unexpected {key} shape: expected {expected}, got {actual}."
            )
    expected_pose_shape = (frame_count, 4, 4)
    actual_pose_shape = tuple(results["camera_poses"].shape)
    if actual_pose_shape != expected_pose_shape:
        raise ValueError(
            "Unexpected camera_poses shape: expected "
            f"{expected_pose_shape}, got {actual_pose_shape}."
        )
    if not torch.isfinite(results["camera_poses"]).all().item():
        raise ValueError("camera_poses contains NaN or infinite values.")


# =============================================================================
# 3D-FF post-processing
# =============================================================================


def _finite_depth_mask(point_map: np.ndarray, depth_edge_rtol: float) -> np.ndarray:
    """Return finite points with positive camera depth away from depth edges."""
    depth = point_map[..., 2]
    valid = np.isfinite(point_map).all(axis=-1) & (depth > 0)
    depth_edges = utils3d.numpy.depth_edge(
        depth,
        rtol=depth_edge_rtol,
        mask=valid,
    )
    return valid & ~depth_edges


def _frame0_from_camera_transforms(
    camera_poses: torch.Tensor,
    coordinate: str,
) -> np.ndarray:
    """Return transforms from each camera frame to the frame-0 camera frame."""
    frame_count = camera_poses.shape[0]
    if coordinate == "camera_base":
        # camera_base intentionally retains its camera-centric output contract;
        # its model does not provide usable camera poses.
        return np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    if coordinate not in {"world_depthanythingv3"}:
        raise ValueError(f"Unsupported coordinate mode: {coordinate}.")

    poses = camera_poses.detach().cpu().numpy().astype(np.float64, copy=False)
    expected_bottom_row = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(poses[:, 3, :], expected_bottom_row, atol=1e-5, rtol=0.0):
        raise ValueError("World-coordinate camera_poses are not homogeneous transforms.")
    try:
        # Validate every pose even though only pose 0 needs to be inverted below.
        np.linalg.inv(poses)
        world_to_frame0 = np.linalg.inv(poses[0])
    except np.linalg.LinAlgError as exc:
        raise ValueError("World-coordinate camera_poses must be invertible.") from exc
    return world_to_frame0[None] @ poses


def _transform_vertices(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous camera-to-frame-0 transform without changing rows."""
    vertices_f64 = np.asarray(vertices, dtype=np.float64)
    transformed = vertices_f64 @ transform[:3, :3].T + transform[:3, 3]
    return transformed.astype(np.float32, copy=False)


def _publish_output_directory(staging_dir: Path, output_dir: Path) -> None:
    """Replace a complete 3D-FF output directory, restoring the old run on error."""
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"3D-FF output path is not a directory: {output_dir}")

    backup_dir: Path | None = None
    if output_dir.exists():
        backup_dir = output_dir.with_name(
            f".{output_dir.name}.backup-{uuid.uuid4().hex}"
        )
        output_dir.replace(backup_dir)

    try:
        staging_dir.replace(output_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            backup_dir.replace(output_dir)
        raise

    if backup_dir is not None:
        try:
            shutil.rmtree(backup_dir)
        except OSError as exc:
            logger.warning("Could not remove old 3D-FF backup %s: %s", backup_dir, exc)


def build_flow_mask(
    results: TensorResults,
    depth_edge_rtol: float,
) -> np.ndarray:
    """Build one reference-grid mask shared by every saved 3D-FF flow cloud.

    A fixed mask preserves vertex identity over time, which is required by
    ``visualization/vis_3d_ff.py`` when it connects equal row indices into
    trajectories.
    """
    points = results["points"]
    tracked_points = results["traj_3d"]
    flow_mask = results["masks"][0].cpu().numpy().astype(bool)
    flow_mask &= _finite_depth_mask(points[0].cpu().numpy(), depth_edge_rtol)
    for frame_index in range(tracked_points.shape[0]):
        flow_mask &= _finite_depth_mask(
            tracked_points[frame_index].cpu().numpy(),
            depth_edge_rtol,
        )
    return flow_mask


def save_3dff_results(
    results: TensorResults,
    output_dir: Path,
    depth_edge_rtol: float,
    coordinate: str,
) -> None:
    """Save complete per-frame geometry, tracked points, UVs, and poses."""
    validate_3dff_results(results)
    frame_count, _, height, width = results["rgbs"].shape
    points = results["points"]
    tracked_points = results["traj_3d"]
    model_masks = results["masks"]
    confidence = results["visconf"][:, 0] * results["visconf"][:, 1]

    # Integer pixel coordinates in the resized model input: (H, W, 2), (u, v).
    image_pixel_uv = utils3d.numpy.image_pixel(width=width, height=height)
    flow_mask = build_flow_mask(results, depth_edge_rtol)
    _, flow_pixel_uv = utils3d.numpy.image_mesh(
        image_pixel_uv,
        mask=flow_mask,
        tri=True,
    )
    flow_pixel_uv = flow_pixel_uv.astype(np.int32, copy=False)
    if flow_pixel_uv.ndim != 2 or flow_pixel_uv.shape[1] != 2:
        raise RuntimeError(
            f"Unexpected flow pixel UV shape: {flow_pixel_uv.shape}."
        )
    first_rgb = results["rgbs"][0].permute(1, 2, 0).cpu().numpy()
    first_rgb = first_rgb.astype(np.float32) / 255.0
    frame0_from_camera = _frame0_from_camera_transforms(
        results["camera_poses"],
        coordinate,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )

    try:
        np.save(staging_dir / "flow_pixel_uv.npy", flow_pixel_uv)
        for frame_index in range(frame_count):
            logger.info("Saving 3D-FF frame %d/%d.", frame_index, frame_count - 1)
            point_map = points[frame_index].cpu().numpy()
            rgb = results["rgbs"][frame_index].permute(1, 2, 0).cpu().numpy()
            rgb = rgb.astype(np.float32) / 255.0

            # frame_XXX is indexed on frame XXX's image grid. Pass UV through the
            # same image_mesh call so filtering and row ordering are identical.
            # Depth filtering happens in camera-t before vertices are mapped to
            # the common frame-0 coordinate system.
            frame_mask = model_masks[frame_index].cpu().numpy().astype(bool)
            frame_mask &= _finite_depth_mask(point_map, depth_edge_rtol)
            _, frame_vertices, frame_colors, frame_pixel_uv = utils3d.numpy.image_mesh(
                point_map,
                rgb,
                image_pixel_uv,
                mask=frame_mask,
                tri=True,
            )
            frame_vertices = _transform_vertices(
                frame_vertices,
                frame0_from_camera[frame_index],
            )
            if frame_vertices.shape[0] != frame_pixel_uv.shape[0]:
                raise RuntimeError(
                    f"frame_{frame_index:03d} has {frame_vertices.shape[0]} points "
                    f"but {frame_pixel_uv.shape[0]} pixel coordinates."
                )
            if frame_pixel_uv.ndim != 2 or frame_pixel_uv.shape[1] != 2:
                raise RuntimeError(
                    f"Unexpected pixel UV shape for frame {frame_index}: "
                    f"{frame_pixel_uv.shape}."
                )

            np.save(
                staging_dir / f"pixel_uv_{frame_index:03d}.npy",
                frame_pixel_uv.astype(np.int32, copy=False),
            )
            save_ply(
                staging_dir / f"frame_{frame_index:03d}.ply",
                frame_vertices,
                frame_colors,
            )

            # flow_XXX is indexed on the first-frame grid. The mask is shared by
            # all frames so row i remains the same tracked first-frame point.
            flow_map = tracked_points[frame_index].cpu().numpy()
            visibility_map = confidence[frame_index].cpu().numpy()[..., None]
            _, flow_vertices, flow_colors, flow_visibility = utils3d.numpy.image_mesh(
                flow_map,
                first_rgb,
                visibility_map,
                mask=flow_mask,
                tri=True,
            )
            flow_vertices = _transform_vertices(
                flow_vertices,
                frame0_from_camera[frame_index],
            )
            if not (
                flow_vertices.shape[0]
                == flow_visibility.shape[0]
                == flow_pixel_uv.shape[0]
            ):
                raise RuntimeError(
                    f"flow_{frame_index:03d} row mismatch: "
                    f"{flow_vertices.shape[0]} points, "
                    f"{flow_visibility.shape[0]} confidence values, and "
                    f"{flow_pixel_uv.shape[0]} reference UV coordinates."
                )
            np.save(staging_dir / f"vis_{frame_index:03d}.npy", flow_visibility)
            save_ply(
                staging_dir / f"flow_{frame_index:03d}.ply",
                flow_vertices,
                flow_colors,
            )

        np.save(
            staging_dir / "c2w.npy",
            results["camera_poses"].detach().cpu().numpy(),
        )
        _publish_output_directory(staging_dir, output_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    logger.info("Saved 3D-FF results for %d frame(s) to %s.", frame_count, output_dir)


# =============================================================================
# Command-line entry point
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track4World standalone first-frame 3D tracking demo"
    )
    parser.add_argument(
        "--ckpt_init",
        default=None,
        help=(
            "Local checkpoint. By default, use the coordinate-specific file below "
            "checkpoints/ or download it if absent."
        ),
    )
    parser.add_argument("--mp4_path", default="demo_data/cat.mp4")
    parser.add_argument(
        "--rgb_dir",
        default=None,
        help="Ordered RGB image directory; takes precedence over --mp4_path.",
    )
    parser.add_argument(
        "--rgb_fps",
        type=float,
        default=15.0,
        help="Capture frequency for --rgb_dir input (default: 15 Hz).",
    )
    parser.add_argument(
        "--config_path",
        default="track4world/config/eval/v1.json",
    )
    parser.add_argument(
        "--save_base_dir",
        default="results/cat_ego",
        help="Output root; the caller selects the final 3D-FF output subdirectory.",
    )
    parser.add_argument(
        "--mask_dir",
        default=None,
        help="Dynamic-mask directory; defaults to <save_base_dir>/mask.",
    )
    parser.add_argument(
        "--mask_index_base",
        choices=("auto", "0", "1"),
        default="auto",
    )
    parser.add_argument(
        "--coordinate",
        choices=("world_depthanythingv3",),
        default="world_depthanythingv3",
    )
    parser.add_argument("--use_original_backbone", action="store_true")
    parser.add_argument(
        "--metric_scale",
        action="store_true",
        help="Enable metric scale for a compatible DA3 checkpoint.",
    )
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=1000,
        help="Hard input-frame cap; use 0 for no cap.",
    )
    parser.add_argument(
        "--Ts",
        "--num_frames",
        dest="num_frames",
        type=int,
        default=-1,
        help="Number of leading frames to process; -1 means all available frames.",
    )
    parser.add_argument("--inference_iters", type=int, default=4)
    parser.add_argument("--depth_edge_rtol", type=float, default=0.04)
    parser.add_argument(
        "--mode",
        choices=("3d_ff",),
        default="3d_ff",
        help=argparse.SUPPRESS,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_frames != -1 and args.num_frames < 3:
        raise ValueError("--Ts/--num_frames must be -1 or at least 3 for 3D-FF.")
    if args.max_frames < 0:
        raise ValueError("--max_frames must be non-negative.")
    if 0 < args.max_frames < 3:
        raise ValueError("--max_frames must be 0 or at least 3 for 3D-FF.")
    if args.inference_iters < 1:
        raise ValueError("--inference_iters must be positive.")
    if args.rgb_fps <= 0:
        raise ValueError("--rgb_fps must be positive.")
    if args.depth_edge_rtol < 0:
        raise ValueError("--depth_edge_rtol must be non-negative.")
    if args.metric_scale and args.coordinate != "world_depthanythingv3":
        raise ValueError("--metric_scale requires --coordinate world_depthanythingv3.")
