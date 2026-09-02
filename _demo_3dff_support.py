"""Input, checkpoint, and basic output helpers shared by 3D-FF entry points."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch


logger = logging.getLogger(__name__)

TensorResults = dict[str, torch.Tensor]
SUPPORTED_MASK_SUFFIXES = {".png", ".jpg", ".jpeg"}
SUPPORTED_RGB_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CHECKPOINT_URLS = {
    "world_depthanythingv3": (
        "https://huggingface.co/cyun9286/Track4World/resolve/main/track4world_da3.pth"
    ),
    "camera_base": (
        "https://huggingface.co/cyun9286/Track4World/resolve/main/track4world_moge.pth"
    ),
}
DEFAULT_CHECKPOINTS = {
    "world_depthanythingv3": Path("checkpoints/track4world_da3.pth"),
    "camera_base": Path("checkpoints/track4world_moge.pth"),
}


def read_video(
    video_path: Path,
    frame_limit: int | None = None,
) -> tuple[list[np.ndarray], float]:
    """Read RGB video frames, optionally stopping at ``frame_limit``."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Cannot open video file: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        logger.warning("Video FPS is unavailable; using 24 FPS for saved RGB video.")
        fps = 24.0

    frames: list[np.ndarray] = []
    while frame_limit is None or len(frames) < frame_limit:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    capture.release()

    if not frames:
        raise ValueError(f"No frames were decoded from {video_path}")
    expected_shape = frames[0].shape
    if any(frame.shape != expected_shape for frame in frames):
        raise ValueError("All video frames must have the same resolution and channels.")
    logger.info("Read %d frame(s) from %s at %.3f FPS.", len(frames), video_path, fps)
    return frames, fps


def read_rgb_sequence(
    rgb_dir: Path,
    fps: float,
    frame_limit: int | None = None,
) -> tuple[list[np.ndarray], float]:
    """Read a lexically ordered RGB image sequence from a directory."""
    if not rgb_dir.is_dir():
        raise NotADirectoryError(f"RGB image directory not found: {rgb_dir}")
    image_paths = sorted(
        path
        for path in rgb_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_RGB_SUFFIXES
    )
    if frame_limit is not None:
        image_paths = image_paths[:frame_limit]
    if not image_paths:
        raise ValueError(f"No supported RGB images found in {rgb_dir}")

    frames: list[np.ndarray] = []
    for image_path in image_paths:
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise OSError(f"Cannot read RGB image: {image_path}")
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    expected_shape = frames[0].shape
    if any(frame.shape != expected_shape for frame in frames):
        raise ValueError("All RGB sequence images must have the same resolution.")
    logger.info("Read %d RGB image(s) from %s at %.3f FPS.", len(frames), rgb_dir, fps)
    return frames, fps


def compute_resized_shape(
    original_height: int,
    original_width: int,
    image_size: int,
    multiple: int = 64,
) -> tuple[int, int]:
    """Fit the largest dimension to ``image_size`` and floor to a multiple."""
    if image_size < multiple:
        raise ValueError(f"--image_size must be at least {multiple}, got {image_size}.")
    scale = image_size / max(original_height, original_width)
    height = (int(original_height * scale) // multiple) * multiple
    width = (int(original_width * scale) // multiple) * multiple
    if height < multiple or width < multiple:
        raise ValueError(
            "The resized frame would have a dimension below 64 pixels. "
            "Increase --image_size or use a less extreme aspect ratio."
        )
    return height, width


def _trailing_frame_index(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def _map_masks_to_frames(
    mask_files: Sequence[Path],
    frame_count: int,
    index_base: str,
) -> dict[int, Path]:
    """Map mask files to zero-based frames without partial misalignment."""
    parsed = [_trailing_frame_index(path) for path in mask_files]
    all_indexed = all(index is not None for index in parsed)
    unique_indices = len(set(parsed)) == len(parsed)
    if all_indexed and not unique_indices:
        duplicates = sorted(index for index in set(parsed) if parsed.count(index) > 1)
        raise ValueError(f"Duplicate mask frame indices: {duplicates}")

    if all_indexed:
        raw_indices = [int(index) for index in parsed if index is not None]
        if index_base == "0":
            offset = 0
        elif index_base == "1":
            offset = 1
        elif len(mask_files) >= frame_count and set(raw_indices) == set(
            range(1, len(mask_files) + 1)
        ):
            offset = 1
            logger.info("Detected one-based mask filenames.")
        else:
            offset = 0
            if raw_indices and 0 not in raw_indices and len(mask_files) < frame_count:
                logger.warning(
                    "Partial mask filenames do not contain frame 0; treating their "
                    "numeric suffixes as zero-based. Use --mask_index_base 1 if needed."
                )

        mapping: dict[int, Path] = {}
        for raw_index, path in zip(raw_indices, mask_files):
            frame_index = raw_index - offset
            if 0 <= frame_index < frame_count:
                if frame_index in mapping:
                    raise ValueError(
                        f"Multiple masks resolve to frame {frame_index}: "
                        f"{mapping[frame_index]} and {path}"
                    )
                mapping[frame_index] = path
            else:
                logger.debug("Ignoring out-of-range mask %s.", path)
        return mapping

    if len(mask_files) != frame_count:
        raise ValueError(
            f"Found {len(mask_files)} mask files for {frame_count} frames, but the "
            "filenames do not provide unique trailing frame indices. Rename masks "
            "like 00000.png or provide either all masks or no masks."
        )
    logger.warning(
        "Mask filenames have no unique trailing indices; aligning them by sorted order."
    )
    return {index: path for index, path in enumerate(mask_files)}


def load_dynamic_masks(
    mask_dir: Path,
    frame_count: int,
    width: int,
    height: int,
    index_base: str = "auto",
) -> torch.Tensor:
    """Load masks as ``(T,H,W)`` floats, filling missing frames with zero."""
    masks = torch.zeros((frame_count, height, width), dtype=torch.float32)
    if not mask_dir.is_dir():
        logger.warning(
            "Mask directory %s does not exist; treating every frame as static.",
            mask_dir,
        )
        return masks
    mask_files = sorted(
        path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_MASK_SUFFIXES
    )
    if not mask_files:
        logger.warning("No mask images found in %s; treating every frame as static.", mask_dir)
        return masks

    frame_to_mask = _map_masks_to_frames(mask_files, frame_count, index_base)
    loaded_count = 0
    for frame_index, mask_path in frame_to_mask.items():
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_image is None:
            logger.warning(
                "Could not read %s; frame %d will use an all-static mask.",
                mask_path,
                frame_index,
            )
            continue
        resized = cv2.resize(
            mask_image,
            dsize=(width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        dynamic = resized > 0 if resized.ndim == 2 else np.any(resized > 0, axis=-1)
        masks[frame_index] = torch.from_numpy(dynamic.astype(np.float32))
        loaded_count += 1
    missing_count = frame_count - loaded_count
    if missing_count:
        logger.warning(
            "%d/%d selected frame(s) have no readable indexed mask; those frames "
            "are static.",
            missing_count,
            frame_count,
        )
    return masks


def prepare_inputs(
    args: argparse.Namespace,
    return_original_size: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, float]
    | tuple[torch.Tensor, torch.Tensor, float, tuple[int, int]]
):
    """Read, select, resize, and stack 3D-FF RGB frames and dynamic masks.

    The default return value is kept backward compatible with the original
    three-tuple API.  ``return_original_size=True`` additionally returns the
    source frame size as ``(height, width)``.  The latter is needed when an
    output calibration has to be converted from the resized model grid back to
    the original image pixel units.
    """
    limits = [limit for limit in (args.max_frames, args.num_frames) if limit > 0]
    frame_limit = min(limits) if limits else None
    if args.rgb_dir is not None:
        frames, fps = read_rgb_sequence(
            Path(args.rgb_dir), fps=args.rgb_fps, frame_limit=frame_limit
        )
    else:
        frames, fps = read_video(Path(args.mp4_path), frame_limit=frame_limit)
    if len(frames) < 2:
        raise ValueError("3D-FF input requires at least two frames.")

    original_height, original_width = frames[0].shape[:2]
    height, width = compute_resized_shape(
        original_height, original_width, args.image_size
    )
    logger.info(
        "Resizing selected frames from (%d,%d) to (%d,%d).",
        original_height,
        original_width,
        height,
        width,
    )
    resized_frames = [
        cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        for frame in frames
    ]
    rgb_tensor = (
        torch.stack(
            [torch.from_numpy(frame).permute(2, 0, 1) for frame in resized_frames]
        )
        .unsqueeze(0)
        .float()
    )
    mask_dir = (
        Path(args.mask_dir)
        if args.mask_dir is not None
        else Path(args.save_base_dir) / "mask"
    )
    logger.info("Loading dynamic masks from %s.", mask_dir)
    dynamic_masks = load_dynamic_masks(
        mask_dir,
        frame_count=len(frames),
        width=width,
        height=height,
        index_base=args.mask_index_base,
    )
    logger.info("3D-FF input tensor shape: %s.", tuple(rgb_tensor.shape))
    result = (rgb_tensor, dynamic_masks, fps)
    if return_original_size:
        return rgb_tensor, dynamic_masks, fps, (original_height, original_width)
    return result


def save_preprocessed_inputs(
    rgb_tensor: torch.Tensor,
    dynamic_masks: torch.Tensor,
    save_base_dir: Path,
    fps: float,
) -> None:
    """Save selected resized RGB frames and masks in the 3D-FF layout."""
    rgb_dir = save_base_dir / "final_rgb"
    mask_dir = save_base_dir / "final_dyn_mask"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    _, frame_count, _, height, width = rgb_tensor.shape
    writer = cv2.VideoWriter(
        str(rgb_dir / "rgb.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise OSError(f"Cannot create output video: {rgb_dir / 'rgb.mp4'}")
    try:
        for frame_index in range(frame_count):
            rgb = (
                rgb_tensor[0, frame_index]
                .permute(1, 2, 0)
                .clamp(0, 255)
                .byte()
                .numpy()
            )
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rgb_path = rgb_dir / f"frame_{frame_index:04d}.png"
            mask_path = mask_dir / f"mask_{frame_index:04d}.png"
            if not cv2.imwrite(str(rgb_path), rgb_bgr):
                raise OSError(f"Cannot write RGB frame: {rgb_path}")
            mask = (dynamic_masks[frame_index].numpy() * 255).astype(np.uint8)
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError(f"Cannot write dynamic mask: {mask_path}")
            writer.write(rgb_bgr)
    finally:
        writer.release()
    logger.info("Saved %d preprocessed RGB frame(s) and mask(s).", frame_count)


def save_ply(
    path: Path,
    vertices: np.ndarray,
    vertex_colors: np.ndarray,
) -> None:
    """Save a colored point cloud as PLY without geometry reprocessing."""
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Saving 3D-FF point clouds requires trimesh.") from exc
    point_cloud = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.empty((0, 3), dtype=np.int32),
        vertex_colors=vertex_colors,
        process=False,
    )
    point_cloud.export(path)


def copy_input_video(video_path: Path, save_base_dir: Path) -> None:
    """Copy the source video into the result root unless it is already there."""
    destination = save_base_dir / "input_copy.mp4"
    if video_path.resolve() == destination.resolve():
        logger.info("Input video already equals %s; skipping reference copy.", destination)
        return
    shutil.copy2(video_path, destination)
    logger.info("Copied input video to %s.", destination)
