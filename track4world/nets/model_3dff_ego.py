"""Isolated 3D-FF variant with one shared four-parameter DA3 intrinsic matrix.

The Track4World weights and DA3 predictions are unchanged.  An adapter records
DA3's raw per-frame pixel intrinsics, replaces them with their temporal mean,
and lets the final 3D-FF pipeline reconstruct every geometric output with that
the same temporal-mean K.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from track4world.nets._model_3dff_core import (
    _Track4World3DFFCore,
    _crop_normalized_intrinsics,
    _normalize_da3_intrinsics,
)
from track4world.nets.blocks import InputPadder


INTRINSIC_PARAMETER_NAMES = ("fx", "fy", "cx", "cy")


@dataclass(frozen=True)
class SequenceCache:
    """Full-sequence DA3 geometry and flow features for one input video.

    Geometry tensors retain the spatial padding used by the legacy inference
    path.  Temporal feature tensors use ``(B,T,...)`` while geometry and pose
    tensors retain the legacy flattened ``(B*T,...)`` layout.  B is fixed to
    one so slicing either representation is unambiguous.
    """

    t_eff: int
    batch_size: int
    image_height: int
    image_width: int
    padded_height: int
    padded_width: int
    pad: tuple[int, int, int, int]
    image_dtype: torch.dtype
    device: torch.device
    model_window_length: int
    mask_threshold: float
    metric_scale: torch.Tensor
    metric_scale_enabled: bool
    raw_da3_intrinsics_px: torch.Tensor
    shared_da3_intrinsics_px: torch.Tensor
    shared_intrinsics: torch.Tensor
    ego_output_px: torch.Tensor
    fmaps: torch.Tensor
    ctxfeats: torch.Tensor
    fmaps3d_detail: torch.Tensor
    pms: torch.Tensor
    points: torch.Tensor
    masks: torch.Tensor
    world_points: torch.Tensor
    camera_poses: torch.Tensor

    def unpad(self, tensor: torch.Tensor) -> torch.Tensor:
        """Crop a padded ``(...,H,W)`` tensor to the encoded image grid."""
        left, right, top, bottom = self.pad
        height_end = tensor.shape[-2] - bottom
        width_end = tensor.shape[-1] - right
        return tensor[..., top:height_end, left:width_end]

    def validate(self) -> None:
        """Reject a stale or malformed cache before the tracking head runs."""
        if self.batch_size != 1:
            raise ValueError(f"SequenceCache requires B=1, got {self.batch_size}.")
        if self.t_eff <= 0 or self.t_eff > 128:
            raise ValueError(
                f"SequenceCache T_eff must satisfy 0 < T_eff <= 128, got {self.t_eff}."
            )
        if min(self.image_height, self.image_width) <= 0:
            raise ValueError("SequenceCache image dimensions must be positive.")
        if self.model_window_length < 2 or self.model_window_length % 2:
            raise ValueError("SequenceCache model window length must be even and >= 2.")
        if not torch.isfinite(torch.tensor(self.mask_threshold)):
            raise ValueError("SequenceCache mask threshold must be finite.")
        left, right, top, bottom = self.pad
        if min(left, right, top, bottom) < 0:
            raise ValueError("SequenceCache padding must be non-negative.")
        if self.padded_height != self.image_height + top + bottom:
            raise ValueError("SequenceCache padded height is inconsistent with pad.")
        if self.padded_width != self.image_width + left + right:
            raise ValueError("SequenceCache padded width is inconsistent with pad.")
        if self.padded_height % 64 or self.padded_width % 64:
            raise ValueError("SequenceCache padded dimensions must be divisible by 64.")

        temporal_shapes = {
            "fmaps": self.fmaps.shape[:2],
            "ctxfeats": self.ctxfeats.shape[:2],
            "fmaps3d_detail": self.fmaps3d_detail.shape[:2],
            "pms": self.pms.shape[:2],
            "shared_intrinsics": self.shared_intrinsics.shape[:2],
            "raw_da3_intrinsics_px": self.raw_da3_intrinsics_px.shape[:2],
            "shared_da3_intrinsics_px": self.shared_da3_intrinsics_px.shape[:2],
        }
        expected_temporal = (self.batch_size, self.t_eff)
        for name, shape in temporal_shapes.items():
            if tuple(shape) != expected_temporal:
                raise ValueError(
                    f"SequenceCache {name} time axes are {tuple(shape)}, expected "
                    f"{expected_temporal}."
                )

        expected_ranks = {
            "fmaps": (self.fmaps, 5),
            "ctxfeats": (self.ctxfeats, 5),
            "fmaps3d_detail": (self.fmaps3d_detail, 5),
            "pms": (self.pms, 5),
            "points": (self.points, 4),
            "masks": (self.masks, 4),
            "world_points": (self.world_points, 4),
        }
        for name, (tensor, rank) in expected_ranks.items():
            if tensor.ndim != rank:
                raise ValueError(
                    f"SequenceCache {name} must have rank {rank}, got "
                    f"shape {tuple(tensor.shape)}."
                )

        expected_feature_hw = (self.padded_height // 8, self.padded_width // 8)
        feature_shapes = {
            "fmaps": self.fmaps.shape[-2:],
            "ctxfeats": self.ctxfeats.shape[-2:],
            "fmaps3d_detail": self.fmaps3d_detail.shape[-2:],
            "pms": self.pms.shape[-2:],
        }
        for name, shape in feature_shapes.items():
            if tuple(shape) != expected_feature_hw:
                raise ValueError(
                    f"SequenceCache {name} spatial shape is {tuple(shape)}, "
                    f"expected {expected_feature_hw}."
                )
        expected_channels = {
            "fmaps": (self.fmaps.shape[2], 128),
            "ctxfeats": (self.ctxfeats.shape[2], 128),
            "fmaps3d_detail": (self.fmaps3d_detail.shape[2], 256),
            "pms": (self.pms.shape[2], 3),
            "points": (self.points.shape[1], 3),
            "masks": (self.masks.shape[1], 1),
            "world_points": (self.world_points.shape[1], 3),
        }
        for name, (actual, expected) in expected_channels.items():
            if actual != expected:
                raise ValueError(
                    f"SequenceCache {name} has {actual} channels, expected {expected}."
                )

        flattened_shapes = {
            "points": self.points.shape,
            "masks": self.masks.shape,
            "world_points": self.world_points.shape,
        }
        expected_count = self.batch_size * self.t_eff
        for name, shape in flattened_shapes.items():
            if shape[0] != expected_count or tuple(shape[-2:]) != (
                self.padded_height,
                self.padded_width,
            ):
                raise ValueError(
                    f"SequenceCache {name} has incompatible shape {tuple(shape)}."
                )
        if tuple(self.camera_poses.shape) != (expected_count, 4, 4):
            raise ValueError(
                "SequenceCache camera_poses must have shape "
                f"({expected_count},4,4), got {tuple(self.camera_poses.shape)}."
            )
        if tuple(self.shared_intrinsics.shape[-2:]) != (3, 3):
            raise ValueError("SequenceCache shared intrinsics must be 3x3 matrices.")
        if tuple(self.raw_da3_intrinsics_px.shape[-2:]) != (3, 3):
            raise ValueError("SequenceCache raw DA3 intrinsics must be 3x3 matrices.")
        if tuple(self.shared_da3_intrinsics_px.shape[-2:]) != (3, 3):
            raise ValueError("SequenceCache shared DA3 intrinsics must be 3x3 matrices.")
        if tuple(self.ego_output_px.shape) != (3, 3):
            raise ValueError("SequenceCache output ego K must have shape (3,3).")
        if self.metric_scale.numel() != 1:
            raise ValueError("SequenceCache metric_scale must be scalar.")

        tensors = {
            "fmaps": self.fmaps,
            "ctxfeats": self.ctxfeats,
            "fmaps3d_detail": self.fmaps3d_detail,
            "pms": self.pms,
            "points": self.points,
            "masks": self.masks,
            "world_points": self.world_points,
            "camera_poses": self.camera_poses,
            "shared_intrinsics": self.shared_intrinsics,
            "raw_da3_intrinsics_px": self.raw_da3_intrinsics_px,
            "shared_da3_intrinsics_px": self.shared_da3_intrinsics_px,
            "ego_output_px": self.ego_output_px,
            "metric_scale": self.metric_scale,
        }
        for name, tensor in tensors.items():
            if tensor.device != self.device:
                raise ValueError(
                    f"SequenceCache {name} is on {tensor.device}, expected {self.device}."
                )
        for name, tensor in {
            "fmaps": self.fmaps,
            "ctxfeats": self.ctxfeats,
            "fmaps3d_detail": self.fmaps3d_detail,
            "pms": self.pms,
        }.items():
            if tensor.dtype != self.image_dtype:
                raise ValueError(
                    f"SequenceCache {name} has dtype {tensor.dtype}, expected "
                    f"{self.image_dtype}."
                )
        if not torch.isfinite(self.metric_scale).all() or self.metric_scale.item() <= 0:
            raise ValueError("SequenceCache metric_scale must be finite and positive.")
        if not torch.isfinite(self.shared_intrinsics).all():
            raise ValueError("SequenceCache shared intrinsics contain NaN or infinity.")
        if not torch.isfinite(self.raw_da3_intrinsics_px).all():
            raise ValueError("SequenceCache raw DA3 intrinsics contain NaN or infinity.")
        if not torch.isfinite(self.shared_da3_intrinsics_px).all():
            raise ValueError("SequenceCache shared DA3 intrinsics contain NaN or infinity.")
        if not torch.isfinite(self.ego_output_px).all():
            raise ValueError("SequenceCache output ego K contains NaN or infinity.")
        if not torch.isfinite(self.camera_poses).all():
            raise ValueError("SequenceCache camera poses contain NaN or infinity.")
        expected_bottom = self.camera_poses.new_tensor([0.0, 0.0, 0.0, 1.0])
        if not torch.allclose(
            self.camera_poses[:, 3],
            expected_bottom.expand_as(self.camera_poses[:, 3]),
            atol=1e-5,
            rtol=0.0,
        ):
            raise ValueError("SequenceCache camera poses have malformed bottom rows.")
        pose_determinants = torch.linalg.det(self.camera_poses.float())
        if not torch.isfinite(pose_determinants).all() or (
            pose_determinants.abs() <= torch.finfo(torch.float32).eps
        ).any():
            raise ValueError("SequenceCache camera poses must be invertible.")
        first_k = self.shared_intrinsics[:, :1]
        if not torch.equal(self.shared_intrinsics, first_k.expand_as(self.shared_intrinsics)):
            raise ValueError("SequenceCache shared intrinsics are not temporally constant.")
        first_da3_k = self.shared_da3_intrinsics_px[:, :1]
        if not torch.equal(
            self.shared_da3_intrinsics_px,
            first_da3_k.expand_as(self.shared_da3_intrinsics_px),
        ):
            raise ValueError("SequenceCache shared DA3 intrinsics are not constant.")
        if (self.ego_output_px.diagonal()[:2] <= 0).any():
            raise ValueError("SequenceCache output ego K has non-positive focal length.")
        canonical_bottom = self.ego_output_px.new_tensor([0.0, 0.0, 1.0])
        if not torch.equal(self.ego_output_px[2], canonical_bottom):
            raise ValueError("SequenceCache output ego K has a malformed bottom row.")

    def window_eval_dict(self, source: int, end: int) -> dict[str, torch.Tensor]:
        """Slice all cached time axes into a window-local legacy eval cache."""
        self.validate()
        if source < 0 or end <= source or end > self.t_eff:
            raise ValueError(
                f"Invalid cached window [{source}:{end}) for T_eff={self.t_eff}."
            )

        def flattened_window(tensor: torch.Tensor) -> torch.Tensor:
            shaped = tensor.reshape(self.batch_size, self.t_eff, *tensor.shape[1:])
            return shaped[:, source:end].reshape(-1, *tensor.shape[1:])

        return {
            "fmaps": self.fmaps[:, source:end],
            "ctxfeats": self.ctxfeats[:, source:end],
            "fmaps3d_detail": self.fmaps3d_detail[:, source:end],
            "pms": self.pms[:, source:end],
            "points": flattened_window(self.points),
            "masks": flattened_window(self.masks),
            "world_points": flattened_window(self.world_points),
            "camera_poses": flattened_window(self.camera_poses),
            "intrinsics": self.shared_intrinsics[:, source:end],
        }


def _mean_intrinsics_px(intrinsics_px: torch.Tensor) -> torch.Tensor:
    """Repeat the temporal mean ``fx, fy, cx, cy`` for each batch item.

    Only the four pinhole parameters are averaged.  The remaining entries are
    rebuilt as a canonical camera matrix instead of being treated as learned
    values, so an accidental non-zero skew or malformed homogeneous row cannot
    leak into the shared calibration.
    """
    if intrinsics_px.ndim != 4 or intrinsics_px.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected DA3 intrinsics with shape (B, T, 3, 3), got "
            f"{tuple(intrinsics_px.shape)}."
        )
    if intrinsics_px.shape[1] == 0:
        raise ValueError("Cannot estimate shared intrinsics from an empty sequence.")

    with torch.autocast(device_type=intrinsics_px.device.type, enabled=False):
        intrinsics_f32 = intrinsics_px.float()
        parameters = _intrinsics_parameters_px(intrinsics_f32)
        if not torch.isfinite(parameters).all():
            raise ValueError("DA3 intrinsic parameters contain NaN or infinity.")
        if (parameters[..., :2] <= 0).any():
            raise ValueError("DA3 focal lengths must be positive.")

        mean_parameters = parameters.mean(dim=1, keepdim=True)
        shared = torch.zeros(
            (*mean_parameters.shape[:-1], 3, 3),
            dtype=intrinsics_f32.dtype,
            device=intrinsics_f32.device,
        )
        shared[..., 0, 0] = mean_parameters[..., 0]
        shared[..., 1, 1] = mean_parameters[..., 1]
        shared[..., 0, 2] = mean_parameters[..., 2]
        shared[..., 1, 2] = mean_parameters[..., 3]
        shared[..., 2, 2] = 1.0
        return shared.expand_as(intrinsics_f32).clone()


def _intrinsics_parameters_px(intrinsics_px: torch.Tensor) -> torch.Tensor:
    """Extract ``fx, fy, cx, cy`` from matrices of shape ``(..., 3, 3)``."""
    if intrinsics_px.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {tuple(intrinsics_px.shape)}.")
    return torch.stack(
        [
            intrinsics_px[..., 0, 0],
            intrinsics_px[..., 1, 1],
            intrinsics_px[..., 0, 2],
            intrinsics_px[..., 1, 2],
        ],
        dim=-1,
    )


def _intrinsics_at_unpadded_output_px(
    intrinsics_px: torch.Tensor,
    *,
    da3_height: int,
    da3_width: int,
    output_height: int,
    output_width: int,
    uses_input_padder: bool = True,
) -> torch.Tensor:
    """Convert DA3-grid K to pixel units on the returned image grid.

    ``infer()`` first pads to a multiple of 64 and then crops, whereas
    ``infer_pure_point()`` resizes directly without ``InputPadder``.  The flag
    makes that spatial path explicit instead of inventing a crop for the pure
    point wrapper.
    """
    normalized = _normalize_da3_intrinsics(
        intrinsics_px,
        height=da3_height,
        width=da3_width,
    )

    if uses_input_padder:
        pad_height = (-output_height) % 64
        pad_width = (-output_width) % 64
        normalized = _crop_normalized_intrinsics(
            normalized,
            padded_height=output_height + pad_height,
            padded_width=output_width + pad_width,
            output_height=output_height,
            output_width=output_width,
            crop_top=pad_height // 2,
            crop_left=pad_width // 2,
        )
        if normalized is None:
            raise AssertionError("DA3 intrinsics unexpectedly became None.")

    pixel_intrinsics = normalized.clone()
    pixel_intrinsics[..., 0, :] *= output_width
    pixel_intrinsics[..., 1, :] *= output_height
    pixel_intrinsics[..., 0, 2] -= 0.5
    pixel_intrinsics[..., 1, 2] -= 0.5
    return _intrinsics_parameters_px(pixel_intrinsics)


def summarize_intrinsics_px(parameters_px: torch.Tensor) -> dict[str, Any]:
    """Return interpretable population statistics for ``(T,4)`` pixel values."""
    if parameters_px.ndim != 2 or parameters_px.shape[-1] != 4:
        raise ValueError(
            "Expected per-frame intrinsic parameters with shape (T, 4), got "
            f"{tuple(parameters_px.shape)}."
        )
    if parameters_px.shape[0] == 0:
        raise ValueError("Cannot summarize an empty intrinsic sequence.")

    values = parameters_px.detach().double().cpu()
    report: dict[str, Any] = {
        "frame_count": int(values.shape[0]),
        "variance_definition": "population variance (ddof=0)",
        "parameters": {},
    }
    for index, name in enumerate(INTRINSIC_PARAMETER_NAMES):
        parameter = values[:, index]
        mean = parameter.mean()
        variance = parameter.var(unbiased=False)
        std = variance.sqrt()
        minimum = parameter.min()
        maximum = parameter.max()
        if parameter.numel() > 1:
            difference = parameter[1:] - parameter[:-1]
            difference_rms = difference.square().mean().sqrt().item()
            difference_median_abs = torch.quantile(
                difference.abs(), 0.5
            ).item()
            frame_index = torch.arange(
                parameter.numel(), dtype=parameter.dtype, device=parameter.device
            )
            centered_index = frame_index - frame_index.mean()
            slope = (
                (centered_index * (parameter - mean)).sum()
                / centered_index.square().sum()
            ).item()
        else:
            difference_rms = 0.0
            difference_median_abs = 0.0
            slope = 0.0
        report["parameters"][name] = {
            "mean_px": mean.item(),
            "variance_px2": variance.item(),
            "std_px": std.item(),
            "min_px": minimum.item(),
            "max_px": maximum.item(),
            "range_px": (maximum - minimum).item(),
            "max_abs_deviation_px": (parameter - mean).abs().max().item(),
            "relative_std_percent": (
                100.0 * std / mean.abs().clamp_min(torch.finfo(torch.float64).eps)
            ).item(),
            "first_difference_rms_px": difference_rms,
            "first_difference_median_abs_px": difference_median_abs,
            "linear_slope_px_per_frame": slope,
        }
    return report


class _EgoDepthAnythingAdapter(nn.Module):
    """Intercept DA3 outputs while keeping the original backbone as a submodule."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self._observations: list[
            tuple[torch.Tensor, torch.Tensor, int, int]
        ] = []

    def reset_observations(self) -> None:
        self._observations.clear()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Preserve the wrapped DA3 module's ordinary forward interface."""
        return self.backbone(*args, **kwargs)

    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        """Expose the wrapped backbone under its original checkpoint prefix."""
        return self.backbone.state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )

    def observed_intrinsics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if not self._observations:
            raise RuntimeError("No DA3 intrinsic observations are available.")
        sizes = {
            (height, width) for _, _, height, width in self._observations
        }
        if len(sizes) != 1:
            raise RuntimeError(
                "DA3 image size changed across chunks; one pixel-unit report would "
                f"be ambiguous: {sorted(sizes)}."
            )
        batch_sizes = {
            raw.shape[0] for raw, _, _, _ in self._observations
        }
        if len(batch_sizes) != 1:
            raise RuntimeError("DA3 batch size changed across intrinsic observations.")
        height, width = next(iter(sizes))
        raw = torch.cat(
            [chunk for chunk, _, _, _ in self._observations], dim=1
        )
        shared = torch.cat(
            [chunk for _, chunk, _, _ in self._observations], dim=1
        )
        return raw, shared, height, width

    def inference_v2(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        results = self.backbone.inference_v2(*args, **kwargs)
        if "intrinsics" not in results or "depth" not in results:
            raise KeyError("DA3 inference_v2 output must contain depth and intrinsics.")

        raw_intrinsics = results["intrinsics"]
        shared_intrinsics = _mean_intrinsics_px(raw_intrinsics)
        da3_height, da3_width = results["depth"].shape[-2:]
        self._observations.append(
            (
                raw_intrinsics.detach().float().cpu(),
                shared_intrinsics.detach().float().cpu(),
                da3_height,
                da3_width,
            )
        )

        shared_results = dict(results)
        shared_results["intrinsics"] = shared_intrinsics
        return shared_results


class Track4World3DFFEgo(_Track4World3DFFCore):
    """3D-FF model using one temporal-mean ``fx/fy/cx/cy`` DA3 calibration."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Ego-centric has no meaningful implementation for the legacy backbone
        # variants.  Validate before constructing the parent so an accidental
        # non-DA3 request fails without downloading or initializing another
        # backbone.  Omitting the keyword selects the only supported model.
        requested_model = kwargs.get("use_model", "depthanythingv3")
        if requested_model != "depthanythingv3":
            raise ValueError(
                "Track4World3DFFEgo requires use_model='depthanythingv3'; "
                f"got {requested_model!r}."
            )
        kwargs["use_model"] = "depthanythingv3"
        super().__init__(*args, **kwargs)
        self._install_ego_adapter()
        self._observations_use_input_padder: bool | None = None

    def _install_ego_adapter(self) -> _EgoDepthAnythingAdapter:
        if not isinstance(self.backbone, _EgoDepthAnythingAdapter):
            self.backbone = _EgoDepthAnythingAdapter(self.backbone)
        return self.backbone

    def load_state_dict(self, state_dict: Mapping[str, Any], *args: Any, **kwargs: Any):
        """Load either ordinary or adapter-prefixed Track4World checkpoints."""
        normalized_state_dict: dict[str, Any] = {}
        adapter_prefix = "backbone.backbone."
        ordinary_prefix = "backbone."
        for key, value in state_dict.items():
            if key.startswith(adapter_prefix):
                key = ordinary_prefix + key[len(adapter_prefix) :]
            if key in normalized_state_dict:
                raise ValueError(f"Duplicate checkpoint key after adapter removal: {key}")
            normalized_state_dict[key] = value

        adapter = self._install_ego_adapter()
        self.backbone = adapter.backbone
        try:
            return super().load_state_dict(normalized_state_dict, *args, **kwargs)
        finally:
            self._install_ego_adapter()

    def switch_to_original_backbone(self) -> None:
        if isinstance(self.backbone, _EgoDepthAnythingAdapter):
            self.backbone = self.backbone.backbone
        try:
            super().switch_to_original_backbone()
        finally:
            self._install_ego_adapter()

    def reset_da3_intrinsics_observations(self) -> None:
        self._install_ego_adapter().reset_observations()

    def get_fmaps(self, images_, B, T, sw, is_training):
        # The base DA3 path chunks at 128 frames.  A chunk-wise mean would not
        # be one K for the whole clip, so reject that ambiguous case explicitly.
        if T > 128:
            raise ValueError(
                "Ego-centric inference currently supports at most 128 real frames; "
                f"got {T}."
            )
        self._observations_use_input_padder = True
        self.reset_da3_intrinsics_observations()
        return super().get_fmaps(images_, B, T, sw, is_training)

    def infer_pure_point(self, *args: Any, **kwargs: Any):
        self._observations_use_input_padder = False
        self.reset_da3_intrinsics_observations()
        return super().infer_pure_point(*args, **kwargs)

    def observed_da3_intrinsics_px(
        self,
        *,
        output_height: int,
        output_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return raw, per-frame output, and actually used shared DA3 K."""
        raw_intrinsics, shared_intrinsics, da3_height, da3_width = (
            self._install_ego_adapter().observed_intrinsics()
        )
        if self._observations_use_input_padder is None:
            raise RuntimeError(
                "Intrinsic observations were not produced by infer(), "
                "infer_pair(), or infer_pure_point()."
            )
        raw_parameters_px = _intrinsics_parameters_px(raw_intrinsics)
        output_parameters_px = _intrinsics_at_unpadded_output_px(
            raw_intrinsics,
            da3_height=da3_height,
            da3_width=da3_width,
            output_height=output_height,
            output_width=output_width,
            uses_input_padder=self._observations_use_input_padder,
        )
        shared_output_parameters_px = _intrinsics_at_unpadded_output_px(
            shared_intrinsics,
            da3_height=da3_height,
            da3_width=da3_width,
            output_height=output_height,
            output_width=output_width,
            uses_input_padder=self._observations_use_input_padder,
        )
        metadata = {
            "da3_image_size_hw": [da3_height, da3_width],
            "output_image_size_hw": [output_height, output_width],
            "parameter_order": list(INTRINSIC_PARAMETER_NAMES),
            "spatial_postprocess": (
                "input_padder_then_unpad"
                if self._observations_use_input_padder
                else "direct_forward_point_resize"
            ),
        }
        return (
            raw_parameters_px,
            output_parameters_px,
            shared_output_parameters_px,
            metadata,
        )

    @torch.inference_mode()
    def encode_sequence(self, images: torch.Tensor) -> SequenceCache:
        """Run DA3 and feature extraction exactly once for all real frames.

        This mirrors the normalization and spatial padding at the start of the
        legacy ``forward_sliding`` implementation, but deliberately
        stops before ``forward_window_unified`` (the tracking head).
        """
        if images.ndim != 5:
            raise ValueError(
                f"Expected images with shape (B,T,3,H,W), got {tuple(images.shape)}."
            )
        batch_size, t_eff, channels, height, width = images.shape
        if batch_size != 1:
            raise ValueError(f"encode_sequence requires B=1, got B={batch_size}.")
        if channels != 3:
            raise ValueError(f"encode_sequence requires three channels, got {channels}.")
        if t_eff <= 0 or t_eff > 128:
            raise ValueError(
                f"encode_sequence requires 0 < T_eff < 128, got {t_eff}."
            )
        if images.device != self.device:
            raise ValueError(
                f"Images are on {images.device}, but the model is on {self.device}."
            )
        if self.seqlen < 2 or self.seqlen % 2:
            raise ValueError(
                f"Model seqlen must be even and at least 2, got {self.seqlen}."
            )

        image_dtype = images.dtype
        normalized = images / 255.0
        normalized = (normalized - self.image_mean) / self.image_std
        flattened = normalized.contiguous().reshape(
            batch_size * t_eff, 3, height, width
        )
        padder = InputPadder(flattened.shape)
        padded = padder.pad(flattened)[0]
        padded_height, padded_width = padded.shape[-2:]
        h8, w8 = padded_height // 8, padded_width // 8

        (
            fmaps,
            ctxfeats,
            fmaps3d_detail,
            pms,
            points,
            masks,
            world_points,
            camera_poses,
            shared_intrinsics,
        ) = self.get_fmaps(
            padded,
            batch_size,
            t_eff,
            sw=None,
            is_training=False,
        )
        if shared_intrinsics is None:
            raise RuntimeError("DA3 did not return intrinsics while encoding the sequence.")

        fmaps = fmaps.to(image_dtype).reshape(
            batch_size, t_eff, self.flow_dim, h8, w8
        )
        ctxfeats = ctxfeats.to(image_dtype).reshape(
            batch_size, t_eff, 128, h8, w8
        )
        fmaps3d_detail = fmaps3d_detail.to(image_dtype).reshape(
            batch_size, t_eff, self.flow3d_dim, h8, w8
        )
        pms = pms.to(image_dtype).reshape(batch_size, t_eff, 3, h8, w8)
        shared_intrinsics = shared_intrinsics.reshape(batch_size, t_eff, 3, 3)

        metric_scale = getattr(self, "_metric_scale", None)
        if metric_scale is None:
            raise RuntimeError("DA3 did not expose the full-sequence metric scale.")
        metric_scale = metric_scale.detach().clone().to(device=images.device)

        raw_da3, shared_da3, _, _ = self._install_ego_adapter().observed_intrinsics()
        if tuple(raw_da3.shape[:2]) != (batch_size, t_eff):
            raise RuntimeError(
                "DA3 observations do not cover the complete encoded sequence: "
                f"{tuple(raw_da3.shape[:2])} vs {(batch_size, t_eff)}."
            )
        raw_da3 = raw_da3.to(device=images.device)
        shared_da3 = shared_da3.to(device=images.device)

        output_intrinsics = _crop_normalized_intrinsics(
            shared_intrinsics,
            padded_height=padded_height,
            padded_width=padded_width,
            output_height=height,
            output_width=width,
            crop_top=padder._pad[2],
            crop_left=padder._pad[0],
        )
        if output_intrinsics is None:
            raise AssertionError("Shared DA3 intrinsics unexpectedly became None.")
        ego_output_px = output_intrinsics[0, 0].float().clone()
        ego_output_px[0, :] *= width
        ego_output_px[1, :] *= height
        ego_output_px[0, 2] -= 0.5
        ego_output_px[1, 2] -= 0.5

        cache = SequenceCache(
            t_eff=t_eff,
            batch_size=batch_size,
            image_height=height,
            image_width=width,
            padded_height=padded_height,
            padded_width=padded_width,
            pad=tuple(padder._pad),
            image_dtype=image_dtype,
            device=images.device,
            model_window_length=self.seqlen,
            mask_threshold=float(self.mask_threshold),
            metric_scale=metric_scale,
            metric_scale_enabled=bool(self.use_metric_scale),
            raw_da3_intrinsics_px=raw_da3,
            shared_da3_intrinsics_px=shared_da3,
            shared_intrinsics=shared_intrinsics,
            ego_output_px=ego_output_px,
            fmaps=fmaps,
            ctxfeats=ctxfeats,
            fmaps3d_detail=fmaps3d_detail,
            pms=pms,
            points=points,
            masks=masks,
            world_points=world_points,
            camera_poses=camera_poses,
        )
        cache.validate()
        return cache

    @torch.inference_mode()
    def track_cached_window(
        self,
        cache: SequenceCache,
        images: torch.Tensor,
        source: int,
        end: int,
        iters: int,
    ) -> list[dict[str, torch.Tensor]]:
        """Run the unchanged legacy tracking head on one cached horizon."""
        cache.validate()
        expected_shape = (
            cache.batch_size,
            cache.t_eff,
            3,
            cache.image_height,
            cache.image_width,
        )
        if tuple(images.shape) != expected_shape:
            raise ValueError(
                f"Images do not match SequenceCache: {tuple(images.shape)} vs "
                f"{expected_shape}."
            )
        if images.dtype != cache.image_dtype or images.device != cache.device:
            raise ValueError(
                "Images and SequenceCache must have identical dtype/device; got "
                f"{images.dtype}/{images.device} and "
                f"{cache.image_dtype}/{cache.device}."
            )
        if cache.device != self.device:
            raise ValueError(
                f"Model moved to {self.device} after cache encoding on {cache.device}."
            )
        if cache.fmaps.shape[2] != self.flow_dim or (
            cache.fmaps3d_detail.shape[2] != self.flow3d_dim
        ):
            raise ValueError(
                "SequenceCache feature dimensions do not match the current model."
            )
        if self.seqlen != cache.model_window_length:
            raise ValueError(
                f"Model seqlen changed after encoding: {self.seqlen} vs "
                f"{cache.model_window_length}."
            )
        if bool(self.use_metric_scale) != cache.metric_scale_enabled:
            raise ValueError(
                "Model metric-scale state changed after encoding: "
                f"{self.use_metric_scale} vs {cache.metric_scale_enabled}."
            )
        horizon = end - source
        if horizon < max(3, self.seqlen) or horizon % self.seqlen:
            raise ValueError(
                "Cached tracking horizon must satisfy the normal legacy T>=3 "
                f"contract, be at least W, and be a multiple of W={self.seqlen}; "
                f"got {horizon}."
            )
        if iters < 1:
            raise ValueError(f"iters must be positive, got {iters}.")

        self._metric_scale = cache.metric_scale.detach().clone().to(self.device)
        if not torch.equal(self._metric_scale, cache.metric_scale.to(self.device)):
            raise RuntimeError("Could not restore the full-sequence metric scale.")

        output, _ = self.infer(
            images[:, source:end],
            iters=iters,
            sw=None,
            is_training=False,
            window_len=self.seqlen,
            stride=None,
            tracking3d=True,
            force_projection=True,
            eval_dict=cache.window_eval_dict(source, end),
        )
        for family in output:
            output_scale = family.get("metric_scale")
            if output_scale is None or not torch.equal(
                output_scale.to(cache.metric_scale), cache.metric_scale
            ):
                raise RuntimeError(
                    "Tracking output did not preserve the full-sequence metric scale."
                )
        return output
