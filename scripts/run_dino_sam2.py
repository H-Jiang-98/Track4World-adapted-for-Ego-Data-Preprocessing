import sys
import os

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "submodules", "Grounded-SAM-2")  # adjust if needed
sys.path.append(os.path.abspath(project_root))
import argparse
import cv2
import json
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from supervision.draw.color import ColorPalette

sys.path.insert(0, 'submodules/Grounded-SAM-2')

from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision 
from tqdm import tqdm

"""
Hyper parameters
"""
parser = argparse.ArgumentParser(
    description="Process a video, image, or RGB image directory with Grounded SAM 2"
)
parser.add_argument(
    '--input-path', '--video-path', '--rgb-dir',
    dest='input_path', required=True,
    help="Path to a video, a single image, or an ordered RGB image directory"
)
parser.add_argument('--text-prompt', required=True, 
help="Text prompts for detection, e.g., 'car. person. dog'")
parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-base")
parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--output-dir", default="output_single_video", help="Directory to save results")
parser.add_argument("--force-cpu", action="store_true")
parser.add_argument("--box-threshold", type=float, default=0.25)
parser.add_argument("--text-threshold", type=float, default=0.25)
parser.add_argument(
    "--max-frames", type=int, default=0,
    help="Maximum number of frames/images to process; 0 means all"
)
parser.add_argument(
    "--preserve-source-names", action="store_true",
    help=(
        "Keep source image stems for output files. By default outputs use "
        "zero-based sequential names (00000.png, 00001.png, ...), which are "
        "compatible with the 3D-FF input loader."
    )
)
args = parser.parse_args()
if args.max_frames < 0:
    parser.error("--max-frames must be greater than or equal to 0")

input_path = Path(args.input_path)
if not input_path.exists():
    parser.error(f"input path does not exist: {input_path}")

# Constants
GROUNDING_MODEL = args.grounding_model
INPUT_PATH = input_path
TEXT_PROMPT = args.text_prompt
SAM2_CHECKPOINT = args.sam2_checkpoint
SAM2_MODEL_CONFIG = args.sam2_model_config
DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
OUTPUT_DIR = Path(args.output_dir)

# Create Output Dirs
output_path_vis = OUTPUT_DIR / "vis"
output_path_mask = OUTPUT_DIR / "mask"
# --- CHANGE 1: Define color path one level up from OUTPUT_DIR ---
# output_path_color = OUTPUT_DIR.parent / "color"

output_path_vis.mkdir(parents=True, exist_ok=True)
output_path_mask.mkdir(parents=True, exist_ok=True)
# output_path_color.mkdir(parents=True, exist_ok=True)

# Environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if DEVICE == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"Loading models on {DEVICE}...")

# Build SAM2 image predictor
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Build Grounding DINO
processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)

def id_to_colors(id): # id to color
    rgb = np.zeros((3, ), dtype=np.uint8)
    for i in range(3):
        rgb[i] = id % 256
        id = id // 256
    return rgb

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def get_frames_generator(source_path, max_frames=0):
    """
    Yields frames from a video, a single image, or an image directory.
    Returns: (frame_index, source_name, frame_rgb_numpy, original_pil_image)
    """
    source = Path(source_path)
    frame_limit = max_frames if max_frames > 0 else None

    if source.is_dir():
        image_paths = sorted(
            path for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if frame_limit is not None:
            image_paths = image_paths[:frame_limit]
        if not image_paths:
            raise ValueError(f"No supported images found in: {source}")

        print(f"Processing image directory: {source} ({len(image_paths)} images)")
        for frame_idx, image_path in enumerate(
            tqdm(image_paths, desc="Processing Images")
        ):
            with Image.open(image_path) as source_image:
                image_pil = source_image.convert("RGB")
            yield frame_idx, image_path.name, np.array(image_pil), image_pil
        return

    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        print(f"Processing single image: {source}")
        with Image.open(source) as source_image:
            image_pil = source_image.convert("RGB")
        yield 0, source.name, np.array(image_pil), image_pil
        return

    print(f"Processing video file: {source}")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {source}")

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_limit is not None:
        total_frames = min(total_frames, frame_limit)

    try:
        with tqdm(total=total_frames, desc="Processing Frames") as pbar:
            while frame_limit is None or frame_idx < frame_limit:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(frame_rgb)
                source_name = f"{frame_idx:05d}.jpg"
                yield frame_idx, source_name, frame_rgb, image_pil

                frame_idx += 1
                pbar.update(1)
    finally:
        cap.release()

# --- Main Processing Loop ---

print(f"Starting inference with prompt: '{TEXT_PROMPT}'")

frame_generator = get_frames_generator(INPUT_PATH, max_frames=args.max_frames)
frame_manifest = []

for frame_idx, source_name, image, image_pil in frame_generator:
    output_stem = (
        Path(source_name).stem
        if args.preserve_source_names
        else f"{frame_idx:05d}"
    )
    vis_name = f"{output_stem}.png"
    png_name = f"{output_stem}.png"
    json_name = f"{output_stem}.json"
    frame_manifest.append({
        "frame_index": frame_idx,
        "source_name": source_name,
        "mask_name": png_name,
    })
    
    # Convert to BGR for OpenCV visualization output.
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 1. Prepare SAM2
    sam2_predictor.set_image(image)

    # 2. Run Grounding DINO
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[image_pil.size[::-1]]
    )

    # 3. Get box prompts and run SAM2
    input_boxes = results[0]["boxes"].cpu().numpy()

    if input_boxes.shape[0] != 0:
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Squeeze dim if needed (n, 1, H, W) -> (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))

        # 4. Supervision Visualization Logic
        # img_bgr is already defined at the top of the loop
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids,
            confidence=np.array(confidences)
        )

        # Non-Maximum Suppression (NMS)
        nms_idx = torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy).float(), 
                    torch.from_numpy(detections.confidence).float(), 
                    0.5
                ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.mask = detections.mask[nms_idx]

        # Annotate
        labels_vis = [
            f"{class_names[id]} {confidence:.2f}"
            for id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(scene=img_bgr.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels_vis)

        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        # 5. Save Output
        # Prepare Mask Data
        masks_final = detections.mask
        labels_final = detections.class_id
        
        color_mask = np.zeros(image.shape, dtype=np.uint8)
        obj_info_json = []

        # Sort masks by size (largest first) for better rendering
        mask_size = [np.sum(m) for m in masks_final]
        sorted_mask_idx = np.argsort(mask_size)[::-1]

        for idx in sorted_mask_idx:
            m = masks_final[idx]
            # Reserve zero for the static/background region.
            object_id = int(idx) + 1
            color_mask[m] = id_to_colors(object_id)

            obj_info_json.append({
                "id": object_id,
                "label": class_names[labels_final[idx]],
                "score": float(detections.confidence[idx]),
            })

        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

        # Save Visualizations
        cv2.imwrite(str(output_path_vis / vis_name), annotated_frame)

        # Save Masks (PNG + JSON)
        cv2.imwrite(str(output_path_mask / png_name), color_mask_bgr)
        with open(output_path_mask / json_name, "w") as f:
            json.dump(obj_info_json, f)

    else:
        # No detections
        # img_bgr is already defined at the top of the loop
        cv2.imwrite(str(output_path_vis / vis_name), img_bgr)

        cv2.imwrite(str(output_path_mask / png_name), np.zeros(image.shape, dtype=np.uint8))
        with open(output_path_mask / json_name, "w") as f:
            json.dump([], f)

with open(OUTPUT_DIR / "frame_manifest.json", "w") as f:
    json.dump(frame_manifest, f, indent=2)

print(f"Processing complete. Results saved to {OUTPUT_DIR}")
