"""
seg_generation.py
-----------------
Standalone segmentation mask generator using FastSAM.

Extracts per-frame binary segmentation masks from a folder of images.
Supports local image directories or HuggingFace dataset repos as input.

Output mask format:
    - Single-channel PNG
    - Pixel values: 0 = background, 1 = foreground (spacecraft)
    - Shape: (H, W), dtype uint8

Usage:
  python generate_seg.py \
    --src_img_dir ./output/Cheops_1000f_left_to_right_cv_bright/image/ \
    --out_seg_dir ./output/Cheops_1000f_left_to_right_cv_bright/seg_v2/ \
    --out_debug_dir ./output/Cheops_1000f_left_to_right_cv_bright/seg_debug_v2/ \
    --fastsam_weights /home/haihong/kubric/FastSAM/weights/FastSAM-x.pt \
    --prompt_type box \
    --init_box 200 300 900 900 \
    --tracker_prompt_type box \
    --union_box_prompt \
    --union_box_padding 100 \
    --iou_threshold 0.05 \
    --start_frame 200 \
    --end_frame 1000 \
    --device cpu \
    --save_debug
"""

import os
import sys
import cv2
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Optional
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 0: FastSAM Path Setup
# =============================================================================

PATH = os.getcwd()
module_dir = os.path.abspath(f"{PATH}/../../FastSAM")
if module_dir not in sys.path:
    sys.path.append(module_dir)

from fastsam import FastSAM, FastSAMPrompt


# =============================================================================
# SECTION 1: HuggingFace Cache Resolution
# =============================================================================

def resolve_hf_dataset_path(
    repo_id: str,
    subfolder: str = None,
    revision: str = None,
    local_dir: str = None,
) -> Path:
    """
    Resolves the local path of a HuggingFace dataset, downloading if needed.

    Args:
        repo_id:   e.g. "itshaihong/reconstruction-tracking-synthetic"
        subfolder: Optional sub-path within the repo root
        revision:  Optional git branch/tag/commit hash
        local_dir: If set, downloads here instead of ~/.cache

    Returns:
        Resolved Path object pointing to the local dataset root (or subfolder).
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        )

    print(f"[HF] Resolving dataset '{repo_id}' from cache (or downloading)...")

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False if local_dir else "auto",
    )

    resolved = Path(snapshot_path)
    if subfolder:
        resolved = resolved / subfolder

    if not resolved.exists():
        raise FileNotFoundError(
            f"Resolved path does not exist: {resolved}\n"
            f"Check that the subfolder '{subfolder}' exists in the repo."
        )

    print(f"[HF] Dataset root: {resolved}")
    return resolved


# =============================================================================
# SECTION 2: ObjectTracker
# =============================================================================

class ObjectTracker:
    """
    Tracks a single object across frames using centroid or bounding box prompts.
    Rejects masks that are too small or have low IoU with the previous frame.
    """

    def __init__(self, init_mask, prompt_type='point', min_mask_area=100, iou_threshold=0.1):
        self.prompt_type  = prompt_type
        self.min_mask_area = min_mask_area
        self.iou_threshold = iou_threshold
        self.centroid = None
        self.box = None
        self.last_mask = None
        self._update_state(init_mask)
        if self.centroid is None or self.box is None:
            raise ValueError(
                "Initial FastSAM prompt returned an empty mask. "
                "Move --init_point onto the spacecraft or use --prompt_type box with --init_box."
            )
        self.initial_centroid = self.centroid
        self.initial_box      = self.box

    def _update_state(self, mask):
        self.last_mask = mask.astype(bool)
        y_coords, x_coords = np.where(self.last_mask)
        if len(y_coords) == 0:
            return
        self.centroid = [int(np.mean(x_coords)), int(np.mean(y_coords))]
        self.box = [
            int(x_coords.min()), int(y_coords.min()),
            int(x_coords.max()), int(y_coords.max()),
        ]

    def compute_iou(self, new_mask):
        intersection = np.logical_and(self.last_mask, new_mask).sum()
        union        = np.logical_or(self.last_mask,  new_mask).sum()
        return intersection / union if union > 0 else 0.0

    def get_prompt(self):
        return self.box if self.prompt_type == 'box' else self.centroid

    def update(self, new_mask):
        area = new_mask.astype(bool).sum()
        if area < self.min_mask_area:
            print(f"  [Tracker] Rejected: area {area} < min {self.min_mask_area}")
            return False
        iou = self.compute_iou(new_mask)
        if iou < self.iou_threshold:
            print(f"  [Tracker] Low IoU ({iou:.3f}) — keeping last known prompt.")
            return False
        self._update_state(new_mask)
        return True


# =============================================================================
# SECTION 3: FastSAM Helpers
# =============================================================================

def _parse_fastsam_ann(ann):
    """Convert FastSAM annotation output to a boolean numpy mask."""
    if ann is None or (hasattr(ann, '__len__') and len(ann) == 0):
        return None
    raw = ann[0]
    if isinstance(raw, torch.Tensor):
        return raw.cpu().numpy().astype(bool)
    return np.array(raw).astype(bool)


def _candidate_masks_and_boxes(everything_results):
    """Return FastSAM candidate masks and xyxy boxes from the raw model result."""
    if not everything_results:
        return [], []
    result = everything_results[0]
    if getattr(result, 'masks', None) is None:
        return [], []

    masks = result.masks.data
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    masks = np.asarray(masks).astype(bool)

    boxes = []
    if getattr(result, 'boxes', None) is not None:
        xyxy = result.boxes.xyxy
        if isinstance(xyxy, torch.Tensor):
            xyxy = xyxy.cpu().numpy()
        boxes = np.asarray(xyxy).tolist()

    if not boxes:
        for mask in masks:
            y_coords, x_coords = np.where(mask)
            if len(y_coords) == 0:
                boxes.append(None)
            else:
                boxes.append([
                    float(x_coords.min()), float(y_coords.min()),
                    float(x_coords.max()), float(y_coords.max()),
                ])

    return list(masks), boxes


def union_candidate_masks_in_box(everything_results, box, padding=80):
    """Union candidate masks belonging to the spacecraft prompt region.

    FastSAM may split the spacecraft into body/panel/appendage candidates.
    Center-only selection can miss a fragment whose bbox center lies just
    outside the current tracker box, so nearby bbox-overlapping candidates are
    also included. Very large image-covering candidates are excluded.
    """
    x0, y0, x1, y1 = [float(value) for value in box]
    masks, boxes = _candidate_masks_and_boxes(everything_results)
    if not masks:
        return None

    image_h, image_w = masks[0].shape[:2]
    x0 = max(0.0, x0 - padding)
    y0 = max(0.0, y0 - padding)
    x1 = min(float(image_w - 1), x1 + padding)
    y1 = min(float(image_h - 1), y1 + padding)
    prompt_area = max(1.0, (x1 - x0) * (y1 - y0))
    selected = []

    for mask, candidate_box in zip(masks, boxes):
        if candidate_box is None:
            continue
        bx0, by0, bx1, by1 = [float(value) for value in candidate_box]
        candidate_area = max(1.0, (bx1 - bx0) * (by1 - by0))
        if candidate_area > 0.75 * image_w * image_h:
            continue

        intersection = max(0.0, min(x1, bx1) - max(x0, bx0)) * max(
            0.0, min(y1, by1) - max(y0, by0)
        )
        if intersection <= 0.0:
            continue

        bbox_iou = intersection / max(
            1.0, prompt_area + candidate_area - intersection
        )
        center_inside = (
            x0 <= 0.5 * (bx0 + bx1) <= x1 and
            y0 <= 0.5 * (by0 + by1) <= y1
        )
        if center_inside or bbox_iou >= 0.01:
            selected.append(mask)

    if not selected:
        return None

    union = np.zeros_like(selected[0], dtype=bool)
    for mask in selected:
        union |= mask
    print(f"  [BoxUnion] Unioned {len(selected)} candidate masks in padded box "
          f"[{x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f}]")  
    return union


def initialize_object_on_first_frame(
    img_pil, everything_results, device,
    prompt_type, point=None, box=None, text=None, union_box_prompt=False, union_box_padding=80
):
    """
    Run FastSAM with a user-supplied prompt on the first frame to initialise tracking.

    Args:
        prompt_type: 'point', 'box', or 'text'
        point:       [x, y] pixel coordinate (for prompt_type='point')
        box:         [x0, y0, x1, y1]         (for prompt_type='box')
        text:        string description        (for prompt_type='text')

    Returns:
        Boolean numpy mask, or None if initialisation failed.
    """
    prompt_process = FastSAMPrompt(img_pil, everything_results, device=device)

    if prompt_type == 'point':
        if point is None:
            raise ValueError("prompt_type='point' requires --init_point x y")
        print(f"  [Init] Point prompt at {point}")
        ann = prompt_process.point_prompt(points=[point], pointlabel=[1])

    elif prompt_type == 'box':
        if box is None:
            raise ValueError("prompt_type='box' requires --init_box x0 y0 x1 y1")
        print(f"  [Init] Box prompt: {box}")
        if union_box_prompt:
            return union_candidate_masks_in_box(everything_results, box, padding=union_box_padding)
        ann = prompt_process.box_prompt(bbox=box)

    elif prompt_type == 'text':
        if text is None:
            raise ValueError("prompt_type='text' requires --init_text '...'")
        print(f"  [Init] Text prompt: '{text}'")
        ann = prompt_process.text_prompt(text=text)

    else:
        raise ValueError(f"Unknown prompt_type '{prompt_type}'")

    return _parse_fastsam_ann(ann)


def get_mask_for_frame(img_pil, everything_results, tracker, device, union_box_prompt=False, union_box_padding=80):
    """
    Run FastSAM on a subsequent frame using the tracker's current prompt.

    Returns:
        Boolean numpy mask, or None if FastSAM returned no result.
    """
    prompt_process  = FastSAMPrompt(img_pil, everything_results, device=device)
    current_prompt  = tracker.get_prompt()

    if tracker.prompt_type == 'box':
        print(f"  [Track] Box prompt: {current_prompt}")
        if union_box_prompt:
            mask = union_candidate_masks_in_box(everything_results, current_prompt, padding=union_box_padding)
            if mask is not None:
                accepted = tracker.update(mask)
                return mask if accepted else tracker.last_mask
            return tracker.last_mask
        ann = prompt_process.box_prompt(bbox=current_prompt)
    else:
        print(f"  [Track] Point prompt: {current_prompt}")
        ann = prompt_process.point_prompt(points=[current_prompt], pointlabel=[1])

    mask = _parse_fastsam_ann(ann)
    if mask is not None:
        accepted = tracker.update(mask)
        return mask if accepted else tracker.last_mask
    return tracker.last_mask


# =============================================================================
# SECTION 4: Mask I/O
# =============================================================================

def save_seg_mask(mask, image_shape: tuple, seg_out: Path):
    """
    Save a binary segmentation mask as a single-channel PNG.

    Output pixel values:
        0 → background
        1 → foreground (spacecraft)

    Note: values {0, 1} look black in image viewers.
          Multiply by 255 for visual inspection.

    Args:
        mask:        boolean/uint8 numpy array, or None (saves all-zero mask)
        image_shape: (H, W)
        seg_out:     output file path
    """
    H, W = image_shape
    if mask is None:
        seg = np.zeros((H, W), dtype=np.uint8)
    else:
        seg = np.asarray(mask, dtype=bool).astype(np.uint8)   # values: {0, 1}

    cv2.imwrite(str(seg_out), seg)


def save_debug_visualization(img_bgr, mask, out_path: Path):
    """
    Save a side-by-side debug image: original | mask overlaid.
    Uses 0-255 scaling so the mask is visible.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    if mask is not None:
        # Scale {0,1} → {0,255} for visibility
        axes[1].imshow(mask.astype(np.uint8) * 255, cmap="gray", interpolation="nearest")
    else:
        axes[1].imshow(np.zeros(img_bgr.shape[:2], dtype=np.uint8), cmap="gray")
    axes[1].set_title("Segmentation Mask (white=foreground)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# SECTION 5: Main Pipeline
# =============================================================================

def generate_segmentation_masks(
    # --- Image source (one of the two must be provided) ---
    src_img_dir: str          = None,

    # --- Output ---
    out_seg_dir: str          = "./seg_output",
    save_debug: bool          = False,
    out_debug_dir: str        = None,
    # --- FastSAM ---
    fastsam_weights: str      = "../FastSAM/weights/FastSAM-x.pt",
    fastsam_conf: float       = 0.25,
    fastsam_iou: float        = 0.9,
    fastsam_imgsz: int        = 1024,
    device: str               = "cuda" if torch.cuda.is_available() else "cpu",
    # --- Prompts ---
    prompt_type: str          = "point",
    init_point: list          = None,
    init_box: list            = None,
    init_text: str            = None,
    # --- Tracker ---
    tracker_prompt_type: str  = "point",
    union_box_prompt: bool    = False,
    union_box_padding: int    = 80,
    min_mask_area: int        = 100,
    iou_threshold: float      = 0.1,
    # --- Frame selection ---
    num_frames: int           = None,
    sample_method: str        = "uniform",
    frame_list: list          = None,
    start_frame: int          = 0,
    end_frame: int            = None,
    resume: bool              = False,
    progress_file: str        = None,
):
    """
    Generate per-frame binary segmentation masks using FastSAM + ObjectTracker.

    Output:
        {out_seg_dir}/{frame_idx:06d}.png
        Single-channel PNG, values {0=background, 1=foreground}
    """

    # --- Resolve image source ---
    if src_img_dir is not None:
        img_dir = Path(src_img_dir)
        if not img_dir.exists():
            raise FileNotFoundError(f"src_img_dir does not exist: {img_dir}")
    else:
        raise ValueError("Provide either --src_img_dir (local)")

    print(f"[SegGen] Source images : {img_dir}")
    print(f"[SegGen] Output seg dir: {out_seg_dir}")
    print(f"[SegGen] Device        : {device}")
    print(f"[SegGen] Loading FastSAM weights: {fastsam_weights}")

    model = FastSAM(fastsam_weights)

    # --- Gather & select frames ---
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ])
    if not img_files:
        raise ValueError(f"No images found in {img_dir}")

    total_available = len(img_files)
    print(f"[SegGen] Total images found: {total_available}")

    if num_frames is not None and num_frames < total_available:
        if sample_method == 'uniform':
            indices = np.linspace(0, total_available - 1, num_frames, dtype=int).tolist()
        elif sample_method == 'user_defined' and frame_list is not None:
            indices = [i for i in frame_list if 0 <= i < total_available]
        else:
            indices = list(range(num_frames))
    else:
        indices = list(range(total_available))

    indices = [i for i in indices if i >= start_frame and
               (end_frame is None or i < end_frame)]
    selected_files = [img_files[i] for i in indices]
    if not selected_files:
        raise ValueError("No frames selected after applying --start_frame/--end_frame.")
    print(f"[SegGen] Processing {len(selected_files)} frames "
          f"(range {indices[0]}..{indices[-1]}).\n")

    # --- Create output directories ---
    seg_dir   = Path(out_seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = None
    if save_debug:
        debug_dir = Path(out_debug_dir) if out_debug_dir else seg_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SegGen] Debug visualizations → {debug_dir}")

    # --- Per-frame loop ---
    tracker: Optional[ObjectTracker] = None
    progress_path = Path(progress_file) if progress_file else seg_dir / "progress.json"

    if resume and indices[0] > 0:
        previous_mask_path = seg_dir / f"{indices[0] - 1:06d}.png"
        if previous_mask_path.exists():
            previous_mask = cv2.imread(str(previous_mask_path), cv2.IMREAD_GRAYSCALE)
            if previous_mask is not None and np.any(previous_mask > 0):
                tracker = ObjectTracker(
                    previous_mask > 0, tracker_prompt_type, min_mask_area, iou_threshold
                )
                print(f"[Resume] Tracker initialized from {previous_mask_path}")

    for frame_idx, img_file in enumerate(selected_files):
        frame_num = indices[frame_idx]
        print(f"  [{frame_idx+1}/{len(selected_files)}] Frame {frame_num:06d} ({img_file})")

        img_bgr = cv2.imread(str(img_dir / img_file))
        if img_bgr is None:
            print(f"    Warning: could not read {img_file}, saving empty mask.")
            H, W = 1200, 1920   # fallback shape — adjust if needed
            save_seg_mask(None, (H, W), seg_dir / f"{frame_num:06d}.png")
            continue

        H, W  = img_bgr.shape[:2]
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        seg_out = seg_dir / f"{frame_num:06d}.png"

        if resume and seg_out.exists():
            existing_mask = cv2.imread(str(seg_out), cv2.IMREAD_GRAYSCALE)
            if existing_mask is not None and np.any(existing_mask > 0):
                tracker = ObjectTracker(
                    existing_mask > 0, tracker_prompt_type, min_mask_area, iou_threshold
                )
                print(f"    [Resume] Existing mask kept: {seg_out}")
                progress_path.write_text(json.dumps({
                    "last_completed_frame": frame_num,
                    "status": "running",
                }, indent=2))
                continue

        # Run FastSAM
        try:
            everything_results = model(
                img_pil,
                device=device,
                retina_masks=True,
                imgsz=fastsam_imgsz,
                conf=fastsam_conf,
                iou=fastsam_iou,
            )
        except Exception as exc:
            print(f"    ERROR: FastSAM failed at frame {frame_num}: {exc}")
            mask = tracker.last_mask if tracker is not None else None
            save_seg_mask(mask, (H, W), seg_out)
            progress_path.write_text(json.dumps({
                "last_completed_frame": frame_num,
                "status": "warning_fastsam_error",
                "error": str(exc),
            }, indent=2))
            continue

        # --- Initialise tracker on first frame ---
        if tracker is None:
            mask = initialize_object_on_first_frame(
                img_pil, everything_results, device,
                prompt_type, init_point, init_box, init_text, union_box_prompt,
                union_box_padding,
            )
            if mask is None:
                print("    ERROR: Initialisation failed. Check prompt and first frame.")
                save_seg_mask(None, (H, W), seg_out)
                continue

            tracker = ObjectTracker(
                mask, tracker_prompt_type, min_mask_area, iou_threshold
            )
            print(f"    Tracker initialised. Centroid: {tracker.centroid}, Box: {tracker.box}")

        # --- Track on subsequent frames ---
        else:
            mask = get_mask_for_frame(img_pil, everything_results, tracker, device, union_box_prompt, union_box_padding)
            if mask is None:
                print(f"    Warning: object lost at frame {frame_num}. Saving empty mask.")

        # --- Save binary mask ---
        save_seg_mask(mask, (H, W), seg_out)
        print(f"    Saved: {seg_out}  (foreground px: {mask.sum() if mask is not None else 0})")

        # --- Optional debug visualisation ---
        if save_debug and debug_dir is not None:
            save_debug_visualization(
                img_bgr, mask,
                debug_dir / f"{frame_num:06d}_debug.png"
            )

        progress_path.write_text(json.dumps({
            "last_completed_frame": frame_num,
            "status": "running",
        }, indent=2))

    print(f"\n[SegGen] Done. Masks saved to: {seg_dir}")
    print(f"[SegGen] Mask format: single-channel PNG, values {{0=background, 1=foreground}}")
    progress_path.write_text(json.dumps({
        "last_completed_frame": indices[-1],
        "status": "complete",
    }, indent=2))
    print(f"[SegGen] Note: multiply by 255 to visualise masks in image viewers.")


# =============================================================================
# SECTION 6: CLI Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate per-frame binary segmentation masks using FastSAM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Image source
    parser.add_argument("--src_img_dir", type=str, default=None,
                        help="Path to source image directory (local).")
    parser.add_argument("--out_seg_dir", type=str, default="./seg_output",
                        help="Directory to save output segmentation masks.")
    parser.add_argument("--out_debug_dir", type=str, default=None,
                        help="Directory to save debug visualizations.")

    # Output
    parser.add_argument("--save_debug",    action="store_true",
                        help="Save side-by-side debug visualizations.")

    # FastSAM
    parser.add_argument("--fastsam_weights", type=str,
                        default="../FastSAM/weights/FastSAM-x.pt",
                        help="Path to FastSAM model weights (.pt file).")
    parser.add_argument("--fastsam_conf",    type=float, default=0.25,
                        help="FastSAM confidence threshold.")
    parser.add_argument("--fastsam_iou",     type=float, default=0.9,
                        help="FastSAM IoU threshold.")
    parser.add_argument("--fastsam_imgsz",   type=int,   default=1024,
                        help="FastSAM input image size.")
    parser.add_argument("--device",          type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device: 'cuda' or 'cpu'.")

    # Prompts
    parser.add_argument("--prompt_type", type=str, default="point",
                        choices=["point", "box", "text"],
                        help="FastSAM prompt type for first-frame initialisation.")
    parser.add_argument("--init_point",  type=int, nargs=2, default=None,
                        metavar=("X", "Y"),
                        help="Initial point prompt [x y] (for prompt_type=point).")
    parser.add_argument("--init_box",    type=int, nargs=4, default=None,
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="Initial box prompt [x0 y0 x1 y1] (for prompt_type=box).")
    parser.add_argument("--init_text",   type=str, default=None,
                        help="Initial text prompt (for prompt_type=text).")

    # Tracker
    parser.add_argument("--tracker_prompt_type", type=str, default="point",
                        choices=["point", "box"],
                        help="Tracker prompt type for subsequent frames.")
    parser.add_argument("--union_box_prompt", action="store_true",
                        help="Union FastSAM candidates belonging to the spacecraft box.")
    parser.add_argument("--union_box_padding", type=int, default=80,
                        help="Pixels added around each tracking box when collecting split spacecraft parts.")
    parser.add_argument("--min_mask_area",  type=int,   default=100,
                        help="Minimum mask area in pixels to accept a detection.")
    parser.add_argument("--iou_threshold",  type=float, default=0.1,
                        help="Minimum IoU with previous mask to accept a detection.")

    # Frame selection
    parser.add_argument("--num_frames",    type=int,   default=None,
                        help="Number of frames to process (None = all).")
    parser.add_argument("--sample_method", type=str,   default="uniform",
                        choices=["uniform", "user_defined"],
                        help="Frame sampling strategy.")
    parser.add_argument("--frame_list",    type=int,   nargs="+", default=None,
                        help="Explicit list of frame indices (for sample_method=user_defined).")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="First source frame index, inclusive.")
    parser.add_argument("--end_frame", type=int, default=None,
                        help="Last source frame index, exclusive.")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse existing masks and initialize from the previous batch mask.")
    parser.add_argument("--progress_file", type=str, default=None,
                        help="Progress JSON path. Defaults to <out_seg_dir>/progress.json.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    object_name = "Cheops"
    output_dir = f"./output/Cheops_textured_bright"

    generate_segmentation_masks(
        src_img_dir          = args.src_img_dir if args.src_img_dir else f"{output_dir}/image/",
        out_seg_dir          = args.out_seg_dir if args.out_seg_dir else f"{output_dir}/seg/",
        save_debug           = args.save_debug,
        out_debug_dir        = args.out_debug_dir if args.out_debug_dir else f"{output_dir}/debug/",
        fastsam_weights      = args.fastsam_weights,
        fastsam_conf         = args.fastsam_conf,
        fastsam_iou          = args.fastsam_iou,
        fastsam_imgsz        = args.fastsam_imgsz,
        device               = args.device,
        prompt_type          = args.prompt_type,
        init_point           = args.init_point,
        init_box             = args.init_box,
        init_text            = args.init_text,
        tracker_prompt_type  = args.tracker_prompt_type,
        union_box_prompt     = args.union_box_prompt,
        union_box_padding    = args.union_box_padding,
        min_mask_area        = args.min_mask_area,
        iou_threshold        = args.iou_threshold,
        num_frames           = args.num_frames,
        sample_method        = args.sample_method,
        frame_list           = args.frame_list,
        start_frame          = args.start_frame,
        end_frame            = args.end_frame,
        resume               = args.resume,
        progress_file        = args.progress_file,
    )
