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
    python seg_generation.py \
        --src_img_dir /path/to/images \
        --out_seg_dir /path/to/output/seg \
        --fastsam_weights /path/to/FastSAM-x.pt \
        --init_point 544 449 \
        --device cpu

    # Or with HuggingFace:
    python seg_generation.py \
        --hf_repo_id itshaihong/reconstruction-tracking-synthetic \
        --hf_subfolder Cheops/orbit_45deg_right_back/image \
        --out_seg_dir /path/to/output/seg \
        --fastsam_weights /path/to/FastSAM-x.pt \
        --init_point 544 449 \
        --device cuda
"""

import os
import sys
import cv2
import argparse
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
module_dir = os.path.abspath(f"{PATH}/../FastSAM")
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
        self._update_state(init_mask)
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


def initialize_object_on_first_frame(
    img_pil, everything_results, device,
    prompt_type, point=None, box=None, text=None,
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
        ann = prompt_process.box_prompt(bbox=box)

    elif prompt_type == 'text':
        if text is None:
            raise ValueError("prompt_type='text' requires --init_text '...'")
        print(f"  [Init] Text prompt: '{text}'")
        ann = prompt_process.text_prompt(text=text)

    else:
        raise ValueError(f"Unknown prompt_type '{prompt_type}'")

    return _parse_fastsam_ann(ann)


def get_mask_for_frame(img_pil, everything_results, tracker, device):
    """
    Run FastSAM on a subsequent frame using the tracker's current prompt.

    Returns:
        Boolean numpy mask, or None if FastSAM returned no result.
    """
    prompt_process  = FastSAMPrompt(img_pil, everything_results, device=device)
    current_prompt  = tracker.get_prompt()

    if tracker.prompt_type == 'box':
        print(f"  [Track] Box prompt: {current_prompt}")
        ann = prompt_process.box_prompt(bbox=current_prompt)
    else:
        print(f"  [Track] Point prompt: {current_prompt}")
        ann = prompt_process.point_prompt(points=[current_prompt], pointlabel=[1])

    mask = _parse_fastsam_ann(ann)
    if mask is not None:
        tracker.update(mask)
    return mask


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
    fastsam_conf: float       = 0.4,
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
    min_mask_area: int        = 100,
    iou_threshold: float      = 0.1,
    # --- Frame selection ---
    num_frames: int           = None,
    sample_method: str        = "uniform",
    frame_list: list          = None,
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

    selected_files = [img_files[i] for i in indices]
    print(f"[SegGen] Processing {len(selected_files)} frames.\n")

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

        # Run FastSAM
        everything_results = model(
            img_pil,
            device=device,
            retina_masks=True,
            imgsz=fastsam_imgsz,
            conf=fastsam_conf,
            iou=fastsam_iou,
        )

        # --- Initialise tracker on first frame ---
        if tracker is None:
            mask = initialize_object_on_first_frame(
                img_pil, everything_results, device,
                prompt_type, init_point, init_box, init_text,
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
            mask = get_mask_for_frame(img_pil, everything_results, tracker, device)
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

    print(f"\n[SegGen] Done. Masks saved to: {seg_dir}")
    print(f"[SegGen] Mask format: single-channel PNG, values {{0=background, 1=foreground}}")
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


    # Output
    parser.add_argument("--save_debug",    action="store_true",
                        help="Save side-by-side debug visualizations.")

    # FastSAM
    parser.add_argument("--fastsam_weights", type=str,
                        default="../FastSAM/weights/FastSAM-x.pt",
                        help="Path to FastSAM model weights (.pt file).")
    parser.add_argument("--fastsam_conf",    type=float, default=0.8,
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    object_name = "Cheops"
    output_dir = f"/home/haihong/tracking_dataset/kubric_sim/reconstruction-tracking-synthetic/{object_name}/"

    generate_segmentation_masks(
        src_img_dir          = f"{output_dir}/image/",
        out_seg_dir          = f"{output_dir}/seg/",
        save_debug           = args.save_debug,
        out_debug_dir        = f"{output_dir}/seg_debug/",
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
        min_mask_area        = args.min_mask_area,
        iou_threshold        = args.iou_threshold,
        num_frames           = args.num_frames,
        sample_method        = args.sample_method,
        frame_list           = args.frame_list,
    )
