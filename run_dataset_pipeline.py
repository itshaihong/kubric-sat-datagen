#!/usr/bin/env python3
"""
Run the full satellite dataset pipeline:

1. Generate rendered RGB/depth/flow data with Kubric.
2. Segment the spacecraft from generated RGB frames using FastSAM.
3. Parse camera and object ground truth into TUM-format text files.

Run inside the same environment used by the component scripts. For rendering,
that usually means the Kubric Docker image. For segmentation, FastSAM and its
weights must be available to that same environment.

Example:

    /usr/bin/python3 /kubric/run_dataset_pipeline.py \
      --generator spacecraft \
      --seed 123 \
      --lighting cv_bright \
      --fastsam-weights /FastSAM/weights/FastSAM-x.pt \
      --init-point 960 600 \
      --device cuda

Pass extra generator options after --generator-extra, for example:

    /usr/bin/python3 /kubric/run_dataset_pipeline.py \
      --generator orbit \
      --seed 7 \
      --lighting space \
      --num-frames 8 \
      --generator-extra --orbit-elevation 15 --sun-direction -0.5 -0.2 0.8
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images, segment spacecraft, and parse ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--generator",
        choices=("spacecraft", "orbit"),
        default="spacecraft",
        help="Which image generator to run.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Generator seed.")
    parser.add_argument(
        "--lighting",
        choices=("space", "balanced", "cv_bright"),
        default="cv_bright",
        help="Generator lighting preset.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Frames/snapshots to render. Passed as --num-frames for spacecraft, --num-snapshots for orbit.",
    )
    parser.add_argument("--object-name", default="Cheops", help="Base asset filename without extension.")
    parser.add_argument("--asset-dir", default=None, help="Generator asset directory override.")
    parser.add_argument("--output-dir", default=None, help="Dataset output directory override.")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the selected output directory before running. Use only for a disposable pipeline run.",
    )

    parser.add_argument("--skip-generation", action="store_true", help="Reuse existing images/depth/flow in output dir.")
    parser.add_argument("--skip-segmentation", action="store_true", help="Skip FastSAM mask generation.")
    parser.add_argument("--skip-ground-truth", action="store_true", help="Skip TUM ground-truth parsing.")

    parser.add_argument(
        "--fastsam-weights",
        default="../FastSAM/weights/FastSAM-x.pt",
        help="Path to FastSAM weights used by generate_seg.py.",
    )
    parser.add_argument("--device", default=None, help="FastSAM device, e.g. cuda or cpu. Defaults to generate_seg.py behavior.")
    parser.add_argument("--fastsam-conf", type=float, default=0.8, help="FastSAM confidence threshold.")
    parser.add_argument("--fastsam-iou", type=float, default=0.9, help="FastSAM IoU threshold.")
    parser.add_argument("--fastsam-imgsz", type=int, default=1024, help="FastSAM inference image size.")
    parser.add_argument(
        "--prompt-type",
        choices=("point", "box", "text"),
        default="point",
        help="FastSAM first-frame prompt type.",
    )
    parser.add_argument(
        "--init-point",
        nargs=2,
        type=int,
        default=(960, 600),
        metavar=("X", "Y"),
        help="Initial point prompt for prompt_type=point. Defaults to image center for 1920x1200 renders.",
    )
    parser.add_argument(
        "--init-box",
        nargs=4,
        type=int,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Initial box prompt for prompt_type=box.",
    )
    parser.add_argument("--init-text", default=None, help="Initial text prompt for prompt_type=text.")
    parser.add_argument(
        "--tracker-prompt-type",
        choices=("point", "box"),
        default="point",
        help="Tracker prompt type for subsequent frames.",
    )
    parser.add_argument(
        "--union-box-prompt",
        action="store_true",
        help="For box prompts, union all FastSAM candidate masks whose centers are inside the box.",
    )
    parser.add_argument("--min-mask-area", type=int, default=100, help="Minimum accepted mask area.")
    parser.add_argument("--iou-threshold", type=float, default=0.1, help="Minimum IoU with previous mask.")
    parser.add_argument("--save-debug", action="store_true", help="Write segmentation debug visualizations.")

    parser.add_argument("--frame-rate", type=float, default=None, help="Override frame rate for parse_ground_truth.py.")
    parser.add_argument(
        "--skip-camera-frame-object",
        action="store_true",
        help="Ask parse_ground_truth.py not to write camera-frame object poses.",
    )

    parser.add_argument(
        "--generator-extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments appended to generate_spacecraft*.py. Put this last.",
    )

    return parser.parse_args()


def default_output_dir(args):
    if args.generator == "orbit":
        name = f"{args.object_name}_orbit_{args.lighting}_seed{args.seed}"
    else:
        name = f"{args.object_name}_{args.lighting}_seed{args.seed}"
    return SCRIPT_DIR / "output" / name


def ensure_clean_output_dir(output_dir):
    output_dir = output_dir.resolve()
    repo_output_dir = (SCRIPT_DIR / "output").resolve()
    if output_dir == SCRIPT_DIR.resolve() or output_dir == Path(output_dir.anchor):
        raise ValueError(f"Refusing to clean broad output path: {output_dir}")
    if repo_output_dir not in output_dir.parents and output_dir.exists():
        raise ValueError(
            f"Refusing to clean path outside repo output/ directory: {output_dir}"
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)


def run_command(command, stage_name):
    print(f"\n[Pipeline] {stage_name}")
    print("[Pipeline] Command:", " ".join(str(part) for part in command))
    subprocess.run(command, cwd=str(SCRIPT_DIR), check=True)


def build_generation_command(args, output_dir):
    script_name = "generate_spacecraft_orbit.py" if args.generator == "orbit" else "generate_spacecraft.py"
    command = [
        sys.executable,
        str(SCRIPT_DIR / script_name),
        "--seed", str(args.seed),
        "--lighting", args.lighting,
        "--object-name", args.object_name,
        "--output-dir", str(output_dir),
    ]
    if args.asset_dir:
        command.extend(["--asset-dir", args.asset_dir])
    if args.num_frames is not None:
        frame_arg = "--num-snapshots" if args.generator == "orbit" else "--num-frames"
        command.extend([frame_arg, str(args.num_frames)])
    command.extend(args.generator_extra)
    return command


def run_segmentation(args, output_dir):
    from generate_seg import generate_segmentation_masks

    image_dir = output_dir / "image"
    seg_dir = output_dir / "seg"
    debug_dir = output_dir / "seg_debug"

    if not image_dir.exists():
        raise FileNotFoundError(f"Generated image directory not found: {image_dir}")

    kwargs = {
        "src_img_dir": str(image_dir),
        "out_seg_dir": str(seg_dir),
        "save_debug": args.save_debug,
        "out_debug_dir": str(debug_dir),
        "fastsam_weights": args.fastsam_weights,
        "fastsam_conf": args.fastsam_conf,
        "fastsam_iou": args.fastsam_iou,
        "fastsam_imgsz": args.fastsam_imgsz,
        "prompt_type": args.prompt_type,
        "init_point": list(args.init_point) if args.init_point is not None else None,
        "init_box": list(args.init_box) if args.init_box is not None else None,
        "init_text": args.init_text,
        "tracker_prompt_type": args.tracker_prompt_type,
        "union_box_prompt": args.union_box_prompt,
        "min_mask_area": args.min_mask_area,
        "iou_threshold": args.iou_threshold,
        "num_frames": args.num_frames,
    }
    if args.device:
        kwargs["device"] = args.device

    print("\n[Pipeline] Segmenting generated images with FastSAM")
    generate_segmentation_masks(**kwargs)


def build_ground_truth_command(args, output_dir):
    command = [
        sys.executable,
        str(SCRIPT_DIR / "parse_ground_truth.py"),
        "--metadata", str(output_dir / "metadata.json"),
        "--pose_labels", str(output_dir / "pose_labels.json"),
        "--output_dir", str(output_dir),
    ]
    if args.frame_rate is not None:
        command.extend(["--frame_rate", str(args.frame_rate)])
    if args.skip_camera_frame_object:
        command.append("--skip_camera_frame_object")
    return command


def write_pipeline_manifest(args, output_dir):
    manifest = {
        "generator": args.generator,
        "seed": args.seed,
        "lighting": args.lighting,
        "num_frames_or_snapshots": args.num_frames,
        "object_name": args.object_name,
        "asset_dir": args.asset_dir,
        "output_dir": str(output_dir),
        "segmentation": {
            "fastsam_weights": args.fastsam_weights,
            "device": args.device,
            "prompt_type": args.prompt_type,
            "init_point": list(args.init_point) if args.init_point is not None else None,
            "init_box": list(args.init_box) if args.init_box is not None else None,
            "init_text": args.init_text,
            "tracker_prompt_type": args.tracker_prompt_type,
            "union_box_prompt": args.union_box_prompt,
            "fastsam_conf": args.fastsam_conf,
            "fastsam_iou": args.fastsam_iou,
            "fastsam_imgsz": args.fastsam_imgsz,
        },
        "stages": {
            "generation": not args.skip_generation,
            "segmentation": not args.skip_segmentation,
            "ground_truth": not args.skip_ground_truth,
        },
        "generator_extra": args.generator_extra,
    }
    path = output_dir / "pipeline_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[Pipeline] Manifest written: {path}")


def main():
    os.chdir(SCRIPT_DIR)
    args = parse_args()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(args).resolve()

    if args.clean_output:
        ensure_clean_output_dir(output_dir)

    if not args.skip_generation:
        run_command(build_generation_command(args, output_dir), "Generating simulation images")

    if not args.skip_segmentation:
        run_segmentation(args, output_dir)

    if not args.skip_ground_truth:
        run_command(build_ground_truth_command(args, output_dir), "Parsing camera/object ground truth")

    write_pipeline_manifest(args, output_dir)
    print(f"\n[Pipeline] Done: {output_dir}")


if __name__ == "__main__":
    main()
