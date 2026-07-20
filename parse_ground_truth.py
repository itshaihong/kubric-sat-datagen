"""
parse_ground_truth.py
=====================
Parses Kubric simulation outputs into TUM-format ground truth files.

Outputs:
    camera_ground_truth.txt  — camera pose in WORLD frame
    object_ground_truth.txt  — object pose in CAMERA frame
    times.txt                — timestamps matching each frame

TUM Format (per line):
    timestamp tx ty tz qx qy qz qw

Quaternion Conventions (verified by cross-checking both files):
    metadata.json  camera.quaternions[i]     → [w, x, y, z]  WORLD frame
    metadata.json  instances.quaternions[i]  → [x, y, z, w]  WORLD frame  (scipy)
    pose_labels.json  q_obj2cam             → [x, y, z, w]  CAMERA frame (scipy)

Usage:
    python parse_ground_truth.py \\
        --metadata   path/to/metadata.json \\
        --pose_labels path/to/pose_labels.json \\
        --output_dir path/to/output/

    # or with explicit frame rate override
    python parse_ground_truth.py \\
        --metadata metadata.json \\
        --pose_labels pose_labels.json \\
        --output_dir ./gt \\
        --frame_rate 24
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path


# =============================================================================
# Helpers
# =============================================================================

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def format_tum_line(timestamp: float,
                    tx: float, ty: float, tz: float,
                    qx: float, qy: float, qz: float, qw: float) -> str:
    """
    Format a single TUM ground truth line.
    TUM format: timestamp tx ty tz qx qy qz qw
    Uses high-precision float formatting to avoid rounding errors.
    """
    return (f"{timestamp:.10f} "
            f"{tx:.10f} {ty:.10f} {tz:.10f} "
            f"{qx:.10f} {qy:.10f} {qz:.10f} {qw:.10f}")


def write_tum_file(output_path: str,
                   lines: list,
                   header_comment: str = ""):
    """
    Write a TUM-format ground truth file with an optional comment header.
    Comment lines start with '#' and are ignored by TUM evaluation tools.
    """
    with open(output_path, "w") as f:
        # Header comments
        if header_comment:
            for line in header_comment.strip().split("\n"):
                f.write(f"# {line}\n")
        f.write("# timestamp tx ty tz qx qy qz qw\n")

        # Data lines
        for line in lines:
            f.write(line + "\n")

    print(f"  Written: {output_path}  ({len(lines)} frames)")


def write_times_file(output_path: str, timestamps: list):
    """
    Write times.txt — one timestamp per line, matching frame order.
    This is used by algorithms that need a separate timing file.
    """
    with open(output_path, "w") as f:
        f.write("# timestamp\n")
        for ts in timestamps:
            f.write(f"{ts:.10f}\n")
    print(f"  Written: {output_path}  ({len(timestamps)} timestamps)")


# =============================================================================
# Camera Pose Parser
# =============================================================================

def parse_camera_poses(metadata: dict, frame_rate: float) -> tuple:
    """
    Parse camera poses from metadata.json.

    Camera is FIXED in Kubric (static observer), so all frames have
    the same world-frame position and orientation.

    Quaternion convention in metadata.json camera section: [w, x, y, z]
    Output convention (TUM):                               [qx, qy, qz, qw]

    Returns:
        timestamps (list of float)
        tum_lines  (list of str)
    """
    cam = metadata["camera"]
    positions   = cam["positions"]    # list of [tx, ty, tz], world frame
    quaternions = cam["quaternions"]  # list of [w, x, y, z], world frame

    num_frames = len(positions)
    timestamps = []
    tum_lines  = []

    for i in range(num_frames):
        ts = i / frame_rate

        tx, ty, tz = positions[i]

        # metadata camera quaternion: [w, x, y, z] → reorder to TUM [qx, qy, qz, qw]
        w, x, y, z = quaternions[i]
        qx, qy, qz, qw = x, y, z, w

        # Sanity check: unit quaternion
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        if abs(norm - 1.0) > 1e-4:
            print(f"  [WARN] Camera quaternion frame {i} norm={norm:.6f} (expected 1.0)")

        timestamps.append(ts)
        tum_lines.append(format_tum_line(ts, tx, ty, tz, qx, qy, qz, qw))

    return timestamps, tum_lines


# =============================================================================
# Object Pose Parser
# =============================================================================

def parse_object_poses(pose_labels: list, frame_rate: float) -> list:
    """
    Parse object poses from pose_labels.json.

    pose_labels.json contains the object pose relative to the CAMERA frame.
    This is the most directly useful format for pose estimation algorithms.

    Quaternion convention in pose_labels.json: [x, y, z, w]  (scipy convention)
    Output convention (TUM):                   [qx, qy, qz, qw]  ← same order, no reordering needed

    Position:  r_obj2cam = [tx, ty, tz] in CAMERA frame, meters
    Orientation: q_obj2cam = [qx, qy, qz, qw] in CAMERA frame

    Returns:
        tum_lines (list of str)
    """
    tum_lines = []

    for i, entry in enumerate(pose_labels):
        ts = i / frame_rate

        # Position: [tx, ty, tz] in camera frame, meters
        tx, ty, tz = entry["r_obj2cam"]

        # Quaternion: [x, y, z, w] → TUM needs [qx, qy, qz, qw] — same order
        qx, qy, qz, qw = entry["q_obj2cam"]

        # Sanity check: unit quaternion
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        if abs(norm - 1.0) > 1e-4:
            print(f"  [WARN] Object quaternion frame {i} norm={norm:.6f} (expected 1.0)")

        # Verify filename matches expected frame index
        expected_filename = f"{i:06d}.png"
        actual_filename   = entry.get("filename", expected_filename)
        if actual_filename != expected_filename:
            print(f"  [WARN] Frame {i}: expected '{expected_filename}', "
                  f"got '{actual_filename}' — using index-based timestamp")

        tum_lines.append(format_tum_line(ts, tx, ty, tz, qx, qy, qz, qw))

    return tum_lines


# =============================================================================
# Optional: World-Frame Object Pose from metadata instances
# =============================================================================

def parse_object_poses_world(metadata: dict, frame_rate: float) -> list:
    """
    Parse object poses in WORLD frame from metadata.json instances section.

    Use this if your algorithm needs world-frame object poses instead of
    camera-relative poses.

    Quaternion convention in metadata instances: [x, y, z, w]  (scipy convention)
    Position: in Kubric simulation units — scaled by step_rate internally.
    
    NOTE: positions here are in Kubric's internal simulation units, not directly
    in meters. The pose_labels.json r_obj2cam is in meters in camera frame and
    is the recommended source for metric ground truth.

    Returns:
        tum_lines (list of str)
    """
    instances = metadata.get("instances", [])
    if not instances:
        print("  [WARN] No instances found in metadata.json")
        return []

    obj = instances[0]  # first (and typically only) object
    positions   = obj["positions"]    # [tx, ty, tz] world frame, simulation units
    quaternions = obj["quaternions"]  # [qx, qy, qz, qw] world frame (scipy)

    num_frames = len(positions)
    tum_lines  = []

    for i in range(num_frames):
        ts = i / frame_rate

        tx, ty, tz = positions[i]

        # instances quaternion: [x, y, z, w] → TUM [qx, qy, qz, qw] — same order
        qx, qy, qz, qw = quaternions[i]

        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        if abs(norm - 1.0) > 1e-4:
            print(f"  [WARN] World object quaternion frame {i} norm={norm:.6f}")

        tum_lines.append(format_tum_line(ts, tx, ty, tz, qx, qy, qz, qw))

    return tum_lines


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parse Kubric metadata.json and pose_labels.json into TUM ground truth files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files:
  camera_ground_truth.txt  — camera pose in world frame (from metadata.json)
  object_ground_truth.txt  — object pose in camera frame (from pose_labels.json)
  object_world_ground_truth.txt — object pose in world frame (from metadata.json instances)
  times.txt                — timestamps for each frame

TUM format:
  timestamp tx ty tz qx qy qz qw

Quaternion conventions (verified):
  camera_ground_truth.txt       → metadata camera.quaternions [w,x,y,z] → reordered to [qx,qy,qz,qw]
  object_ground_truth.txt       → pose_labels q_obj2cam [x,y,z,w]       → output as   [qx,qy,qz,qw]
  object_world_ground_truth.txt → metadata instances.quaternions [x,y,z,w] → output as [qx,qy,qz,qw]
        """
    )


    object_name = "Cheops"
    output_dir = f"/home/haihong/tracking_dataset/kubric_sim/reconstruction-tracking-synthetic/{object_name}/"

    parser.add_argument(
        "--metadata",
        type=str,
        default=f"{output_dir}/metadata.json",
        help="Path to metadata.json (Kubric output)"
    )
    parser.add_argument(
        "--pose_labels",
        type=str,
        default=f"{output_dir}/pose_labels.json",
        help="Path to pose_labels.json"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        help="Directory to write output files (default: ./ground_truth)"
    )
    parser.add_argument(
        "--frame_rate",
        type=float,
        default=None,
        help="Frame rate in Hz. If not set, reads from metadata.json (recommended)."
    )
    parser.add_argument(
        "--also_write_world_object",
        action="store_true",
        default=False,
        help="Also write object_world_ground_truth.txt from metadata instances (world frame, simulation units)"
    )

    args = parser.parse_args()

    # --- Load files ---
    print(f"\n[Parser] Loading metadata:    {args.metadata}")
    metadata = load_json(args.metadata)

    print(f"[Parser] Loading pose_labels: {args.pose_labels}")
    pose_labels = load_json(args.pose_labels)

    # --- Determine frame rate ---
    if args.frame_rate is not None:
        frame_rate = args.frame_rate
        print(f"[Parser] Frame rate: {frame_rate} fps (from --frame_rate argument)")
    else:
        frame_rate = metadata["metadata"]["frame_rate"]
        print(f"[Parser] Frame rate: {frame_rate} fps (from metadata.json)")

    num_frames_meta  = metadata["metadata"]["num_frames"]
    num_frames_pose  = len(pose_labels)
    num_frames_cam   = len(metadata["camera"]["positions"])

    print(f"\n[Parser] Frame counts:")
    print(f"  metadata.num_frames:              {num_frames_meta}")
    print(f"  metadata.camera.positions:        {num_frames_cam}")
    print(f"  pose_labels entries:              {num_frames_pose}")

    if not (num_frames_meta == num_frames_cam == num_frames_pose):
        print(f"\n  [WARN] Frame count mismatch! Using min({num_frames_cam}, {num_frames_pose})")

    num_frames = min(num_frames_cam, num_frames_pose)

    # --- Output directory ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Parser] Output directory: {out_dir.resolve()}")

    # --- Parse camera poses ---
    print("\n[Parser] Parsing camera poses (world frame)...")
    timestamps, cam_tum_lines = parse_camera_poses(metadata, frame_rate)

    cam_header = (
        f"Camera poses in WORLD frame\n"
        f"Source: metadata.json camera.positions + camera.quaternions\n"
        f"Quaternion input format:  [w, x, y, z]\n"
        f"Quaternion output format: [qx, qy, qz, qw]  (TUM standard)\n"
        f"Position units: meters\n"
        f"Frame rate: {frame_rate} fps\n"
        f"Num frames: {num_frames}"
    )
    write_tum_file(
        str(out_dir / "pose_ground_truth.txt"),
        cam_tum_lines[:num_frames],
        cam_header
    )

    # --- Parse object poses (camera frame) ---
    print("\n[Parser] Parsing object poses (camera frame)...")
    obj_tum_lines = parse_object_poses(pose_labels, frame_rate)

    obj_header = (
        f"Object poses in CAMERA frame\n"
        f"Source: pose_labels.json r_obj2cam + q_obj2cam\n"
        f"Quaternion input format:  [x, y, z, w]  (scipy convention)\n"
        f"Quaternion output format: [qx, qy, qz, qw]  (TUM standard)\n"
        f"Position units: meters\n"
        f"Frame rate: {frame_rate} fps\n"
        f"Num frames: {num_frames}"
    )
    write_tum_file(
        str(out_dir / "object_pose_ground_truth.txt"),
        obj_tum_lines[:num_frames],
        obj_header
    )

    # --- Parse object poses (world frame) — optional ---
    if args.also_write_world_object:
        print("\n[Parser] Parsing object poses (world frame from metadata instances)...")
        obj_world_lines = parse_object_poses_world(metadata, frame_rate)

        obj_world_header = (
            f"Object poses in WORLD frame\n"
            f"Source: metadata.json instances[0].positions + instances[0].quaternions\n"
            f"Quaternion input format:  [x, y, z, w]  (scipy convention)\n"
            f"Quaternion output format: [qx, qy, qz, qw]  (TUM standard)\n"
            f"Position units: Kubric simulation units (NOT directly meters)\n"
            f"Frame rate: {frame_rate} fps\n"
            f"Num frames: {num_frames}"
        )
        write_tum_file(
            str(out_dir / "object_world_ground_truth.txt"),
            obj_world_lines[:num_frames],
            obj_world_header
        )

    # --- Write times.txt ---
    print("\n[Parser] Writing times.txt...")
    write_times_file(
        str(out_dir / "times.txt"),
        timestamps[:num_frames]
    )

    # --- Print first 3 lines of each file for quick sanity check ---
    print("\n" + "="*60)
    print("SANITY CHECK — First 3 lines of each output file:")
    print("="*60)

    print(f"\n[camera_ground_truth.txt]")
    print(f"  # timestamp tx ty tz qx qy qz qw")
    for line in cam_tum_lines[:3]:
        print(f"  {line}")

    print(f"\n[object_ground_truth.txt]")
    print(f"  # timestamp tx ty tz qx qy qz qw")
    for line in obj_tum_lines[:3]:
        print(f"  {line}")

    print(f"\n[times.txt]")
    for ts in timestamps[:3]:
        print(f"  {ts:.10f}")

    print(f"\n[Parser] Done. All files written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
