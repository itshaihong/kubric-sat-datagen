"""Parse current Kubric metadata and pose labels into TUM ground truth files.

Current authoritative schema:
  pose_labels.json:
    camera_position_world_m
    camera_quaternion_world_wxyz
    object_position_world_m
    object_quaternion_world_wxyz
    r_obj2cam
    q_obj2cam              # [w, x, y, z]

The parser also accepts older metadata files containing per-frame arrays.
"""

import argparse
import json
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
    return (f"{timestamp:.10f} "
            f"{tx:.10f} {ty:.10f} {tz:.10f} "
            f"{qx:.10f} {qy:.10f} {qz:.10f} {qw:.10f}")


def write_tum_file(output_path: str, lines: list, header_comment: str = ""):
    with open(output_path, "w") as f:
        if header_comment:
            for line in header_comment.strip().split("\n"):
                f.write(f"# {line}\n")
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for line in lines:
            f.write(line + "\n")
    print(f"  Written: {output_path}  ({len(lines)} frames)")


def write_times_file(output_path: str, timestamps: list):
    with open(output_path, "w") as f:
        f.write("# timestamp\n")
        for ts in timestamps:
            f.write(f"{ts:.10f}\n")
    print(f"  Written: {output_path}  ({len(timestamps)} timestamps)")


def check_unit_quaternion(q: np.ndarray, label: str, frame_idx: int):
    norm = np.linalg.norm(q)
    if abs(norm - 1.0) > 1e-4:
        print(f"  [WARN] {label} quaternion frame {frame_idx} norm={norm:.6f} (expected 1.0)")


# =============================================================================
# Camera Pose Parser
# =============================================================================

def parse_camera_poses(metadata: dict, frame_rate: float) -> tuple:
    """
    Parse camera poses from metadata.json camera section.

    Camera is FIXED in this dataset — all frames have identical pose.
    Quaternion convention: [w, x, y, z] → reordered to TUM [qx, qy, qz, qw]
    Position units: metres (world frame)

    Returns:
        timestamps (list[float])
        tum_lines  (list[str])
    """
    cam         = metadata["camera"]
    positions   = cam["positions"]    # list of [tx, ty, tz]
    quaternions = cam["quaternions"]  # list of [w, x, y, z]

    timestamps = []
    tum_lines  = []

    for i in range(len(positions)):
        ts       = i / frame_rate
        tx, ty, tz = positions[i]

        w, x, y, z = quaternions[i]            # metadata cam: [w, x, y, z]
        qx, qy, qz, qw = x, y, z, w            # → TUM: [qx, qy, qz, qw]

        check_unit_quaternion(np.array([qx, qy, qz, qw]), "camera", i)

        timestamps.append(ts)
        tum_lines.append(format_tum_line(ts, tx, ty, tz, qx, qy, qz, qw))

    return timestamps, tum_lines


# =============================================================================
# Object Pose Parser — WORLD frame (primary GT source)
# =============================================================================

def parse_object_poses_world(metadata: dict, frame_rate: float) -> list:
    """
    Parse object poses in WORLD frame from metadata.json instances section.
    This is the PRIMARY ground truth source — directly from PyBullet simulation.

    Quaternion convention in metadata instances: [w, x, y, z]  (same as camera)
    Position units: metres in world frame (scale=MM_TO_M applies to mesh only,
                    not to PyBullet rigid body positions)

    Returns:
        tum_lines (list[str])
    """
    instances = metadata.get("instances", [])
    if not instances:
        print("  [WARN] No instances found in metadata.json — skipping world object poses")
        return []

    obj         = instances[0]
    positions   = obj["positions"]    # [tx, ty, tz] world frame, metres
    quaternions = obj["quaternions"]  # [w, x, y, z] world frame

    tum_lines = []

    for i in range(len(positions)):
        ts         = i / frame_rate
        tx, ty, tz = positions[i]

        w, x, y, z     = quaternions[i]        # metadata instances: [w, x, y, z]
        qx, qy, qz, qw = x, y, z, w            # → TUM: [qx, qy, qz, qw]

        check_unit_quaternion(np.array([qx, qy, qz, qw]), "world object", i)

        tum_lines.append(format_tum_line(ts, tx, ty, tz, qx, qy, qz, qw))

    return tum_lines


# =============================================================================
# Object Pose Parser — CAMERA frame (from pose_labels.json)
# =============================================================================

def parse_object_poses_camera(pose_labels: list, frame_rate: float) -> list:
    """
    Parse object poses in CAMERA frame from pose_labels.json.

    NOTE: pose_labels.json r_obj2cam was generated from generate_trajectory()
    which used wrong time units (m/frame vs m/s). Prefer world frame GT from
    parse_object_poses_world() and transform to camera frame if needed.

    Quaternion convention: [x, y, z, w] (scipy) → TUM [qx, qy, qz, qw] (no reorder needed)
    Position units: metres in camera frame
    """
    tum_lines = []

    for i, entry in enumerate(pose_labels):
        ts         = i / frame_rate
        tx, ty, tz = entry["r_obj2cam"]
        qx, qy, qz, qw = entry["q_obj2cam"]    # already [x,y,z,w] = TUM order

        check_unit_quaternion(np.array([qx, qy, qz, qw]), "camera-frame object", i)

        expected_fn = f"{i:06d}.png"
        if entry.get("filename", expected_fn) != expected_fn:
            print(f"  [WARN] Frame {i}: filename mismatch in pose_labels.json")

        tum_lines.append(format_tum_line(ts, tx, ty, tz, qx, qy, qz, qw))

    return tum_lines


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parse Kubric outputs into TUM ground truth files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    object_name = "Cheops"
    output_dir  = (f"/home/haihong/tracking_dataset/kubric_sim/"
                   f"reconstruction-tracking-synthetic/{object_name}/")

    parser.add_argument("--metadata",    type=str, default=f"{output_dir}/metadata.json")
    parser.add_argument("--pose_labels", type=str, default=f"{output_dir}/pose_labels.json")
    parser.add_argument("--output_dir",  type=str, default=output_dir)
    parser.add_argument("--frame_rate",  type=float, default=None,
                        help="Override frame rate (default: read from metadata.json)")
    parser.add_argument("--skip_camera_frame_object", action="store_true", default=False,
                        help="Skip writing object_pose_ground_truth.txt (camera frame, unreliable)")

    args = parser.parse_args()

    print(f"\n[Parser] Loading metadata:    {args.metadata}")
    metadata    = load_json(args.metadata)
    print(f"[Parser] Loading pose_labels: {args.pose_labels}")
    pose_labels = load_json(args.pose_labels)

    frame_rate = args.frame_rate or metadata["metadata"]["frame_rate"]
    print(f"[Parser] Frame rate: {frame_rate} fps")

    num_frames_meta  = metadata["metadata"]["num_frames"]
    num_frames_cam   = len(metadata["camera"]["positions"])
    num_frames_inst  = len(metadata["instances"][0]["positions"])
    num_frames_pose  = len(pose_labels)

    print(f"\n[Parser] Frame counts:")
    print(f"  metadata.num_frames:              {num_frames_meta}")
    print(f"  metadata.camera.positions:        {num_frames_cam}")
    print(f"  metadata.instances[0].positions:  {num_frames_inst}")
    print(f"  pose_labels entries:              {num_frames_pose}")

    num_frames = min(num_frames_cam, num_frames_inst, num_frames_pose)
    if not (num_frames_meta == num_frames_cam == num_frames_inst == num_frames_pose):
        print(f"  [WARN] Frame count mismatch — using min = {num_frames}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Parser] Output directory: {out_dir.resolve()}")

    # --- Camera poses (world frame) ---
    print("\n[Parser] Parsing camera poses (world frame)...")
    timestamps, cam_lines = parse_camera_poses(metadata, frame_rate)
    write_tum_file(
        str(out_dir / "pose_ground_truth.txt"),
        cam_lines[:num_frames],
        ("Camera poses in WORLD frame\n"
         "Source: metadata.json camera.positions + camera.quaternions\n"
         "Quaternion input [w,x,y,z] → output [qx,qy,qz,qw] (TUM)\n"
         f"Position units: metres | Frame rate: {frame_rate} fps | Frames: {num_frames}")
    )

    # --- Object poses (world frame) — PRIMARY GT ---
    print("\n[Parser] Parsing object poses (world frame) — PRIMARY GT...")
    obj_world_lines = parse_object_poses_world(metadata, frame_rate)
    write_tum_file(
        str(out_dir / "object_pose_ground_truth.txt"),
        obj_world_lines[:num_frames],
        ("Object poses in WORLD frame  *** PRIMARY GROUND TRUTH ***\n"
         "Source: metadata.json instances[0].positions + instances[0].quaternions\n"
         "Directly from PyBullet simulation — use this in preference to pose_labels.json\n"
         "Quaternion input [w,x,y,z] → output [qx,qy,qz,qw] (TUM)\n"
         f"Position units: metres | Frame rate: {frame_rate} fps | Frames: {num_frames}")
    )

    # --- Object poses (camera frame) — secondary, potentially unreliable ---
    if not args.skip_camera_frame_object:
        print("\n[Parser] Parsing object poses (camera frame from pose_labels.json)...")
        obj_cam_lines = parse_object_poses_camera(pose_labels, frame_rate)
        write_tum_file(
            str(out_dir / "object_pose_ground_truth_unreliable.txt"),
            obj_cam_lines[:num_frames],
            ("Object poses in CAMERA frame\n"
             "Source: pose_labels.json r_obj2cam + q_obj2cam\n"
             "WARNING: generated from generate_trajectory() which may not match rendered frames.\n"
             "Prefer object_pose_ground_truth.txt transformed to camera frame.\n"
             "Quaternion input [x,y,z,w] → output [qx,qy,qz,qw] (TUM)\n"
             f"Position units: metres | Frame rate: {frame_rate} fps | Frames: {num_frames}")
        )

    # --- times.txt ---
    print("\n[Parser] Writing times.txt...")
    write_times_file(str(out_dir / "times.txt"), timestamps[:num_frames])

    # --- Sanity check printout ---
    print("\n" + "=" * 60)
    print("SANITY CHECK — First 3 lines of each output:")
    print("=" * 60)

    print("\n[pose_ground_truth.txt]  (camera, world frame)")
    for line in cam_lines[:3]:
        print(f"  {line}")

    print("\n[object_pose_ground_truth.txt]  (object, world frame — PRIMARY)")
    for line in obj_world_lines[:3]:
        print(f"  {line}")

    print(f"\n[Parser] Done → {out_dir.resolve()}")


if __name__ == "__main__":
    main()