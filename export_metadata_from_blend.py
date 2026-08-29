"""Recover pose labels and metadata from an already rendered Kubric Blender scene.

This does not render images. Run inside the Kubric Docker image, for example:
  docker run --rm --user "$(id -u):$(id -g)" \
    --volume "$(pwd):/kubric" kubricdockerhub/kubruntu \
    /usr/bin/python3 /kubric/export_metadata_from_blend.py \
    --blend /kubric/output/Cheops_1000f_left_to_right_cv_bright/blender_scene.blend \
    --output-dir /kubric/output/Cheops_1000f_left_to_right_cv_bright
"""

import argparse
import json
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation
from kubric.safeimport.bpy import bpy


def quat_wxyz_from_rotation(rotation):
    q = rotation.as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def normalize(vector, name):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if length < 1e-8:
        raise ValueError(f"{name} is zero-length.")
    return vector / length


def find_object(object_name):
    obj = bpy.data.objects.get(object_name)
    if obj is not None:
        return obj

    meshes = [item for item in bpy.data.objects if item.type == "MESH"]
    if len(meshes) == 1:
        return meshes[0]
    if not meshes:
        raise RuntimeError("Could not find a satellite mesh/object in the blend file.")
    raise RuntimeError(
        f"Object '{object_name}' not found. Mesh objects are: "
        + ", ".join(item.name for item in meshes)
    )


def camera_cv_basis(camera):
    """World-to-camera proper rotation: x right, y down, z forward."""
    # Blender camera local axes are x=right, y=up, -z=forward.
    camera_to_world = np.asarray(camera.matrix_world.to_3x3(), dtype=float)
    return np.vstack([
        camera_to_world[:, 0],
        -camera_to_world[:, 1],
        -camera_to_world[:, 2],
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True, help="Saved blender_scene.blend")
    parser.add_argument("--output-dir", required=True, help="Existing dataset output directory")
    parser.add_argument("--object-name", default="cheops_satellite")
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()

    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend))
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        raise RuntimeError("The blend file has no active camera.")
    satellite = find_object(args.object_name)

    frame_start = int(scene.frame_start)
    frame_end = int(scene.frame_end)
    frame_numbers = list(range(frame_start, frame_end + 1))
    fps = float(args.fps if args.fps is not None else scene.render.fps)

    pose_labels = []
    camera_positions = []
    camera_quaternions = []
    object_positions = []
    object_quaternions = []

    for frame in frame_numbers:
        scene.frame_set(frame)
        camera_basis = camera_cv_basis(camera)
        camera_position = np.asarray(camera.matrix_world.translation, dtype=float)
        object_position = np.asarray(satellite.matrix_world.translation, dtype=float)

        object_rotation_world = satellite.matrix_world.to_quaternion().normalized()
        object_rotation_world = Rotation.from_quat([
            object_rotation_world.x, object_rotation_world.y,
            object_rotation_world.z, object_rotation_world.w,
        ])
        object_rotation_camera = (
            Rotation.from_matrix(camera_basis) * object_rotation_world
        )
        relative_position = camera_basis @ (object_position - camera_position)

        if not np.isclose(np.linalg.det(camera_basis), 1.0, atol=1e-5):
            raise RuntimeError(f"Invalid camera rotation at frame {frame}.")
        if relative_position[2] <= 0.0:
            raise RuntimeError(
                f"Object is behind camera at frame {frame}: {relative_position.tolist()}"
            )

        camera_to_world = Rotation.from_matrix(camera_basis.T)
        camera_positions.append(camera_position.tolist())
        camera_quaternions.append(quat_wxyz_from_rotation(camera_to_world))
        object_positions.append(object_position.tolist())
        object_quaternions.append([
            float(object_rotation_world.as_quat()[3]),
            float(object_rotation_world.as_quat()[0]),
            float(object_rotation_world.as_quat()[1]),
            float(object_rotation_world.as_quat()[2]),
        ])

        pose_labels.append({
            "filename": f"{frame - frame_start:06d}.png",
            "frame": frame,
            "camera_position_world_m": camera_position.tolist(),
            "camera_quaternion_world_wxyz": camera_quaternions[-1],
            "q_obj2cam": quat_wxyz_from_rotation(object_rotation_camera),
            "r_obj2cam": relative_position.tolist(),
            "object_position_world_m": object_position.tolist(),
            "object_quaternion_world_wxyz": object_quaternions[-1],
        })

    camera_basis = camera_cv_basis(camera)
    scene.frame_set(frame_start)
    camera_position = np.asarray(camera.matrix_world.translation, dtype=float)
    look_direction = -np.asarray(camera.matrix_world.to_3x3(), dtype=float)[:, 2]
    object_position_start = np.asarray(object_positions[0])
    object_position_end = np.asarray(object_positions[-1])

    if len({tuple(np.round(value, 8)) for value in camera_positions}) != 1:
        print("[Pose Check] WARNING: camera position changes over the sequence.")
    if len({tuple(np.round(value, 8)) for value in camera_quaternions}) != 1:
        print("[Pose Check] WARNING: camera orientation changes over the sequence.")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "pose_labels.json"), "w") as handle:
        json.dump(pose_labels, handle, indent=2)

    camera_data = camera.data
    metadata = {
        "source": "recovered from saved Blender scene; no rerender performed",
        "camera": {
            "name": camera.name,
            "position_world_m": camera_positions[0],
            "quaternion_world_wxyz": camera_quaternions[0],
            "focal_length_mm": float(camera_data.lens),
            "sensor_width_mm": float(camera_data.sensor_width),
            "resolution": [int(scene.render.resolution_x), int(scene.render.resolution_y)],
            "fps": fps,
            "pose_frame": {
                "convention": "CV: x right, y down, z forward",
                "world_to_camera_rotation": camera_basis.tolist(),
                "camera_boresight_world": normalize(look_direction, "camera boresight").tolist(),
            },
        },
        "generation_config": {
            "output_dir": args.output_dir,
            "num_frames": len(frame_numbers),
            "fps": fps,
            "duration_seconds": len(frame_numbers) / fps,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "object_name": satellite.name,
            "object_position_world_first_m": object_position_start.tolist(),
            "object_position_world_last_m": object_position_end.tolist(),
            "recovered_from_blend": os.path.abspath(args.blend),
        },
        "instances": [{
            "name": satellite.name,
            "position": object_positions[0],
            "quaternion": object_quaternions[0],
        }],
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"[Recovery] Wrote {len(pose_labels)} pose labels.")
    print(f"[Recovery] Camera position: {np.round(camera_positions[0], 6)}")
    print(f"[Recovery] Object camera pose first: {np.round(pose_labels[0]['r_obj2cam'], 6)}")
    print(f"[Recovery] Object camera pose last:  {np.round(pose_labels[-1]['r_obj2cam'], 6)}")
    print(f"[Recovery] Output: {args.output_dir}")


if __name__ == "__main__":
    main()
