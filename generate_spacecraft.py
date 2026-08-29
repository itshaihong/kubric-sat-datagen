"""
Spacecraft Trajectory Dataset Generator
========================================
Combines:
- Original debris simulation pipeline (RGB, Depth, Flow export)
- SPEED-UE-Cube lighting conditions (sun angle constraints, physically correct shadow)
- Randomized SO(3) orientation sampling
- Trajectory-based sequential pose simulation

Lighting follows SPEED-UE-Cube conventions:
  Constraint 1: Sun is never blocked by Earth (not simulated, but sun always illuminates spacecraft)
  Constraint 2: Angle between camera boresight and sun direction >= 75 deg

Camera follows SPEED / SPEED-UE-Cube:
  Point Grey Grasshopper 3 + Xenoplan 1.4/17mm lens
  Resolution: 1920 x 1200 (overrideable)
  Horizontal FOV: 35.6 deg
  Sensor width: 11.2512 mm (5.86 um pixel pitch x 1920 pixels)
  Focal length: 17.5217 mm (back-calculated to match 35.6 deg FOV exactly)

Docker (Linux):
    docker run --rm --interactive \
        --user $(id -u):$(id -g) \
        --volume "$(pwd):/kubric" \
        --volume "$HOME/tracking_dataset:/dataset" \
        kubricdockerhub/kubruntu \
        /usr/bin/python3 kubric-sat-datagen/generate_spacecraft.py

Docker (PowerShell):
    docker run --rm --interactive --volume "%cd%:/kubric" kubricdockerhub/kubruntu \
        /usr/bin/python3 kubric-sat-datagen/generate_spacecraft.py
"""

import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.simulator.pybullet import PyBullet as KubricSimulator
import numpy as np
import os
import shutil
import argparse
import json
import math
import re
from PIL import Image
import imageio
from scipy.spatial.transform import Rotation
from scene_setup import setup_scene_actors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
"""
Suggested presets
Goal	                    SUN_INTENSITY	FILL_INTENSITY	AMBIENT_LEVEL	SUN_SHADOW_SOFTNESS
Realistic space lighting	15.0	        0.0	            0.0	            0.00462
Balanced CV-friendly	    6.0	            1.2	            0.08	        0.08
Very easy tracking	        5.0	            2.0	            0.12            0.15
"""

# =============================================================================
# CONSTANTS
# =============================================================================
MAX_DEPTH = 9999.0   # metres — background/sky pixels replaced with 0.0
MM_TO_M   = 0.001
CM_TO_M   = 0.01

# Camera — Point Grey Grasshopper 3 + Xenoplan 1.4/17mm
# Sensor width  = 5.86 um pixel pitch x 1920 pixels = 11.2512 mm
# Focal length  = 17.5217 mm (back-calculated from paper's stated 35.6 deg horizontal FOV)
# Ref: SPEED-UE-Cube paper, Section "Camera Effects"
CAMERA_FOCAL_LENGTH_MM = 17.5217   # effective focal length (mm)
CAMERA_SENSOR_WIDTH_MM = 11.2512   # sensor width (mm)
CAMERA_SENSOR_HEIGHT_MM = 7.0320   # sensor height = 5.86 um x 1200 pixels (mm)
CAMERA_FOV_H_DEG       = 35.6      # horizontal FOV as stated in paper (deg)

# Sun — physically correct angular diameter of the Sun as seen from space
# shadow_softness = tan(sun_angular_radius) = tan(0.265 deg) ~ 0.00462
# Ref: Sun subtends ~0.53 deg diameter => radius = 0.265 deg
SUN_ANGULAR_RADIUS_DEG  = 0.265
SUN_SHADOW_SOFTNESS     = math.tan(math.radians(SUN_ANGULAR_RADIUS_DEG))  # ~ 0.00462

# Lighting constraints from SPEED-UE-Cube
# Constraint: angle between camera boresight and sun direction >= 75 deg
# This keeps the Sun outside the camera FOV at all times
SUN_MIN_ANGLE_FROM_BORESIGHT_DEG = 75.0


SUN_DISTANCE = 100.0  # arbitrary large distance for DirectionalLight

CAMERA_POSITION = np.array([0.0, -10.0, 0.0])
LOOK_AT         = np.array([0.0,  0.0,  0.0])

# Safe long-sequence motion defaults. At 20 seconds this moves only ~0.22 m,
# keeping Cheops centered in the fixed 35.6 deg camera FOV at 10 m range.
DEFAULT_FPS = 60
DEFAULT_DURATION_SECONDS = None
DEFAULT_LINEAR_VELOCITY_MPS = (0.010, 0.005, 0.000)
DEFAULT_ANGULAR_VELOCITY_DPS = (9.0, 3.0, 12.0)

# Default lighting matches the balanced preset below. Override from the CLI.
SUN_SHADOW_SOFTNESS = 0.08
SUN_INTENSITY = 6.0
FILL_INTENSITY = 1.2
AMBIENT_LEVEL = 0.08
SUN_MAX_ANGLE_FROM_BORESIGHT_DEG = 115.0

LIGHTING_PRESETS = {
    # High contrast, physically closer to direct sunlight in space.
    # Dark sides and silhouette edges stay dark.
    "space": {
        "sun_intensity": 15.0,
        "fill_intensity": 0.0,
        "ambient_level": 0.0,
        "sun_shadow_softness": math.tan(math.radians(SUN_ANGULAR_RADIUS_DEG)),
        "sun_min_angle": SUN_MIN_ANGLE_FROM_BORESIGHT_DEG,
        "sun_max_angle": 115.0,
        "description": "direct sun only, hard shadows, strong dark edges",
    },
    # Moderate visibility while still retaining directional lighting.
    "balanced": {
        "sun_intensity": 6.0,
        "fill_intensity": 1.2,
        "ambient_level": 0.08,
        "sun_shadow_softness": 0.08,
        "sun_min_angle": SUN_MIN_ANGLE_FROM_BORESIGHT_DEG,
        "sun_max_angle": 115.0,
        "description": "directional sunlight with weak fill for tracking",
    },
    # Bright, CV-friendly images where most faces remain visible.
    "cv_bright": {
        "sun_intensity": 5.0,
        "fill_intensity": 3.5,
        "ambient_level": 0.35,
        "sun_shadow_softness": 0.15,
        "sun_min_angle": SUN_MIN_ANGLE_FROM_BORESIGHT_DEG,
        "sun_max_angle": 115.0,
        "description": "soft shadows, strong camera fill, bright ambient floor",
    },
}


# =============================================================================
# CONFIG HELPERS
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a reproducible textured spacecraft sequence."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampled sun and satellite orientation.")
    parser.add_argument(
        "--lighting",
        choices=sorted(LIGHTING_PRESETS),
        default="balanced",
        help="Lighting preset: space has hard dark edges; cv_bright keeps the satellite visible.",
    )
    parser.add_argument("--sun-intensity", type=float, default=None, help="Override preset sun intensity.")
    parser.add_argument("--fill-intensity", type=float, default=None, help="Override preset camera-side fill intensity.")
    parser.add_argument("--ambient-level", type=float, default=None, help="Override preset ambient illumination.")
    parser.add_argument("--sun-shadow-softness", type=float, default=None, help="Override preset sun shadow softness.")
    parser.add_argument("--sun-min-angle", type=float, default=None, help="Minimum camera boresight to sun angle in degrees.")
    parser.add_argument("--sun-max-angle", type=float, default=None, help="Maximum camera boresight to sun angle in degrees.")
    parser.add_argument("--sun-direction", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"), help="Use an exact world-space sun direction instead of sampling.")
    parser.add_argument("--camera-position", nargs=3, type=float, default=tuple(CAMERA_POSITION), metavar=("X", "Y", "Z"), help="Camera position in metres.")
    parser.add_argument("--look-at", nargs=3, type=float, default=tuple(LOOK_AT), metavar=("X", "Y", "Z"), help="Camera target point in metres.")
    parser.add_argument("--initial-position", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"), help="Initial satellite position in metres.")
    parser.add_argument("--initial-quaternion", nargs=4, type=float, default=None, metavar=("W", "X", "Y", "Z"), help="Use an exact initial satellite quaternion instead of sampling.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Simulation/render frame rate in frames per second.")
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS, help="Duration to render. If set, overrides --num-frames via round(duration * fps).")
    parser.add_argument("--num-frames", type=int, default=24, help="Number of frames to render when --duration-seconds is not set.")
    parser.add_argument("--object-name", default="Cheops", help="Base object filename without extension.")
    parser.add_argument("--asset-dir", default=None, help="Asset directory containing OBJ, MTL, textures, and URDF.")
    parser.add_argument("--output-dir", default=None, help="Output directory for rendered dataset.")
    parser.add_argument("--render-chunk-size", type=int, default=120, help="Number of frames rendered/postprocessed at once. Keeps long sequences from loading all frames into RAM.")
    parser.add_argument("--linear-velocity-mps", nargs=3, type=float, default=DEFAULT_LINEAR_VELOCITY_MPS, metavar=("X", "Y", "Z"), help="Satellite linear velocity in metres per second.")
    parser.add_argument("--angular-velocity-dps", nargs=3, type=float, default=DEFAULT_ANGULAR_VELOCITY_DPS, metavar=("X", "Y", "Z"), help="Satellite angular velocity in degrees per second.")
    parser.add_argument("--linear-velocity", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"), help="Deprecated: satellite linear velocity in metres per frame. Converted to m/s using --fps.")
    parser.add_argument("--angular-velocity", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"), help="Deprecated: satellite angular velocity in degrees per frame. Converted to deg/s using --fps.")
    return parser.parse_args()


def resolve_lighting_config(args):
    config = dict(LIGHTING_PRESETS[args.lighting])
    overrides = {
        "sun_intensity": args.sun_intensity,
        "fill_intensity": args.fill_intensity,
        "ambient_level": args.ambient_level,
        "sun_shadow_softness": args.sun_shadow_softness,
        "sun_min_angle": args.sun_min_angle,
        "sun_max_angle": args.sun_max_angle,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def normalize_vector(values, name):
    vector = np.array(values, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError(f"{name} must not be the zero vector.")
    return vector / norm


def quaternion_wxyz_to_kb(quaternion_wxyz):
    return kb.Quaternion(
        w=float(quaternion_wxyz[0]),
        x=float(quaternion_wxyz[1]),
        y=float(quaternion_wxyz[2]),
        z=float(quaternion_wxyz[3]),
    )


def recenter_satellite_geometry(asset_id):
    """Move imported mesh vertices so the object's origin is its bbox center."""
    import bpy
    from mathutils import Vector
    root = bpy.data.objects.get(asset_id)
    meshes = find_satellite_meshes(asset_id)
    if not meshes:
        raise RuntimeError(f"No mesh geometry found for '{asset_id}' to recenter.")
    root_matrix = root.matrix_world if root is not None else meshes[0].matrix_world
    root_inverse = root_matrix.inverted()
    points_root = np.asarray([
        root_inverse @ (obj.matrix_world @ Vector(corner))
        for obj in meshes for corner in obj.bound_box
    ], dtype=float)
    minimum = points_root.min(axis=0)
    maximum = points_root.max(axis=0)
    pivot_root = (minimum + maximum) / 2.0
    pivot_world = root_matrix @ Vector(pivot_root.tolist())
    for obj in meshes:
        delta_local = obj.matrix_world.inverted().to_3x3() @ (-Vector(pivot_world))
        for vertex in obj.data.vertices:
            vertex.co += delta_local
        obj.data.update()
    print(f"[Object] Recentered bbox pivot in root coordinates: {np.round(pivot_root, 6)}")
    return {
        "pivot_root_before_recenter_m": pivot_root.tolist(),
        "bbox_min_root_m": minimum.tolist(),
        "bbox_max_root_m": maximum.tolist(),
        "mesh_count": len(meshes),
    }


def camera_frame_basis(camera_position, look_at):
    """Return CV-style world-to-camera rotation: x right, y up, z forward."""
    forward = normalize_vector(np.asarray(look_at) - np.asarray(camera_position), "camera boresight")
    up_reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up_reference))) > 0.999:
        up_reference = np.array([0.0, 1.0, 0.0])
    right = normalize_vector(np.cross(forward, up_reference), "camera right")
    up = normalize_vector(np.cross(right, forward), "camera up")
    # A proper CV rotation uses x=right, y=down, z=forward. Using y=up
    # together with z=forward would be a reflection (determinant -1), not a
    # valid quaternion-representable rotation.
    down = -up
    return np.vstack([right, down, forward])


def rotation_to_wxyz(rotation):
    q_xyzw = rotation.as_quat()
    return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]


# =============================================================================
# FILE I/O HELPERS  (unchanged from original)
# =============================================================================
def write_flo(filename, flow):
    """Write HxWx2 float32 array to Middlebury .flo format."""
    assert flow.ndim == 3 and flow.shape[2] == 2, "Flow must be HxWx2"
    with open(filename, "wb") as f:
        np.array([202021.25], dtype=np.float32).tofile(f)
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)

def write_flo_batch(flows, output_dir, name="forward_flow", start_index=0):
    os.makedirs(output_dir, exist_ok=True)
    for i, flow in enumerate(flows):
        frame_idx = start_index + i
        write_flo(os.path.join(output_dir, f"{frame_idx:06d}.flo"), flow[..., :2])

def write_rgb_batch(rgb_frames, output_dir, start_index=0):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(rgb_frames):
        frame_idx = start_index + i
        Image.fromarray(frame[..., :3], mode="RGB").save(
            os.path.join(output_dir, f"{frame_idx:06d}.png"))

def write_tiff_depth_batch(depth_f64, output_dir, start_index=0):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(depth_f64):
        frame_idx = start_index + i
        imageio.imwrite(
            os.path.join(output_dir, f"{frame_idx:06d}.tiff"),
            frame.squeeze().astype(np.float64),
            format="tiff")

def clamp_depth_batch(depth_frames, max_depth=MAX_DEPTH):
    """
    Replace inf/nan/<=0/beyond max_depth with 0.0 (invalid sentinel).
    Background pixels from Kubric come in as inf — zeroed here.
    """
    depth = np.array(depth_frames, dtype=np.float32)
    invalid = ~np.isfinite(depth) | (depth <= 0.0) | (depth > max_depth)
    depth[invalid] = 9999.0
    return depth


def clear_render_scratch(scratch_dir):
    """Clear only Kubric/Blender per-frame scratch outputs before a chunk render."""
    for subdir in ("exr", "images"):
        path = os.path.join(scratch_dir, subdir)
        if os.path.isdir(path):
            shutil.rmtree(path)


def iter_frame_chunks(frame_start, frame_end, chunk_size):
    if chunk_size < 1:
        raise ValueError("--render-chunk-size must be >= 1")
    start = frame_start
    while start <= frame_end:
        end = min(frame_end, start + chunk_size - 1)
        yield start, end, list(range(start, end + 1))
        start = end + 1




# =============================================================================
# MATERIAL / TEXTURE HELPERS
# =============================================================================
def parse_mtl_file(mtl_path):
    """
    Parse the small Wavefront MTL subset needed for this asset.

    Returns:
        dict keyed by material name with Kd/Ks/Ns/d/illum/map_Kd entries where
        present. Texture paths are resolved relative to the MTL file.
    """
    materials = {}
    current_name = None
    mtl_dir = os.path.dirname(os.path.abspath(mtl_path))

    with open(mtl_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            parts = line.split()
            key = parts[0]
            values = parts[1:]

            if key == "newmtl" and values:
                current_name = values[0]
                materials[current_name] = {}
            elif current_name is None:
                continue
            elif key in {"Ka", "Kd", "Ks"} and len(values) >= 3:
                materials[current_name][key] = tuple(float(v) for v in values[:3])
            elif key in {"Ns", "d", "illum"} and values:
                materials[current_name][key] = float(values[0])
            elif key == "map_Kd" and values:
                texture_name = " ".join(values)
                materials[current_name][key] = os.path.join(mtl_dir, texture_name)

    return materials


def material_base_name(name):
    """Strip Blender's duplicate suffix so blinn3SG.001 maps to blinn3SG."""
    return re.sub(r"\.\d{3}$", "", name)


def find_satellite_meshes(asset_id):
    """
    Return mesh objects belonging to the imported spacecraft.

    Kubric usually imports a FileBasedObject as a Blender object named after
    asset_id, but OBJ import details can vary by Blender/Kubric version.
    """
    import bpy

    root = bpy.data.objects.get(asset_id)
    if root is not None:
        if root.type == "MESH":
            return [root]
        children = [obj for obj in root.children_recursive if obj.type == "MESH"]
        if children:
            return children

    candidates = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        slot_names = {material_base_name(slot.material.name)
                      for slot in obj.material_slots if slot.material}
        if slot_names:
            candidates.append(obj)

    return candidates


def apply_mtl_textures_to_imported_object(asset_id, mtl_path):
    """
    Force Blender materials imported by Kubric's OBJ path to use the MTL
    texture files as shader image nodes.

    This keeps geometry, UVs, physics, keyframing, and Kubric metadata intact.
    """
    import bpy

    if not os.path.exists(mtl_path):
        raise FileNotFoundError(f"MTL file not found: {mtl_path}")

    material_defs = parse_mtl_file(mtl_path)
    if not material_defs:
        raise ValueError(f"No materials found in MTL file: {mtl_path}")

    meshes = find_satellite_meshes(asset_id)
    if not meshes:
        raise RuntimeError(f"No Blender mesh objects found for asset_id='{asset_id}'.")

    applied = {}
    missing_textures = []
    unmatched_slots = []

    for obj in meshes:
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue

            base_name = material_base_name(mat.name)
            props = material_defs.get(base_name)
            if props is None:
                unmatched_slots.append(mat.name)
                continue

            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            bsdf = nodes.get("Principled BSDF")
            if bsdf is None:
                bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

            kd = props.get("Kd", (1.0, 1.0, 1.0))
            mat.diffuse_color = (kd[0], kd[1], kd[2], props.get("d", 1.0))
            if "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = mat.diffuse_color
            if "Alpha" in bsdf.inputs:
                bsdf.inputs["Alpha"].default_value = props.get("d", 1.0)
            if "Specular" in bsdf.inputs and "Ks" in props:
                bsdf.inputs["Specular"].default_value = max(props["Ks"])
            if "Roughness" in bsdf.inputs and "Ns" in props:
                bsdf.inputs["Roughness"].default_value = max(
                    0.02, min(1.0, 1.0 - props["Ns"] / 100.0)
                )

            texture_path = props.get("map_Kd")
            if texture_path:
                if not os.path.exists(texture_path):
                    missing_textures.append(texture_path)
                    continue

                image = bpy.data.images.load(texture_path, check_existing=True)
                tex_node = nodes.new(type="ShaderNodeTexImage")
                tex_node.name = f"{base_name}_diffuse_texture"
                tex_node.image = image
                tex_node.extension = "REPEAT"
                links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
                applied[base_name] = os.path.basename(texture_path)
            else:
                applied.setdefault(base_name, "[flat color]")

    print("[Materials] Applied material/texture mapping:")
    for mat_name in sorted(applied):
        print(f"  {mat_name} -> {applied[mat_name]}")
    if unmatched_slots:
        print(f"[Materials] Unmatched Blender material slots: {sorted(set(unmatched_slots))}")
    if missing_textures:
        raise FileNotFoundError(
            "Referenced texture file(s) missing: " + ", ".join(sorted(set(missing_textures)))
        )

    return applied


# =============================================================================
# LIGHTING HELPERS  (SPEED-UE-Cube conventions)
# =============================================================================
# def sample_sun_direction(
#     camera_position=CAMERA_POSITION,
#     look_at=LOOK_AT,
#     sun_distance=SUN_DISTANCE,
#     min_angle_deg=SUN_MIN_ANGLE_FROM_BORESIGHT_DEG,
#     max_attempts=1000
# ):
#     """
#     Sample a random sun position (as a unit direction scaled by sun_distance)
#     such that the angle between the camera boresight and the direction FROM
#     the camera TO the sun is >= min_angle_deg.

#     This correctly replicates SPEED-UE-Cube Constraint 2:
#         "The angle between the camera boresight and the vector from the
#          camera to the Sun must be >= 75 degrees."

#     Args:
#         camera_position (np.ndarray): Camera position in world space.
#         look_at         (np.ndarray): Point the camera is looking at.
#         sun_distance    (float)     : Distance to place the sun (arbitrary for
#                                       DirectionalLight, but needed to compute
#                                       the camera-to-sun vector correctly).
#         min_angle_deg   (float)     : Minimum angle in degrees between camera
#                                       boresight and camera-to-sun direction.
#         max_attempts    (int)       : Max rejection sampling attempts.

#     Returns:
#         sun_dir      (np.ndarray): Unit vector pointing FROM scene TOWARD sun.
#                                    Use as: sun.position = tuple(sun_dir * sun_distance)
#                                            sun.look_at((0, 0, 0))
#         sun_position (np.ndarray): Actual sun position in world space.
#         angle_deg    (float)     : Actual angle between boresight and sun (for logging).
#     """
#     # --- Camera boresight: unit vector FROM camera TOWARD look_at target ---
#     boresight = look_at - camera_position
#     boresight /= np.linalg.norm(boresight)   # = [0, 1, 0] for your setup

#     min_angle_rad = math.radians(min_angle_deg)

#     for attempt in range(max_attempts):
#         # 1. Sample a random unit vector on the sphere
#         sun_dir = np.random.randn(3)
#         sun_dir /= np.linalg.norm(sun_dir)

#         # 2. Compute actual sun position in world space
#         sun_position = sun_dir * sun_distance

#         # 3. Compute direction FROM CAMERA TO SUN (this is what matters)
#         #    Not just sun_dir — because camera is not at the origin
#         cam_to_sun = sun_position - camera_position
#         cam_to_sun /= np.linalg.norm(cam_to_sun)

#         # 4. Angle between boresight and camera-to-sun direction
#         cos_angle = np.clip(np.dot(boresight, cam_to_sun), -1.0, 1.0)
#         angle_deg = math.degrees(math.acos(cos_angle))

#         # 5. Accept if constraint satisfied
#         if angle_deg >= min_angle_deg:
#             return sun_dir, sun_position, angle_deg

#     raise RuntimeError(
#         f"Could not sample valid sun direction after {max_attempts} attempts. "
#         f"Check min_angle_deg={min_angle_deg}."
#     )
def sample_sun_direction(
    camera_position=CAMERA_POSITION,
    look_at=LOOK_AT,
    sun_distance=SUN_DISTANCE,
    min_angle_deg=SUN_MIN_ANGLE_FROM_BORESIGHT_DEG,
    max_angle_deg=SUN_MAX_ANGLE_FROM_BORESIGHT_DEG,
    max_attempts=1000,
    rng=None,
):
    """
    Sample a sun direction that is CV-friendly:
    - keeps Sun outside camera FOV
    - avoids very strong backlighting
    - avoids grazing illumination that produces large dark shadows
    """

    boresight = look_at - camera_position
    boresight /= np.linalg.norm(boresight)

    if rng is None:
        rng = np.random.default_rng()

    for attempt in range(max_attempts):
        sun_dir = rng.normal(size=3)
        sun_dir /= np.linalg.norm(sun_dir)

        sun_position = sun_dir * sun_distance

        cam_to_sun = sun_position - camera_position
        cam_to_sun /= np.linalg.norm(cam_to_sun)

        cos_angle = np.clip(np.dot(boresight, cam_to_sun), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))

        if min_angle_deg <= angle_deg <= max_angle_deg:
            return sun_dir, sun_position, angle_deg

    raise RuntimeError(
        f"Could not sample valid sun direction after {max_attempts} attempts. "
        f"Need angle in [{min_angle_deg}, {max_angle_deg}] deg."
    )


# def make_sun_light(sun_direction):
#     """
#     Create a kb.DirectionalLight pointing in sun_direction.

#     Kubric's look_at() sets the rotation so the light points FROM
#     sun_position TOWARD the look_at point. We place the sun far away
#     along the OPPOSITE of sun_direction so it illuminates the scene
#     from sun_direction.

#     Args:
#         sun_direction (np.ndarray): unit vector [x, y, z], direction light comes FROM

#     Returns:
#         kb.DirectionalLight
#     """
#     sun_position = tuple(sun_direction * SUN_DISTANCE)

#     sun = kb.DirectionalLight(
#         color=kb.get_color("white"),
#         # Physically correct: Sun's angular radius as seen from space ~ 0.265 deg
#         # shadow_softness = tan(angular_radius_rad) ~ 0.00462
#         # Ref: SPEED-UE-Cube paper + solar angular diameter literature
#         shadow_softness=SUN_SHADOW_SOFTNESS,
#         intensity=15.0,          # tune to your spacecraft material
#     )
#     sun.position = sun_position
#     sun.look_at((0, 0, 0))     # always point toward scene origin (spacecraft)
#     return sun
def make_sun_light(sun_direction, intensity=SUN_INTENSITY, shadow_softness=SUN_SHADOW_SOFTNESS):
    """Create a directional sun light from the selected lighting configuration."""

    sun_position = tuple(sun_direction * SUN_DISTANCE)

    sun = kb.DirectionalLight(
        color=kb.get_color("white"),
        shadow_softness=shadow_softness,
        intensity=intensity,
    )
    sun.position = sun_position
    sun.look_at((0, 0, 0))

    return sun

def make_camera_fill_light(position=CAMERA_POSITION, target=LOOK_AT, intensity=FILL_INTENSITY, shadow_softness=0.3):
    """Create an optional weak fill light from near the camera direction."""

    fill = kb.DirectionalLight(
        color=kb.Color(0.75, 0.82, 1.0),   # slightly cool fill
        shadow_softness=shadow_softness,
        intensity=intensity,
    )

    # Put fill near camera side, pointing toward spacecraft
    fill.position = tuple(position)
    fill.look_at(tuple(target))

    return fill


# =============================================================================
# TRAJECTORY HELPERS
# =============================================================================
def generate_trajectory(num_frames,
                        initial_position=(0.0, 0.0, 0.0),
                        linear_velocity_mps=DEFAULT_LINEAR_VELOCITY_MPS,
                        angular_velocity_dps=DEFAULT_ANGULAR_VELOCITY_DPS,
                        fps=DEFAULT_FPS,
                        initial_quaternion=None,
                        random_state=None):
    """
    Generate a simple free-space trajectory (no gravity, no orbital mechanics).
    Matches the original code's tumbling + drifting behaviour but exposes
    all parameters explicitly and returns per-frame pose labels.

    Args:
        num_frames (int): total number of frames
        initial_position (tuple): starting [x, y, z] in metres
        linear_velocity_mps (tuple): constant [vx, vy, vz] in metres/second
        angular_velocity_dps (tuple): rotation rate around [x, y, z] in degrees/second
        fps (float): frame rate used to convert rates to per-frame increments
        initial_quaternion (np.ndarray or None): [w, x, y, z], random if None

    Returns:
        positions   (np.ndarray): shape (num_frames, 3)
        quaternions (np.ndarray): shape (num_frames, 4) as [w, x, y, z]
    """
    if initial_quaternion is None:
        # Uniformly random SO(3), made reproducible by random_state when provided.
        r0 = Rotation.random(random_state=random_state)
    else:
        r0 = Rotation.from_quat([initial_quaternion[1],   # scipy uses [x,y,z,w]
                                  initial_quaternion[2],
                                  initial_quaternion[3],
                                  initial_quaternion[0]])

    pos = np.array(initial_position, dtype=float)
    vel_per_frame = np.array(linear_velocity_mps, dtype=float) / float(fps)

    # Angular velocity converted to a rotation applied each frame.
    ang_vel_rad_per_frame = np.radians(angular_velocity_dps) / float(fps)
    delta_rot = Rotation.from_rotvec(ang_vel_rad_per_frame)

    positions   = np.zeros((num_frames, 3))
    quaternions = np.zeros((num_frames, 4))   # [w, x, y, z]

    current_rot = r0
    for f in range(num_frames):
        positions[f] = pos
        q_xyzw = current_rot.as_quat()              # scipy: [x, y, z, w]
        quaternions[f] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]  # → [w,x,y,z]

        pos         = pos + vel_per_frame
        current_rot = delta_rot * current_rot        # accumulate rotation

    return positions, quaternions


# =============================================================================
# MAIN
# =============================================================================
def main():

    # -------------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------------
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    scipy_random_state = np.random.RandomState(args.seed)
    np.random.seed(args.seed)

    object_name = args.object_name
    asset_dir = args.asset_dir or os.path.join(SCRIPT_DIR, "assets", f"{object_name}_from_fbx")
    output_dir = args.output_dir or os.path.join(
        SCRIPT_DIR, "output", f"{object_name}_{args.lighting}_seed{args.seed}"
    )
    lighting = resolve_lighting_config(args)
    camera_position = np.array(args.camera_position, dtype=float)
    look_at = np.array(args.look_at, dtype=float)
    initial_position = tuple(args.initial_position)

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be > 0")
    if args.render_chunk_size < 1:
        raise ValueError("--render-chunk-size must be >= 1")

    if args.duration_seconds is not None:
        args.num_frames = int(round(args.duration_seconds * args.fps))

    linear_velocity_mps = tuple(
        np.array(args.linear_velocity, dtype=float) * args.fps
        if args.linear_velocity is not None else
        np.array(args.linear_velocity_mps, dtype=float)
    )
    angular_velocity_dps = tuple(
        np.array(args.angular_velocity, dtype=float) * args.fps
        if args.angular_velocity is not None else
        np.array(args.angular_velocity_dps, dtype=float)
    )

    if args.num_frames < 1:
        raise ValueError("--num-frames must be >= 1")
    if lighting["sun_min_angle"] > lighting["sun_max_angle"]:
        raise ValueError("--sun-min-angle must be <= --sun-max-angle")

    num_frames = args.num_frames
    frame_start = 1
    frame_end = num_frames

    os.makedirs(f"{output_dir}/image/",    exist_ok=True)
    os.makedirs(f"{output_dir}/depth/",    exist_ok=True)
    os.makedirs(f"{output_dir}/flow/",     exist_ok=True)
    os.makedirs(f"{output_dir}/tmp/",      exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. SCENE
    # -------------------------------------------------------------------------
    scene = kb.Scene(
        resolution=(1920, 1200),    # Point Grey Grasshopper 3 native resolution
                                    # Ref: SPEED-UE-Cube paper, Section "Camera Effects"
        frame_start=frame_start,
        frame_end=frame_end,
        frame_rate=args.fps,
    )

    # -------------------------------------------------------------------------
    # 2. SIMULATOR (PyBullet)
    # -------------------------------------------------------------------------
    simulator = KubricSimulator(scene, scratch_dir=f"{output_dir}/tmp")
    scene.gravity = (0, 0, 0)      # zero gravity - free space

    # -------------------------------------------------------------------------
    # 3. RENDERER (Blender)
    # -------------------------------------------------------------------------
    renderer = KubricRenderer(scene, scratch_dir=f"{output_dir}/tmp")

    # No ambient light - in space there is no atmospheric scattering or bounce light.
    # Only the DirectionalLight (Sun) contributes.
    # Ref: SPEED-UE-Cube uses black background + single directional sun lamp.
    # renderer.ambient_illumination = kb.Color(0.0, 0.0, 0.0)
    ambient_level = lighting["ambient_level"]
    renderer.ambient_illumination = kb.Color(
        ambient_level,
        ambient_level,
        ambient_level,
    )

    # -------------------------------------------------------------------------
    # 4. CAMERA
    #    Point Grey Grasshopper 3 + Xenoplan 1.4/17mm
    #    focal_length = 17.5217 mm  (back-calculated from stated 35.6 deg H-FOV)
    #    sensor_width = 11.2512 mm  (5.86 um pixel pitch x 1920 pixels)
    #    Ref: SPEED-UE-Cube paper, Section "Camera Effects"
    # -------------------------------------------------------------------------
    scene.camera = kb.PerspectiveCamera(
        focal_length=CAMERA_FOCAL_LENGTH_MM,
        sensor_width=CAMERA_SENSOR_WIDTH_MM,
    )
    scene.camera.position = tuple(camera_position)
    scene.camera.look_at(tuple(look_at))

    # -------------------------------------------------------------------------
    # 5. LIGHTING — SPEED-UE-Cube conventions
    #    Single DirectionalLight (Sun), no ambient.
    #    Sun direction sampled once per sequence (fixed lighting for trajectory).
    #    Constraint: angle(sun_dir, camera_boresight) >= 75 deg
    #    Ref: SPEED-UE-Cube paper, Section "Training Dataset Pose Labels", Constraint 2
    # -------------------------------------------------------------------------
    if args.sun_direction is None:
        sun_direction, sun_position, angle_deg = sample_sun_direction(
            camera_position=camera_position,
            look_at=look_at,
            min_angle_deg=lighting["sun_min_angle"],
            max_angle_deg=lighting["sun_max_angle"],
            rng=rng,
        )
    else:
        sun_direction = normalize_vector(args.sun_direction, "--sun-direction")
        sun_position = sun_direction * SUN_DISTANCE
        boresight = normalize_vector(look_at - camera_position, "camera boresight")
        cam_to_sun = normalize_vector(sun_position - camera_position, "camera-to-sun vector")
        angle_deg = math.degrees(math.acos(np.clip(np.dot(boresight, cam_to_sun), -1.0, 1.0)))

    sun = make_sun_light(
        sun_direction,
        intensity=lighting["sun_intensity"],
        shadow_softness=lighting["sun_shadow_softness"],
    )
    scene += sun

    fill = None
    if lighting["fill_intensity"] > 0.0:
        fill = make_camera_fill_light(
            position=camera_position,
            target=look_at,
            intensity=lighting["fill_intensity"],
        )
        scene += fill

    print(f"[Lighting] Sun direction (world): {np.round(sun_direction, 4)}")
    angle_from_boresight = math.degrees(
        math.acos(np.clip(np.dot(sun_direction, [0, 1, 0]), -1, 1)))
    # print(f"[Lighting] Angle from camera boresight: {angle_from_boresight:.2f} deg "
    #       f"(must be >= {SUN_MIN_ANGLE_FROM_BORESIGHT_DEG} deg) ✓")
    print(f"[Lighting] Preset: {args.lighting} ({lighting['description']})")
    print(f"[Lighting] Angle from camera boresight: {angle_deg:.2f} deg "
          f"(target range: {lighting['sun_min_angle']} to "
          f"{lighting['sun_max_angle']} deg)")
    print(f"[Repro] seed={args.seed} output_dir={output_dir}")

    # -------------------------------------------------------------------------
    # 6. SPACECRAFT OBJECT
    # -------------------------------------------------------------------------
    debris = kb.FileBasedObject(
        asset_id="cheops_satellite",
        render_filename=f"{asset_dir}/{object_name}.obj",
        simulation_filename=f"{asset_dir}/{object_name}.urdf",
        # bounds omitted — defaults to ((0,0,0),(0,0,0))
        # Safe to omit: camera is fixed, collision uses URDF, no auto-framing needed
        position=initial_position,
        mass=10.0,
        scale=MM_TO_M,
    )
    scene += debris
    material_texture_map = apply_mtl_textures_to_imported_object(
        asset_id="cheops_satellite",
        mtl_path=f"{asset_dir}/{object_name}.mtl",
    )
    pivot_info = recenter_satellite_geometry("cheops_satellite")

    # -------------------------------------------------------------------------
    # 7. TRAJECTORY — pre-compute per-frame poses
    #    Simple free-space tumbling + drifting (no orbital mechanics needed)
    #    Orientation: uniform SO(3) sampling via scipy subgroup algorithm
    #    Ref: SPEED-UE-Cube paper, Section "Training Dataset Pose Labels"
    # -------------------------------------------------------------------------
    positions, quaternions = generate_trajectory(
        num_frames=num_frames,
        initial_position=initial_position,
        linear_velocity_mps=linear_velocity_mps,
        angular_velocity_dps=angular_velocity_dps,
        fps=args.fps,
        initial_quaternion=args.initial_quaternion,
        random_state=scipy_random_state,
    )

    # Explicit keyframes make the rendered sequence exactly match pose_labels.json.
    # The camera remains static; only the spacecraft pose is animated.
    for frame_idx in range(num_frames):
        frame_id = frame_start + frame_idx
        debris.position = tuple(positions[frame_idx])
        debris.quaternion = quaternion_wxyz_to_kb(quaternions[frame_idx])
        debris.keyframe_insert("position", frame_id)
        debris.keyframe_insert("quaternion", frame_id)

    print(
        f"[Motion] fps={args.fps}, frames={num_frames}, "
        f"duration={num_frames / args.fps:.3f}s, "
        f"linear_velocity_mps={np.round(linear_velocity_mps, 6)}, "
        f"angular_velocity_dps={np.round(angular_velocity_dps, 6)}"
    )
    print(
        f"[Motion] position range x/y/z: "
        f"min={np.round(positions.min(axis=0), 4)} max={np.round(positions.max(axis=0), 4)}"
    )

    # -------------------------------------------------------------------------
    # 8. PHYSICS
    #    Motion is keyframed for reproducible long sequences, so no PyBullet
    #    integration step is needed here.
    # -------------------------------------------------------------------------
    print("\n[Physics] Skipped: using explicit deterministic pose keyframes.")
    # -------------------------------------------------------------------------
    # 9. RUN RENDERING (Blender)
    # -------------------------------------------------------------------------
    print("[Render] Running Blender renderer in chunks...")
    renderer.save_state(f"{output_dir}/blender_scene.blend")   # save for debugging

    depth_sum = 0.0
    depth_min = np.inf
    depth_max = -np.inf
    valid_count = 0
    total_count = 0
    invalid_raw_count = 0
    invalid_clamped_count = 0

    for chunk_start, chunk_end, chunk_frames in iter_frame_chunks(
        frame_start, frame_end, args.render_chunk_size
    ):
        chunk_offset = chunk_start - frame_start
        print(
            f"[Render] Chunk frames {chunk_start}-{chunk_end} "
            f"({len(chunk_frames)} frames)"
        )
        clear_render_scratch(f"{output_dir}/tmp")
        frames_dict = renderer.render(frames=chunk_frames)

        print(f"[Depth] Clamping chunk {chunk_start}-{chunk_end} to MAX_DEPTH={MAX_DEPTH:.1f} m ...")
        depth_raw = frames_dict["depth"]
        depth_clamped = clamp_depth_batch(depth_raw, max_depth=MAX_DEPTH)
        depth_f64 = depth_clamped.astype(np.float64)

        raw_invalid = ~np.isfinite(depth_raw.astype(np.float32)) | (depth_raw <= 0.0)
        clamped_invalid = depth_clamped == 0.0
        invalid_raw_count += int(np.sum(raw_invalid))
        invalid_clamped_count += int(np.sum(clamped_invalid))

        valid_mask = depth_clamped > 0.0
        if np.any(valid_mask):
            valid_values = depth_clamped[valid_mask]
            depth_sum += float(np.sum(valid_values))
            depth_min = min(depth_min, float(np.min(valid_values)))
            depth_max = max(depth_max, float(np.max(valid_values)))
            valid_count += int(valid_values.size)
        total_count += int(depth_clamped.size)

        print(f"[Export] Writing chunk {chunk_start}-{chunk_end} RGB, depth, flow ...")
        write_rgb_batch(frames_dict["rgba"], f"{output_dir}/image/", start_index=chunk_offset)
        write_tiff_depth_batch(depth_f64, f"{output_dir}/depth/", start_index=chunk_offset)
        write_flo_batch(frames_dict["forward_flow"], f"{output_dir}/flow/", start_index=chunk_offset)

        del frames_dict, depth_raw, depth_clamped, depth_f64

    # -------------------------------------------------------------------------
    # 12. SAVE POSE LABELS
    #     Explicit CV camera frame: x right, y down, z forward.
    # -------------------------------------------------------------------------
    world_to_camera = camera_frame_basis(camera_position, look_at)
    camera_to_world_rotation = Rotation.from_matrix(world_to_camera.T)
    camera_quaternion_wxyz = rotation_to_wxyz(camera_to_world_rotation)
    camera_boresight = world_to_camera[2]
    if not np.allclose(world_to_camera @ world_to_camera.T, np.eye(3), atol=1e-6):
        raise RuntimeError("Camera pose basis is not orthonormal.")
    if not np.isclose(np.linalg.det(world_to_camera), 1.0, atol=1e-6):
        raise RuntimeError("Camera pose basis has invalid handedness.")
    relative_positions_all = (world_to_camera @ (positions - camera_position).T).T
    if np.any(relative_positions_all[:, 2] <= 0.0):
        raise RuntimeError("At least one object pose is behind the camera.")
    quaternion_norms = np.linalg.norm(quaternions, axis=1)
    if not np.allclose(quaternion_norms, 1.0, atol=1e-5):
        raise RuntimeError("Object pose quaternions are not normalized.")
    print(
        "[Pose Check] camera_world="
        f"{np.round(camera_position, 4)}, "
        f"object_camera_xyz first/last="
        f"{np.round(relative_positions_all[0], 4)} / "
        f"{np.round(relative_positions_all[-1], 4)}, "
        f"camera_z_range="
        f"{relative_positions_all[:, 2].min():.4f}.."
        f"{relative_positions_all[:, 2].max():.4f} m"
    )
    pose_labels = []
    for frame_idx in range(num_frames):
        object_rotation_world = Rotation.from_quat([
            quaternions[frame_idx][1], quaternions[frame_idx][2],
            quaternions[frame_idx][3], quaternions[frame_idx][0]
        ])
        object_rotation_camera = Rotation.from_matrix(world_to_camera) * object_rotation_world
        relative_position = world_to_camera @ (positions[frame_idx] - camera_position)
        pose_labels.append({
            "filename": f"{frame_idx:06d}.png",
            "frame": frame_start + frame_idx,
            "camera_position_world_m": camera_position.tolist(),
            "camera_quaternion_world_wxyz": camera_quaternion_wxyz,
            "q_obj2cam": rotation_to_wxyz(object_rotation_camera),
            "r_obj2cam": relative_position.tolist(),
            "object_position_world_m": positions[frame_idx].tolist(),
            "object_quaternion_world_wxyz": quaternions[frame_idx].tolist(),
            "sun_direction_world": sun_direction.tolist(),
            "sun_boresight_angle_deg": round(angle_deg, 4),
        })

    with open(f"{output_dir}/pose_labels.json", "w") as f:
        json.dump(pose_labels, f, indent=2)

    # -------------------------------------------------------------------------
    # 13. SCENE METADATA
    # -------------------------------------------------------------------------
    kb.file_io.write_json(filename=f"{output_dir}/metadata.json", data={
        "metadata":  kb.get_scene_metadata(scene),
        "generation_config": {
            "seed": args.seed,
            "lighting_preset": args.lighting,
            "asset_dir": asset_dir,
            "output_dir": output_dir,
            "num_frames": num_frames,
            "fps": args.fps,
            "duration_seconds": num_frames / args.fps,
            "render_chunk_size": args.render_chunk_size,
            "camera_position": camera_position.tolist(),
            "look_at": look_at.tolist(),
            "initial_position": list(initial_position),
            "linear_velocity_mps": list(linear_velocity_mps),
            "angular_velocity_dps": list(angular_velocity_dps),
            "legacy_linear_velocity_per_frame": args.linear_velocity,
            "legacy_angular_velocity_per_frame": args.angular_velocity,
            "initial_quaternion_wxyz": quaternions[0].tolist(),
            "pivot_correction": pivot_info,
            "explicit_initial_quaternion": args.initial_quaternion is not None,
            "explicit_sun_direction": args.sun_direction is not None,
        },
        "camera": {
            **kb.get_camera_info(scene.camera),
            # Explicit camera intrinsics for SPEED-UE-Cube reproducibility
            "focal_length_mm":    CAMERA_FOCAL_LENGTH_MM,
            "sensor_width_mm":    CAMERA_SENSOR_WIDTH_MM,
            "sensor_height_mm":   CAMERA_SENSOR_HEIGHT_MM,
            "fov_horizontal_deg": CAMERA_FOV_H_DEG,
            "pose_frame": {
                "convention": "CV: x right, y down, z forward",
                "world_to_camera_rotation": world_to_camera.tolist(),
                "camera_boresight_world": camera_boresight.tolist(),
            },
        },

        "material_textures": material_texture_map,
        "lighting": {
            "type": "DirectionalLight Sun + weak camera fill + low ambient",
            "sun_direction_world": sun_direction.tolist(),
            "sun_boresight_angle_deg": round(angle_deg, 4),
            "shadow_softness": lighting["sun_shadow_softness"],
            "sun_intensity": lighting["sun_intensity"],
            "fill_intensity": lighting["fill_intensity"],
            "ambient_illumination": lighting["ambient_level"],
            "preset_description": lighting["description"],
            "note": (
                "Lighting values are resolved from the selected preset plus any "
                "explicit CLI overrides. Use lighting='space' for strong dark edges "
                "or lighting='cv_bright' for high object visibility."
            ),
        },
        "instances": kb.get_instance_info(scene),
    })

    # -------------------------------------------------------------------------
    # 14. DEPTH STATS
    # -------------------------------------------------------------------------
    print(f"\n[Depth Stats] (valid pixels only, depth > 0)")
    if valid_count > 0:
        print(f"  mean:  {depth_sum / valid_count:.4f} m")
        print(f"  min:   {depth_min:.4f} m")
        print(f"  max:   {depth_max:.4f} m")
    else:
        print("  mean/min/max: no valid depth pixels")
    print(f"  invalid raw pixels:     {invalid_raw_count}")
    print(f"  invalid clamped pixels: {invalid_clamped_count}")
    print(f"  valid: {valid_count} / {total_count} pixels "
          f"({100 * valid_count / total_count:.1f}%)")

    print("\nDataset generation complete! Check:", output_dir)

if __name__ == "__main__":
    main()
