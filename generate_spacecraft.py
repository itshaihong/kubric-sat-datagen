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
import sys
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
MAX_DEPTH = 9999.0   # metres â€” background/sky pixels replaced with 0.0
MM_TO_M   = 0.001
CM_TO_M   = 0.01

# Camera â€” Point Grey Grasshopper 3 + Xenoplan 1.4/17mm
# Sensor width  = 5.86 um pixel pitch x 1920 pixels = 11.2512 mm
# Focal length  = 17.5217 mm (back-calculated from paper's stated 35.6 deg horizontal FOV)
# Ref: SPEED-UE-Cube paper, Section "Camera Effects"
CAMERA_FOCAL_LENGTH_MM = 17.5217   # effective focal length (mm)
CAMERA_SENSOR_WIDTH_MM = 11.2512   # sensor width (mm)
CAMERA_SENSOR_HEIGHT_MM = 7.0320   # sensor height = 5.86 um x 1200 pixels (mm)
CAMERA_FOV_H_DEG       = 35.6      # horizontal FOV as stated in paper (deg)

# Sun â€” physically correct angular diameter of the Sun as seen from space
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
    parser.add_argument("--trajectory-mode", choices=("static", "curved_flyby", "tumbling_approach", "tumbling_fly_across"), default="static", help="Camera/object trajectory mode. 'static' preserves the original fixed camera behavior.")
    parser.add_argument("--flyby-start-range", type=float, default=11.0, help="curved_flyby start range from --look-at in metres.")
    parser.add_argument("--flyby-end-range", type=float, default=5.5, help="curved_flyby close/end range from --look-at in metres.")
    parser.add_argument("--flyby-arc-deg", type=float, default=110.0, help="curved_flyby accumulated azimuth arc in degrees.")
    parser.add_argument("--flyby-elevation-deg", type=float, default=18.0, help="curved_flyby accumulated elevation change in degrees.")
    parser.add_argument("--flyby-start-azimuth-deg", type=float, default=0.0, help="curved_flyby starting azimuth in degrees around +Z; 0 starts on negative Y.")
    parser.add_argument("--flyby-start-elevation-deg", type=float, default=0.0, help="curved_flyby starting elevation in degrees.")
    parser.add_argument("--framing-fov-fraction", type=float, default=0.55, help="Fraction of the limiting half-FOV occupied by the object's bounding sphere for auto-scaled trajectories.")
    parser.add_argument("--approach-near-multiplier", type=float, default=1.25, help="tumbling_approach near range multiplier applied to the safe framing distance.")
    parser.add_argument("--approach-far-multiplier", type=float, default=2.35, help="tumbling_approach far range multiplier applied to the safe framing distance.")
    parser.add_argument("--fly-across-distance-multiplier", type=float, default=1.65, help="tumbling_fly_across observation distance multiplier applied to the safe framing distance.")
    parser.add_argument("--fly-across-lateral-fraction", type=float, default=0.55, help="Fraction of the safe frustum center offset used on each side for tumbling_fly_across.")
    parser.add_argument("--max-viewpoint-deg-per-frame", type=float, default=1.5, help="Soft auto-scaling limit for adjacent relative-view direction change in new trajectory modes.")
    parser.add_argument("--auto-tumble-deg-per-frame", type=float, default=0.6, help="Default tumble speed per frame for new trajectory modes when no angular velocity is explicitly supplied.")
    parser.add_argument("--initial-position", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"), help="Initial satellite position in metres.")
    parser.add_argument("--initial-quaternion", nargs=4, type=float, default=None, metavar=("W", "X", "Y", "Z"), help="Use an exact initial satellite quaternion instead of sampling.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Simulation/render frame rate in frames per second.")
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS, help="Duration to render. If set, overrides --num-frames via round(duration * fps).")
    parser.add_argument("--num-frames", type=int, default=24, help="Number of frames to render when --duration-seconds is not set.")
    parser.add_argument("--object-name", default="Cheops", help="Base object filename without extension.")
    parser.add_argument("--asset-dir", default=None, help="Asset directory containing OBJ, MTL, textures, and URDF.")
    parser.add_argument("--output-dir", default=None, help="Output directory for rendered dataset.")
    parser.add_argument("--render-chunk-size", type=int, default=120, help="Number of frames rendered/postprocessed at once. Keeps long sequences from loading all frames into RAM.")
    parser.add_argument("--max-render-chunks", type=int, default=None, help="Stop after this many rendered chunks. Useful for pilot previews.")
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


def compute_satellite_bounds(asset_id, unit_scale=1.0):
    """Return world-space bounds for the imported, scaled spacecraft meshes."""
    import bpy
    from mathutils import Vector

    meshes = find_satellite_meshes(asset_id)
    if not meshes:
        raise RuntimeError(f"No mesh geometry found for '{asset_id}' bounds.")
    points = np.asarray([
        obj.matrix_world @ Vector(corner)
        for obj in meshes for corner in obj.bound_box
    ], dtype=float) * float(unit_scale)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    dimensions = bbox_max - bbox_min
    radius = float(np.max(np.linalg.norm(points - center[None, :], axis=1)))
    return {
        "bbox_min_m": bbox_min,
        "bbox_max_m": bbox_max,
        "bbox_center_m": center,
        "bbox_dimensions_m": dimensions,
        "max_span_m": float(np.max(dimensions)),
        "bounding_radius_m": radius,
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


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def camera_fov_degrees():
    fov_h = 2.0 * math.degrees(math.atan((CAMERA_SENSOR_WIDTH_MM * 0.5) / CAMERA_FOCAL_LENGTH_MM))
    fov_v = 2.0 * math.degrees(math.atan((CAMERA_SENSOR_HEIGHT_MM * 0.5) / CAMERA_FOCAL_LENGTH_MM))
    return fov_h, fov_v


def safe_framing_distance(radius_m, fov_fraction):
    if radius_m <= 0.0:
        raise ValueError("Object bounding radius must be positive for auto-scaled trajectories.")
    if not 0.05 <= fov_fraction <= 0.95:
        raise ValueError("--framing-fov-fraction must be in [0.05, 0.95].")
    _, fov_v = camera_fov_degrees()
    limiting_half_angle = math.radians(min(CAMERA_FOV_H_DEG, fov_v) * 0.5 * fov_fraction)
    return radius_m / math.sin(limiting_half_angle)


def trajectory_motion_stats(camera_positions, camera_look_ats, object_positions, angular_velocity_dps, fps):
    relative = object_positions - camera_positions
    ranges = np.linalg.norm(relative, axis=1)
    adjacent_steps = np.linalg.norm(np.diff(relative, axis=0), axis=1)
    adjacent_velocity = adjacent_steps * float(fps)
    adjacent_view_angles = adjacent_angles_deg(relative)
    angular_speed = float(np.linalg.norm(np.asarray(angular_velocity_dps, dtype=float)))
    deg_per_frame = angular_speed / float(fps)
    total_tumble = deg_per_frame * max(len(object_positions) - 1, 0)
    pointing_errors = []
    for position, target in zip(camera_positions, camera_look_ats):
        basis = camera_frame_basis(position, target)
        target_dir = normalize_vector(target - position, "camera target direction")
        pointing_errors.append(math.degrees(math.acos(np.clip(np.dot(basis[2], target_dir), -1.0, 1.0))))
    return {
        "range_m": ranges,
        "adjacent_step_m": adjacent_steps,
        "adjacent_velocity_mps": adjacent_velocity,
        "adjacent_viewpoint_deg": adjacent_view_angles,
        "total_viewpoint_deg": float(np.sum(adjacent_view_angles)) if len(adjacent_view_angles) else 0.0,
        "tumble_deg_per_frame": deg_per_frame,
        "tumble_deg_per_second": angular_speed,
        "total_tumble_deg": total_tumble,
        "object_tumble_deg": total_tumble,
        "pointing_error_deg": np.asarray(pointing_errors),
    }


def print_min_median_max(label, values, suffix=""):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        print(f"  {label}: n/a")
        return
    print(f"  {label} min/median/max: {values.min():.4f} / {np.median(values):.4f} / {values.max():.4f}{suffix}")


def summarize_trajectory(mode, camera_positions, camera_look_ats, object_positions, angular_velocity_dps, fps, bounds, extra=None):
    fov_h, fov_v = camera_fov_degrees()
    stats = trajectory_motion_stats(camera_positions, camera_look_ats, object_positions, angular_velocity_dps, fps)
    print(f"\n[Trajectory] {mode} summary")
    print(f"  object bbox dimensions: {np.round(bounds['bbox_dimensions_m'], 6)} m")
    print(f"  object max span: {bounds['max_span_m']:.6f} m")
    print(f"  object bounding radius: {bounds['bounding_radius_m']:.6f} m")
    print(f"  camera FOV h/v: {fov_h:.4f} / {fov_v:.4f} deg")
    print(f"  frames: {len(camera_positions)}, fps: {fps}")
    if extra:
        for key, value in extra.items():
            print(f"  {key}: {value}")
    ranges = stats["range_m"]
    print(f"  range start/end: {ranges[0]:.4f} / {ranges[-1]:.4f} m")
    print_min_median_max("range", ranges, " m")
    print_min_median_max("adjacent translation", stats["adjacent_step_m"], " m/frame")
    print_min_median_max("linear velocity", stats["adjacent_velocity_mps"], " m/s")
    print_min_median_max("adjacent viewpoint change", stats["adjacent_viewpoint_deg"], " deg/frame")
    print(f"  tumble: {stats['tumble_deg_per_frame']:.4f} deg/frame, {stats['tumble_deg_per_second']:.4f} deg/s, total {stats['total_tumble_deg']:.4f} deg")
    print(f"  look-at pointing error max: {stats['pointing_error_deg'].max():.8f} deg")
    return stats


def generate_camera_trajectory(
    mode,
    num_frames,
    camera_position,
    look_at,
    start_range=11.0,
    end_range=5.5,
    arc_deg=110.0,
    elevation_deg=18.0,
    start_azimuth_deg=0.0,
    start_elevation_deg=0.0,
):
    """Return per-frame camera positions and look-at targets."""
    look_at = np.asarray(look_at, dtype=float)

    if mode == "static":
        positions = np.repeat(np.asarray(camera_position, dtype=float)[None, :], num_frames, axis=0)
        targets = np.repeat(look_at[None, :], num_frames, axis=0)
        return positions, targets

    if mode != "curved_flyby":
        raise ValueError(f"Unsupported trajectory mode: {mode}")
    if start_range <= 0.0 or end_range <= 0.0:
        raise ValueError("curved_flyby ranges must be positive.")

    if num_frames == 1:
        s = np.array([0.0], dtype=float)
    else:
        s = np.linspace(0.0, 1.0, num_frames, dtype=float)

    eased = smoothstep(s)
    radius = start_range + (end_range - start_range) * eased
    azimuth = np.radians(start_azimuth_deg + arc_deg * eased)
    elevation = np.radians(start_elevation_deg + elevation_deg * eased)

    cos_el = np.cos(elevation)
    directions = np.column_stack([
        np.sin(azimuth) * cos_el,
        -np.cos(azimuth) * cos_el,
        np.sin(elevation),
    ])
    positions = look_at[None, :] + radius[:, None] * directions
    targets = np.repeat(look_at[None, :], num_frames, axis=0)
    return positions, targets


def generate_tumbling_approach_trajectory(num_frames, look_at, safe_distance, near_multiplier, far_multiplier):
    if near_multiplier <= 1.0:
        raise ValueError("--approach-near-multiplier must be > 1.0")
    if far_multiplier <= near_multiplier:
        raise ValueError("--approach-far-multiplier must be greater than --approach-near-multiplier")
    if num_frames == 1:
        s = np.array([0.0], dtype=float)
    else:
        s = np.linspace(0.0, 1.0, num_frames, dtype=float)
    eased = smoothstep(s)
    far_distance = far_multiplier * safe_distance
    near_distance = near_multiplier * safe_distance
    ranges = far_distance + (near_distance - far_distance) * eased
    look_at = np.asarray(look_at, dtype=float)
    camera_positions = look_at[None, :] + ranges[:, None] * np.array([[0.0, -1.0, 0.0]])
    camera_look_ats = np.repeat(look_at[None, :], num_frames, axis=0)
    object_positions = np.repeat(look_at[None, :], num_frames, axis=0)
    return camera_positions, camera_look_ats, object_positions, {
        "safe_framing_distance_m": f"{safe_distance:.4f}",
        "auto near/far distance m": f"{near_distance:.4f} / {far_distance:.4f}",
    }


def generate_tumbling_fly_across_trajectory(
    num_frames,
    look_at,
    radius_m,
    safe_distance,
    distance_multiplier,
    lateral_fraction,
    max_viewpoint_deg_per_frame,
):
    if distance_multiplier <= 1.0:
        raise ValueError("--fly-across-distance-multiplier must be > 1.0")
    if not 0.05 <= lateral_fraction <= 0.95:
        raise ValueError("--fly-across-lateral-fraction must be in [0.05, 0.95].")
    if num_frames == 1:
        s = np.array([0.0], dtype=float)
    else:
        s = np.linspace(0.0, 1.0, num_frames, dtype=float)
    eased = smoothstep(s)
    distance = distance_multiplier * safe_distance
    half_fov_h = math.radians(CAMERA_FOV_H_DEG * 0.5)
    half_width_at_depth = distance * math.tan(half_fov_h)
    center_limit = max(0.0, half_width_at_depth - 2.0 * radius_m)
    lateral_limit = lateral_fraction * center_limit
    look_at = np.asarray(look_at, dtype=float)
    camera_position = look_at + np.array([0.0, -distance, 0.0])
    camera_positions = np.repeat(camera_position[None, :], num_frames, axis=0)
    camera_look_ats = np.repeat(look_at[None, :], num_frames, axis=0)
    lateral = -lateral_limit + 2.0 * lateral_limit * eased
    object_positions = look_at[None, :] + np.column_stack([
        lateral,
        np.zeros(num_frames, dtype=float),
        np.zeros(num_frames, dtype=float),
    ])
    relative = object_positions - camera_positions
    adjacent_view = adjacent_angles_deg(relative)
    if len(adjacent_view) and adjacent_view.max() > max_viewpoint_deg_per_frame:
        scale = max_viewpoint_deg_per_frame / adjacent_view.max()
        lateral_limit *= 0.95 * scale
        lateral = -lateral_limit + 2.0 * lateral_limit * eased
        object_positions = look_at[None, :] + np.column_stack([
            lateral,
            np.zeros(num_frames, dtype=float),
            np.zeros(num_frames, dtype=float),
        ])
    return camera_positions, camera_look_ats, object_positions, {
        "safe_framing_distance_m": f"{safe_distance:.4f}",
        "auto observation distance m": f"{distance:.4f}",
        "auto left/right lateral limits m": f"{-lateral_limit:.4f} / {lateral_limit:.4f}",
    }


def keyframe_camera_trajectory(camera, camera_positions, camera_look_ats, frame_start):
    """Keyframe a Kubric camera using Blender's local -Z optical axis convention."""
    for frame_idx, (position, target) in enumerate(zip(camera_positions, camera_look_ats)):
        frame_id = frame_start + frame_idx
        camera.position = tuple(position)
        camera.look_at(tuple(target))
        camera.keyframe_insert("position", frame_id)
        camera.keyframe_insert("quaternion", frame_id)


def camera_pose_arrays(camera_positions, camera_look_ats):
    world_to_camera = []
    camera_quaternions = []
    boresights = []
    for position, target in zip(camera_positions, camera_look_ats):
        basis = camera_frame_basis(position, target)
        world_to_camera.append(basis)
        boresights.append(basis[2])
        camera_quaternions.append(rotation_to_wxyz(Rotation.from_matrix(basis.T)))
    return np.asarray(world_to_camera), np.asarray(camera_quaternions), np.asarray(boresights)


def adjacent_angles_deg(vectors):
    vectors = np.asarray(vectors, dtype=float)
    if len(vectors) < 2:
        return np.array([], dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    dots = np.sum(vectors[:-1] * vectors[1:], axis=1)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def summarize_curved_flyby(camera_positions, camera_look_ats, quaternions, angular_velocity_dps, fps):
    ranges = np.linalg.norm(camera_positions - camera_look_ats, axis=1)
    viewpoint_dirs = camera_positions - camera_look_ats
    adjacent_view_angles = adjacent_angles_deg(viewpoint_dirs)
    adjacent_steps = np.linalg.norm(np.diff(camera_positions, axis=0), axis=1)
    total_view_angle = float(np.sum(adjacent_view_angles)) if len(adjacent_view_angles) else 0.0
    angular_speed = np.linalg.norm(np.asarray(angular_velocity_dps, dtype=float))
    object_tumble = angular_speed * max(len(camera_positions) - 1, 0) / float(fps)
    boresight_to_target = []
    for position, target in zip(camera_positions, camera_look_ats):
        basis = camera_frame_basis(position, target)
        target_dir = normalize_vector(target - position, "camera target direction")
        boresight_to_target.append(
            math.degrees(math.acos(np.clip(np.dot(basis[2], target_dir), -1.0, 1.0)))
        )
    boresight_to_target = np.asarray(boresight_to_target)

    print("\n[Trajectory] curved_flyby summary")
    print(f"  frames: {len(camera_positions)}")
    print(f"  range start/end: {ranges[0]:.4f} / {ranges[-1]:.4f} m")
    print(f"  range min/median/max: {ranges.min():.4f} / {np.median(ranges):.4f} / {ranges.max():.4f} m")
    print(f"  total accumulated viewpoint change: {total_view_angle:.4f} deg")
    if len(adjacent_view_angles):
        print(
            "  adjacent viewpoint change min/median/max: "
            f"{adjacent_view_angles.min():.4f} / {np.median(adjacent_view_angles):.4f} / "
            f"{adjacent_view_angles.max():.4f} deg"
        )
        print(
            "  adjacent camera step min/median/max: "
            f"{adjacent_steps.min():.4f} / {np.median(adjacent_steps):.4f} / "
            f"{adjacent_steps.max():.4f} m"
        )
    print(f"  camera center start/end: {np.round(camera_positions[0], 4)} / {np.round(camera_positions[-1], 4)}")
    print(f"  object accumulated tumble estimate: {object_tumble:.4f} deg")
    print(f"  look-at pointing error max: {boresight_to_target.max():.8f} deg")
    return {
        "range_m": ranges,
        "adjacent_viewpoint_deg": adjacent_view_angles,
        "adjacent_step_m": adjacent_steps,
        "total_viewpoint_deg": total_view_angle,
        "object_tumble_deg": object_tumble,
        "pointing_error_deg": boresight_to_target,
    }


def write_trajectory_sidecars(
    output_dir,
    frame_start,
    fps,
    camera_positions,
    camera_quaternions_wxyz,
    object_positions,
    object_quaternions_wxyz,
    camera_look_ats,
):
    os.makedirs(output_dir, exist_ok=True)
    camera_trajectory = []
    for frame_idx, (cam_pos, cam_quat, target) in enumerate(
        zip(camera_positions, camera_quaternions_wxyz, camera_look_ats)
    ):
        camera_trajectory.append({
            "frame": frame_start + frame_idx,
            "time_s": frame_idx / float(fps),
            "camera_position_world_m": cam_pos.tolist(),
            "camera_quaternion_world_wxyz": cam_quat.tolist(),
            "look_at_world_m": target.tolist(),
            "optical_axis": "Blender/Kubric local -Z; exported pose frame is CV z-forward",
        })

    with open(os.path.join(output_dir, "camera_trajectory.json"), "w") as f:
        json.dump(camera_trajectory, f, indent=2)

    with open(os.path.join(output_dir, "times.txt"), "w") as f:
        for frame_idx in range(len(camera_positions)):
            f.write(f"{frame_idx / float(fps):.9f}\n")

    with open(os.path.join(output_dir, "pose_ground_truth.txt"), "w") as f:
        for frame_idx, (position, quaternion) in enumerate(zip(camera_positions, camera_quaternions_wxyz)):
            values = [frame_start + frame_idx, *position.tolist(), *quaternion.tolist()]
            f.write(" ".join(str(v) for v in values) + "\n")

    with open(os.path.join(output_dir, "object_pose_ground_truth.txt"), "w") as f:
        for frame_idx, (position, quaternion) in enumerate(zip(object_positions, object_quaternions_wxyz)):
            values = [frame_start + frame_idx, *position.tolist(), *quaternion.tolist()]
            f.write(" ".join(str(v) for v in values) + "\n")


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

def write_png_depth_batch(depth_m, output_dir, start_index=0):
    """Write depth as uint16 PNG in millimetres; 0 marks invalid/out-of-range."""
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(depth_m):
        frame_idx = start_index + i
        depth = np.asarray(frame).squeeze().astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0.0) & (depth < 65.535)
        depth_mm = np.zeros(depth.shape, dtype=np.uint16)
        depth_mm[valid] = np.round(depth[valid] * 1000.0).astype(np.uint16)
        Image.fromarray(depth_mm, mode="I;16").save(
            os.path.join(output_dir, f"{frame_idx:06d}.png"))

def write_segmentation_batch(seg_frames, output_dir, debug_dir=None, start_index=0):
    """Write Kubric renderer segmentation IDs and return foreground pixel counts."""
    os.makedirs(output_dir, exist_ok=True)
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
    mask_areas = []
    for i, frame in enumerate(seg_frames):
        frame_idx = start_index + i
        seg = np.asarray(frame).squeeze()
        mask = seg > 0
        mask_areas.append(int(np.count_nonzero(mask)))
        if not np.issubdtype(seg.dtype, np.integer):
            raise ValueError(f"Segmentation frame must contain integer IDs, got {seg.dtype}.")
        if int(np.max(seg)) <= 255:
            seg_image = Image.fromarray(seg.astype(np.uint8), mode="L")
        else:
            seg_image = Image.fromarray(seg.astype(np.uint16), mode="I;16")
        seg_image.save(os.path.join(output_dir, f"{frame_idx:06d}.png"))
        if debug_dir is not None:
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                os.path.join(debug_dir, f"{frame_idx:06d}.png"))
    return mask_areas


def segmentation_batch_stats(seg_frames):
    """Return per-frame mask area, border margin, and adjacent IoU for a rendered chunk."""
    areas = []
    margins = []
    ious = []
    previous = None
    for frame in seg_frames:
        mask = np.asarray(frame).squeeze() > 0
        areas.append(int(np.count_nonzero(mask)))
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            margins.append(0)
        else:
            height, width = mask.shape
            margins.append(int(min(xs.min(), ys.min(), width - 1 - xs.max(), height - 1 - ys.max())))
        if previous is not None:
            union = np.logical_or(previous, mask)
            if np.any(union):
                ious.append(float(np.count_nonzero(np.logical_and(previous, mask)) / np.count_nonzero(union)))
            else:
                ious.append(0.0)
        previous = mask
    return areas, margins, ious

def clamp_depth_batch(depth_frames, max_depth=MAX_DEPTH):
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
#         #    Not just sun_dir â€” because camera is not at the origin
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
        quaternions[f] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]  # â†’ [w,x,y,z]

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
    if args.max_render_chunks is not None and args.max_render_chunks < 1:
        raise ValueError("--max-render-chunks must be >= 1 when set")

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
    os.makedirs(f"{output_dir}/seg/",      exist_ok=True)
    os.makedirs(f"{output_dir}/seg_debug/", exist_ok=True)
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
    # 5. SPACECRAFT OBJECT
    # -------------------------------------------------------------------------
    debris = kb.FileBasedObject(
        asset_id="cheops_satellite",
        render_filename=f"{asset_dir}/{object_name}.obj",
        simulation_filename=f"{asset_dir}/{object_name}.urdf",
        # bounds omitted: auto-framing reads the imported Blender mesh bounds.
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
    object_bounds = compute_satellite_bounds("cheops_satellite", unit_scale=MM_TO_M)
    print(f"[Object] bbox dimensions m: {np.round(object_bounds['bbox_dimensions_m'], 6)}")
    print(f"[Object] max span / bounding radius m: {object_bounds['max_span_m']:.6f} / {object_bounds['bounding_radius_m']:.6f}")

    # -------------------------------------------------------------------------
    # 6. TRAJECTORY - pre-compute per-frame camera/object poses
    # -------------------------------------------------------------------------
    auto_modes = {"tumbling_approach", "tumbling_fly_across"}
    camera_positions, camera_look_ats = generate_camera_trajectory(
        mode="static" if args.trajectory_mode in auto_modes else args.trajectory_mode,
        num_frames=num_frames,
        camera_position=camera_position,
        look_at=look_at,
        start_range=args.flyby_start_range,
        end_range=args.flyby_end_range,
        arc_deg=args.flyby_arc_deg,
        elevation_deg=args.flyby_elevation_deg,
        start_azimuth_deg=args.flyby_start_azimuth_deg,
        start_elevation_deg=args.flyby_start_elevation_deg,
    )
    positions = None
    trajectory_extra = None
    safe_distance = safe_framing_distance(
        object_bounds["bounding_radius_m"],
        args.framing_fov_fraction,
    )
    if args.trajectory_mode == "tumbling_approach":
        camera_positions, camera_look_ats, positions, trajectory_extra = generate_tumbling_approach_trajectory(
            num_frames=num_frames,
            look_at=look_at,
            safe_distance=safe_distance,
            near_multiplier=args.approach_near_multiplier,
            far_multiplier=args.approach_far_multiplier,
        )
    elif args.trajectory_mode == "tumbling_fly_across":
        camera_positions, camera_look_ats, positions, trajectory_extra = generate_tumbling_fly_across_trajectory(
            num_frames=num_frames,
            look_at=look_at,
            radius_m=object_bounds["bounding_radius_m"],
            safe_distance=safe_distance,
            distance_multiplier=args.fly_across_distance_multiplier,
            lateral_fraction=args.fly_across_lateral_fraction,
            max_viewpoint_deg_per_frame=args.max_viewpoint_deg_per_frame,
        )

    camera_position = camera_positions[0]
    look_at = camera_look_ats[0]
    scene.camera.position = tuple(camera_positions[0])
    scene.camera.look_at(tuple(camera_look_ats[0]))
    if args.trajectory_mode != "static":
        keyframe_camera_trajectory(
            scene.camera,
            camera_positions=camera_positions,
            camera_look_ats=camera_look_ats,
            frame_start=frame_start,
        )

    angular_velocity_explicit = any(token.startswith("--angular-velocity") for token in sys.argv[1:])
    if args.trajectory_mode in auto_modes and not angular_velocity_explicit:
        tumble_axis = normalize_vector(rng.normal(size=3), "auto tumble axis")
        angular_velocity_dps = tuple(tumble_axis * args.auto_tumble_deg_per_frame * args.fps)
        print(f"[Motion] Auto tumble axis: {np.round(tumble_axis, 6)}")

    if positions is None:
        positions, quaternions = generate_trajectory(
            num_frames=num_frames,
            initial_position=initial_position,
            linear_velocity_mps=linear_velocity_mps,
            angular_velocity_dps=angular_velocity_dps,
            fps=args.fps,
            initial_quaternion=args.initial_quaternion,
            random_state=scipy_random_state,
        )
    else:
        _, quaternions = generate_trajectory(
            num_frames=num_frames,
            initial_position=positions[0],
            linear_velocity_mps=(0.0, 0.0, 0.0),
            angular_velocity_dps=angular_velocity_dps,
            fps=args.fps,
            initial_quaternion=args.initial_quaternion,
            random_state=scipy_random_state,
        )

    trajectory_diagnostics = None
    if args.trajectory_mode == "curved_flyby":
        trajectory_diagnostics = summarize_curved_flyby(
            camera_positions=camera_positions,
            camera_look_ats=camera_look_ats,
            quaternions=quaternions,
            angular_velocity_dps=angular_velocity_dps,
            fps=args.fps,
        )
    elif args.trajectory_mode in auto_modes:
        trajectory_diagnostics = summarize_trajectory(
            mode=args.trajectory_mode,
            camera_positions=camera_positions,
            camera_look_ats=camera_look_ats,
            object_positions=positions,
            angular_velocity_dps=angular_velocity_dps,
            fps=args.fps,
            bounds=object_bounds,
            extra=trajectory_extra,
        )

    # -------------------------------------------------------------------------
    # 5. LIGHTING â€” SPEED-UE-Cube conventions
    #    Single DirectionalLight (Sun), no ambient.
    #    Sun direction sampled once per sequence (fixed lighting for trajectory).
    #    Constraint: angle(sun_dir, camera_boresight) >= 75 deg
    #    Ref: SPEED-UE-Cube paper, Section "Training Dataset Pose Labels", Constraint 2
    # -------------------------------------------------------------------------
    if args.sun_direction is None:
        sun_direction, sun_position, angle_deg = sample_sun_direction(
            camera_position=camera_position,
            look_at=look_at,
            sun_distance=max(SUN_DISTANCE, 3.0 * float(np.linalg.norm(camera_position))),
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
            position=camera_positions[0],
            target=camera_look_ats[0],
            intensity=lighting["fill_intensity"],
        )
        scene += fill

    print(f"[Lighting] Sun direction (world): {np.round(sun_direction, 4)}")
    angle_from_boresight = math.degrees(
        math.acos(np.clip(np.dot(sun_direction, [0, 1, 0]), -1, 1)))
    # print(f"[Lighting] Angle from camera boresight: {angle_from_boresight:.2f} deg "
    #       f"(must be >= {SUN_MIN_ANGLE_FROM_BORESIGHT_DEG} deg) âœ“")
    print(f"[Lighting] Preset: {args.lighting} ({lighting['description']})")
    print(f"[Lighting] Angle from camera boresight: {angle_deg:.2f} deg "
          f"(target range: {lighting['sun_min_angle']} to "
          f"{lighting['sun_max_angle']} deg)")
    print(f"[Repro] seed={args.seed} output_dir={output_dir}")
    # Explicit keyframes make the rendered sequence exactly match pose_labels.json.
    # The camera is keyframed for curved_flyby; the spacecraft pose is always animated.
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
    mask_areas = []
    mask_margins = []
    mask_ious = []
    previous_mask = None
    rendered_frame_count = 0

    for chunk_index, (chunk_start, chunk_end, chunk_frames) in enumerate(
        iter_frame_chunks(frame_start, frame_end, args.render_chunk_size),
        start=1,
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

        print(f"[Export] Writing chunk {chunk_start}-{chunk_end} RGB and depth ...")
        write_rgb_batch(frames_dict["rgba"], f"{output_dir}/image/", start_index=chunk_offset)
        write_png_depth_batch(depth_f64, f"{output_dir}/depth/", start_index=chunk_offset)
        if "segmentation" in frames_dict:
            mask_areas.extend(write_segmentation_batch(
                frames_dict["segmentation"],
                f"{output_dir}/seg/",
                debug_dir=f"{output_dir}/seg_debug/",
                start_index=chunk_offset,
            ))
            for seg_frame in frames_dict["segmentation"]:
                mask = np.asarray(seg_frame).squeeze() > 0
                ys, xs = np.nonzero(mask)
                if len(xs) == 0:
                    mask_margins.append(0)
                else:
                    height, width = mask.shape
                    mask_margins.append(int(min(xs.min(), ys.min(), width - 1 - xs.max(), height - 1 - ys.max())))
                if previous_mask is not None:
                    union = np.logical_or(previous_mask, mask)
                    mask_ious.append(
                        float(np.count_nonzero(np.logical_and(previous_mask, mask)) / np.count_nonzero(union))
                        if np.any(union) else 0.0
                    )
                previous_mask = mask
        else:
            print("[Export] segmentation buffer not present in renderer output; skipping seg masks.")

        del frames_dict, depth_raw, depth_clamped, depth_f64
        rendered_frame_count += len(chunk_frames)
        if args.max_render_chunks is not None and chunk_index >= args.max_render_chunks:
            print(f"[Render] Stopping after {chunk_index} chunk(s) by --max-render-chunks.")
            break

    export_num_frames = rendered_frame_count if args.max_render_chunks is not None else num_frames

    # -------------------------------------------------------------------------
    # 12. SAVE POSE LABELS
    #     Explicit CV camera frame: x right, y down, z forward.
    # -------------------------------------------------------------------------
    world_to_camera_all, camera_quaternions_wxyz, camera_boresights = camera_pose_arrays(
        camera_positions, camera_look_ats
    )
    if not np.allclose(
        np.matmul(world_to_camera_all, np.transpose(world_to_camera_all, (0, 2, 1))),
        np.eye(3)[None, :, :],
        atol=1e-6,
    ):
        raise RuntimeError("Camera pose basis is not orthonormal.")
    if not np.allclose(np.linalg.det(world_to_camera_all), 1.0, atol=1e-6):
        raise RuntimeError("Camera pose basis has invalid handedness.")
    relative_positions_all = np.asarray([
        world_to_camera_all[i] @ (positions[i] - camera_positions[i])
        for i in range(num_frames)
    ])
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
    for frame_idx in range(export_num_frames):
        world_to_camera = world_to_camera_all[frame_idx]
        object_rotation_world = Rotation.from_quat([
            quaternions[frame_idx][1], quaternions[frame_idx][2],
            quaternions[frame_idx][3], quaternions[frame_idx][0]
        ])
        object_rotation_camera = Rotation.from_matrix(world_to_camera) * object_rotation_world
        relative_position = relative_positions_all[frame_idx]
        pose_labels.append({
            "filename": f"{frame_idx:06d}.png",
            "frame": frame_start + frame_idx,
            "camera_position_world_m": camera_positions[frame_idx].tolist(),
            "camera_quaternion_world_wxyz": camera_quaternions_wxyz[frame_idx].tolist(),
            "q_obj2cam": rotation_to_wxyz(object_rotation_camera),
            "r_obj2cam": relative_position.tolist(),
            "object_position_world_m": positions[frame_idx].tolist(),
            "object_quaternion_world_wxyz": quaternions[frame_idx].tolist(),
            "sun_direction_world": sun_direction.tolist(),
            "sun_boresight_angle_deg": round(angle_deg, 4),
        })

    with open(f"{output_dir}/pose_labels.json", "w") as f:
        json.dump(pose_labels, f, indent=2)

    write_trajectory_sidecars(
        output_dir=output_dir,
        frame_start=frame_start,
        fps=args.fps,
        camera_positions=camera_positions[:export_num_frames],
        camera_quaternions_wxyz=camera_quaternions_wxyz[:export_num_frames],
        object_positions=positions[:export_num_frames],
        object_quaternions_wxyz=quaternions[:export_num_frames],
        camera_look_ats=camera_look_ats[:export_num_frames],
    )

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
            "max_render_chunks": args.max_render_chunks,
            "exported_frames": export_num_frames,
            "trajectory_mode": args.trajectory_mode,
            "camera_position": camera_positions[0].tolist(),
            "look_at": camera_look_ats[0].tolist(),
            "flyby_start_range": args.flyby_start_range,
            "flyby_end_range": args.flyby_end_range,
            "flyby_arc_deg": args.flyby_arc_deg,
            "flyby_elevation_deg": args.flyby_elevation_deg,
            "flyby_start_azimuth_deg": args.flyby_start_azimuth_deg,
            "flyby_start_elevation_deg": args.flyby_start_elevation_deg,
            "framing_fov_fraction": args.framing_fov_fraction,
            "approach_near_multiplier": args.approach_near_multiplier,
            "approach_far_multiplier": args.approach_far_multiplier,
            "fly_across_distance_multiplier": args.fly_across_distance_multiplier,
            "fly_across_lateral_fraction": args.fly_across_lateral_fraction,
            "max_viewpoint_deg_per_frame": args.max_viewpoint_deg_per_frame,
            "auto_tumble_deg_per_frame": args.auto_tumble_deg_per_frame,
            "initial_position": list(initial_position),
            "linear_velocity_mps": list(linear_velocity_mps),
            "angular_velocity_dps": list(angular_velocity_dps),
            "legacy_linear_velocity_per_frame": args.linear_velocity,
            "legacy_angular_velocity_per_frame": args.angular_velocity,
            "initial_quaternion_wxyz": quaternions[0].tolist(),
            "pivot_correction": pivot_info,
            "explicit_initial_quaternion": args.initial_quaternion is not None,
            "explicit_sun_direction": args.sun_direction is not None,
            "trajectory_diagnostics": None if trajectory_diagnostics is None else {
                "range_m_min": float(np.min(trajectory_diagnostics["range_m"])),
                "range_m_median": float(np.median(trajectory_diagnostics["range_m"])),
                "range_m_max": float(np.max(trajectory_diagnostics["range_m"])),
                "total_viewpoint_deg": float(trajectory_diagnostics["total_viewpoint_deg"]),
                "adjacent_viewpoint_deg_min": (
                    float(np.min(trajectory_diagnostics["adjacent_viewpoint_deg"]))
                    if len(trajectory_diagnostics["adjacent_viewpoint_deg"]) else 0.0
                ),
                "adjacent_viewpoint_deg_median": (
                    float(np.median(trajectory_diagnostics["adjacent_viewpoint_deg"]))
                    if len(trajectory_diagnostics["adjacent_viewpoint_deg"]) else 0.0
                ),
                "adjacent_viewpoint_deg_max": (
                    float(np.max(trajectory_diagnostics["adjacent_viewpoint_deg"]))
                    if len(trajectory_diagnostics["adjacent_viewpoint_deg"]) else 0.0
                ),
                "object_tumble_deg": float(trajectory_diagnostics["object_tumble_deg"]),
                "pointing_error_deg_max": float(np.max(trajectory_diagnostics["pointing_error_deg"])),
            },
            "object_geometry": {
                "bbox_min_m": object_bounds["bbox_min_m"].tolist(),
                "bbox_max_m": object_bounds["bbox_max_m"].tolist(),
                "bbox_center_m": object_bounds["bbox_center_m"].tolist(),
                "bbox_dimensions_m": object_bounds["bbox_dimensions_m"].tolist(),
                "max_span_m": object_bounds["max_span_m"],
                "bounding_radius_m": object_bounds["bounding_radius_m"],
            },
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
                "world_to_camera_rotation": world_to_camera_all[0].tolist(),
                "camera_boresight_world": camera_boresights[0].tolist(),
                "world_to_camera_rotation_frame0": world_to_camera_all[0].tolist(),
                "camera_boresight_world_frame0": camera_boresights[0].tolist(),
                "camera_trajectory_file": "camera_trajectory.json",
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
    if mask_areas:
        mask_areas_np = np.asarray(mask_areas, dtype=int)
        print("\n[Mask Stats] (segmentation > 0)")
        print(
            "  area pixels min/median/max: "
            f"{mask_areas_np.min()} / {int(np.median(mask_areas_np))} / {mask_areas_np.max()}"
        )
        if mask_ious:
            mask_ious_np = np.asarray(mask_ious, dtype=float)
            print(
                "  adjacent IoU min/median/max: "
                f"{mask_ious_np.min():.4f} / {np.median(mask_ious_np):.4f} / {mask_ious_np.max():.4f}"
            )
        if mask_margins:
            margins_np = np.asarray(mask_margins, dtype=int)
            print(
                "  image-border margin px min/median/max: "
                f"{margins_np.min()} / {int(np.median(margins_np))} / {margins_np.max()}"
            )

    print("\nDataset generation complete! Check:", output_dir)

if __name__ == "__main__":
    main()
