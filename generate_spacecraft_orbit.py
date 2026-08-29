"""
Spacecraft Orbit Snapshot Dataset Generator
============================================
Camera-revolves variant of the trajectory generator.

DIFFERENCE FROM ORIGINAL:
- The satellite is STATIONARY at the origin with a fixed orientation
  (no tumbling, no drifting).
- The CAMERA revolves around the satellite on a circular orbit and takes
  8 evenly-spaced snapshots.

IDENTICAL TO ORIGINAL (ambient conditions unchanged):
- Sun-direction sampling (SPEED-UE-Cube 75-115 deg boresight constraint)
- CV-friendly lighting overrides (SUN_INTENSITY, FILL_INTENSITY, AMBIENT_LEVEL,
  SUN_SHADOW_SOFTNESS)
- Weak camera-side fill light + low ambient illumination
- Camera intrinsics (Point Grey Grasshopper 3 + Xenoplan 1.4/17mm)
- Resolution 1920 x 1200, zero gravity, black-space background
- The sun is sampled ONCE and FIXED in the world frame, so lighting never
  changes as the camera moves.

Export format is unchanged:
  output/<obj>_orbit/image/000000.png
  output/<obj>_orbit/depth/000000.tiff
  output/<obj>_orbit/flow/000000.flo
  output/<obj>_orbit/pose_labels.json
  output/<obj>_orbit/metadata.json

Docker (Linux):
    docker run --rm --interactive \
        --user $(id -u):$(id -g) \
        --volume "$(pwd):/kubric" \
        --volume "$HOME/tracking_dataset:/dataset" \
        kubricdockerhub/kubruntu \
        /usr/bin/python3 kubric-sat-datagen/generate_spacecraft_orbit.py
"""

import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.simulator.pybullet import PyBullet as KubricSimulator
import numpy as np
import os
import argparse
import json
import math
from PIL import Image
import imageio
from scipy.spatial.transform import Rotation
from scene_setup import setup_scene_actors
from generate_spacecraft import (
    LIGHTING_PRESETS,
    apply_mtl_textures_to_imported_object,
    normalize_vector,
    quaternion_wxyz_to_kb,
    resolve_lighting_config,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONSTANTS  (unchanged from original)
# =============================================================================
MAX_DEPTH = 9999.0   # metres — background/sky pixels replaced with 0.0
MM_TO_M   = 0.001
CM_TO_M   = 0.01

# Camera — Point Grey Grasshopper 3 + Xenoplan 1.4/17mm
CAMERA_FOCAL_LENGTH_MM  = 17.5217   # effective focal length (mm)
CAMERA_SENSOR_WIDTH_MM  = 11.2512   # sensor width (mm)
CAMERA_SENSOR_HEIGHT_MM = 7.0320    # sensor height = 5.86 um x 1200 pixels (mm)
CAMERA_FOV_H_DEG        = 35.6      # horizontal FOV as stated in paper (deg)

# Sun — physically correct angular diameter of the Sun as seen from space
SUN_ANGULAR_RADIUS_DEG = 0.265
SUN_SHADOW_SOFTNESS    = math.tan(math.radians(SUN_ANGULAR_RADIUS_DEG))  # ~0.00462

# Lighting constraints from SPEED-UE-Cube
SUN_MIN_ANGLE_FROM_BORESIGHT_DEG = 75.0

SUN_DISTANCE = 100.0  # arbitrary large distance for DirectionalLight

CAMERA_POSITION = np.array([0.0, -10.0, 0.0])
LOOK_AT         = np.array([0.0,  0.0,  0.0])

# CV-friendly lighting overrides (unchanged from original)
SUN_SHADOW_SOFTNESS = 0.15
SUN_INTENSITY  = 5.0
FILL_INTENSITY = 3.5      # was 2.0 — stronger camera-side fill
AMBIENT_LEVEL  = 0.35     # was 0.12 — raise the floor so no face goes black
SUN_MAX_ANGLE_FROM_BORESIGHT_DEG = 115.0

# Orbit configuration (NEW)
NUM_SNAPSHOTS = 8            # number of camera viewpoints around the satellite
ORBIT_RADIUS  = 10.0         # metres — same stand-off distance as original
ORBIT_ELEVATION_DEG = 0.0    # 0 = equatorial sweep; >0 tilts the orbit upward


# =============================================================================
# CONFIG HELPERS
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reproducible orbit snapshots of a static textured spacecraft."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampled sun and optional satellite orientation.")
    parser.add_argument(
        "--lighting",
        choices=sorted(LIGHTING_PRESETS),
        default="cv_bright",
        help="Lighting preset: space has hard dark edges; cv_bright keeps the satellite visible.",
    )
    parser.add_argument("--sun-intensity", type=float, default=None, help="Override preset sun intensity.")
    parser.add_argument("--fill-intensity", type=float, default=None, help="Override preset camera-side fill intensity.")
    parser.add_argument("--ambient-level", type=float, default=None, help="Override preset ambient illumination.")
    parser.add_argument("--sun-shadow-softness", type=float, default=None, help="Override preset sun shadow softness.")
    parser.add_argument("--sun-min-angle", type=float, default=None, help="Minimum reference boresight to sun angle in degrees.")
    parser.add_argument("--sun-max-angle", type=float, default=None, help="Maximum reference boresight to sun angle in degrees.")
    parser.add_argument("--sun-direction", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"), help="Use an exact world-space sun direction instead of sampling.")
    parser.add_argument("--initial-position", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"), help="Static satellite position in metres.")
    parser.add_argument("--initial-quaternion", nargs=4, type=float, default=(1.0, 0.0, 0.0, 0.0), metavar=("W", "X", "Y", "Z"), help="Static satellite quaternion in w x y z order.")
    parser.add_argument("--random-initial-quaternion", action="store_true", help="Sample the static satellite orientation from --seed instead of using --initial-quaternion.")
    parser.add_argument("--num-snapshots", type=int, default=NUM_SNAPSHOTS, help="Number of camera snapshots around the orbit.")
    parser.add_argument("--orbit-radius", type=float, default=ORBIT_RADIUS, help="Camera orbit radius in metres.")
    parser.add_argument("--orbit-elevation", type=float, default=ORBIT_ELEVATION_DEG, help="Camera orbit elevation angle in degrees.")
    parser.add_argument("--object-name", default="Cheops", help="Base object filename without extension.")
    parser.add_argument("--asset-dir", default=None, help="Asset directory containing OBJ, MTL, textures, and URDF.")
    parser.add_argument("--output-dir", default=None, help="Output directory for rendered dataset.")
    return parser.parse_args()


def rotation_from_quaternion_wxyz(quaternion_wxyz):
    return Rotation.from_quat([
        quaternion_wxyz[1],
        quaternion_wxyz[2],
        quaternion_wxyz[3],
        quaternion_wxyz[0],
    ])


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

def write_flo_batch(flows, output_dir, name="forward_flow"):
    os.makedirs(output_dir, exist_ok=True)
    for i, flow in enumerate(flows):
        write_flo(os.path.join(output_dir, f"{i:06d}.flo"), flow[..., :2])

def write_rgb_batch(rgb_frames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(rgb_frames):
        Image.fromarray(frame[..., :3], mode="RGB").save(
            os.path.join(output_dir, f"{i:06d}.png"))

def write_tiff_depth_batch(depth_f64, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(depth_f64):
        imageio.imwrite(
            os.path.join(output_dir, f"{i:06d}.tiff"),
            frame.squeeze().astype(np.float64),
            format="tiff")

def clamp_depth_batch(depth_frames, max_depth=MAX_DEPTH):
    """Replace inf/nan/<=0/beyond max_depth with 0.0 (invalid sentinel)."""
    depth = np.array(depth_frames, dtype=np.float32)
    invalid = ~np.isfinite(depth) | (depth <= 0.0) | (depth > max_depth)
    depth[invalid] = 9999.0
    return depth


# =============================================================================
# LIGHTING HELPERS  (SPEED-UE-Cube conventions — unchanged from original)
# =============================================================================
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
    Sample a CV-friendly sun direction:
    - keeps Sun outside camera FOV
    - avoids very strong backlighting
    - avoids grazing illumination that produces large dark shadows

    Note: the angle is evaluated against the ORIGINAL reference camera
    position/boresight so the fixed world-frame sun matches the original
    ambient conditions exactly.
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


# def make_camera_fill_light():
#     """Weak fill light from near the (reference) camera direction."""
#     fill = kb.DirectionalLight(
#         color=kb.Color(0.75, 0.82, 1.0),   # slightly cool fill
#         shadow_softness=0.3,
#         intensity=FILL_INTENSITY,
#     )
#     fill.position = tuple(CAMERA_POSITION)
#     fill.look_at(tuple(object_position))
#     return fill
def make_camera_fill_light(position=CAMERA_POSITION, target=LOOK_AT, intensity=FILL_INTENSITY, shadow_softness=0.3):
    """Weak fill light co-located with the camera so the viewed face is illuminated."""
    fill = kb.DirectionalLight(
        color=kb.Color(0.75, 0.82, 1.0),   # slightly cool fill
        shadow_softness=shadow_softness,
        intensity=intensity,
    )
    fill.position = tuple(position)
    fill.look_at(tuple(target))
    return fill


# =============================================================================
# CAMERA ORBIT HELPERS  (NEW)
# =============================================================================
def generate_camera_orbit(num_snapshots=NUM_SNAPSHOTS,
                          radius=ORBIT_RADIUS,
                          elevation_deg=ORBIT_ELEVATION_DEG):
    """
    Generate camera positions evenly spaced on a circular orbit around the
    origin (where the stationary satellite sits).

    View 0 starts along -Y so it matches the original camera position
    (0, -radius, 0). Azimuth then increases about the +Z axis.

    Args:
        num_snapshots (int): number of viewpoints
        radius        (float): orbit radius in metres
        elevation_deg (float): tilt of the orbit above the XY-plane in degrees

    Returns:
        positions   (np.ndarray): shape (num_snapshots, 3)
        azimuths_deg(np.ndarray): shape (num_snapshots,)
    """
    elev = math.radians(elevation_deg)
    z = radius * math.sin(elev)
    r_xy = radius * math.cos(elev)

    positions = np.zeros((num_snapshots, 3))
    azimuths_deg = np.zeros(num_snapshots)

    for i in range(num_snapshots):
        # start at -Y (azimuth 0 -> pointing along -Y), sweep around +Z
        az = 2.0 * math.pi * i / num_snapshots
        x = r_xy * math.sin(az)
        y = -r_xy * math.cos(az)
        positions[i] = [x, y, z]
        azimuths_deg[i] = math.degrees(az)

    return positions, azimuths_deg


def look_at_rotation(cam_pos, target=np.array([0.0, 0.0, 0.0]),
                     up=np.array([0.0, 0.0, 1.0])):
    """
    Compute the world->camera rotation matrix for a camera at cam_pos looking
    at target, using the given world up vector. Uses the OpenGL/Blender
    convention where the camera looks down its local -Z axis.

    Returns:
        R_world2cam (np.ndarray): 3x3 rotation matrix (world -> camera)
        R_cam2world (np.ndarray): 3x3 rotation matrix (camera -> world)
    """
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)          # camera looks along -Z (world)

    # Guard against forward parallel to up
    if abs(np.dot(forward, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    # Blender camera basis: X=right, Y=up, Z=-forward
    R_cam2world = np.column_stack((right, true_up, -forward))
    R_world2cam = R_cam2world.T
    return R_world2cam, R_cam2world


def relative_object_pose(obj_pos_world, obj_rot_world, cam_pos_world, R_world2cam):
    """
    Compute the object's pose relative to the camera.

    r_obj2cam = R_world2cam @ (obj_pos_world - cam_pos_world)
    R_obj2cam = R_world2cam @ R_obj2world

    Args:
        obj_pos_world (np.ndarray): object position in world (3,)
        obj_rot_world (Rotation)  : object orientation in world
        cam_pos_world (np.ndarray): camera position in world (3,)
        R_world2cam   (np.ndarray): 3x3 world->camera rotation

    Returns:
        r_obj2cam (np.ndarray): position in camera frame (3,)
        q_obj2cam (np.ndarray): quaternion [w, x, y, z]
    """
    r_obj2cam = R_world2cam @ (obj_pos_world - cam_pos_world)

    R_obj2world = obj_rot_world.as_matrix()
    R_obj2cam = R_world2cam @ R_obj2world
    q_xyzw = Rotation.from_matrix(R_obj2cam).as_quat()   # scipy: [x,y,z,w]
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return r_obj2cam, q_wxyz


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
        SCRIPT_DIR, "output", f"{object_name}_orbit_{args.lighting}_seed{args.seed}"
    )
    lighting = resolve_lighting_config(args)
    object_position = np.array(args.initial_position, dtype=float)

    if args.num_snapshots < 1:
        raise ValueError("--num-snapshots must be >= 1")
    if args.orbit_radius <= 0.0:
        raise ValueError("--orbit-radius must be > 0")
    if lighting["sun_min_angle"] > lighting["sun_max_angle"]:
        raise ValueError("--sun-min-angle must be <= --sun-max-angle")

    num_frames = args.num_snapshots
    frame_start = 1
    frame_end = num_frames

    os.makedirs(f"{output_dir}/image/", exist_ok=True)
    os.makedirs(f"{output_dir}/depth/", exist_ok=True)
    os.makedirs(f"{output_dir}/flow/",  exist_ok=True)
    os.makedirs(f"{output_dir}/tmp/",   exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. SCENE
    # -------------------------------------------------------------------------
    scene = kb.Scene(
        resolution=(1920, 1200),    # Point Grey Grasshopper 3 native resolution
        frame_start=frame_start,
        frame_end=frame_end,
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
    ambient_level = lighting["ambient_level"]
    renderer.ambient_illumination = kb.Color(
        ambient_level, ambient_level, ambient_level,
    )

    # -------------------------------------------------------------------------
    # 4. CAMERA
    # -------------------------------------------------------------------------
    scene.camera = kb.PerspectiveCamera(
        focal_length=CAMERA_FOCAL_LENGTH_MM,
        sensor_width=CAMERA_SENSOR_WIDTH_MM,
    )

    # -------------------------------------------------------------------------
    # 5. LIGHTING — SPEED-UE-Cube conventions (sampled ONCE, fixed in world)
    # -------------------------------------------------------------------------
    reference_camera_position = np.array([0.0, -args.orbit_radius, 0.0])
    if args.sun_direction is None:
        sun_direction, sun_position, ref_angle_deg = sample_sun_direction(
            camera_position=reference_camera_position,
            look_at=object_position,
            min_angle_deg=lighting["sun_min_angle"],
            max_angle_deg=lighting["sun_max_angle"],
            rng=rng,
        )
    else:
        sun_direction = normalize_vector(args.sun_direction, "--sun-direction")
        sun_position = sun_direction * SUN_DISTANCE
        boresight = normalize_vector(object_position - reference_camera_position, "reference camera boresight")
        cam_to_sun = normalize_vector(sun_position - reference_camera_position, "reference camera-to-sun vector")
        ref_angle_deg = math.degrees(math.acos(np.clip(np.dot(boresight, cam_to_sun), -1.0, 1.0)))

    sun = make_sun_light(
        sun_direction,
        intensity=lighting["sun_intensity"],
        shadow_softness=lighting["sun_shadow_softness"],
    )
    scene += sun

    fill = None
    if lighting["fill_intensity"] > 0.0:
        fill = make_camera_fill_light(
            position=reference_camera_position,
            target=object_position,
            intensity=lighting["fill_intensity"],
        )
        scene += fill

    print(f"[Lighting] Sun direction (world): {np.round(sun_direction, 4)}")
    print(f"[Lighting] Preset: {args.lighting} ({lighting['description']})")
    print(f"[Lighting] Reference angle from view 0 boresight: {ref_angle_deg:.2f} deg "
          f"(target range: {lighting['sun_min_angle']} to "
          f"{lighting['sun_max_angle']} deg)")
    print(f"[Repro] seed={args.seed} output_dir={output_dir}")

    # -------------------------------------------------------------------------
    # 6. SPACECRAFT OBJECT — STATIONARY at origin, fixed orientation
    # -------------------------------------------------------------------------
    if args.random_initial_quaternion:
        object_rotation = Rotation.random(random_state=scipy_random_state)
        q_xyzw = object_rotation.as_quat()
        obj_quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    else:
        obj_quat_wxyz = np.array(args.initial_quaternion, dtype=float)
        quat_norm = np.linalg.norm(obj_quat_wxyz)
        if quat_norm < 1e-8:
            raise ValueError("--initial-quaternion must not be the zero quaternion")
        obj_quat_wxyz = obj_quat_wxyz / quat_norm
        object_rotation = rotation_from_quaternion_wxyz(obj_quat_wxyz)

    debris = kb.FileBasedObject(
        asset_id="cheops_satellite",
        render_filename=f"{asset_dir}/{object_name}.obj",
        simulation_filename=f"{asset_dir}/{object_name}.urdf",
        position=tuple(object_position),
        mass=10.0,
        scale=MM_TO_M,
    )
    scene += debris
    material_texture_map = apply_mtl_textures_to_imported_object(
        asset_id="cheops_satellite",
        mtl_path=f"{asset_dir}/{object_name}.mtl",
    )

    # Keep the satellite perfectly still: fixed orientation, no motion.
    debris.quaternion = quaternion_wxyz_to_kb(obj_quat_wxyz)
    debris.velocity = (0.0, 0.0, 0.0)
    debris.angular_velocity = (0.0, 0.0, 0.0)

    # -------------------------------------------------------------------------
    # 7. CAMERA ORBIT — keyframe camera pose per snapshot
    # -------------------------------------------------------------------------
    cam_positions, cam_azimuths = generate_camera_orbit(
        num_snapshots=num_frames,
        radius=args.orbit_radius,
        elevation_deg=args.orbit_elevation,
    )

    # Pre-compute per-frame relative poses + per-view sun angle for labels
    pose_records = []
    for i in range(num_frames):
        frame_id = frame_start + i
        cam_pos = cam_positions[i]

        # Set + keyframe the camera pose for this frame
        scene.camera.position = tuple(cam_pos)
        scene.camera.look_at(tuple(object_position))
        scene.camera.keyframe_insert("position", frame_id)
        scene.camera.keyframe_insert("quaternion", frame_id)

        # Move the optional fill light with the camera so the viewed face is lit.
        if fill is not None:
            fill.position = tuple(cam_pos)
            fill.look_at(tuple(object_position))
            fill.keyframe_insert("position", frame_id)
            fill.keyframe_insert("quaternion", frame_id)
        # World->camera rotation (matches look_at above)
        R_world2cam, _ = look_at_rotation(cam_pos, target=object_position)

        # Relative object pose in this camera frame
        r_obj2cam, q_obj2cam = relative_object_pose(
            obj_pos_world=object_position,
            obj_rot_world=object_rotation,
            cam_pos_world=cam_pos,
            R_world2cam=R_world2cam,
        )

        # Per-view sun-to-boresight angle (camera boresight = world +? -> use look dir)
        boresight = (object_position - cam_pos)
        boresight /= np.linalg.norm(boresight)
        cam_to_sun = (sun_direction * SUN_DISTANCE) - cam_pos
        cam_to_sun /= np.linalg.norm(cam_to_sun)
        view_sun_angle = math.degrees(
            math.acos(np.clip(np.dot(boresight, cam_to_sun), -1.0, 1.0)))

        pose_records.append({
            "r_obj2cam": r_obj2cam,
            "q_obj2cam": q_obj2cam,
            "camera_position_world": cam_pos,
            "camera_azimuth_deg": float(cam_azimuths[i]),
            "view_sun_angle_deg": view_sun_angle,
        })

    # -------------------------------------------------------------------------
    # 8. RUN PHYSICS SIMULATION (PyBullet)
    # -------------------------------------------------------------------------
    print("\n[Physics] Running PyBullet simulation (satellite stationary)...")
    simulator.run()

    # -------------------------------------------------------------------------
    # 9. RUN RENDERING (Blender)
    # -------------------------------------------------------------------------
    print("[Render] Running Blender renderer...")
    renderer.save_state(f"{output_dir}/blender_scene.blend")
    frames_dict = renderer.render()

    # -------------------------------------------------------------------------
    # 10. PROCESS DEPTH
    # -------------------------------------------------------------------------
    print(f"\n[Depth] Clamping to MAX_DEPTH={MAX_DEPTH:.1f} m ...")
    depth_raw     = frames_dict["depth"]
    depth_clamped = clamp_depth_batch(depth_raw, max_depth=MAX_DEPTH)
    depth_f64     = depth_clamped.astype(np.float64)

    n_invalid_raw     = np.sum(~np.isfinite(depth_raw.astype(np.float32)) | (depth_raw <= 0.0))
    n_invalid_clamped = np.sum(depth_clamped == 0.0)
    print(f"  Invalid pixels before clamping (inf/nan/<=0): {n_invalid_raw}")
    print(f"  Invalid pixels after  clamping (zeroed):      {n_invalid_clamped}")

    # -------------------------------------------------------------------------
    # 11. EXPORT DATA  (same format as original)
    # -------------------------------------------------------------------------
    print("\n[Export] Writing RGB, depth, flow ...")
    write_rgb_batch(frames_dict["rgba"],         f"{output_dir}/image/")
    write_tiff_depth_batch(depth_f64,            f"{output_dir}/depth/")
    write_flo_batch(frames_dict["forward_flow"], f"{output_dir}/flow/")

    # -------------------------------------------------------------------------
    # 12. SAVE POSE LABELS (SPEED-UE-Cube format)
    # -------------------------------------------------------------------------
    pose_labels = []
    for frame_idx in range(num_frames):
        rec = pose_records[frame_idx]
        pose_labels.append({
            "filename": f"{frame_idx:06d}.png",
            # Quaternion [w, x, y, z] — orientation of spacecraft w.r.t. camera
            "q_obj2cam": rec["q_obj2cam"].tolist(),
            # Position [x, y, z] in metres — spacecraft position in camera frame
            "r_obj2cam": rec["r_obj2cam"].tolist(),
            # Sun direction unit vector in world frame (fixed for all views)
            "sun_direction_world": sun_direction.tolist(),
            # Per-view angle between sun and this camera's boresight
            "sun_boresight_angle_deg": round(rec["view_sun_angle_deg"], 4),
            # Extra info: where the camera was for this snapshot
            "camera_position_world": rec["camera_position_world"].tolist(),
            "camera_azimuth_deg": round(rec["camera_azimuth_deg"], 4),
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
            "num_snapshots": num_frames,
            "orbit_radius_m": args.orbit_radius,
            "orbit_elevation_deg": args.orbit_elevation,
            "initial_position": object_position.tolist(),
            "initial_quaternion_wxyz": obj_quat_wxyz.tolist(),
            "random_initial_quaternion": args.random_initial_quaternion,
            "explicit_sun_direction": args.sun_direction is not None,
        },
        "camera": {
            **kb.get_camera_info(scene.camera),
            "focal_length_mm":    CAMERA_FOCAL_LENGTH_MM,
            "sensor_width_mm":    CAMERA_SENSOR_WIDTH_MM,
            "sensor_height_mm":   CAMERA_SENSOR_HEIGHT_MM,
            "fov_horizontal_deg": CAMERA_FOV_H_DEG,
        },
        "orbit": {
            "mode": "camera revolves around stationary satellite",
            "num_snapshots": num_frames,
            "orbit_radius_m": args.orbit_radius,
            "orbit_elevation_deg": args.orbit_elevation,
            "camera_positions_world": cam_positions.tolist(),
            "camera_azimuths_deg": cam_azimuths.tolist(),
        },
        "material_textures": material_texture_map,
        "lighting": {
            "type": "DirectionalLight Sun + weak camera fill + low ambient",
            "preset_description": lighting["description"],
            "sun_direction_world": sun_direction.tolist(),
            "sun_reference_boresight_angle_deg": round(ref_angle_deg, 4),
            "shadow_softness": lighting["sun_shadow_softness"],
            "sun_intensity": lighting["sun_intensity"],
            "fill_intensity": lighting["fill_intensity"],
            "ambient_illumination": lighting["ambient_level"],
            "note": (
                "Ambient conditions identical to the trajectory generator. Sun is "
                "sampled once and fixed in the world frame; the camera orbits a "
                "stationary satellite, so per-view sun-to-boresight angle varies "
                "and is recorded per frame in pose_labels.json."
            ),
        },
        "instances": kb.get_instance_info(scene),
    })

    # -------------------------------------------------------------------------
    # 14. DEPTH STATS
    # -------------------------------------------------------------------------
    valid_mask = depth_clamped > 0.0
    print(f"\n[Depth Stats] (valid pixels only, depth > 0)")
    print(f"  mean:  {np.mean(depth_clamped[valid_mask]):.4f} m")
    print(f"  min:   {np.min(depth_clamped[valid_mask]):.4f} m")
    print(f"  max:   {np.max(depth_clamped[valid_mask]):.4f} m")
    print(f"  valid: {np.sum(valid_mask)} / {depth_clamped.size} pixels "
          f"({100 * np.mean(valid_mask):.1f}%)")

    print("\n[DONE] Orbit snapshot generation complete! Check:", output_dir)


if __name__ == "__main__":
    main()
