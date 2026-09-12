"""
View-Rich Spacecraft Snapshot Dataset Generator
================================================

Drop-in replacement for generate_spacecraft_orbit.py.

Main change:
- Satellite stays perfectly stationary.
- Fibonacci mode distributes cameras over the full viewing sphere to provide
  complementary 3D surface coverage.
- Camera radius is fixed.
- Every camera looks at the stationary spacecraft center.
- Existing RGB / depth / optical-flow / pose-label / metadata outputs are kept.

Recommended diagnostic run:
    python generate_spacecraft_orbit.py \
        --trajectory fibonacci \
        --num-snapshots 48 \
        --orbit-radius 10.0 \
        --lighting cv_bright \
        --seed 0

Legacy equatorial orbit:
    python generate_spacecraft_orbit.py \
        --trajectory ring \
        --num-snapshots 48 \
        --orbit-radius 10.0 \
        --orbit-elevation 0

Notes:
- The sun is still sampled once against the reference frame-0 boresight and
  remains fixed in the world frame, matching the previous script's convention.
"""

import argparse
import json
import math
import os
import shutil

import imageio
import kubric as kb
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from kubric.renderer.blender import Blender as KubricRenderer
from kubric.simulator.pybullet import PyBullet as KubricSimulator

from generate_spacecraft import (
    LIGHTING_PRESETS,
    apply_mtl_textures_to_imported_object,
    normalize_vector,
    quaternion_wxyz_to_kb,
    resolve_lighting_config,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONSTANTS
# =============================================================================

MAX_DEPTH = 9999.0
MM_TO_M = 0.001

# Point Grey Grasshopper 3 + Xenoplan 1.4/17mm
CAMERA_FOCAL_LENGTH_MM = 17.5217
CAMERA_SENSOR_WIDTH_MM = 11.2512
CAMERA_SENSOR_HEIGHT_MM = 7.0320
CAMERA_FOV_H_DEG = 35.6

SUN_DISTANCE = 100.0
SUN_MIN_ANGLE_FROM_BORESIGHT_DEG = 75.0
SUN_MAX_ANGLE_FROM_BORESIGHT_DEG = 115.0

# Defaults for the new view-rich dataset.
NUM_SNAPSHOTS = 96
ORBIT_RADIUS = 10.0
ORBIT_ELEVATION_DEG = 0.0
MAX_ABS_ELEVATION_DEG = 70.0

CAMERA_POSITION = np.array([0.0, -ORBIT_RADIUS, 0.0], dtype=float)
LOOK_AT = np.array([0.0, 0.0, 0.0], dtype=float)

# Kept for function defaults; actual values normally come from lighting preset.
SUN_INTENSITY = 5.0
FILL_INTENSITY = 3.5
SUN_SHADOW_SOFTNESS = 0.15


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a view-rich synthetic RGB-D spacecraft dataset with a "
            "stationary target and moving camera."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for sampled sun and optional satellite orientation.",
    )

    parser.add_argument(
        "--lighting",
        choices=sorted(LIGHTING_PRESETS),
        default="cv_bright",
        help="Lighting preset.",
    )
    parser.add_argument("--sun-intensity", type=float, default=None)
    parser.add_argument("--fill-intensity", type=float, default=None)
    parser.add_argument("--ambient-level", type=float, default=None)
    parser.add_argument("--sun-shadow-softness", type=float, default=None)
    parser.add_argument("--sun-min-angle", type=float, default=None)
    parser.add_argument("--sun-max-angle", type=float, default=None)
    parser.add_argument(
        "--sun-direction",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Use exact fixed world-space sun direction instead of sampling.",
    )

    parser.add_argument(
        "--initial-position",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Static spacecraft position in metres.",
    )
    parser.add_argument(
        "--initial-quaternion",
        nargs=4,
        type=float,
        default=(1.0, 0.0, 0.0, 0.0),
        metavar=("W", "X", "Y", "Z"),
        help="Static spacecraft quaternion in w x y z order.",
    )
    parser.add_argument(
        "--random-initial-quaternion",
        action="store_true",
        help="Sample static spacecraft orientation from --seed.",
    )

    parser.add_argument(
        "--trajectory",
        choices=("fibonacci", "ring"),
        default="fibonacci",
        help=(
            "'fibonacci' = view-rich full sphere; "
            "'ring' = legacy single-elevation circular orbit."
        ),
    )
    parser.add_argument(
        "--num-snapshots",
        type=int,
        default=NUM_SNAPSHOTS,
        help="Number of rendered viewpoints.",
    )
    parser.add_argument(
        "--render-chunk-size",
        type=int,
        default=10,
        help="Number of frames to render and export per batch.",
    )
    parser.add_argument(
        "--orbit-radius",
        type=float,
        default=ORBIT_RADIUS,
        help="Fixed camera distance from target in metres.",
    )
    parser.add_argument(
        "--max-abs-elevation",
        type=float,
        default=MAX_ABS_ELEVATION_DEG,
        help=(
            "Deprecated compatibility option; fibonacci mode now covers the "
            "full sphere."
        ),
    )
    parser.add_argument(
        "--orbit-elevation",
        type=float,
        default=ORBIT_ELEVATION_DEG,
        help="For legacy ring trajectory only.",
    )

    parser.add_argument(
        "--object-name",
        default="Cheops",
        help="Base object filename without extension.",
    )
    parser.add_argument(
        "--asset-dir",
        default=None,
        help="Asset directory containing OBJ, MTL, textures, and URDF.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dataset directory.",
    )

    return parser.parse_args()


# =============================================================================
# BASIC HELPERS
# =============================================================================

def rotation_from_quaternion_wxyz(quaternion_wxyz):
    return Rotation.from_quat([
        quaternion_wxyz[1],
        quaternion_wxyz[2],
        quaternion_wxyz[3],
        quaternion_wxyz[0],
    ])


def write_flo(filename, flow):
    """Write HxWx2 float32 array to Middlebury .flo format."""
    assert flow.ndim == 3 and flow.shape[2] == 2, "Flow must be HxWx2"
    with open(filename, "wb") as f:
        np.array([202021.25], dtype=np.float32).tofile(f)
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)


def write_flo_batch(flows, output_dir, start_index=0):
    os.makedirs(output_dir, exist_ok=True)
    for i, flow in enumerate(flows):
        frame_idx = start_index + i
        write_flo(
            os.path.join(output_dir, f"{frame_idx:06d}.flo"),
            flow[..., :2],
        )


def write_rgb_batch(rgb_frames, output_dir, start_index=0):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(rgb_frames):
        frame_idx = start_index + i
        Image.fromarray(frame[..., :3], mode="RGB").save(
            os.path.join(output_dir, f"{frame_idx:06d}.png")
        )


def write_tiff_depth_batch(depth_f64, output_dir, start_index=0):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(depth_f64):
        frame_idx = start_index + i
        imageio.imwrite(
            os.path.join(output_dir, f"{frame_idx:06d}.tiff"),
            frame.squeeze().astype(np.float64),
            format="tiff",
        )


def clamp_depth_batch(depth_frames, max_depth=MAX_DEPTH):
    """
    Replace invalid/background depth with 9999.0.

    This intentionally preserves the previous script's output convention.
    """
    depth = np.array(depth_frames, dtype=np.float32)
    invalid = (
        ~np.isfinite(depth)
        | (depth <= 0.0)
        | (depth > max_depth)
    )
    depth[invalid] = 9999.0
    return depth


def clear_render_scratch(scratch_dir):
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
# LIGHTING
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
    Sample one fixed world-space sun direction using the reference camera.

    This deliberately keeps the same convention as the old orbit script:
    the angular constraint applies to the reference frame-0 camera, not every
    view on the spherical trajectory.
    """
    boresight = np.asarray(look_at, dtype=float) - np.asarray(
        camera_position, dtype=float
    )
    boresight /= np.linalg.norm(boresight)

    if rng is None:
        rng = np.random.default_rng()

    for _ in range(max_attempts):
        sun_dir = rng.normal(size=3)
        sun_dir /= np.linalg.norm(sun_dir)

        sun_position = sun_dir * sun_distance
        cam_to_sun = sun_position - camera_position
        cam_to_sun /= np.linalg.norm(cam_to_sun)

        cos_angle = np.clip(
            np.dot(boresight, cam_to_sun),
            -1.0,
            1.0,
        )
        angle_deg = math.degrees(math.acos(cos_angle))

        if min_angle_deg <= angle_deg <= max_angle_deg:
            return sun_dir, sun_position, angle_deg

    raise RuntimeError(
        f"Could not sample a valid sun direction after {max_attempts} attempts. "
        f"Need angle in [{min_angle_deg}, {max_angle_deg}] deg."
    )


def make_sun_light(
    sun_direction,
    intensity=SUN_INTENSITY,
    shadow_softness=SUN_SHADOW_SOFTNESS,
):
    sun = kb.DirectionalLight(
        color=kb.get_color("white"),
        shadow_softness=shadow_softness,
        intensity=intensity,
    )
    sun.position = tuple(np.asarray(sun_direction) * SUN_DISTANCE)
    sun.look_at((0.0, 0.0, 0.0))
    return sun


def make_camera_fill_light(
    position=CAMERA_POSITION,
    target=LOOK_AT,
    intensity=FILL_INTENSITY,
    shadow_softness=0.3,
):
    fill = kb.DirectionalLight(
        color=kb.Color(0.75, 0.82, 1.0),
        shadow_softness=shadow_softness,
        intensity=intensity,
    )
    fill.position = tuple(position)
    fill.look_at(tuple(target))
    return fill


# =============================================================================
# CAMERA TRAJECTORIES
# =============================================================================

def camera_angles_from_position(cam_pos, target):
    """
    Return azimuth/elevation of camera position around target.

    Convention:
      azimuth = 0 deg at target -Y direction
      azimuth increases toward +X
      elevation = asin(z / radius)
    """
    rel = np.asarray(cam_pos, dtype=float) - np.asarray(target, dtype=float)
    r = np.linalg.norm(rel)

    if r < 1e-12:
        raise ValueError("Camera cannot coincide with target.")

    elevation_deg = math.degrees(
        math.asin(np.clip(rel[2] / r, -1.0, 1.0))
    )
    azimuth_deg = math.degrees(
        math.atan2(rel[0], -rel[1])
    ) % 360.0

    return azimuth_deg, elevation_deg


def generate_camera_ring(
    num_snapshots,
    radius,
    elevation_deg=0.0,
    target=np.zeros(3),
):
    """
    Legacy single-elevation circular orbit.
    View 0 is exactly target + (0, -radius*cos(elev), radius*sin(elev)).
    """
    elev = math.radians(elevation_deg)
    z = radius * math.sin(elev)
    r_xy = radius * math.cos(elev)

    positions = np.zeros((num_snapshots, 3), dtype=float)

    for i in range(num_snapshots):
        az = 2.0 * math.pi * i / num_snapshots

        # Same convention as the original script:
        # azimuth 0 -> -Y, then sweep toward +X.
        x = r_xy * math.sin(az)
        y = -r_xy * math.cos(az)

        positions[i] = np.asarray(target) + np.array([x, y, z])

    return positions


def generate_camera_fibonacci(
    num_snapshots,
    radius,
    max_abs_elevation_deg=70.0,
    target=np.zeros(3),
):
    """
    Generate a full Fibonacci-sphere trajectory.

    Design choices:
    1. Every frame uses approximately uniform equal-area spherical sampling.
    2. No camera is placed exactly on a pole, avoiding look-at degeneracy.
    3. The fixed radius removes scale/distance as an experimental variable.

    max_abs_elevation_deg is kept in the signature for CLI compatibility, but
    Fibonacci mode now intentionally covers the full sphere.
    """
    if num_snapshots < 1:
        raise ValueError("num_snapshots must be >= 1")

    target = np.asarray(target, dtype=float)
    positions = []

    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0

    for i in range(num_snapshots):
        # Standard equal-area Fibonacci sphere.
        z = 1.0 - 2.0 * (i + 0.5) / num_snapshots

        theta = 2.0 * math.pi * i / golden_ratio
        rho = math.sqrt(max(0.0, 1.0 - z * z))

        # Keep the existing azimuth convention: 0 deg is target -Y.
        direction = np.array([
            rho * math.sin(theta),
            -rho * math.cos(theta),
            z,
        ])
        direction /= np.linalg.norm(direction)

        positions.append(target + radius * direction)

    return np.asarray(positions, dtype=float)


def generate_camera_trajectory(args, target):
    if args.trajectory == "ring":
        positions = generate_camera_ring(
            num_snapshots=args.num_snapshots,
            radius=args.orbit_radius,
            elevation_deg=args.orbit_elevation,
            target=target,
        )
    elif args.trajectory == "fibonacci":
        positions = generate_camera_fibonacci(
            num_snapshots=args.num_snapshots,
            radius=args.orbit_radius,
            max_abs_elevation_deg=args.max_abs_elevation,
            target=target,
        )
    else:
        raise ValueError(f"Unknown trajectory mode: {args.trajectory}")

    azimuths = []
    elevations = []

    for pos in positions:
        az, el = camera_angles_from_position(pos, target)
        azimuths.append(az)
        elevations.append(el)

    return (
        positions,
        np.asarray(azimuths, dtype=float),
        np.asarray(elevations, dtype=float),
    )


# =============================================================================
# POSE GEOMETRY
# =============================================================================

def look_at_rotation(
    cam_pos,
    target=np.array([0.0, 0.0, 0.0]),
    up=np.array([0.0, 0.0, 1.0]),
):
    """
    Compute world->camera and camera->world rotation matrices.

    Blender camera convention:
      local +X = image right
      local +Y = image up
      local -Z = forward/look direction
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float).copy()

    forward = target - cam_pos
    forward /= np.linalg.norm(forward)

    # Avoid degeneracy if camera is very close to a pole.
    if abs(np.dot(forward, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)

    R_cam2world = np.column_stack(
        (right, true_up, -forward)
    )
    R_world2cam = R_cam2world.T

    return R_world2cam, R_cam2world


def relative_object_pose(
    obj_pos_world,
    obj_rot_world,
    cam_pos_world,
    R_world2cam,
):
    """
    Compute spacecraft pose relative to camera.

    Returns:
      r_obj2cam : [x,y,z] metres
      q_obj2cam : quaternion [w,x,y,z]
    """
    r_obj2cam = R_world2cam @ (
        obj_pos_world - cam_pos_world
    )

    R_obj2world = obj_rot_world.as_matrix()
    R_obj2cam = R_world2cam @ R_obj2world

    q_xyzw = Rotation.from_matrix(R_obj2cam).as_quat()
    q_wxyz = np.array([
        q_xyzw[3],
        q_xyzw[0],
        q_xyzw[1],
        q_xyzw[2],
    ])

    return r_obj2cam, q_wxyz


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    scipy_random_state = np.random.RandomState(args.seed)
    np.random.seed(args.seed)

    object_name = args.object_name

    asset_dir = args.asset_dir or os.path.join(
        SCRIPT_DIR,
        "assets",
        f"{object_name}_from_fbx",
    )

    if args.output_dir is None:
        if args.trajectory == "fibonacci":
            dataset_name = (
                f"{object_name}_viewrich_"
                f"{args.num_snapshots}f_"
                "fullsphere_"
                f"{args.lighting}_seed{args.seed}"
            )
        else:
            dataset_name = (
                f"{object_name}_ring_"
                f"{args.num_snapshots}f_"
                f"e{int(round(args.orbit_elevation))}_"
                f"{args.lighting}_seed{args.seed}"
            )

        output_dir = os.path.join(
            SCRIPT_DIR,
            "output",
            dataset_name,
        )
    else:
        output_dir = args.output_dir

    lighting = resolve_lighting_config(args)

    object_position = np.array(
        args.initial_position,
        dtype=float,
    )

    if args.num_snapshots < 1:
        raise ValueError("--num-snapshots must be >= 1")

    if args.render_chunk_size < 1:
        raise ValueError("--render-chunk-size must be >= 1")

    if args.orbit_radius <= 0.0:
        raise ValueError("--orbit-radius must be > 0")

    if lighting["sun_min_angle"] > lighting["sun_max_angle"]:
        raise ValueError(
            "--sun-min-angle must be <= --sun-max-angle"
        )

    # Generate camera positions before lighting so frame 0 is the actual
    # reference camera used by the trajectory.
    (
        cam_positions,
        cam_azimuths,
        cam_elevations,
    ) = generate_camera_trajectory(
        args,
        target=object_position,
    )

    num_frames = len(cam_positions)
    frame_start = 1
    frame_end = num_frames

    os.makedirs(os.path.join(output_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "flow"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tmp"), exist_ok=True)

    print("\n============================================================")
    print("VIEW-RICH SPACECRAFT DATASET")
    print("============================================================")
    print(f"[Trajectory] mode={args.trajectory}")
    print(f"[Trajectory] frames={num_frames}")
    print(f"[Trajectory] radius={args.orbit_radius:.3f} m")
    print(
        f"[Trajectory] elevation range="
        f"[{cam_elevations.min():.2f}, {cam_elevations.max():.2f}] deg"
    )
    print(f"[Trajectory] frame-0 position={cam_positions[0]}")
    print(f"[Output] {output_dir}")

    # -------------------------------------------------------------------------
    # 1. SCENE
    # -------------------------------------------------------------------------

    scene = kb.Scene(
        resolution=(1920, 1200),
        frame_start=frame_start,
        frame_end=frame_end,
    )

    # -------------------------------------------------------------------------
    # 2. SIMULATOR
    # -------------------------------------------------------------------------

    simulator = KubricSimulator(
        scene,
        scratch_dir=os.path.join(output_dir, "tmp"),
    )
    scene.gravity = (0.0, 0.0, 0.0)

    # -------------------------------------------------------------------------
    # 3. RENDERER
    # -------------------------------------------------------------------------

    renderer = KubricRenderer(
        scene,
        scratch_dir=os.path.join(output_dir, "tmp"),
    )

    ambient_level = lighting["ambient_level"]
    renderer.ambient_illumination = kb.Color(
        ambient_level,
        ambient_level,
        ambient_level,
    )

    # -------------------------------------------------------------------------
    # 4. CAMERA
    # -------------------------------------------------------------------------

    scene.camera = kb.PerspectiveCamera(
        focal_length=CAMERA_FOCAL_LENGTH_MM,
        sensor_width=CAMERA_SENSOR_WIDTH_MM,
    )

    # -------------------------------------------------------------------------
    # 5. LIGHTING
    # -------------------------------------------------------------------------

    # IMPORTANT: use actual frame-0 camera as the reference.
    reference_camera_position = cam_positions[0]

    if args.sun_direction is None:
        (
            sun_direction,
            sun_position,
            ref_angle_deg,
        ) = sample_sun_direction(
            camera_position=reference_camera_position,
            look_at=object_position,
            min_angle_deg=lighting["sun_min_angle"],
            max_angle_deg=lighting["sun_max_angle"],
            rng=rng,
        )
    else:
        sun_direction = normalize_vector(
            args.sun_direction,
            "--sun-direction",
        )
        sun_position = sun_direction * SUN_DISTANCE

        boresight = normalize_vector(
            object_position - reference_camera_position,
            "reference camera boresight",
        )
        cam_to_sun = normalize_vector(
            sun_position - reference_camera_position,
            "reference camera-to-sun vector",
        )

        ref_angle_deg = math.degrees(
            math.acos(
                np.clip(
                    np.dot(boresight, cam_to_sun),
                    -1.0,
                    1.0,
                )
            )
        )

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

    print(
        f"[Lighting] Sun direction world="
        f"{np.round(sun_direction, 4)}"
    )
    print(
        f"[Lighting] preset={args.lighting}: "
        f"{lighting['description']}"
    )
    print(
        f"[Lighting] frame-0 sun/boresight angle="
        f"{ref_angle_deg:.2f} deg"
    )

    # -------------------------------------------------------------------------
    # 6. STATIC SPACECRAFT
    # -------------------------------------------------------------------------

    if args.random_initial_quaternion:
        object_rotation = Rotation.random(
            random_state=scipy_random_state
        )

        q_xyzw = object_rotation.as_quat()
        obj_quat_wxyz = np.array([
            q_xyzw[3],
            q_xyzw[0],
            q_xyzw[1],
            q_xyzw[2],
        ])
    else:
        obj_quat_wxyz = np.array(
            args.initial_quaternion,
            dtype=float,
        )

        quat_norm = np.linalg.norm(obj_quat_wxyz)
        if quat_norm < 1e-8:
            raise ValueError(
                "--initial-quaternion must not be the zero quaternion"
            )

        obj_quat_wxyz /= quat_norm
        object_rotation = rotation_from_quaternion_wxyz(
            obj_quat_wxyz
        )

    debris = kb.FileBasedObject(
        asset_id="cheops_satellite",
        render_filename=os.path.join(
            asset_dir,
            f"{object_name}.obj",
        ),
        simulation_filename=os.path.join(
            asset_dir,
            f"{object_name}.urdf",
        ),
        position=tuple(object_position),
        mass=10.0,
        scale=MM_TO_M,
    )

    scene += debris

    material_texture_map = (
        apply_mtl_textures_to_imported_object(
            asset_id="cheops_satellite",
            mtl_path=os.path.join(
                asset_dir,
                f"{object_name}.mtl",
            ),
        )
    )

    debris.quaternion = quaternion_wxyz_to_kb(
        obj_quat_wxyz
    )
    debris.velocity = (0.0, 0.0, 0.0)
    debris.angular_velocity = (0.0, 0.0, 0.0)

    # -------------------------------------------------------------------------
    # 7. KEYFRAME CAMERA / FILL AND BUILD LABELS
    # -------------------------------------------------------------------------

    pose_records = []

    for i in range(num_frames):
        frame_id = frame_start + i
        cam_pos = cam_positions[i]

        scene.camera.position = tuple(cam_pos)
        scene.camera.look_at(tuple(object_position))
        scene.camera.keyframe_insert("position", frame_id)
        scene.camera.keyframe_insert("quaternion", frame_id)

        if fill is not None:
            fill.position = tuple(cam_pos)
            fill.look_at(tuple(object_position))
            fill.keyframe_insert("position", frame_id)
            fill.keyframe_insert("quaternion", frame_id)

        R_world2cam, R_cam2world = look_at_rotation(
            cam_pos,
            target=object_position,
        )

        r_obj2cam, q_obj2cam = relative_object_pose(
            obj_pos_world=object_position,
            obj_rot_world=object_rotation,
            cam_pos_world=cam_pos,
            R_world2cam=R_world2cam,
        )

        boresight = object_position - cam_pos
        boresight /= np.linalg.norm(boresight)

        cam_to_sun = (
            sun_direction * SUN_DISTANCE
        ) - cam_pos
        cam_to_sun /= np.linalg.norm(cam_to_sun)

        view_sun_angle = math.degrees(
            math.acos(
                np.clip(
                    np.dot(boresight, cam_to_sun),
                    -1.0,
                    1.0,
                )
            )
        )

        pose_records.append({
            "r_obj2cam": r_obj2cam,
            "q_obj2cam": q_obj2cam,
            "camera_position_world": cam_pos,
            "camera_azimuth_deg": float(cam_azimuths[i]),
            "camera_elevation_deg": float(cam_elevations[i]),
            "R_world2cam": R_world2cam,
            "R_cam2world": R_cam2world,
            "view_sun_angle_deg": view_sun_angle,
        })

    # Save the camera trajectory immediately, even before rendering.
    # This is handy for plotting/debugging.
    trajectory_json = []

    for i, rec in enumerate(pose_records):
        trajectory_json.append({
            "frame_index": i,
            "filename": f"{i:06d}.png",
            "camera_position_world": rec[
                "camera_position_world"
            ].tolist(),
            "camera_azimuth_deg": round(
                rec["camera_azimuth_deg"],
                6,
            ),
            "camera_elevation_deg": round(
                rec["camera_elevation_deg"],
                6,
            ),
            "R_world2cam": rec[
                "R_world2cam"
            ].tolist(),
        })

    with open(
        os.path.join(output_dir, "camera_trajectory.json"),
        "w",
    ) as f:
        json.dump(trajectory_json, f, indent=2)

    # -------------------------------------------------------------------------
    # 8. PHYSICS
    # -------------------------------------------------------------------------

    print(
        "\n[Physics] Running PyBullet "
        "(spacecraft remains stationary)..."
    )
    simulator.run()

    # -------------------------------------------------------------------------
    # 9. RENDER
    # -------------------------------------------------------------------------

    print("[Render] Running Blender renderer in chunks...")

    renderer.save_state(
        os.path.join(
            output_dir,
            "blender_scene.blend",
        )
    )

    depth_sum = 0.0
    depth_min = np.inf
    depth_max = -np.inf
    valid_count = 0
    total_count = 0

    for chunk_start, chunk_end, chunk_frames in iter_frame_chunks(
        frame_start,
        frame_end,
        args.render_chunk_size,
    ):
        chunk_offset = chunk_start - frame_start
        print(
            f"[Render] Chunk frames {chunk_start}-{chunk_end} "
            f"({len(chunk_frames)} frames)"
        )

        clear_render_scratch(os.path.join(output_dir, "tmp"))
        frames_dict = renderer.render(frames=chunk_frames)

        # ---------------------------------------------------------------------
        # 10. DEPTH
        # ---------------------------------------------------------------------

        print(
            f"[Depth] Clamping chunk {chunk_start}-{chunk_end} "
            f"to {MAX_DEPTH:.1f} m ..."
        )

        depth_raw = frames_dict["depth"]
        depth_clamped = clamp_depth_batch(
            depth_raw,
            max_depth=MAX_DEPTH,
        )
        depth_f64 = depth_clamped.astype(np.float64)

        valid_mask = (
            np.isfinite(depth_clamped)
            & (depth_clamped > 0.0)
            & (depth_clamped < MAX_DEPTH)
        )
        if np.any(valid_mask):
            valid_values = depth_clamped[valid_mask]
            depth_sum += float(np.sum(valid_values))
            depth_min = min(depth_min, float(np.min(valid_values)))
            depth_max = max(depth_max, float(np.max(valid_values)))
            valid_count += int(valid_values.size)
        total_count += int(depth_clamped.size)

        # ---------------------------------------------------------------------
        # 11. EXPORT RGB / DEPTH / FLOW
        # ---------------------------------------------------------------------

        print(
            f"[Export] Writing chunk {chunk_start}-{chunk_end} "
            "RGB, depth, flow ..."
        )

        write_rgb_batch(
            frames_dict["rgba"],
            os.path.join(output_dir, "image"),
            start_index=chunk_offset,
        )

        write_tiff_depth_batch(
            depth_f64,
            os.path.join(output_dir, "depth"),
            start_index=chunk_offset,
        )

        write_flo_batch(
            frames_dict["forward_flow"],
            os.path.join(output_dir, "flow"),
            start_index=chunk_offset,
        )

        del frames_dict, depth_raw, depth_clamped, depth_f64

    # -------------------------------------------------------------------------
    # 12. POSE LABELS
    # -------------------------------------------------------------------------

    pose_labels = []

    for frame_idx in range(num_frames):
        rec = pose_records[frame_idx]

        pose_labels.append({
            "filename": f"{frame_idx:06d}.png",

            # Existing SPEED-style quantities.
            "q_obj2cam": rec["q_obj2cam"].tolist(),
            "r_obj2cam": rec["r_obj2cam"].tolist(),

            # Fixed world illumination.
            "sun_direction_world": sun_direction.tolist(),
            "sun_boresight_angle_deg": round(
                rec["view_sun_angle_deg"],
                4,
            ),

            # Camera trajectory diagnostics.
            "camera_position_world": rec[
                "camera_position_world"
            ].tolist(),
            "camera_azimuth_deg": round(
                rec["camera_azimuth_deg"],
                4,
            ),
            "camera_elevation_deg": round(
                rec["camera_elevation_deg"],
                4,
            ),
        })

    with open(
        os.path.join(output_dir, "pose_labels.json"),
        "w",
    ) as f:
        json.dump(pose_labels, f, indent=2)

    # -------------------------------------------------------------------------
    # 13. METADATA
    # -------------------------------------------------------------------------

    kb.file_io.write_json(
        filename=os.path.join(
            output_dir,
            "metadata.json",
        ),
        data={
            "metadata": kb.get_scene_metadata(scene),

            "generation_config": {
                "seed": args.seed,
                "lighting_preset": args.lighting,
                "asset_dir": asset_dir,
                "output_dir": output_dir,

                "trajectory": args.trajectory,
                "num_snapshots": num_frames,
                "render_chunk_size": args.render_chunk_size,
                "orbit_radius_m": args.orbit_radius,

                "max_abs_elevation_deg": (
                    None
                    if args.trajectory == "fibonacci"
                    else args.max_abs_elevation
                ),
                "orbit_elevation_deg": (
                    args.orbit_elevation
                    if args.trajectory == "ring"
                    else None
                ),

                "initial_position": object_position.tolist(),
                "initial_quaternion_wxyz": (
                    obj_quat_wxyz.tolist()
                ),
                "random_initial_quaternion": (
                    args.random_initial_quaternion
                ),
                "explicit_sun_direction": (
                    args.sun_direction is not None
                ),
            },

            "camera": {
                **kb.get_camera_info(scene.camera),
                "focal_length_mm": CAMERA_FOCAL_LENGTH_MM,
                "sensor_width_mm": CAMERA_SENSOR_WIDTH_MM,
                "sensor_height_mm": CAMERA_SENSOR_HEIGHT_MM,
                "fov_horizontal_deg": CAMERA_FOV_H_DEG,
            },

            "trajectory": {
                "mode": args.trajectory,
                "description": (
                    "stationary spacecraft; camera moves on a "
                    "full Fibonacci sphere"
                    if args.trajectory == "fibonacci"
                    else
                    "stationary spacecraft; camera moves on "
                    "a single circular ring"
                ),
                "frame_0_is_legacy_anchor": False,
                "num_snapshots": num_frames,
                "radius_m": args.orbit_radius,
                "camera_positions_world": (
                    cam_positions.tolist()
                ),
                "camera_azimuths_deg": (
                    cam_azimuths.tolist()
                ),
                "camera_elevations_deg": (
                    cam_elevations.tolist()
                ),
            },

            "material_textures": material_texture_map,

            "lighting": {
                "type": (
                    "DirectionalLight Sun + camera-following "
                    "fill + ambient"
                ),
                "preset_description": lighting[
                    "description"
                ],
                "sun_direction_world": (
                    sun_direction.tolist()
                ),
                "sun_reference_boresight_angle_deg": round(
                    ref_angle_deg,
                    4,
                ),
                "shadow_softness": lighting[
                    "sun_shadow_softness"
                ],
                "sun_intensity": lighting[
                    "sun_intensity"
                ],
                "fill_intensity": lighting[
                    "fill_intensity"
                ],
                "ambient_illumination": lighting[
                    "ambient_level"
                ],
                "note": (
                    "Sun is sampled once using frame 0 as the "
                    "reference and stays fixed in world coordinates. "
                    "The optional fill light follows each camera."
                ),
            },

            "instances": kb.get_instance_info(scene),
        },
    )

    # -------------------------------------------------------------------------
    # 14. STATS
    # -------------------------------------------------------------------------

    print("\n[Depth Stats] valid spacecraft/scene depth pixels")

    if valid_count > 0:
        print(
            f"  mean: {depth_sum / valid_count:.4f} m"
        )
        print(
            f"  min:  {depth_min:.4f} m"
        )
        print(
            f"  max:  {depth_max:.4f} m"
        )
        print(
            f"  valid: {valid_count} / "
            f"{total_count} pixels "
            f"({100.0 * valid_count / total_count:.2f}%)"
        )
    else:
        print("  WARNING: no valid depth pixels found.")

    print("\n[Camera Coverage]")
    print(
        f"  azimuth span: "
        f"{cam_azimuths.min():.2f} -> "
        f"{cam_azimuths.max():.2f} deg"
    )
    print(
        f"  elevation span: "
        f"{cam_elevations.min():.2f} -> "
        f"{cam_elevations.max():.2f} deg"
    )
    print(
        f"  frame 0: az={cam_azimuths[0]:.2f}, "
        f"el={cam_elevations[0]:.2f}, "
        f"pos={np.round(cam_positions[0], 4)}"
    )

    print(
        "\n[DONE] View-rich snapshot generation complete:"
    )
    print(output_dir)


if __name__ == "__main__":
    main()
