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
        /usr/bin/python3 reconstruction-tracking-synthetic/generate_spacecraft.py

Docker (PowerShell):
    docker run --rm --interactive --volume "%cd%:/kubric" kubricdockerhub/kubruntu \
        /usr/bin/python3 reconstruction-tracking-synthetic/generate_spacecraft.py
"""

import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.simulator.pybullet import PyBullet as KubricSimulator
import numpy as np
import os
import json
import math
from PIL import Image
import imageio
from scipy.spatial.transform import Rotation
from scene_setup import setup_scene_actors


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
    """
    Replace inf/nan/<=0/beyond max_depth with 0.0 (invalid sentinel).
    Background pixels from Kubric come in as inf — zeroed here.
    """
    depth = np.array(depth_frames, dtype=np.float32)
    invalid = ~np.isfinite(depth) | (depth <= 0.0) | (depth > max_depth)
    depth[invalid] = 9999.0
    return depth


# =============================================================================
# LIGHTING HELPERS  (SPEED-UE-Cube conventions)
# =============================================================================
def sample_sun_direction(
    camera_position=CAMERA_POSITION,
    look_at=LOOK_AT,
    sun_distance=SUN_DISTANCE,
    min_angle_deg=SUN_MIN_ANGLE_FROM_BORESIGHT_DEG,
    max_attempts=1000
):
    """
    Sample a random sun position (as a unit direction scaled by sun_distance)
    such that the angle between the camera boresight and the direction FROM
    the camera TO the sun is >= min_angle_deg.

    This correctly replicates SPEED-UE-Cube Constraint 2:
        "The angle between the camera boresight and the vector from the
         camera to the Sun must be >= 75 degrees."

    Args:
        camera_position (np.ndarray): Camera position in world space.
        look_at         (np.ndarray): Point the camera is looking at.
        sun_distance    (float)     : Distance to place the sun (arbitrary for
                                      DirectionalLight, but needed to compute
                                      the camera-to-sun vector correctly).
        min_angle_deg   (float)     : Minimum angle in degrees between camera
                                      boresight and camera-to-sun direction.
        max_attempts    (int)       : Max rejection sampling attempts.

    Returns:
        sun_dir      (np.ndarray): Unit vector pointing FROM scene TOWARD sun.
                                   Use as: sun.position = tuple(sun_dir * sun_distance)
                                           sun.look_at((0, 0, 0))
        sun_position (np.ndarray): Actual sun position in world space.
        angle_deg    (float)     : Actual angle between boresight and sun (for logging).
    """
    # --- Camera boresight: unit vector FROM camera TOWARD look_at target ---
    boresight = look_at - camera_position
    boresight /= np.linalg.norm(boresight)   # = [0, 1, 0] for your setup

    min_angle_rad = math.radians(min_angle_deg)

    for attempt in range(max_attempts):
        # 1. Sample a random unit vector on the sphere
        sun_dir = np.random.randn(3)
        sun_dir /= np.linalg.norm(sun_dir)

        # 2. Compute actual sun position in world space
        sun_position = sun_dir * sun_distance

        # 3. Compute direction FROM CAMERA TO SUN (this is what matters)
        #    Not just sun_dir — because camera is not at the origin
        cam_to_sun = sun_position - camera_position
        cam_to_sun /= np.linalg.norm(cam_to_sun)

        # 4. Angle between boresight and camera-to-sun direction
        cos_angle = np.clip(np.dot(boresight, cam_to_sun), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))

        # 5. Accept if constraint satisfied
        if angle_deg >= min_angle_deg:
            return sun_dir, sun_position, angle_deg

    raise RuntimeError(
        f"Could not sample valid sun direction after {max_attempts} attempts. "
        f"Check min_angle_deg={min_angle_deg}."
    )


def make_sun_light(sun_direction):
    """
    Create a kb.DirectionalLight pointing in sun_direction.

    Kubric's look_at() sets the rotation so the light points FROM
    sun_position TOWARD the look_at point. We place the sun far away
    along the OPPOSITE of sun_direction so it illuminates the scene
    from sun_direction.

    Args:
        sun_direction (np.ndarray): unit vector [x, y, z], direction light comes FROM

    Returns:
        kb.DirectionalLight
    """
    sun_position = tuple(sun_direction * SUN_DISTANCE)

    sun = kb.DirectionalLight(
        color=kb.get_color("white"),
        # Physically correct: Sun's angular radius as seen from space ~ 0.265 deg
        # shadow_softness = tan(angular_radius_rad) ~ 0.00462
        # Ref: SPEED-UE-Cube paper + solar angular diameter literature
        shadow_softness=SUN_SHADOW_SOFTNESS,
        intensity=15.0,          # tune to your spacecraft material
    )
    sun.position = sun_position
    sun.look_at((0, 0, 0))     # always point toward scene origin (spacecraft)
    return sun


# =============================================================================
# TRAJECTORY HELPERS
# =============================================================================
def generate_trajectory(num_frames,
                        initial_position=(0.0, 0.0, 0.0),
                        linear_velocity=(0.5, 0.1, 0.0),
                        angular_velocity_deg_per_frame=(1.5, 0.5, 2.0),
                        initial_quaternion=None):
    """
    Generate a simple free-space trajectory (no gravity, no orbital mechanics).
    Matches the original code's tumbling + drifting behaviour but exposes
    all parameters explicitly and returns per-frame pose labels.

    Args:
        num_frames (int): total number of frames
        initial_position (tuple): starting [x, y, z] in metres
        linear_velocity (tuple): constant [vx, vy, vz] in metres/frame
        angular_velocity_deg_per_frame (tuple): rotation per frame around [x, y, z] in degrees
        initial_quaternion (np.ndarray or None): [w, x, y, z], random if None

    Returns:
        positions   (np.ndarray): shape (num_frames, 3)
        quaternions (np.ndarray): shape (num_frames, 4) as [w, x, y, z]
    """
    if initial_quaternion is None:
        # Uniformly random SO(3) — same subgroup algorithm as SPEED / SPEED-UE-Cube
        r0 = Rotation.random()
    else:
        r0 = Rotation.from_quat([initial_quaternion[1],   # scipy uses [x,y,z,w]
                                  initial_quaternion[2],
                                  initial_quaternion[3],
                                  initial_quaternion[0]])

    pos = np.array(initial_position, dtype=float)
    vel = np.array(linear_velocity, dtype=float)

    # Angular velocity as a rotation applied each frame
    ang_vel_rad = np.radians(angular_velocity_deg_per_frame)
    delta_rot = Rotation.from_rotvec(ang_vel_rad)   # rotation per frame

    positions   = np.zeros((num_frames, 3))
    quaternions = np.zeros((num_frames, 4))   # [w, x, y, z]

    current_rot = r0
    for f in range(num_frames):
        positions[f] = pos
        q_xyzw = current_rot.as_quat()              # scipy: [x, y, z, w]
        quaternions[f] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]  # → [w,x,y,z]

        pos         = pos + vel
        current_rot = delta_rot * current_rot        # accumulate rotation

    return positions, quaternions


# =============================================================================
# MAIN
# =============================================================================
def main():

    # -------------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------------
    object_name = "Cheops"
    asset_dir   = f"reconstruction-tracking-synthetic/assets/{object_name}"
    output_dir  = f"/dataset/kubric_sim/reconstruction-tracking-synthetic/{object_name}/"

    num_frames  = 48        # 1-second video at 24 fps
    frame_start = 1
    frame_end   = num_frames

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
    renderer.ambient_illumination = kb.Color(0.0, 0.0, 0.0)

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
    scene.camera.position = (0, -10, 0)     # camera sits 10 m along -Y axis
    scene.camera.look_at((0, 0, 0))         # always pointing at spacecraft (origin)

    # -------------------------------------------------------------------------
    # 5. LIGHTING — SPEED-UE-Cube conventions
    #    Single DirectionalLight (Sun), no ambient.
    #    Sun direction sampled once per sequence (fixed lighting for trajectory).
    #    Constraint: angle(sun_dir, camera_boresight) >= 75 deg
    #    Ref: SPEED-UE-Cube paper, Section "Training Dataset Pose Labels", Constraint 2
    # -------------------------------------------------------------------------
    sun_direction, sun_position, angle_deg = sample_sun_direction(
        min_angle_deg=SUN_MIN_ANGLE_FROM_BORESIGHT_DEG
    )
    sun = make_sun_light(sun_direction)
    scene += sun

    print(f"[Lighting] Sun direction (world): {np.round(sun_direction, 4)}")
    angle_from_boresight = math.degrees(
        math.acos(np.clip(np.dot(sun_direction, [0, 1, 0]), -1, 1)))
    print(f"[Lighting] Angle from camera boresight: {angle_from_boresight:.2f} deg "
          f"(must be >= {SUN_MIN_ANGLE_FROM_BORESIGHT_DEG} deg) ✓")

    # -------------------------------------------------------------------------
    # 6. SPACECRAFT OBJECT
    # -------------------------------------------------------------------------
    debris = kb.FileBasedObject(
        asset_id="cheops_satellite",
        render_filename=f"{asset_dir}/{object_name}.obj",
        simulation_filename=f"{asset_dir}/{object_name}.urdf",
        # bounds omitted — defaults to ((0,0,0),(0,0,0))
        # Safe to omit: camera is fixed, collision uses URDF, no auto-framing needed
        position=(0.0, 0.0, 0.0),
        mass=10.0,
        scale=MM_TO_M,
    )
    scene += debris

    # -------------------------------------------------------------------------
    # 7. TRAJECTORY — pre-compute per-frame poses
    #    Simple free-space tumbling + drifting (no orbital mechanics needed)
    #    Orientation: uniform SO(3) sampling via scipy subgroup algorithm
    #    Ref: SPEED-UE-Cube paper, Section "Training Dataset Pose Labels"
    # -------------------------------------------------------------------------
    positions, quaternions = generate_trajectory(
        num_frames=num_frames,
        initial_position=(0.0, 0.0, 0.0),
        linear_velocity=(0.5, 0.1, 0.0),           # metres per frame
        angular_velocity_deg_per_frame=(1.5, 0.5, 2.0),
        initial_quaternion=None,                    # random initial orientation
    )



    # Give the object a random starting rotation
    debris.quaternion = kb.random_rotation()
    # Give it a slight linear velocity (drifting)
    debris.velocity = (0.5, 0.1, 0.0)
    # Give it an angular velocity (tumbling)
    debris.angular_velocity = (1.5, 0.5, 2.0)

    # -------------------------------------------------------------------------
    # 8. RUN PHYSICS SIMULATION (PyBullet)
    #    Note: physics sim respects keyframed poses above.
    #    Zero gravity ensures no drift beyond what we specified.
    # -------------------------------------------------------------------------
    print("\n[Physics] Running PyBullet simulation...")
    simulator.run()

    # -------------------------------------------------------------------------
    # 9. RUN RENDERING (Blender)
    # -------------------------------------------------------------------------
    print("[Render] Running Blender renderer...")
    renderer.save_state(f"{output_dir}/blender_scene.blend")   # save for debugging
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
    # 11. EXPORT DATA
    # -------------------------------------------------------------------------
    print("\n[Export] Writing RGB, depth, flow ...")
    write_rgb_batch(frames_dict["rgba"],        f"{output_dir}/image/")
    write_tiff_depth_batch(depth_f64,           f"{output_dir}/depth/")
    write_flo_batch(frames_dict["forward_flow"],f"{output_dir}/flow/")

    # -------------------------------------------------------------------------
    # 12. SAVE POSE LABELS (SPEED-UE-Cube format)
    #     Includes per-frame quaternion, position, and sun direction
    # -------------------------------------------------------------------------
    pose_labels = []
    for frame_idx in range(num_frames):
        pose_labels.append({
            "filename": f"{frame_idx:06d}.png",
            # Quaternion [w, x, y, z] — relative orientation of spacecraft w.r.t. camera
            "q_obj2cam": quaternions[frame_idx].tolist(),
            # Position [x, y, z] in metres — spacecraft position in camera frame
            "r_obj2cam": positions[frame_idx].tolist(),
            # Sun direction unit vector in world frame
            # Useful for lighting-aware pose estimation
            "sun_direction_world": sun_direction.tolist(),
            # Angle between sun and camera boresight (sanity check)
            "sun_boresight_angle_deg": round(angle_from_boresight, 4),
        })

    with open(f"{output_dir}/pose_labels.json", "w") as f:
        json.dump(pose_labels, f, indent=2)

    # -------------------------------------------------------------------------
    # 13. SCENE METADATA
    # -------------------------------------------------------------------------
    kb.file_io.write_json(filename=f"{output_dir}/metadata.json", data={
        "metadata":  kb.get_scene_metadata(scene),
        "camera": {
            **kb.get_camera_info(scene.camera),
            # Explicit camera intrinsics for SPEED-UE-Cube reproducibility
            "focal_length_mm":    CAMERA_FOCAL_LENGTH_MM,
            "sensor_width_mm":    CAMERA_SENSOR_WIDTH_MM,
            "sensor_height_mm":   CAMERA_SENSOR_HEIGHT_MM,
            "fov_horizontal_deg": CAMERA_FOV_H_DEG,
        },
        "lighting": {
            "type":                   "DirectionalLight (Sun)",
            "sun_direction_world":    sun_direction.tolist(),
            "sun_boresight_angle_deg": round(angle_from_boresight, 4),
            "shadow_softness":        SUN_SHADOW_SOFTNESS,
            "sun_angular_radius_deg": SUN_ANGULAR_RADIUS_DEG,
            "ambient_illumination":   0.0,
            "note": (
                "Lighting follows SPEED-UE-Cube conventions: single directional sun lamp, "
                "zero ambient, sun direction constrained to >= 75 deg from camera boresight. "
                "Ref: SPEED-UE-Cube paper, Section Training Dataset Pose Labels, Constraint 2."
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

    print("\n✅ Dataset generation complete! Check:", output_dir)


if __name__ == "__main__":
    main()
