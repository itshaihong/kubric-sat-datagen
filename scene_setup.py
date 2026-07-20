"""
scene_setup.py
Combined camera + object + lighting setup for Kubric spacecraft trajectory simulation.
Replaces the separate camera and object keyframing blocks.
"""

import bpy
import math
import numpy as np
import kubric as kb

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
# SECTION 1 — CAMERA SETUP + TRAJECTORY KEYFRAMING
# =============================================================================

def setup_camera_and_keyframe(
    scene,
    camera_positions,       # np.ndarray (N, 3)
    camera_look_ats,        # np.ndarray (N, 3)
    focal_length_mm,
    sensor_width_mm,
):
    """
    Creates the Kubric PerspectiveCamera, adds it to the scene,
    then keyframes its position and rotation at every frame.

    The initial position/look_at is set from frame-0 of the trajectory
    so the Kubric camera object and the Blender object stay in sync.

    Args:
        scene:              kb.Scene instance
        camera_positions:   (N, 3) array of camera world positions in meters
        camera_look_ats:    (N, 3) array of look-at targets in meters
        focal_length_mm:    e.g. 17.5217
        sensor_width_mm:    e.g. 11.2512

    Returns:
        camera:             the kb.PerspectiveCamera added to the scene
        blender_cam_name:   the Blender object name (for debugging)
    """

    # --- Create Kubric camera, initialised at frame-0 pose ---
    camera = kb.PerspectiveCamera(
        focal_length=focal_length_mm,
        sensor_width=sensor_width_mm,
    )
    camera.position = tuple(camera_positions[0])   # frame-0 position
    camera.look_at(tuple(camera_look_ats[0]))       # frame-0 look-at
    scene += camera
    scene.camera = camera

    # --- Find the camera's Blender object name ---
    blender_cam_name = None
    print("\n[DEBUG] All objects in Blender scene:")
    for obj in bpy.data.objects:
        print(f"  name='{obj.name}'  type={obj.type}  location={list(obj.location)}")
        if obj.type == 'CAMERA':
            blender_cam_name = obj.name

    if blender_cam_name is None:
        raise RuntimeError("No CAMERA object found in Blender scene after adding kb.PerspectiveCamera.")
    print(f"[Camera] Blender camera object name: '{blender_cam_name}'")

    # --- Keyframe every frame ---
    print(f"[Camera] Keyframing {len(camera_positions)} frames...")
    for i, (pos, look_at) in enumerate(zip(camera_positions, camera_look_ats)):
        kubric_frame = scene.frame_start + i
        _keyframe_camera(
            blender_cam_name=blender_cam_name,
            position=tuple(pos),
            look_at_point=tuple(look_at),
            frame=kubric_frame,
        )

    print(f"[Camera] Done. Trajectory type covers frames "
          f"{scene.frame_start} → {scene.frame_start + len(camera_positions) - 1}")

    return camera, blender_cam_name


def _keyframe_camera(blender_cam_name, position, look_at_point, frame):
    """
    Low-level bpy keyframe insertion for a camera.
    Sets location and rotation_quaternion at the given frame.
    """
    import mathutils

    cam = bpy.data.objects.get(blender_cam_name)
    if cam is None:
        available = [o.name for o in bpy.data.objects if o.type == 'CAMERA']
        raise ValueError(
            f"Camera '{blender_cam_name}' not found. Available: {available}"
        )

    # Position
    cam.location = position

    # Rotation — point camera toward look_at_point
    cam_pos = np.array(position)
    target  = np.array(look_at_point)
    forward = target - cam_pos
    norm    = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError(
            f"Camera position {position} and look_at {look_at_point} are too close."
        )
    forward /= norm

    # Blender camera looks along its local -Z axis
    # rotation_difference gives us the quaternion to rotate -Z onto forward
    rot = mathutils.Vector((0.0, 0.0, -1.0)).rotation_difference(
          mathutils.Vector(forward.tolist()))

    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = rot

    cam.keyframe_insert(data_path="location",            frame=frame)
    cam.keyframe_insert(data_path="rotation_quaternion", frame=frame)


# =============================================================================
# SECTION 2 — OBJECT SETUP + TRAJECTORY KEYFRAMING
# =============================================================================

def setup_object_and_keyframe(
    scene,
    asset_id,               # str  e.g. "cheops_satellite"
    render_filename,        # str  path to .obj
    simulation_filename,    # str  path to .urdf
    positions,              # np.ndarray (N, 3)  in meters
    quaternions_xyzw,       # np.ndarray (N, 4)  scipy convention (x,y,z,w)
    mass=10.0,
    scale=0.001,            # MM_TO_M by default
):
    """
    Creates the Kubric FileBasedObject, adds it to the scene,
    then keyframes its position and quaternion at every frame.

    Args:
        scene:               kb.Scene instance
        asset_id:            Kubric asset identifier string
        render_filename:     path to .obj file
        simulation_filename: path to .urdf file
        positions:           (N, 3) array of world positions in meters
        quaternions_xyzw:    (N, 4) array of quaternions in scipy (x,y,z,w) order
        mass:                object mass in kg for PyBullet
        scale:               uniform scale factor (e.g. 0.001 for mm → m)

    Returns:
        debris:              the kb.FileBasedObject added to the scene
    """

    # --- Create object at frame-0 pose ---
    q0_xyzw = quaternions_xyzw[0]
    debris = kb.FileBasedObject(
        asset_id=asset_id,
        render_filename=render_filename,
        simulation_filename=simulation_filename,
        position=tuple(positions[0]),
        quaternion=kb.Quaternion(
            w=float(q0_xyzw[3]),
            x=float(q0_xyzw[0]),
            y=float(q0_xyzw[1]),
            z=float(q0_xyzw[2]),
        ),
        mass=mass,
        scale=scale,
    )
    scene += debris

    # --- Verify the Blender object name ---
    blender_obj = bpy.data.objects.get(asset_id)
    if blender_obj is None:
        # Blender may have appended .001 etc — search by type MESH
        mesh_names = [o.name for o in bpy.data.objects if o.type == 'MESH']
        print(f"[Object] WARNING: '{asset_id}' not found directly. "
              f"MESH objects in scene: {mesh_names}")
        if mesh_names:
            actual_name = mesh_names[0]
            print(f"[Object] Using '{actual_name}' instead.")
        else:
            raise RuntimeError("No MESH object found in Blender scene.")
    else:
        actual_name = asset_id

    print(f"[Object] Blender object name: '{actual_name}'")

    # --- Keyframe every frame ---
    print(f"[Object] Keyframing {len(positions)} frames...")
    for i, (pos, quat_xyzw) in enumerate(zip(positions, quaternions_xyzw)):
        kubric_frame = scene.frame_start + i

        # scipy (x,y,z,w) → bpy (w,x,y,z)
        quat_wxyz = (
            float(quat_xyzw[3]),
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
        )

        _keyframe_object(
            blender_obj_name=actual_name,
            position=tuple(pos),
            quaternion_wxyz=quat_wxyz,
            frame=kubric_frame,
        )

        if i % 10 == 0 or i == len(positions) - 1:
            print(f"  Frame {kubric_frame}: "
                  f"pos={np.round(pos, 3)}  "
                  f"quat_wxyz={np.round(quat_wxyz, 3)}")

    print(f"[Object] Done. Trajectory covers frames "
          f"{scene.frame_start} → {scene.frame_start + len(positions) - 1}")

    return debris


def _keyframe_object(blender_obj_name, position, quaternion_wxyz, frame):
    """
    Low-level bpy keyframe insertion for a mesh object.
    Sets location and rotation_quaternion at the given frame.

    Args:
        blender_obj_name:  exact name in bpy.data.objects
        position:          (x, y, z) tuple in meters
        quaternion_wxyz:   (w, x, y, z) tuple  — bpy convention
        frame:             integer Blender frame number
    """
    obj = bpy.data.objects.get(blender_obj_name)
    if obj is None:
        available = [o.name for o in bpy.data.objects]
        raise ValueError(
            f"Object '{blender_obj_name}' not found in Blender scene.\n"
            f"Available objects: {available}"
        )

    obj.location = position
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = quaternion_wxyz

    obj.keyframe_insert(data_path="location",            frame=frame)
    obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)


# =============================================================================
# SECTION 3 — LIGHTING SETUP (fixed sun, sampled once from frame-0 boresight)
# =============================================================================

def setup_sun_light(
    scene,
    camera_positions,       # np.ndarray (N, 3) — only frame 0 is used
    camera_look_ats,        # np.ndarray (N, 3) — only frame 0 is used
    min_angle_deg,          # float, e.g. 75.0
    sun_intensity,          # float, e.g. 5.0
    shadow_softness=0.00462,  # tan(0.265 deg) — solar angular radius
    max_attempts=1000,
):
    """
    Samples a random sun direction validated against the frame-0 camera boresight,
    then creates a fixed DirectionalLight for the entire sequence.

    The sun is NEVER moved after this — one call, fixed for all frames.

    Args:
        scene:            kb.Scene instance
        camera_positions: (N,3) array — frame 0 position used for constraint check
        camera_look_ats:  (N,3) array — frame 0 look-at used for constraint check
        min_angle_deg:    minimum angle between sun and camera boresight (default 75)
        sun_intensity:    DirectionalLight intensity value
        shadow_softness:  angular radius of sun disk in radians
        max_attempts:     rejection sampling limit

    Returns:
        sun:              the kb.DirectionalLight added to the scene
        sun_direction:    (3,) unit vector in world space
        angle_deg:        actual angle from frame-0 boresight in degrees
    """

    cam_pos_f0   = np.array(camera_positions[0])
    look_at_f0   = np.array(camera_look_ats[0])

    sun_direction, sun_position, angle_deg = sample_sun_direction(
        camera_position=cam_pos_f0,
        look_at=look_at_f0,
        min_angle_deg=min_angle_deg,
        max_attempts=max_attempts,
    )

    sun = kb.DirectionalLight(
        color=kb.get_color("white"),
        shadow_softness=shadow_softness,
        intensity=sun_intensity,
    )
    sun.position = tuple(sun_position)   # position along -sun_direction * 100
    sun.look_at((0.0, 0.0, 0.0))        # points toward scene origin along sun_direction
    scene += sun

    print(f"[Lighting] Sun direction (world): {np.round(sun_direction, 4)}")
    print(f"[Lighting] Sun position  (world): {np.round(sun_position,  2)}")
    print(f"[Lighting] Angle from frame-0 boresight: {angle_deg:.2f} deg "
          f"(must be >= {min_angle_deg} deg) ✓")
    print(f"[Lighting] Sun is FIXED for the entire sequence.")

    return sun, sun_direction, angle_deg


# =============================================================================
# SECTION 4 — COMBINED ENTRY POINT (drop-in replacement for your main() block)
# =============================================================================

def setup_scene_actors(
    scene,
    cfg,
    num_frames,
    asset_dir,
    object_name,
    positions,              # (N,3) object world positions
    quaternions_xyzw,       # (N,4) object quaternions in scipy (x,y,z,w) order
):
    """
    Single function that replaces all the separate camera / object / lighting
    setup blocks in main(). Call this once after scene creation.

    Order of operations (matters for bpy object name detection):
        1. Camera  → added first  → Blender name found automatically
        2. Sun     → fixed once   → validated against frame-0 boresight
        3. Object  → added last   → Blender name found automatically
        4. All three keyframed    → camera + object only (sun is static)

    Args:
        scene:             kb.Scene
        cfg:               argparse Namespace with fields:
                             camera_trajectory, camera_radius,
                             sun_intensity, asset_id
        num_frames:        int
        asset_dir:         str, path to asset folder
        object_name:       str, e.g. "Cheops"
        positions:         (N,3) ndarray
        quaternions_xyzw:  (N,4) ndarray

    Returns:
        camera:        kb.PerspectiveCamera
        debris:        kb.FileBasedObject
        sun:           kb.DirectionalLight
        camera_positions:  (N,3) ndarray
        camera_look_ats:   (N,3) ndarray
        sun_direction:     (3,)  ndarray
    """

    # ------------------------------------------------------------------
    # 1. Generate camera trajectory
    # ------------------------------------------------------------------
    camera_positions, camera_look_ats = generate_camera_trajectory(
        num_frames=num_frames,
        trajectory_type=cfg.camera_trajectory,
        radius=cfg.camera_radius,
        target=np.array([0.0, 0.0, 0.0]),
    )

    # ------------------------------------------------------------------
    # 2. Camera — created + keyframed
    # ------------------------------------------------------------------
    camera, blender_cam_name = setup_camera_and_keyframe(
        scene=scene,
        camera_positions=camera_positions,
        camera_look_ats=camera_look_ats,
        focal_length_mm=CAMERA_FOCAL_LENGTH_MM,
        sensor_width_mm=CAMERA_SENSOR_WIDTH_MM,
    )

    # ------------------------------------------------------------------
    # 3. Sun — sampled once from frame-0 boresight, fixed for sequence
    # ------------------------------------------------------------------
    sun, sun_direction, angle_deg = setup_sun_light(
        scene=scene,
        camera_positions=camera_positions,
        camera_look_ats=camera_look_ats,
        min_angle_deg=SUN_MIN_ANGLE_FROM_BORESIGHT_DEG,
        sun_intensity=cfg.sun_intensity,
    )

    # ------------------------------------------------------------------
    # 4. Object — created + keyframed
    # ------------------------------------------------------------------
    debris = setup_object_and_keyframe(
        scene=scene,
        asset_id=cfg.asset_id,
        render_filename=f"{asset_dir}/{object_name}.obj",
        simulation_filename=f"{asset_dir}/{object_name}.urdf",
        positions=positions,
        quaternions_xyzw=quaternions_xyzw,
        mass=cfg.mass,
        scale=MM_TO_M,
    )

    return camera, debris, sun, camera_positions, camera_look_ats, sun_direction


# =============================================================================
# USAGE EXAMPLE — how to call setup_scene_actors() from main()
# =============================================================================

"""
Replace this block in your main():

    ── OLD (separate blocks) ────────────────────────────────────────────────

    scene.camera = kb.PerspectiveCamera(...)
    scene.camera.position = (0, -10, 0)
    scene.camera.look_at((0, 0, 0))
    scene += scene.camera

    camera_positions, camera_look_ats = generate_camera_trajectory(...)
    for i, (pos, look_at) in enumerate(...):
        keyframe_camera(...)

    sun_direction, sun_position, angle_deg = sample_sun_direction(...)
    sun = make_sun_light(sun_direction)
    scene += sun

    debris = kb.FileBasedObject(...)
    scene += debris
    for i, (pos, quat_xyzw) in enumerate(...):
        keyframe_object(...)

    ── NEW (single call) ────────────────────────────────────────────────────

    positions, quaternions = generate_trajectory(
        num_frames=num_frames,
        initial_position=(0.0, 0.0, 0.0),
        linear_velocity=(0.5, 0.1, 0.0),
        angular_velocity_deg_per_frame=(1.5, 0.5, 2.0),
        initial_quaternion=None,
    )

    camera, debris, sun, camera_positions, camera_look_ats, sun_direction = (
        setup_scene_actors(
            scene=scene,
            cfg=cfg,
            num_frames=num_frames,
            asset_dir=asset_dir,
            object_name=object_name,
            positions=positions,
            quaternions_xyzw=quaternions,
        )
    )

    # Then continue as before:
    simulator.run()
    frames_dict = renderer.render()
"""
