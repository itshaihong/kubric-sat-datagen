import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.simulator.pybullet import PyBullet as KubricSimulator
import numpy as np
import os
from PIL import Image
import imageio

# Maximum depth in metres — set to match max_depth in your frontend.flags.
# Background/sky pixels (inf/nan from Kubric) are replaced with 0.0 (invalid sentinel).
MAX_DEPTH = 100.0

# .flo writer (Middlebury format)
def write_flo(filename, flow):
    """Write HxWx2 float32 array to Middlebury .flo format."""
    assert flow.ndim == 3 and flow.shape[2] == 2, "Flow must be HxWx2"
    with open(filename, 'wb') as f:
        np.array([202021.25], dtype=np.float32).tofile(f)                    # magic number
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)  # W, H
        flow.astype(np.float32).tofile(f)

# batch writer to match kubric's write_flow_batch naming convention
def write_flo_batch(flows, output_dir, name="forward_flow"):
    """Write a batch of flow frames as .flo files."""
    os.makedirs(output_dir, exist_ok=True)
    for i, flow in enumerate(flows):
        out_path = os.path.join(output_dir, f"{i:06d}.flo")
        write_flo(out_path, flow[..., :2])  # take only (u,v), drop alpha if present

def write_rgb_batch(rgb_frames, output_dir, name="image"):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(rgb_frames):
        rgb = frame[..., :3]  # drop alpha
        out_path = os.path.join(output_dir, f"{i:06d}.png")
        Image.fromarray(rgb, mode="RGB").save(out_path)

def write_tiff_depth_batch(depth_f64, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(depth_f64):
        frame = frame.squeeze().astype(np.float64)  # guarantee 2D
        imageio.imwrite(
            os.path.join(output_dir, f"{i:06d}.tiff"),
            frame,
            format="tiff"
        )

def clamp_depth_batch(depth_frames, max_depth=MAX_DEPTH):
    """
    Clamp a batch of depth frames to [0, max_depth].

    Kubric renders background pixels as inf (no geometry hit). These, along with
    any nan or negative values, are replaced with 0.0 — the standard "invalid"
    sentinel that DynoSAM's frontend checks via depth > 0. Any finite depth
    beyond max_depth is also zeroed out so the pipeline never sees unbounded values.

    Args:
        depth_frames: array of shape (N, H, W) or (N, H, W, 1)
        max_depth:    maximum valid depth in metres

    Returns:
        Clamped float32 array of the same shape.
    """
    depth = np.array(depth_frames, dtype=np.float32)
    invalid_mask = ~np.isfinite(depth) | (depth <= 0.0) | (depth > max_depth)
    depth[invalid_mask] = 100
    return depth


object_name = "Cheops"
asset_dir = f"reconstruction-tracking-synthetic/assets/{object_name}"
output_dir = f"/dataset/kubric_sim/reconstruction-tracking-synthetic/{object_name}/"
os.makedirs(f"{output_dir}/image/", exist_ok=True)
os.makedirs(f"{output_dir}/depth/", exist_ok=True)
os.makedirs(f"{output_dir}/flow/", exist_ok=True)

# --- 1. INITIALIZE SCENE, SIMULATOR, AND RENDERER ---
scene = kb.Scene(resolution=(1024, 1024), frame_start=1, frame_end=24) # 1-second video at 24fps

# PyBullet for Physics (Collisions & Poses)
simulator = KubricSimulator(scene, scratch_dir=f"{output_dir}/tmp")
scene.gravity = (0, 0, 0) # ZERO GRAVITY for space simulation

# Blender for High-Quality Rendering (RGB, Depth, Flow)
renderer = KubricRenderer(scene, scratch_dir=f"{output_dir}/tmp")
renderer.ambient_illumination = kb.Color(0.1, 0.1, 0.1) # Dark ambient for space

# --- 2. ADD CAMERA ---
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
scene.camera.position = (0, -10, 0)
scene.camera.look_at((0, 0, 0))

# --- 3. ADD LIGHTING (Sunlight / High Contrast) ---
sun = kb.DirectionalLight(color=kb.get_color("white"), shadow_softness=0.1, intensity=5.0)
sun.position = (10, -10, 10)
sun.look_at((0, 0, 0))
scene += sun

# --- 4. LOAD YOUR IRREGULAR OBJECT ---
# Replace with the path to your actual .obj file
CM_TO_M = 0.01
MM_TO_M = 0.001
debris = kb.FileBasedObject(
    asset_id="cheops_satellite",
    render_filename=f"{asset_dir}/{object_name}.obj",
    bounds=((-1, -1, -1), (1, 1, 1)), # Approximate bounding box
    simulation_filename=f"{asset_dir}/{object_name}.urdf", # Used for collision geometry
    position=(0, 0, 0),
    mass=10.0,
    scale=MM_TO_M 
)
scene += debris

# --- 5. APPLY KINEMATICS (TUMBLING IN SPACE) ---
# Give the object a random starting rotation
debris.quaternion = kb.random_rotation()
# Give it a slight linear velocity (drifting)
debris.velocity = (0.5, 0.1, 0.0)
# Give it an angular velocity (tumbling)
debris.angular_velocity = (1.5, 0.5, 2.0)

# --- 6. RUN PHYSICS SIMULATION (PyBullet) ---
print("Running Physics Simulation...")
simulator.run()

# --- 7. RUN RENDERING (Blender) ---
print("Running Rendering Engine...")
renderer.save_state(f"{output_dir}/blender_scene.blend") # Highly recommended for debugging
frames_dict = renderer.render()

# --- 8. CLAMP DEPTH ---
print("Clamping depth to MAX_DEPTH={:.1f}m (replacing inf/nan/out-of-range with 0.0)...".format(MAX_DEPTH))
depth_raw = frames_dict["depth"]
depth_clamped = clamp_depth_batch(depth_raw, max_depth=MAX_DEPTH)

n_invalid_raw = np.sum(~np.isfinite(depth_raw.astype(np.float32)) | (depth_raw <= 0.0))
n_invalid_clamped = np.sum(depth_clamped == 0.0)
print(f"  Invalid pixels before clamping (inf/nan/<=0): {n_invalid_raw}")
print(f"  Invalid pixels after  clamping (zeroed out):  {n_invalid_clamped}")

# --- 9. EXPORT DATA ---
print("Exporting Dataset...")
# Fix 1: Strip alpha channel (RGBA -> RGB) before writing
rgb_frames = frames_dict["rgba"][..., :3]
write_rgb_batch(rgb_frames, f"{output_dir}/image/")

# Fix 2: Clamp depth AND cast to float64 to match pipeline expectation
depth_clamped = clamp_depth_batch(frames_dict["depth"], MAX_DEPTH)
depth_f64 = depth_clamped.astype(np.float64)
write_tiff_depth_batch(depth_f64, f"{output_dir}/depth/")

write_flo_batch(frames_dict["forward_flow"], f"{output_dir}/flow", name="forward_flow")

kb.file_io.write_json(filename=f"{output_dir}/metadata.json", data={
    "metadata": kb.get_scene_metadata(scene),
    "camera": kb.get_camera_info(scene.camera),
    "instances": kb.get_instance_info(scene),
})

# Stats over valid pixels only (depth > 0)
valid_mask = depth_clamped > 0.0
print(f"\nDepth stats (valid pixels only, depth > 0):")
print(f"  mean:  {np.mean(depth_clamped[valid_mask]):.4f} m")
print(f"  min:   {np.min(depth_clamped[valid_mask]):.4f} m")
print(f"  max:   {np.max(depth_clamped[valid_mask]):.4f} m")
print(f"  valid: {np.sum(valid_mask)} / {depth_clamped.size} pixels ({100*np.mean(valid_mask):.1f}%)")

print("\nDataset generation complete! Check the /output folder.")


'''
PowerShell
docker run --rm --interactive --volume "%cd%:/kubric" kubricdockerhub/kubruntu /usr/bin/python3 reconstruction-tracking-synthetic/generate_debris.py

Linux command line:
docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$(pwd):/kubric" \
    --volume "$HOME/tracking_dataset:/dataset" \
    kubricdockerhub/kubruntu \
    /usr/bin/python3 reconstruction-tracking-synthetic/generate_debris.py

'''