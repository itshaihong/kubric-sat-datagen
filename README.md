# Kubric Satellite Dataset Generation

This repo contains four main scripts for generating textured satellite imagery and derived labels with Kubric:

- `generate_spacecraft.py`: tumbling/drifting satellite video sequence from a fixed camera.
- `generate_spacecraft_orbit.py`: static satellite with the camera revolving around it for snapshot views.
- `convert_fbx_to_kubric_obj.py`: converts an embedded-texture FBX into OBJ/MTL/textures/URDF files that the Kubric scripts can load.
- `run_dataset_pipeline.py`: automates rendering, FastSAM segmentation, and camera/object ground-truth parsing.

The default asset path for both generators is `assets/Cheops_from_fbx`, which is produced from `assets/Cheops_fbx/cheops.fbx`.

## Runtime

Run these scripts inside the Kubric Docker image so `kubric`, `bpy`, Blender rendering, and PyBullet are available.

From this repo on the WSL host:

```bash
docker run --rm --interactive \
  --volume /home/haihong/kubric/kubric-sat-datagen:/kubric \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 /kubric/generate_spacecraft.py --help
```

## End-to-End Pipeline

`run_dataset_pipeline.py` runs the three dataset stages in order:

1. Render simulation images with `generate_spacecraft.py` or `generate_spacecraft_orbit.py`.
2. Segment the spacecraft in `output/.../image` using FastSAM, following `generate_seg.py`.
3. Parse `metadata.json` and `pose_labels.json` into ground-truth pose files, following `parse_ground_truth.py`.

If FastSAM is outside this repo, mount it into the container too. The examples below assume `/home/haihong/kubric/FastSAM` is mounted as `/FastSAM`.

Example for a reproducible bright fixed-camera video:

```bash
docker run --rm --interactive \
  --volume /home/haihong/kubric/kubric-sat-datagen:/kubric \
  --volume /home/haihong/kubric/FastSAM:/FastSAM \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 /kubric/run_dataset_pipeline.py \
    --generator spacecraft \
    --seed 123 \
    --lighting cv_bright \
    --num-frames 24 \
    --output-dir /kubric/output/Cheops_pipeline_seed123 \
    --fastsam-weights /FastSAM/weights/FastSAM-x.pt \
    --device cuda \
    --init-point 960 600
```

Example for 8 orbit snapshots with strong space lighting:

```bash
docker run --rm --interactive \
  --volume /home/haihong/kubric/kubric-sat-datagen:/kubric \
  --volume /home/haihong/kubric/FastSAM:/FastSAM \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 /kubric/run_dataset_pipeline.py \
    --generator orbit \
    --seed 123 \
    --lighting space \
    --num-frames 8 \
    --output-dir /kubric/output/Cheops_orbit_pipeline_seed123 \
    --fastsam-weights /FastSAM/weights/FastSAM-x.pt \
    --device cuda \
    --init-point 960 600
```

The pipeline writes:

```text
output/<run>/image/                       rendered RGB frames
output/<run>/seg/                         FastSAM masks
output/<run>/seg_debug/                   optional debug overlays when --save-debug is set
output/<run>/metadata.json                Kubric metadata
output/<run>/pose_labels.json             per-frame object/camera labels from the generator
output/<run>/camera_ground_truth.txt      parsed camera poses
output/<run>/object_pose_ground_truth.txt parsed object poses
output/<run>/pipeline_manifest.json       full pipeline configuration
```

Useful pipeline controls:

```text
--clean-output                 remove an existing output/<run> before starting
--skip-generation              segment/parse an existing render
--skip-segmentation            render and parse only
--skip-ground-truth            render and segment only
--prompt-type point|box|text   first-frame FastSAM prompt type
--init-point X Y               point prompt, default 960 600 for 1920x1200 renders
--init-box X0 Y0 X1 Y1         box prompt for difficult first frames
--save-debug                   save FastSAM debug visualizations
--generator-extra ...          pass remaining arguments to the selected generator; keep this option last
```

## Lighting Presets

Both generation scripts support the same lighting presets:

| Preset | Purpose | Settings |
| --- | --- | --- |
| `space` | Space-like hard lighting with strong dark edges. | Direct sun only, no fill, no ambient, physical sun softness. |
| `balanced` | Moderate visibility while retaining directional shadows. | Sun plus weak camera fill and low ambient. |
| `cv_bright` | Computer-vision friendly visibility from most angles. | Softer sun, strong camera-side fill, brighter ambient floor. |

You can override individual lighting values on top of a preset:

```bash
--sun-intensity 8.0 \
--fill-intensity 2.0 \
--ambient-level 0.15 \
--sun-shadow-softness 0.1
```

## Reproducibility

`--seed` controls sampled sun direction and sampled satellite orientation. For exact repeatability independent of random sampling, pass explicit pose and sun values:

```bash
--sun-direction -0.59 -0.22 0.77 \
--initial-position 0 0 0 \
--initial-quaternion 1 0 0 0
```

Each run writes the resolved configuration into `metadata.json`, including seed, lighting preset, asset directory, camera/orbit settings, satellite pose, and sun direction.

## `generate_spacecraft.py`

This script renders a fixed-camera sequence with a satellite that starts at `--initial-position` and then drifts/tumbles according to velocity arguments.

Dark space-style sequence:

```bash
docker run --rm --interactive \
  --volume /home/haihong/kubric/kubric-sat-datagen:/kubric \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 /kubric/generate_spacecraft.py \
    --seed 123 \
    --lighting space
```

CV-friendly sequence:

```bash
docker run --rm --interactive \
  --volume /home/haihong/kubric/kubric-sat-datagen:/kubric \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 /kubric/generate_spacecraft.py \
    --seed 123 \
    --lighting cv_bright
```

Useful options:

```text
--num-frames 24
--camera-position 0 -10 0
--look-at 0 0 0
--linear-velocity 0.5 0.1 0.0
--angular-velocity 1.5 0.5 2.0
--asset-dir /kubric/assets/Cheops_from_fbx
--output-dir /kubric/output/my_sequence
```

## `generate_spacecraft_orbit.py`

This script keeps the satellite static and renders camera snapshots around a circular orbit. By default it produces 8 snapshots.

Dark orbit snapshots:

```bash
docker run --rm --interactive \
  --volume /home/haihong/kubric/kubric-sat-datagen:/kubric \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 /kubric/generate_spacecraft_orbit.py \
    --seed 123 \
    --lighting space
```

Bright CV-friendly orbit snapshots:

```bash
docker run --rm --interactive \
  --volume /home/haihong/kubric/kubric-sat-datagen:/kubric \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 /kubric/generate_spacecraft_orbit.py \
    --seed 123 \
    --lighting cv_bright
```

Useful orbit options:

```text
--num-snapshots 8
--orbit-radius 10.0
--orbit-elevation 0.0
--initial-position 0 0 0
--initial-quaternion 1 0 0 0
--random-initial-quaternion
--sun-direction -0.59 -0.22 0.77
--asset-dir /kubric/assets/Cheops_from_fbx
--output-dir /kubric/output/my_orbit
```

In `space` lighting, the fill light is omitted. In `balanced` and `cv_bright`, the fill light moves with the camera so the viewed face is easier to see.

## FBX to OBJ/MTL/Texture Conversion

Use `convert_fbx_to_kubric_obj.py` when you receive a textured FBX and want a Kubric-compatible asset directory.

```bash
docker run --rm --interactive \
  --volume /home/haihong/kubric/kubric-sat-datagen:/kubric \
  kubricdockerhub/kubruntu \
  /usr/bin/python3 /kubric/convert_fbx_to_kubric_obj.py \
    --input /kubric/assets/Cheops_fbx/cheops.fbx \
    --output-dir /kubric/assets/Cheops_from_fbx \
    --object-name Cheops \
    --overwrite
```

The converter imports the FBX with Blender, saves embedded textures beside the OBJ, exports `Cheops.obj` and `Cheops.mtl`, and writes `Cheops.urdf` for PyBullet collision loading.

The Cheops FBX conversion currently yields these diffuse texture bindings:

```text
Body*        -> MappedTexture2.jpg
Body2        -> MappedTexture1.jpg
base         -> MappedTexture3.jpg
Default      -> MappedTexture4.jpg
Tapa-tubo*   -> MappedTexture4.jpg
solarpanels* -> SolarPanels_FrontBackSides.jpg
```

This is why the FBX-derived asset is preferred over the older `assets/Cheops` OBJ/MTL: the older MTL did not reference `MappedTexture4.jpg`.
