#!/usr/bin/env python3
"""
Convert a textured FBX spacecraft asset into Kubric-friendly OBJ/MTL files.

Run this inside the Kubric Docker image because it needs Blender's bpy module:

    docker run --rm --interactive \
        --volume /home/haihong/kubric/kubric-sat-datagen:/kubric \
        kubricdockerhub/kubruntu \
        /usr/bin/python3 /kubric/convert_fbx_to_kubric_obj.py \
        --input /kubric/assets/Cheops_fbx/cheops.fbx \
        --output-dir /kubric/assets/Cheops_from_fbx \
        --object-name Cheops

The script keeps texture/material assignment from the FBX, writes image files
next to the OBJ/MTL, and creates a simple URDF for PyBullet collision loading.
"""

import argparse
import os
import re
import shutil
from pathlib import Path


def require_bpy():
    try:
        import bpy  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise SystemExit(
            "This script must run with Blender's Python, for example inside "
            "kubricdockerhub/kubruntu using /usr/bin/python3."
        ) from exc
    return bpy


def clean_name(name):
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "texture.png"


def image_basename(image, fallback):
    if image.filepath:
        base = clean_name(Path(image.filepath).name)
        if Path(base).suffix:
            return base
    return clean_name(fallback)


def unique_path(directory, filename):
    path = directory / filename
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    index = 1
    while True:
        candidate = directory / f"{stem}_{index:02d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def choose_image_format(path):
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "JPEG"
    if suffix == ".png":
        return "PNG"
    if suffix in {".tif", ".tiff"}:
        return "TIFF"
    return "PNG"


def save_or_copy_image(bpy, image, output_dir, used_by):
    source_path = Path(bpy.path.abspath(image.filepath)) if image.filepath else None
    target_name = image_basename(image, f"{clean_name(image.name)}.png")
    target_path = unique_path(output_dir, target_name)

    if image.packed_file is not None:
        image.filepath_raw = str(target_path)
        image.file_format = choose_image_format(target_path)
        image.save()
    elif source_path and source_path.exists():
        shutil.copy2(source_path, target_path)
    else:
        raise FileNotFoundError(
            f"Texture image '{image.name}' used by {used_by} is not packed and "
            f"does not exist at '{image.filepath}'."
        )

    image.filepath = str(target_path)
    image.filepath_raw = str(target_path)
    return target_path


def iter_material_image_nodes(bpy):
    for material in bpy.data.materials:
        if not material.node_tree:
            continue
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                yield material, node, node.image


def export_textures(bpy, output_dir):
    image_to_path = {}
    mapping = {}

    for material, node, image in iter_material_image_nodes(bpy):
        image_key = image_basename(image, f"{clean_name(image.name)}.png").lower()
        if image_key not in image_to_path:
            image_to_path[image_key] = save_or_copy_image(
                bpy, image, output_dir, f"material '{material.name}'"
            )
        else:
            image.filepath = str(image_to_path[image_key])
            image.filepath_raw = str(image_to_path[image_key])

        mapping.setdefault(material.name, set()).add(image_to_path[image_key].name)
        node.image = image

    return {key: sorted(value) for key, value in sorted(mapping.items())}


def prepare_output_dir(output_dir, overwrite):
    output_dir = output_dir.resolve()
    if output_dir == Path(output_dir.anchor):
        raise ValueError(f"Refusing to use filesystem root as output directory: {output_dir}")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Pass --overwrite to reuse it."
            )
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)


def delete_default_scene(bpy):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_fbx(bpy, fbx_path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects were imported from {fbx_path}")
    return imported, meshes


def select_meshes(bpy, meshes):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]


def export_obj(bpy, obj_path, meshes, axis_forward, axis_up):
    select_meshes(bpy, meshes)
    bpy.ops.export_scene.obj(
        filepath=str(obj_path),
        use_selection=True,
        use_animation=False,
        use_mesh_modifiers=True,
        use_edges=True,
        use_smooth_groups=True,
        use_materials=True,
        keep_vertex_order=True,
        path_mode="RELATIVE",
        axis_forward=axis_forward,
        axis_up=axis_up,
    )


def write_urdf(urdf_path, obj_filename, robot_name, mass):
    urdf = f'''<?xml version="1.0"?>
<robot name="{robot_name}">
  <link name="base_link">
    <collision>
      <geometry>
        <mesh filename="{obj_filename}" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="1.0" ixy="0" ixz="0"
               iyy="1.0" iyz="0"
               izz="1.0"/>
    </inertial>
  </link>
</robot>
'''
    urdf_path.write_text(urdf, encoding="utf-8")


def print_summary(texture_mapping, output_dir, obj_path, urdf_path):
    print("[OK] Wrote Kubric-compatible asset files")
    print(f"  output_dir: {output_dir}")
    print(f"  obj:        {obj_path}")
    print(f"  mtl:        {obj_path.with_suffix('.mtl')}")
    print(f"  urdf:       {urdf_path}")
    print("[Texture mapping from FBX materials]")
    if not texture_mapping:
        print("  No image texture nodes found.")
    for material, textures in texture_mapping.items():
        print(f"  {material} -> {', '.join(textures)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert embedded-texture FBX to OBJ/MTL/textures/URDF for Kubric."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input FBX file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where OBJ, MTL, textures, and URDF will be written.",
    )
    parser.add_argument(
        "--object-name",
        default="Cheops",
        help="Base filename for OBJ/MTL/URDF output. Default: Cheops.",
    )
    parser.add_argument(
        "--robot-name",
        default=None,
        help="URDF robot name. Defaults to --object-name.",
    )
    parser.add_argument("--mass", default=10.0, type=float, help="URDF mass value.")
    parser.add_argument(
        "--axis-forward",
        default="-Z",
        help="Blender OBJ export forward axis. Default matches Blender OBJ export: -Z.",
    )
    parser.add_argument(
        "--axis-up",
        default="Y",
        help="Blender OBJ export up axis. Default matches Blender OBJ export: Y.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in the output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fbx_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if not fbx_path.exists():
        raise FileNotFoundError(f"Input FBX does not exist: {fbx_path}")

    prepare_output_dir(output_dir, args.overwrite)

    bpy = require_bpy()
    delete_default_scene(bpy)
    _, meshes = import_fbx(bpy, fbx_path)

    texture_mapping = export_textures(bpy, output_dir)

    obj_path = output_dir / f"{args.object_name}.obj"
    export_obj(bpy, obj_path, meshes, args.axis_forward, args.axis_up)

    urdf_path = output_dir / f"{args.object_name}.urdf"
    write_urdf(
        urdf_path=urdf_path,
        obj_filename=obj_path.name,
        robot_name=args.robot_name or args.object_name,
        mass=args.mass,
    )

    print_summary(texture_mapping, output_dir, obj_path, urdf_path)


if __name__ == "__main__":
    main()
