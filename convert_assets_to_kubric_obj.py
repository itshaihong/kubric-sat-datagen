#!/usr/bin/env python3
"""
Convert raw GLB/STL assets into Kubric-friendly OBJ/MTL/URDF folders.

Run conversions inside the Kubric Docker image because Blender's bpy module is
needed for import/export:

    docker run --rm --interactive \
      --volume "C:\\Users\\haiho\\kubric-sat-datagen:/kubric" \
      kubricdockerhub/kubruntu \
      /usr/bin/python3 /kubric/convert_assets_to_kubric_obj.py \
      --batch-known-assets \
      --assets-root /kubric/assets \
      --overwrite

The batch mode treats each Cassini and Deep Space Station STL component as a
separate asset and writes a manifest for all 23 desired assets.
"""

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


READY_ASSETS = [
    ("Cheops", "Cheops_from_fbx"),
    ("Integral", "Integral_from_fbx"),
    ("smart-1", "smart-1_from_fbx"),
]

CASSINI_COMPONENTS = [
    "boom_01.stl",
    "boom_02.stl",
    "bottom.stl",
    "bracket.stl",
    "bus_01.stl",
    "bus_02.stl",
    "bus_top.stl",
    "dish.stl",
    "huygens_01.stl",
    "huygens_02.stl",
    "pins_2x.stl",
    "RTG_3.stl",
]

DSS_COMPONENTS = [
    "Azimuth track.stl",
    "Elevation assembly.stl",
    "Lower alidade.stl",
    "Main reflector.stl",
    "Quadrapod.stl",
    "Upper alidade.stl",
]


@dataclass(frozen=True)
class AssetConversion:
    source: Path
    output_dir: Path
    object_name: str
    source_kind: str

    @staticmethod
    def manifest_path(path):
        return Path(path).as_posix()

    def to_manifest_entry(self):
        asset_dir = self.manifest_path(self.output_dir)
        return {
            "object_name": self.object_name,
            "asset_dir": asset_dir,
            "source": self.manifest_path(self.source),
            "source_kind": self.source_kind,
            "obj": self.manifest_path(self.output_dir / f"{self.object_name}.obj"),
            "mtl": self.manifest_path(self.output_dir / f"{self.object_name}.mtl"),
            "urdf": self.manifest_path(self.output_dir / f"{self.object_name}.urdf"),
            "generate_spacecraft_args": (
                f"--object-name {self.object_name} --asset-dir {asset_dir}"
            ),
        }


def require_bpy():
    try:
        import bpy  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise SystemExit(
            "Conversion must run with Blender's Python, for example inside "
            "kubricdockerhub/kubruntu using /usr/bin/python3."
        ) from exc
    return bpy


def clean_token(value):
    value = Path(value).stem if Path(value).suffix else str(value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value or "asset"


def clean_filename(value):
    value = Path(value).name
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value or "texture.png"


def build_known_asset_plan(assets_root):
    assets_root = Path(assets_root)
    plan = []

    for object_name, folder_name in READY_ASSETS:
        folder = assets_root / folder_name
        plan.append(AssetConversion(
            source=folder,
            output_dir=folder,
            object_name=object_name,
            source_kind="ready",
        ))

    for filename in CASSINI_COMPONENTS:
        component = clean_token(filename)
        object_name = f"Cassini_{component}"
        plan.append(AssetConversion(
            source=assets_root / "Cassini" / filename,
            output_dir=assets_root / "Cassini_components" / object_name,
            object_name=object_name,
            source_kind="stl",
        ))

    for filename in DSS_COMPONENTS:
        component = clean_token(filename)
        object_name = f"DSS_{component}"
        plan.append(AssetConversion(
            source=assets_root / "Deep_Space_Station_Antenna" / filename,
            output_dir=assets_root / "DSS_components" / object_name,
            object_name=object_name,
            source_kind="stl",
        ))

    plan.append(AssetConversion(
        source=assets_root / "Astronaut.glb",
        output_dir=assets_root / "Astronaut_from_glb",
        object_name="Astronaut",
        source_kind="glb",
    ))
    plan.append(AssetConversion(
        source=assets_root / "Block Island.stl",
        output_dir=assets_root / "Block_Island_from_stl",
        object_name="Block_Island",
        source_kind="stl",
    ))

    return plan


def filter_plan(plan, only):
    if not only:
        return list(plan)
    wanted = set(only)
    selected = [asset for asset in plan if asset.object_name in wanted]
    missing = sorted(wanted - {asset.object_name for asset in selected})
    if missing:
        raise ValueError(f"Unknown asset name(s) for --only: {', '.join(missing)}")
    return selected


def infer_source_kind(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".stl":
        return "stl"
    if suffix in {".glb", ".gltf"}:
        return "glb"
    if suffix == ".fbx":
        return "fbx"
    raise ValueError(f"Cannot infer source kind from extension: {path}")


def prepare_output_dir(output_dir, overwrite):
    output_dir = Path(output_dir).resolve()
    if output_dir == Path(output_dir.anchor):
        raise ValueError(f"Refusing to use filesystem root as output directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Pass --overwrite to replace it."
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


def imported_meshes_since(bpy, before):
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects were imported.")
    return meshes


def import_source(bpy, source, source_kind):
    before = set(bpy.data.objects)
    if source_kind == "stl":
        bpy.ops.import_mesh.stl(filepath=str(source))
    elif source_kind == "glb":
        bpy.ops.import_scene.gltf(filepath=str(source))
    elif source_kind == "fbx":
        bpy.ops.import_scene.fbx(filepath=str(source))
    else:
        raise ValueError(f"Cannot convert source kind: {source_kind}")
    return imported_meshes_since(bpy, before)


def assign_flat_material(bpy, meshes, material_name, color):
    material = bpy.data.materials.new(material_name)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None and "Base Color" in bsdf.inputs:
        bsdf.inputs["Base Color"].default_value = color
    for mesh in meshes:
        mesh.data.materials.clear()
        mesh.data.materials.append(material)


def image_basename(image, fallback):
    if image.filepath:
        base = clean_filename(Path(image.filepath).name)
        if Path(base).suffix:
            return base
    return clean_filename(fallback)


def unique_path(directory, filename):
    path = directory / filename
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = directory / f"{path.stem}_{index:02d}{path.suffix}"
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


def iter_material_image_nodes(bpy):
    for material in bpy.data.materials:
        if not material.node_tree:
            continue
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                yield material, node, node.image


def save_or_copy_image(bpy, image, output_dir, used_by):
    source_path = Path(bpy.path.abspath(image.filepath)) if image.filepath else None
    target_name = image_basename(image, f"{clean_filename(image.name)}.png")
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


def export_textures(bpy, output_dir):
    image_to_path = {}
    mapping = {}
    for material, node, image in iter_material_image_nodes(bpy):
        image_key = image_basename(image, f"{clean_filename(image.name)}.png").lower()
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


def convert_asset(asset, overwrite, mass, axis_forward, axis_up, stl_color):
    if asset.source_kind == "ready":
        return {"status": "ready"}
    if not asset.source.exists():
        raise FileNotFoundError(f"Input asset does not exist: {asset.source}")

    bpy = require_bpy()
    delete_default_scene(bpy)
    prepare_output_dir(asset.output_dir, overwrite)
    meshes = import_source(bpy, asset.source, asset.source_kind)
    if asset.source_kind == "stl":
        assign_flat_material(bpy, meshes, f"{asset.object_name}_mat", stl_color)
    texture_mapping = export_textures(bpy, asset.output_dir)

    obj_path = asset.output_dir / f"{asset.object_name}.obj"
    export_obj(bpy, obj_path, meshes, axis_forward, axis_up)
    write_urdf(
        asset.output_dir / f"{asset.object_name}.urdf",
        obj_path.name,
        asset.object_name,
        mass,
    )
    return {"status": "converted", "textures": texture_mapping}


def write_manifest(path, assets, results=None):
    results = results or {}
    data = {
        "asset_count": len(assets),
        "assets": [
            {
                **asset.to_manifest_entry(),
                "conversion": results.get(asset.object_name, {}),
            }
            for asset in assets
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert known raw STL/GLB assets into Kubric OBJ/MTL/URDF asset folders."
    )
    parser.add_argument("--assets-root", type=Path, default=Path("assets"))
    parser.add_argument("--batch-known-assets", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Object names to convert from the known asset plan.")
    parser.add_argument("--input", type=Path, default=None, help="Single raw asset to convert.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for --input conversion.")
    parser.add_argument("--object-name", default=None, help="Object/base filename for --input conversion.")
    parser.add_argument("--source-kind", choices=("stl", "glb", "fbx"), default=None, help="Override source kind for --input.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only write/print the manifest plan.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--mass", type=float, default=10.0)
    parser.add_argument("--axis-forward", default="-Z")
    parser.add_argument("--axis-up", default="Y")
    parser.add_argument(
        "--stl-color",
        nargs=4,
        type=float,
        default=(0.72, 0.72, 0.68, 1.0),
        metavar=("R", "G", "B", "A"),
        help="Flat material color assigned to STL-derived assets.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    assets_root = args.assets_root.resolve()

    if args.batch_known_assets:
        manifest_path = args.manifest or (assets_root / "kubric_asset_manifest.json")
        plan = filter_plan(build_known_asset_plan(assets_root), args.only)
    elif args.input is not None:
        if args.output_dir is None or args.object_name is None:
            raise SystemExit("--input conversion requires --output-dir and --object-name.")
        manifest_path = args.manifest or (Path(args.output_dir).resolve() / "kubric_asset_manifest.json")
        plan = [
            AssetConversion(
                source=Path(args.input).resolve(),
                output_dir=Path(args.output_dir).resolve(),
                object_name=args.object_name,
                source_kind=args.source_kind or infer_source_kind(args.input),
            )
        ]
    else:
        raise SystemExit("Pass --batch-known-assets or --input with --output-dir and --object-name.")

    results = {}

    for asset in plan:
        if args.dry_run:
            results[asset.object_name] = {"status": "planned"}
            continue
        print(f"[Asset] {asset.object_name}: {asset.source_kind} -> {asset.output_dir}")
        results[asset.object_name] = convert_asset(
            asset=asset,
            overwrite=args.overwrite,
            mass=args.mass,
            axis_forward=args.axis_forward,
            axis_up=args.axis_up,
            stl_color=tuple(args.stl_color),
        )

    write_manifest(manifest_path, plan, results)
    print(f"[OK] Wrote manifest: {manifest_path}")
    print(f"[OK] Assets in manifest: {len(plan)}")


if __name__ == "__main__":
    main()
