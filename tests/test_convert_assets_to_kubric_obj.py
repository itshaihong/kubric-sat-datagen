import tempfile
import unittest
from pathlib import Path

import convert_assets_to_kubric_obj as converter


class ConvertAssetsToKubricObjTest(unittest.TestCase):
    def test_known_asset_plan_contains_23_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cheops_from_fbx").mkdir()
            (root / "Integral_from_fbx").mkdir()
            (root / "smart-1_from_fbx").mkdir()
            cassini = root / "Cassini"
            cassini.mkdir()
            for name in [
                "boom_01.stl", "boom_02.stl", "bottom.stl", "bracket.stl",
                "bus_01.stl", "bus_02.stl", "bus_top.stl", "dish.stl",
                "huygens_01.stl", "huygens_02.stl", "pins_2x.stl", "RTG_3.stl",
            ]:
                (cassini / name).write_text("solid x\nendsolid x\n")
            dss = root / "Deep_Space_Station_Antenna"
            dss.mkdir()
            for name in [
                "Azimuth track.stl", "Elevation assembly.stl",
                "Lower alidade.stl", "Main reflector.stl",
                "Quadrapod.stl", "Upper alidade.stl",
            ]:
                (dss / name).write_text("solid x\nendsolid x\n")
            (root / "Astronaut.glb").write_bytes(b"glTF")
            (root / "Block Island.stl").write_text("solid x\nendsolid x\n")

            plan = converter.build_known_asset_plan(root)

        self.assertEqual(23, len(plan))
        names = {asset.object_name for asset in plan}
        self.assertIn("Cassini_boom_01", names)
        self.assertIn("DSS_Main_reflector", names)
        self.assertIn("Astronaut", names)
        self.assertIn("Block_Island", names)

    def test_manifest_entry_matches_generator_arguments(self):
        source = Path("/kubric/assets/Cassini/boom_01.stl")
        output = Path("/kubric/assets/Cassini_components/Cassini_boom_01")
        asset = converter.AssetConversion(
            source=source,
            output_dir=output,
            object_name="Cassini_boom_01",
            source_kind="stl",
        )

        entry = asset.to_manifest_entry()

        self.assertEqual("Cassini_boom_01", entry["object_name"])
        self.assertEqual("/kubric/assets/Cassini/boom_01.stl", entry["source"])
        self.assertEqual("/kubric/assets/Cassini_components/Cassini_boom_01", entry["asset_dir"])
        self.assertEqual(
            "--object-name Cassini_boom_01 --asset-dir /kubric/assets/Cassini_components/Cassini_boom_01",
            entry["generate_spacecraft_args"],
        )

    def test_filter_plan_selects_requested_assets(self):
        plan = [
            converter.AssetConversion(Path("a"), Path("out/a"), "Astronaut", "glb"),
            converter.AssetConversion(Path("b"), Path("out/b"), "Block_Island", "stl"),
            converter.AssetConversion(Path("c"), Path("out/c"), "Cassini_boom_01", "stl"),
        ]

        selected = converter.filter_plan(plan, ["Astronaut", "Block_Island"])

        self.assertEqual(["Astronaut", "Block_Island"], [asset.object_name for asset in selected])


if __name__ == "__main__":
    unittest.main()
