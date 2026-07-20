# generate_urdf.py  — run this once before your main pipeline
import os

object_name = "Cheops"
asset_dir = f"assets/{object_name}"
obj_rel = f"{object_name}.obj"

urdf_content = f"""<?xml version="1.0"?>
<robot name="{object_name}">
  <link name="base_link">
    <collision>
      <geometry>
        <mesh filename="{obj_rel}" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0" ixz="0"
               iyy="1.0" iyz="0"
               izz="1.0"/>
    </inertial>
  </link>
</robot>
"""

urdf_path = os.path.join(asset_dir, f"{object_name}.urdf")
with open(urdf_path, "w") as f:
    f.write(urdf_content)

print(f"URDF written to: {urdf_path}")