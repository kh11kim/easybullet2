import sys, os
import numpy as np
import trimesh
import xml.etree.cElementTree as ET
from pathlib import Path
from typing import *
from tempfile import TemporaryDirectory

class HideOutput:
    '''
    A context manager that block stdout for its scope'''

    def __init__(self, *args, **kw):
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        self._newstdout = os.dup(1)
        os.dup2(self._devnull, 1)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, 1)


def generate_temp_urdf(mesh: trimesh.Trimesh, tempdir: str, rgba: List[int]=[1, 1, 1, 1]):
    mesh_path = Path(tempdir) / "mesh.obj"
    if mesh_path.exists():
        mesh_path.unlink()
    mesh.export(mesh_path, "obj")
    mesh_offset = - mesh.centroid
    
    ndarray_to_str = lambda x: str(x.round(3)).replace("[", "").replace("]", "")
    offset_str = ndarray_to_str(mesh_offset)
    color_str = ndarray_to_str(np.array(rgba))
    
    urdf_name = "temp"
    root = ET.Element("robot", name=urdf_name+".urdf")
    link = ET.SubElement(root, "link", name="mesh")
    inertial = ET.SubElement(link, "inertial")
    visual = ET.SubElement(link, "visual")
    collision = ET.SubElement(link, "collision")
    ET.SubElement(inertial, "mass", value="1")
    ET.SubElement(inertial, "inertia", 
        ixx="0.1", ixy="0", ixz="0", iyy="0.1", iyz="0", izz="0.1")
    for element in [visual, collision]:
        origin = ET.SubElement(element, "origin",
            xyz=offset_str, rpy="0 0 0")
        geometry = ET.SubElement(element, "geometry")
        geometry_mesh = ET.SubElement(geometry, "mesh",
            filename="mesh.obj", scale="1 1 1")
    material = ET.SubElement(visual, "material", name="color")
    ET.SubElement(material, "color", rgba=color_str)

    tree = ET.ElementTree(root)
    ET.indent(tree, '  ')
    urdf_path = Path(tempdir) / "temp.urdf"
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return urdf_path

def generate_frame_urdf(tempdir, length=0.05, radius=0.005):
    urdf_name = "frame"
    root = ET.Element("robot", name=urdf_name)
    base_link = ET.SubElement(root, "link", name="base")

    def add_material(visual, color_name, rgba_str):
        material = ET.SubElement(visual, "material", name=color_name)
        ET.SubElement(material, "color", rgba=rgba_str)
    def add_fixed_joint(name, parent_name, child_name, xyz_str="0 0 0", rpy_str="0 0 0"):
        joint = ET.SubElement(root, "joint", name=name, type="fixed")
        ET.SubElement(joint, "parent", link=parent_name)
        ET.SubElement(joint, "child", link=child_name)
        ET.SubElement(joint, "origin", rpy=rpy_str, xyz=xyz_str)
        ET.SubElement(joint, "axis", xyz="0 0 0")
    def add_axis_link(name, color_name, rgba_str, length=0.05, radius=0.005, ):
        link = ET.SubElement(root, "link", name=name)
        visual = ET.SubElement(link, "visual")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "cylinder", length=f"{length}", radius=f"{radius}")
        ET.SubElement(visual, "origin", xyz=f"0 0 {length/2}", rpy="0 0 0")
        add_material(visual, color_name, rgba_str)
    
    visual = ET.SubElement(base_link, "visual")
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "sphere", radius="0.01")
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    add_material(visual, "white", "1 1 1 1")
    add_axis_link("x_axis", "red", "1 0 0 1", length=length, radius=radius)
    add_axis_link("y_axis", "green", "0 1 0 1", length=length, radius=radius)
    add_axis_link("z_axis", "blue", "0 0 1 1", length=length, radius=radius)
    add_fixed_joint("x_axis_joint", "base", "x_axis", rpy_str="0 1.5708 0", )
    add_fixed_joint("y_axis_joint", "base", "y_axis", rpy_str="-1.5708 0 0", )
    add_fixed_joint("z_axis_joint", "base", "z_axis", rpy_str="0 0 0", )
    
    urdf_path = Path(tempdir) / "frame.urdf"
    tree = ET.ElementTree(root)
    ET.indent(tree, '  ')
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return urdf_path


