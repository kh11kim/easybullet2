import sys, os
import trimesh
import xml.etree.cElementTree as ET
from pathlib import Path
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


def generate_temp_urdf(mesh: trimesh.Trimesh, tempdir: str):
    mesh_path = Path(tempdir) / "mesh.obj"
    if mesh_path.exists():
        mesh_path.unlink()
    mesh.export(mesh_path, "obj")
    mesh_offset = - mesh.centroid
    offset_str = str(mesh_offset.round(3)).replace("[", "").replace("]", "")
    
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
    material = ET.SubElement(visual, "material", name="white")
    ET.SubElement(material, "color", rgba="1 1 1 1")

    tree = ET.ElementTree(root)
    ET.indent(tree, '  ')
    urdf_path = Path(tempdir) / "temp.urdf"
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return urdf_path