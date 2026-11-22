"""Startup FK from the packaged training geometry; no simulator on hardware."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .policy import JOINTS

SCENE = Path(__file__).resolve().parents[1] / 'assets/a2w_z1/scene.xml'


def ee_command(root_r, link_p, link_r, ee_p, ee_r):
    """Training ABI: link0-local XY, world Z, root-relative wxyz rotation."""
    local = link_r.T @ (ee_p - link_p)
    xyzw = Rotation.from_matrix(root_r.T @ ee_r).as_quat()
    if xyzw[3] < 0:
        xyzw = -xyzw
    return np.r_[local[:2], ee_p[2], xyzw[[3, 0, 1, 2]]]


def vector(element, name, default):
    return np.fromstring(element.get(name, default), sep=' ')


def transform(element):
    result = np.eye(4)
    q = vector(element, 'quat', '1 0 0 0')
    result[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = vector(element, 'pos', '0 0 0')
    return result


class Kinematics:
    def __init__(self, scene=SCENE):
        tree = ET.parse(scene).getroot()
        self.bodies = []
        self.wheels = {}
        meshes = {m.get('name'): m for m in tree.findall('asset/mesh')}
        meshdir = Path(scene).parent / tree.find('compiler').get('meshdir', '')

        def visit(body, parent):
            name = body.get('name')
            joints = []
            for joint in body.findall('joint'):
                axis = vector(joint, 'axis', '0 0 1')
                joints.append((JOINTS.index(joint.get('name')), axis / np.linalg.norm(axis),
                               vector(joint, 'pos', '0 0 0')))
            self.bodies.append((name, parent, transform(body), joints))
            if name.endswith('_wheel'):
                geom = next(g for g in body.findall('geom') if g.get('contype') == '2')
                mesh = meshes[geom.get('mesh')]
                with (meshdir / mesh.get('file')).open() as file:
                    vertices = np.array([list(map(float, line.split()[1:4]))
                                         for line in file if line.startswith('v ')])
                vertices *= vector(mesh, 'scale', '1 1 1')
                frame = transform(geom)
                self.wheels[name] = vertices @ frame[:3, :3].T + frame[:3, 3]
            for child in body.findall('body'):
                visit(child, name)

        visit(tree.find('worldbody/body[@name="base"]'), None)

    def forward(self, q, root_quaternion, root_height=0.):
        poses = {}
        for name, parent, local, joints in self.bodies:
            if parent is None:
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_quat(np.asarray(root_quaternion)[[1, 2, 3, 0]]).as_matrix()
                pose[2, 3] = root_height
            else:
                pose = poses[parent] @ local
                for index, axis, pivot in joints:
                    joint = np.eye(4)
                    joint[:3, :3] = Rotation.from_rotvec(axis * q[index]).as_matrix()
                    joint[:3, 3] = pivot - joint[:3, :3] @ pivot
                    pose = pose @ joint
            poses[name] = pose
        return poses

    def command(self, q, root_quaternion, root_height=None):
        poses = self.forward(q, root_quaternion)
        if root_height is None:
            # LowState supplies joints + IMU, not world height. Estimate the
            # standing base height from four wheels on a common horizontal plane.
            heights = []
            for name, vertices in self.wheels.items():
                pose = poses[name]
                heights.append(-np.min(vertices @ pose[2, :3] + pose[2, 3]))
            root_height = float(np.median(heights))
        root, link, ee = (poses[n].copy() for n in ('base', 'link0', 'end_effector'))
        link[2, 3] += root_height
        ee[2, 3] += root_height
        return ee_command(root[:3, :3], link[:3, 3], link[:3, :3], ee[:3, 3], ee[:3, :3])
