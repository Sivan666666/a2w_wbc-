"""Read-only physics/frame extraction; writes diagnostic data, not robot assets."""
import argparse
from pathlib import Path
import json
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
from pxr import Usd, UsdGeom, UsdPhysics
import numpy as np
import mujoco

root = Path('/mnt/sda/ps/leggedmanip-lab/LeggedManip_Lab')
usd = root/'source/LeggedManip_Lab/LeggedManip_Lab/assets/b2_z1/b2_z1.usd'
stage = Usd.Stage.Open(str(usd))
cache = UsdGeom.XformCache()
def vector(x):
    return None if x is None else [float(v) for v in x]
def quat(x):
    return None if x is None else [float(x.GetReal()), *vector(x.GetImaginary())]
result = dict(usd=str(usd), bodies={}, joints={})
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        mass = UsdPhysics.MassAPI(prim)
        result['bodies'][prim.GetName()] = dict(path=str(prim.GetPath()), mass=mass.GetMassAttr().Get(),
            com=vector(mass.GetCenterOfMassAttr().Get()), inertia=vector(mass.GetDiagonalInertiaAttr().Get()),
            principal_axes=quat(mass.GetPrincipalAxesAttr().Get()), matrix=np.array(cache.GetLocalToWorldTransform(prim)).tolist())
    if prim.IsA(UsdPhysics.Joint):
        j = UsdPhysics.Joint(prim)
        result['joints'][prim.GetName()] = dict(type=prim.GetTypeName(),
            parent=[str(v).split('/')[-1] for v in j.GetBody0Rel().GetTargets()], child=[str(v).split('/')[-1] for v in j.GetBody1Rel().GetTargets()],
            pos0=vector(j.GetLocalPos0Attr().Get()), pos1=vector(j.GetLocalPos1Attr().Get()),
            quat0=quat(j.GetLocalRot0Attr().Get()), quat1=quat(j.GetLocalRot1Attr().Get()),
            attributes={v.GetName():str(v.Get()) for v in prim.GetAttributes() if any(k in v.GetName() for k in ('axis','friction','damping','stiffness','maxForce'))})
m = mujoco.MjModel.from_xml_path(str(root/'logs/b2_neowa_comparison_20260915/15cm/scene_training_frames.xml'))
result['mujoco'] = dict(bodies={}, joints={})
for i in range(1,m.nbody):
    result['mujoco']['bodies'][m.body(i).name] = dict(mass=float(m.body_mass[i]), com=m.body_ipos[i].tolist(),
        inertia=m.body_inertia[i].tolist(), principal_axes=m.body_iquat[i].tolist(), pos=m.body_pos[i].tolist(), quat=m.body_quat[i].tolist())
for i in range(1,m.njnt):
    k=m.jnt_dofadr[i]
    result['mujoco']['joints'][m.joint(i).name] = dict(damping=float(m.dof_damping[k]), frictionloss=float(m.dof_frictionloss[k]), armature=float(m.dof_armature[k]), axis=m.jnt_axis[i].tolist())
a.output.mkdir(parents=True, exist_ok=True)
(a.output/'physics.json').write_text(json.dumps(result, indent=2)+'\n')
print('AUDIT_DONE', a.output/'physics.json', flush=True)
app.close()
