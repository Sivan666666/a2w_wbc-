"""Read-only frame audit for the B2 training USD versus official MuJoCo MJCF."""
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'hide_ui': True})
from pxr import Usd, UsdGeom, UsdPhysics
from pathlib import Path
import mujoco

root = Path('/mnt/sda/ps/leggedmanip-lab/LeggedManip_Lab')
stage = Usd.Stage.Open(str(root/'source/LeggedManip_Lab/LeggedManip_Lab/assets/b2_z1/b2_z1.usd'))
cache = UsdGeom.XformCache()
for p in stage.Traverse():
    if p.GetName() in ('base', 'base_link', 'link0', 'link00', 'link6', 'link06', 'end_effector'):
        print('FRAME', p.GetPath(), 'WORLD', cache.GetLocalToWorldTransform(p), 'LOCAL', UsdGeom.Xformable(p).GetLocalTransformation())
    if p.IsA(UsdPhysics.Joint):
        j = UsdPhysics.Joint(p)
        print('JOINT', p.GetPath(), 'BODIES', j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets(), 'POS', j.GetLocalPos0Attr().Get(), j.GetLocalPos1Attr().Get(), 'ROT', j.GetLocalRot0Attr().Get(), j.GetLocalRot1Attr().Get())
m = mujoco.MjModel.from_xml_path(str(root/'mujoco/robots/b2_z1/scene.xml'))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
for n in ('base_link','link00','link01','link02','link03','link04','link05','link06','gripperMover'):
    print('MUJOCO', n, 'LOCAL', m.body(n).pos, m.body(n).quat, 'WORLD', d.body(n).xpos, d.body(n).xquat)
app.close()
