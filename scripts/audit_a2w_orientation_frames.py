"""Read-only URDF/USD mounting-frame audit. Does not load or modify a simulator."""
from pathlib import Path
import hashlib
import json
import os
import xml.etree.ElementTree as ET
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

root = Path(os.environ.get('A2W_FRAME_ASSET_ROOT',str(Path(__file__).resolve().parent)))
output = Path(os.environ.get('A2W_FRAME_REPORT_DIR',str(root)))
urdf = root/'a2wz1_wbc.urdf'
tree = ET.parse(urdf).getroot()
joints = {j.find('child').get('link'):j for j in tree.findall('joint')}
chain=[]
name='link0'
while name in joints:
    j=joints[name]
    origin=j.find('origin')
    chain.append(dict(name=j.get('name'),type=j.get('type'),
        parent=j.find('parent').get('link'),child=name,
        xyz=[float(v) for v in origin.get('xyz','0 0 0').split()],
        rpy=[float(v) for v in origin.get('rpy','0 0 0').split()]))
    name=j.find('parent').get('link')
chain.reverse()
assert name=='base'
assert all(j['type']=='fixed' and j['rpy']==[0.,0.,0.] for j in chain)
manifest=json.loads((root/'asset_manifest.json').read_text())
digest=hashlib.sha256(urdf.read_bytes()).hexdigest()
assert digest==manifest['output_sha256']

stage=Usd.Stage.Open(str(root/'usd/a2wz1_wbc.usd'))
cache=UsdGeom.XformCache()
body={}
joint={}
articulations=[]
def quaternion(q):
    return [float(q.GetReal()), *[float(v) for v in q.GetImaginary()]]
for p in stage.Traverse():
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        articulations.append(str(p.GetPath()))
    if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName() in ['base','z1_mount','link0','end_effector']:
        body[p.GetName()] = dict(path=str(p.GetPath()),matrix=np.array(cache.GetLocalToWorldTransform(p)).tolist())
    if p.IsA(UsdPhysics.Joint) and p.GetName() in [j['name'] for j in chain]:
        j=UsdPhysics.Joint(p)
        joint[p.GetName()]=dict(type=p.GetTypeName(),body0=[str(v) for v in j.GetBody0Rel().GetTargets()],
            body1=[str(v) for v in j.GetBody1Rel().GetTargets()],
            localRot0=quaternion(j.GetLocalRot0Attr().Get()),localRot1=quaternion(j.GetLocalRot1Attr().Get()),
            localPos0=[float(v) for v in j.GetLocalPos0Attr().Get()],localPos1=[float(v) for v in j.GetLocalPos1Attr().Get()])
assert len(joint)==len(chain)
assert all(j['type']=='PhysicsFixedJoint' for j in joint.values())
assert all(np.allclose(j['localRot0'],j['localRot1'],rtol=0,atol=1e-7) for j in joint.values())
# USD matrices use row-vector transforms.
relative=np.array(body['link0']['matrix'])@np.linalg.inv(np.array(body['base']['matrix']))
err=float(np.max(np.abs(relative[:3,:3]-np.eye(3))))
assert err<1e-7
report=dict(urdf_sha256=digest,manifest_hash_matches=True,urdf_chain=chain,
    usd_hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'usd').rglob('*.usd')},
    articulations=articulations,bodies=body,joints=joint,
    base_to_link0_row_transform=relative.tolist(),rotation_max_abs_error_from_identity=err,
    result='No fixed angular offset: base and link0 axes coincide; mounting translation does not affect quaternion composition.',
    scope='Static training URDF and composed USD, including fixed-joint constraints. No dynamic solver residual measured; not a bitwise runtime equality claim.')
output.mkdir(parents=True,exist_ok=True)
(output/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
