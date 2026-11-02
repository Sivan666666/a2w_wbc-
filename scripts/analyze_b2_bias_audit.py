"""Independent FK and paired dynamics-ablation analysis."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a=p.parse_args()
physics=json.loads((a.root/'physics.json').read_text())
baseline=json.loads((a.root/'baseline/trace.json').read_text())
summary=json.loads((a.root/'baseline/summary.json').read_text())
def rotation(q):
    w,x,y,z=np.array(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def frame(q,pos):
    T=np.eye(4);T[:3,:3]=rotation(q);T[:3,3]=pos;return T
fk_errors=[]
for r in baseline[::13]:
    transforms={'base':frame(r['base_quaternion'],r['base'])}
    positions=dict(zip(summary['joint_order'],r['joint_position']))
    pending=dict(physics['joints'])
    while pending:
        progressed=False
        for name,j in list(pending.items()):
            if len(j['parent'])!=1 or j['parent'][0] not in transforms:
                continue
            A=frame(j['quat0'],j['pos0']);B=frame(j['quat1'],j['pos1'])
            Q=np.eye(4)
            if j['type']=='PhysicsRevoluteJoint':
                axis=np.eye(3)['XYZ'.index(j['attributes']['physics:axis'])]
                angle=positions[name]
                Q[:3,:3]=rotation([np.cos(angle/2),*(axis*np.sin(angle/2))])
            transforms[j['child'][0]]=transforms[j['parent'][0]]@A@Q@np.linalg.inv(B)
            del pending[name];progressed=True
        assert progressed, pending.keys()
    fk_errors.append(dict(t=r['t'],ee_m=float(np.linalg.norm(transforms['end_effector'][:3,3]-r['actual'])),
                          link0_m=float(np.linalg.norm(transforms['link0'][:3,3]-r['link_position']))))
cases=[]
for folder in sorted(a.root.iterdir()):
    if not (folder/'trace.json').exists():continue
    trace=json.loads((folder/'trace.json').read_text())
    rows=[r for r in trace if r['phase']=='WRITE']
    if not rows:continue
    assert [(v['t'],v['phase'],v['goal']) for v in trace]==[(v['t'],v['phase'],v['goal']) for v in baseline],folder
    diff=np.array([r['actual'] for r in rows])-np.array([r['goal'] for r in rows])
    errors=np.linalg.norm(diff,axis=1)*100
    cases.append(dict(case=folder.name,mean_cm=float(errors.mean()),rmse_cm=float(np.sqrt(np.mean(errors**2))),
                      mean_xyz_cm=(diff.mean(0)*100).tolist(),
                      p95_cm=float(np.percentile(errors,95)),self_contact_frames=sum(bool(r['self_contacts']) for r in rows),
                      max_base_cm=max(float(np.linalg.norm(r['base'][:2]))*100 for r in trace)))
result=dict(fk=dict(samples=len(fk_errors),max_ee_error_m=max(v['ee_m'] for v in fk_errors),max_link0_error_m=max(v['link0_m'] for v in fk_errors)),cases=cases)
(a.root/'audit_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
