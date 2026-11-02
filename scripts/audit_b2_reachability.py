"""Read-only checkpoint/log audit and bounded IK feasibility checks, never executed as control."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path('/mnt/sda/ps/leggedmanip-lab/LeggedManip_Lab')
run=root/'logs/rsl_rl/b2_z1_wbc/2026-09-14_14-24-22_repro_seed42_20260914_142415'
trace=json.loads((root/'logs/b2_neowa_massmatched_video_20260915/trace.json').read_text())
writing=[r for r in trace if r['phase']=='WRITE']
m=mujoco.MjModel.from_xml_path(str(root/'logs/b2_neowa_massmatched_video_20260915/scene_training_frames.xml'))
d=mujoco.MjData(m)
names=[f'{l}_{j}_joint' for l in ('FR','FL','RR','RL') for j in ('hip','thigh','calf')]+[f'joint{i}' for i in range(1,7)]
ids=[m.joint(n).id for n in names];qids=m.jnt_qposadr[ids]
arm=ids[12:];lo,hi=m.jnt_range[arm].T
ee=m.body('end_effector').id
def rot(q):return Rotation.from_quat([q[1],q[2],q[3],q[0]])
rows=[]
for r in writing[::42]:
    d.qpos[:3]=r['base'];d.qpos[3:7]=r['base_quaternion'];d.qpos[qids]=r['joint_position']
    desired=rot(r['link_quaternion'])*rot(r['pose_command'][3:])
    def residual(q):
        d.qpos[qids[12:]]=q;mujoco.mj_forward(m,d)
        angular=(desired.inv()*rot(d.xquat[ee])).as_rotvec()
        return np.r_[d.xpos[ee]-r['goal'],angular*.2]
    seeds=[np.array(r['joint_position'][12:]),np.array([0,2.2,-1.8,-.4,0,0]),np.array([0,1.5,-2.,.5,0,0])]
    best=None
    for seed in seeds:
        result=least_squares(residual,np.clip(seed,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=160,gtol=1e-10,xtol=1e-10,ftol=1e-10)
        if best is None or result.cost<best.cost:best=result
    err=residual(best.x)
    rows.append(dict(t=r['t'],goal=r['goal'],original_error_cm=r['error_m']*100,
        position_error_cm=float(np.linalg.norm(err[:3])*100),orientation_error_deg=float(np.linalg.norm(err[3:])/.2*180/np.pi),
        solution_joint_angles_deg=(best.x*180/np.pi).tolist(),joint_limit_margin_deg=float(np.minimum(best.x-lo,hi-best.x).min()*180/np.pi)))
    print('IK',rows[-1]['t'],rows[-1]['position_error_cm'],rows[-1]['orientation_error_deg'],flush=True)
events=EventAccumulator(str(run),size_guidance={'scalars':0}).Reload()
scalars={}
for tag in events.Tags()['scalars']:
    if any(v in tag.lower() for v in ('position_error','orientation_error','end_effector_position','end_effector_orientation','pos_cmd_levels')):
        values=events.Scalars(tag)
        scalars[tag]=dict(last_step=values[-1].step,last_value=values[-1].value,last100_mean=float(np.mean([v.value for v in values[-100:]])))
result=dict(ik=rows,training_metrics=scalars,note='Offline reachability only. No joint solutions were applied to a running rollout or video.')
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'reachability_and_training.json').write_text(json.dumps(result,indent=2)+'\n')
print('REACHABILITY_RESULT',json.dumps(result),flush=True)
