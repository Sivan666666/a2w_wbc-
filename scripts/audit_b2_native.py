"""Diagnostic only: PhysX FK replay and native-asset closed-loop writing."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--policy', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
app=AppLauncher(a).app
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse
from LeggedManip_Lab.assets.b2_z1.b2_z1_articulation_cfg import B2_Z1_CFG

a.output.mkdir(parents=True,exist_ok=True)
sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device='cuda:0'))
ground=sim_utils.GroundPlaneCfg()
ground.func('/World/Ground',ground)
cfg=B2_Z1_CFG.replace(prim_path='/World/Robot')
for ac in cfg.actuators.values():
    ac.min_delay=0
    ac.max_delay=0
robot=Articulation(cfg)
sim.reset()
names=[f'{leg}_{j}_joint' for leg in ('FR','FL','RR','RL') for j in ('hip','thigh','calf')]+[f'joint{i}' for i in range(1,7)]
ids,_=robot.find_joints(names,preserve_order=True)
link=robot.find_bodies('link0')[0][0]
ee=robot.find_bodies('end_effector')[0][0]
reference=json.loads(a.reference.read_text())
def as_tensor(x):return torch.tensor(x,dtype=torch.float32,device=robot.device)
fk=[]
for r in reference[::13]:
    state=robot.data.default_root_state.clone()
    state[0,:3]=as_tensor(r['base'])
    state[0,3:7]=as_tensor(r['base_quaternion'])
    state[:,7:]=0
    q=robot.data.default_joint_pos.clone()
    q[0,ids]=as_tensor(r['joint_position'])
    robot.write_root_state_to_sim(state)
    robot.write_joint_state_to_sim(q,torch.zeros_like(q))
    sim.forward()
    robot.update(.005)
    pos=robot.data.body_pos_w[0,ee].cpu().numpy()
    pos_link=robot.data.body_pos_w[0,link].cpu().numpy()
    fk.append(dict(t=r['t'],native_ee=pos.tolist(),mujoco_ee=r['actual'],ee_error_m=float(np.linalg.norm(pos-r['actual'])),link_error_m=float(np.linalg.norm(pos_link-r['link_position']))))
(a.output/'native_fk.json').write_text(json.dumps(fk,indent=2)+'\n')
print('NATIVE_FK_MAX',max(v['ee_error_m'] for v in fk),flush=True)
native_props=dict(joint_names=robot.joint_names,body_names=robot.body_names,
                  default_joint_pos=robot.data.default_joint_pos.cpu().tolist(),
                  masses=robot.root_physx_view.get_masses().cpu().tolist(),
                  inertias=robot.root_physx_view.get_inertias().cpu().tolist(),
                  coms=robot.root_physx_view.get_coms().cpu().tolist())
(a.output/'native_properties.json').write_text(json.dumps(native_props,indent=2)+'\n')
state=robot.data.default_root_state.clone()
state[0,:3]=as_tensor([0.,0.,.55]);state[0,3:7]=as_tensor([1.,0.,0.,0.]);state[:,7:]=0
robot.write_root_state_to_sim(state)
robot.write_joint_state_to_sim(robot.data.default_joint_pos,torch.zeros_like(robot.data.default_joint_vel))
robot.reset()
sim.forward();robot.update(.005)
policy=torch.jit.load(str(a.policy),map_location='cpu').eval()
torch.set_num_threads(2)
history=[np.zeros((3,n),np.float32) for n in (3,3,18,18,18,3,7)]
action=np.zeros(18,np.float32)
rows=[]
for r in reference:
    goal=as_tensor([r['goal']])
    linkpos=robot.data.body_pos_w[:,link]
    linkquat=robot.data.body_quat_w[:,link]
    delta=quat_apply_inverse(linkquat,goal-linkpos)[0].cpu().numpy()
    pitch=np.clip(-np.arctan2(delta[2],np.linalg.norm(delta[:2])),-3.14/4,3.14/3)
    yaw=np.clip(np.arctan2(delta[1],delta[0]),-3.14/4,3.14/3)
    cp,sp,cy,sy=np.cos(pitch/2),np.sin(pitch/2),np.cos(yaw/2),np.sin(yaw/2)
    pose=np.array([delta[0],delta[1],r['goal'][2],cp*cy,-sp*sy,sp*cy,cp*sy],np.float32)
    obs=[robot.data.root_ang_vel_b[0].cpu().numpy()*.2,
         robot.data.projected_gravity_b[0].cpu().numpy(),
         (robot.data.joint_pos[0,ids]-robot.data.default_joint_pos[0,ids]).cpu().numpy(),
         robot.data.joint_vel[0,ids].cpu().numpy()*.05,action,np.zeros(3,np.float32),pose]
    for h,x in zip(history,obs):h[:-1],h[-1]=h[1:],x
    with torch.inference_mode():
        action=policy(torch.from_numpy(np.concatenate([h.ravel() for h in history])).unsqueeze(0)).squeeze(0).numpy()
    target=torch.clamp(robot.data.default_joint_pos[:,ids]+as_tensor(action)[None]*.25,-10,10)
    robot.set_joint_position_target(target,joint_ids=ids)
    actual=robot.data.body_pos_w[0,ee].cpu().numpy()
    rows.append(dict(t=r['t'],phase=r['phase'],letter=r['letter'],goal=r['goal'],actual=actual.tolist(),
        error_m=float(np.linalg.norm(actual-r['goal'])),base=robot.data.root_pos_w[0].cpu().tolist(),
        pose=pose.tolist(),joint_position=robot.data.joint_pos[0,ids].cpu().tolist(),action=action.tolist()))
    for _ in range(4):
        robot.write_data_to_sim();sim.step(render=False);robot.update(.005)
write=[v for v in rows if v['phase']=='WRITE']
err=np.array([v['error_m'] for v in write])*100
result=dict(mean_cm=float(err.mean()),rmse_cm=float(np.sqrt(np.mean(err**2))),
    signed_xyz_cm=(np.mean([np.array(v['actual'])-v['goal'] for v in write],axis=0)*100).tolist(),
    max_base_cm=max(float(np.linalg.norm(v['base'][:2]))*100 for v in rows),
    fk_max_error_m=max(v['ee_error_m'] for v in fk),note='Nominal training USD, no randomization, actuator delay zero; no policy training or persistent asset changes.')
(a.output/'native_trace.json').write_text(json.dumps(rows)+'\n')
(a.output/'native_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('NATIVE_RESULT',json.dumps(result),flush=True)
app.close()
