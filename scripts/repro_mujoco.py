"""Headless sim-to-sim check with name-based joint mapping and train-matched gains.

Does not overwrite official deployment scripts/policies. Outputs video + measured metrics.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import torch
import imageio.v2 as imageio

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", choices=["b2_z1", "a2w_z1"], required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=30.)
    p.add_argument("--xml", type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    xml = a.xml or root / "mujoco/robots" / a.robot / "scene.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    m.opt.timestep = .005
    d = mujoco.MjData(m)
    legs = [f"{leg}_{j}_joint" for leg in ["FR","FL","RR","RL"] for j in ["hip","thigh","calf"]]
    names = legs + [f"joint{i}" for i in range(1,7)]
    wheeled = a.robot == "a2w_z1"
    if wheeled:
        names += [f"{leg}_wheel_joint" for leg in ["FR","FL","RR","RL"]]
    ids = [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    if min(ids) < 0:
        raise ValueError(f"Missing joints: {[n for n,i in zip(names,ids) if i<0]}")
    qids, vids = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
    motor_ids = []
    for j in ids:
        matches = np.flatnonzero(m.actuator_trnid[:,0] == j)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one motor for {m.joint(j).name}: {matches}")
        motor_ids.append(int(matches[0]))
    default = np.zeros(len(ids))
    default[:12] = [0.,.8,-1.5]*4 if wheeled else [-.1,.8,-1.5,.1,.8,-1.5,-.1,1.,-1.5,.1,1.,-1.5]
    kp = np.array(([120.]*12 if wheeled else [250.]*12) + [50.,50.,80.,30.,30.,20.] + ([0.]*4 if wheeled else []))
    kd = np.array(([3.]*12 if wheeled else [5.]*12) + [3.,2.,3.,3.,2.5,1.] + ([2.]*4 if wheeled else []))
    d.qpos[2] = .55
    d.qpos[3:7] = [1.,0.,0.,0.]
    d.qpos[qids] = default
    m.dof_armature[vids] = .01
    mujoco.mj_forward(m,d)
    policy = torch.jit.load(str(a.policy),map_location="cpu").eval()
    torch.set_num_threads(2)
    dims = [3,3,18,len(ids),len(ids),3,7]
    histories = [np.zeros((3,n),dtype=np.float32) for n in dims]
    action = np.zeros(len(ids),dtype=np.float32)
    target = default.copy()
    wheel_target = np.zeros(4)
    a.output.mkdir(parents=True,exist_ok=True)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1280,720
    renderer = mujoco.Renderer(m,height=720,width=1280)
    cam = mujoco.MjvCamera()
    cam.distance,cam.azimuth,cam.elevation = 3.2,135.,-18.
    metrics=[]
    with imageio.get_writer(a.output / "rollout.mp4", fps=25, codec="libx264", quality=8) as writer:
        for step in range(round(a.seconds/.005)):
            if step % 4 == 0:
                t = step*.005
                vel_cmd = np.array([.2 if t>5. else 0.,0.,.15 if t>15. else 0.],dtype=np.float32)
                pose_cmd = np.array([.425,0.,.5,1.,0.,0.,0.],dtype=np.float32)
                R=np.zeros(9)
                mujoco.mju_quat2Mat(R,d.qpos[3:7])
                gravity=R.reshape(3,3).T @ np.array([0.,0.,-1.])
                obs=[d.qvel[3:6]*.2,gravity,d.qpos[qids[:18]]-default[:18],d.qvel[vids]*.05,action,vel_cmd,pose_cmd]
                for h,x in zip(histories,obs):
                    h[:-1]=h[1:]
                    h[-1]=x
                tensor=torch.from_numpy(np.concatenate([h.reshape(-1) for h in histories])).unsqueeze(0)
                with torch.inference_mode():
                    action=policy(tensor).squeeze(0).numpy()
                if action.shape != (len(ids),) or not np.isfinite(action).all():
                    raise ValueError(f"Invalid policy output {action}")
                # Isaac Lab JointAction: affine transform first, then clip the target.
                target[:18]=np.clip(default[:18]+action[:18]*.25,-10.,10.)
                if wheeled:
                    wheel_target=np.clip(action[18:]*10.,-3.,3.)
                metrics.append({"t":t,"height":float(d.qpos[2]),"upright":float(-gravity[2]),
                    "position":d.qpos[:3].tolist(),"command":vel_cmd.tolist(),"action_max":float(np.abs(action).max())})
            tau=kp*(target-d.qpos[qids])-kd*d.qvel[vids]
            if wheeled:
                tau[18:]=kd[18:]*(wheel_target-d.qvel[vids[18:]])
            for k,aid in enumerate(motor_ids):
                if m.actuator_ctrllimited[aid]:
                    tau[k]=np.clip(tau[k],*m.actuator_ctrlrange[aid])
            d.ctrl[motor_ids]=tau
            mujoco.mj_step(m,d)
            if not np.isfinite(d.qpos).all() or d.warning.number.sum():
                raise RuntimeError("MuJoCo numerical warning or invalid state")
            if step%8 == 0:
                cam.lookat[:]=d.qpos[:3]
                renderer.update_scene(d,camera=cam)
                writer.append_data(renderer.render())
    renderer.close()
    summary={"robot":a.robot,"policy":str(a.policy),"seconds":a.seconds,"joint_order":names,
        "actor_input_dim":3*sum(dims),"actions":len(ids),"min_height":min(x["height"] for x in metrics),
        "upright_fraction":float(np.mean([x["upright"]>.7 and x["height"]>.2 for x in metrics])),
        "final_position":d.qpos[:3].tolist(),"note":"Pipeline check only; stability/tracking must be inspected, not a success-rate benchmark."}
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (a.output/"trace.json").write_text(json.dumps(metrics)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
