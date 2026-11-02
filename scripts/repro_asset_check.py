"""Static pose/render checks before launching WBC training."""
import json
import os
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl")
import mujoco
import numpy as np
import imageio.v2 as imageio

root=Path(__file__).resolve().parents[1]
out=root/"logs/asset_checks"
out.mkdir(parents=True,exist_ok=True)
report={}
for robot in ("b2_z1","a2w_z1"):
    m=mujoco.MjModel.from_xml_path(str(root/"mujoco/robots"/robot/"scene.xml"))
    d=mujoco.MjData(m)
    d.qpos[2]=.55
    d.qpos[3:7]=[1,0,0,0]
    for leg in ("FR","FL","RR","RL"):
        angles=(0.,.8,-1.5) if robot=="a2w_z1" else ((-.1 if leg[1]=="R" else .1),(.8 if leg[0]=="F" else 1.),-1.5)
        for name,value in zip(("hip","thigh","calf"),angles):
            j=m.joint(f"{leg}_{name}_joint")
            d.qpos[j.qposadr[0]]=value
    mujoco.mj_forward(m,d)
    bodies={m.body(i).name:d.xpos[i].tolist() for i in range(m.nbody) if any(k in m.body(i).name for k in ("foot","wheel","effector","gripper_link"))}
    report[robot]={"nq":m.nq,"nv":m.nv,"nu":m.nu,"mass":float(m.body_mass.sum()),"body_positions":bodies,"initial_contacts":d.ncon}
    m.vis.global_.offwidth,m.vis.global_.offheight=1280,720
    renderer=mujoco.Renderer(m,height=720,width=1280)
    cam=mujoco.MjvCamera()
    cam.lookat[:]=[0,0,.5]
    cam.distance,cam.azimuth,cam.elevation=2.6,140,-20
    renderer.update_scene(d,camera=cam)
    imageio.imwrite(out/f"{robot}.png",renderer.render())
    renderer.close()
(out/"asset_check.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
