"""A2W WBC-only N tracking; zero base command, free base, no IK override.

Target positions are fixed in the world. Convert to the exact training reward's
hybrid convention: link0-local XY and world Z. Collision geoms are hidden only
for rendering; model masses, collisions and actuator gains remain unchanged.
"""
import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch


def smooth(u):
    u = np.clip(u, 0., 1.)
    return u*u*u*(10.+u*(-15.+6.*u))


def goal_at(t, vertices, segment_seconds, settle):
    if t < settle:
        return vertices[0].copy(), "SETTLE / MOVE TO START"
    elapsed = t-settle
    k = min(int(elapsed/segment_seconds), 2)
    u = smooth((elapsed-k*segment_seconds)/segment_seconds)
    return vertices[k]*(1-u)+vertices[k+1]*u, (f"N: STROKE {k+1}/3" if elapsed < 3*segment_seconds else "HOLD END")


def line(scene, p, q, color, radius=.003):
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(np.asarray(p)-q) < 1.e-7:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(p), np.asarray(q))
    scene.ngeom += 1


def ball(scene, p, color, radius=.01):
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.ones(3)*radius, p, np.eye(3).ravel(), np.array(color))
    scene.ngeom += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--xml", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=float, default=.24)
    p.add_argument("--height", type=float, default=.24)
    p.add_argument("--center-z", type=float, default=.65)
    p.add_argument("--forward", type=float, default=.60)
    p.add_argument("--segment-seconds", type=float, default=8.)
    p.add_argument("--settle-seconds", type=float, default=6.)
    p.add_argument("--orientation", choices=("radial", "identity"), default="radial")
    p.add_argument("--no-video", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    # Correct only visible mount orientation to match a2wz1_mount door asset.
    # The original -90deg body frame and +90deg collision frame are left intact.
    # Both visual and collision mount now have the same composed orientation.
    tree = ET.parse(a.xml)
    compiler = tree.getroot().find("compiler")
    compiler.set("meshdir", str((a.xml.resolve().parent/compiler.get("meshdir", "")).resolve()))
    for body in tree.findall(".//body"):
        if body.get("name") in ("wrist_up_mount", "wrist_down_mount"):
            for geom in body.findall("geom"):
                if geom.get("group") == "1":
                    assert geom.get("quat") is None
                    geom.set("quat", "0.7071067811865476 0.7071067811865476 0 0")
    variant = a.output/"scene_mount_visual_aligned.xml"
    tree.write(variant, encoding="utf-8", xml_declaration=True)
    m = mujoco.MjModel.from_xml_path(str(variant))
    m.opt.timestep = .005
    d = mujoco.MjData(m)
    names = [f"{leg}_{j}_joint" for leg in ("FR", "FL", "RR", "RL") for j in ("hip", "thigh", "calf")]
    names += [f"joint{i}" for i in range(1, 7)] + [f"{leg}_wheel_joint" for leg in ("FR", "FL", "RR", "RL")]
    ids = np.array([m.joint(n).id for n in names])
    qids, vids = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
    motors = [int(np.flatnonzero(m.actuator_trnid[:, 0] == j).item()) for j in ids]
    default = np.array([0., .8, -1.5]*4+[0.]*10)
    kp = np.array([120.]*12+[50., 50., 80., 30., 30., 20.]+[0.]*4)
    kd = np.array([3.]*12+[3., 2., 3., 3., 2.5, 1.]+[2.]*4)
    d.qpos[2], d.qpos[3:7], d.qpos[qids] = .55, [1., 0., 0., 0.], default
    m.dof_armature[vids] = .01
    mujoco.mj_forward(m, d)
    link = m.body("link0").id
    ee = m.body("end_effector").id
    x = d.xpos[link, 0]+a.forward
    y = d.xpos[link, 1]
    # From a front camera, world -Y points left: lower left -> upper left -> lower right -> upper right.
    vertices = np.array([[x, y-a.width/2, a.center_z-a.height/2],
                         [x, y-a.width/2, a.center_z+a.height/2],
                         [x, y+a.width/2, a.center_z-a.height/2],
                         [x, y+a.width/2, a.center_z+a.height/2]])
    start_base = d.qpos[:3].copy()
    policy = torch.jit.load(str(a.policy), map_location="cpu").eval()
    torch.set_num_threads(2)
    histories = [np.zeros((3, n), np.float32) for n in (3, 3, 18, 22, 22, 3, 7)]
    action = np.zeros(22, np.float32)
    target = default.copy()
    wheel_target = np.zeros(4)
    vel_cmd = np.zeros(3, np.float32)
    rows, actual_trail = [], []
    render, writer = None, None
    cyan, orange = (0., .82, 1., 1.), (1., .42, .08, 1.)
    option = mujoco.MjvOption()
    hidden = []
    for i in range(m.ngeom):
        if m.geom_bodyid[i] != 0 and (m.geom_contype[i] or m.geom_conaffinity[i]):
            hidden.append(i)
            m.geom_group[i] = 3
    option.geomgroup[3] = 0
    if not a.no_video:
        m.vis.global_.offwidth, m.vis.global_.offheight = 1600, 900
        render = mujoco.Renderer(m, height=900, width=1600, max_geom=5000)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [.30, 0., .50]
        cam.distance, cam.azimuth, cam.elevation = 2.8, 140., -16.
        writer = imageio.get_writer(a.output/"draw_n.mp4", fps=25, codec="libx264", quality=8, macro_block_size=1)
    duration = a.settle_seconds+3*a.segment_seconds+3.
    try:
        for step in range(round(duration/.005)):
            t = step*.005
            mujoco.mj_forward(m, d)
            goal, phase = goal_at(t, vertices, a.segment_seconds, a.settle_seconds)
            if step % 4 == 0:
                R = d.xmat[link].reshape(3, 3)
                delta = R.T @ (goal-d.xpos[link])
                pitch = np.clip(-np.arctan2(delta[2], np.linalg.norm(delta[:2])), -3.14/4, 3.14/3)
                yaw = np.clip(np.arctan2(delta[1], delta[0]), -3.14/4, 3.14/3)
                cp, sp, cy, sy = np.cos(pitch/2), np.sin(pitch/2), np.cos(yaw/2), np.sin(yaw/2)
                pose_cmd = np.array([delta[0], delta[1], goal[2], cp*cy, -sp*sy, sp*cy, cp*sy], np.float32)
                if a.orientation == "identity":
                    pose_cmd[3:] = [1., 0., 0., 0.]
                mat = np.zeros(9)
                mujoco.mju_quat2Mat(mat, d.qpos[3:7])
                gravity = mat.reshape(3, 3).T @ np.array([0., 0., -1.])
                obs = [d.qvel[3:6]*.2, gravity, d.qpos[qids[:18]]-default[:18], d.qvel[vids]*.05, action, vel_cmd, pose_cmd]
                for h, v in zip(histories, obs):
                    h[:-1] = h[1:]
                    h[-1] = v
                with torch.inference_mode():
                    action = policy(torch.from_numpy(np.concatenate([h.ravel() for h in histories])).unsqueeze(0)).squeeze(0).numpy()
                if action.shape != (22,) or not np.isfinite(action).all():
                    raise RuntimeError("Invalid policy action")
                # Match Isaac Lab JointAction: scale + offset BEFORE target clipping.
                target[:18] = np.clip(default[:18]+.25*action[:18], -10., 10.)
                wheel_target = np.clip(10.*action[18:], -3., 3.)
                actual = d.xpos[ee].copy()
                rows.append(dict(t=t, goal=goal.tolist(), actual=actual.tolist(), base=d.qpos[:3].tolist(),
                                 base_command=vel_cmd.tolist(), pose_command=pose_cmd.tolist(),
                                 wheel_target=wheel_target.tolist(), wheel_actual=d.qvel[vids[18:]].tolist(),
                                 base_quaternion_wxyz=d.qpos[3:7].tolist(),
                                 error_m=float(np.linalg.norm(actual-goal)), upright=float(-gravity[2])))
                if a.settle_seconds <= t <= a.settle_seconds+3*a.segment_seconds and step % 8 == 0:
                    actual_trail.append(actual)
            if render is not None and step % 8 == 0:
                render.update_scene(d, camera=cam, scene_option=option)
                for start, end in zip(vertices[:-1], vertices[1:]):
                    line(render.scene, start, end, cyan, .003)
                trail = actual_trail[::2]
                for start, end in zip(trail[:-1], trail[1:]):
                    line(render.scene, start, end, orange, .0025)
                ball(render.scene, goal, cyan)
                ball(render.scene, d.xpos[ee], orange, .008)
                frame = render.render().copy()
                cv2.rectangle(frame, (0, 0), (1600, 100), (20, 24, 30), -1)
                cv2.putText(frame, "A2W + Z1 | Learned WBC | MuJoCo", (30, 38), cv2.FONT_HERSHEY_SIMPLEX, .9, (245, 245, 245), 2)
                cv2.putText(frame, f"vx = vy = yaw rate = 0  |  {phase}  |  t = {t:.1f}s", (30, 79), cv2.FONT_HERSHEY_SIMPLEX, .67, (235, 235, 235), 2)
                # Fixed-axis world YZ projection: same scale and origin for reference and measured motion.
                left, top, size = 1140, 130, 420
                cv2.rectangle(frame, (left, top), (left+size, top+570), (25, 29, 35), -1)
                scale = 220./max(a.width, a.height, .2)
                def pixel(pt):
                    return (int(left+210+(pt[1]-y)*scale), int(top+230-(pt[2]-a.center_z)*scale))
                def draw_poly(pts, col):
                    if len(pts) > 1:
                        cv2.polylines(frame, [np.array([pixel(v) for v in pts], np.int32)], False, col, 2, cv2.LINE_AA)
                draw_poly(vertices, (0, 209, 255))
                draw_poly(actual_trail, (255, 107, 20))
                cv2.circle(frame, pixel(goal), 6, (0, 209, 255), -1)
                actual_pixel = pixel(d.xpos[ee])
                if left+10 < actual_pixel[0] < left+size-10 and top+60 < actual_pixel[1] < top+345:
                    cv2.circle(frame, actual_pixel, 5, (255, 107, 20), -1)
                labels = [("World Y-Z projection", (230, 230, 230)), ("Cyan: desired N", (0, 209, 255)),
                          ("Orange: actual end effector", (255, 107, 20)),
                          (f"3D error: {rows[-1]['error_m']*100:.1f} cm", (235, 235, 235)),
                          ("Base free; no IK correction", (235, 235, 235))]
                for j, (label, col) in enumerate(labels):
                    yy = top+32 if j == 0 else top+380+34*(j-1)
                    cv2.putText(frame, label, (left+16, yy), cv2.FONT_HERSHEY_SIMPLEX, .60, col, 1, cv2.LINE_AA)
                writer.append_data(frame)
                if step == 0 or abs(t-(a.settle_seconds+3*a.segment_seconds)) < .002:
                    imageio.imwrite(a.output/("start.png" if step == 0 else "end.png"), frame)
            tau = kp*(target-d.qpos[qids])-kd*d.qvel[vids]
            tau[18:] = kd[18:]*(wheel_target-d.qvel[vids[18:]])
            for k, aid in enumerate(motors):
                if m.actuator_ctrllimited[aid]:
                    tau[k] = np.clip(tau[k], *m.actuator_ctrlrange[aid])
            d.ctrl[motors] = tau
            mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all() or d.warning.number.sum():
                raise RuntimeError("MuJoCo numerical warning / invalid state")
    finally:
        if writer is not None:
            writer.close()
        if render is not None:
            render.close()
    draw_rows = [r for r in rows if a.settle_seconds <= r['t'] <= a.settle_seconds+3*a.segment_seconds]
    errors = np.array([r['error_m'] for r in draw_rows])
    xyz = np.array([r['actual'] for r in draw_rows])
    summary = dict(policy=str(a.policy), policy_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest(),
                   xml=str(a.xml), xml_sha256=hashlib.sha256(a.xml.read_bytes()).hexdigest(),
                   base_command=[0., 0., 0.], floating_base=True, ik_correction=False,
                   vertices_world=vertices.tolist(), drawing_seconds=3*a.segment_seconds,
                   command_convention="link0-local XY, world Z; orientation radial in link0 frame",
                   mean_error_cm=float(errors.mean()*100), rmse_cm=float(np.sqrt(np.mean(errors**2))*100),
                   p95_error_cm=float(np.percentile(errors, 95)*100), actual_extent_xyz_m=np.ptp(xyz, axis=0).tolist(),
                   max_base_xy_displacement_m=max(float(np.linalg.norm(np.array(r['base'])[:2]-start_base[:2])) for r in rows),
                   min_base_height_m=min(r['base'][2] for r in rows),
                   upright_fraction=float(np.mean([r['upright'] > .7 and r['base'][2] > .2 for r in rows])),
                   hidden_collision_geoms=hidden, physics_changed=False, orientation=a.orientation,
                   action_processing="Isaac Lab affine_then_clip: position=clip(default+0.25*raw,-10,10); wheel=clip(10*raw,-3,3)",
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   mount_visual_correction="+90deg visual roll to cancel original -90deg mount body roll; collisions/inertia unchanged",
                   note="Actual trace is measured, not snapped to reference. Not a success-rate benchmark.")
    (a.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (a.output/"trace.json").write_text(json.dumps(rows)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
