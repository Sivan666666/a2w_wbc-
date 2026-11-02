"""Continuous NeoWa demo: WBC writing -> retract -> commanded lateral move.

Uses the trained policy only (no IK, teleport, base locking or simulation reset).
Targets and measured trajectories are recorded separately. Movement timeouts and
tracking errors are reported, never hidden by replacing measured motion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from repro_mujoco_draw_n import smooth, line, ball


def arc(start, end, count=100):
    theta = np.linspace(start, end, count)
    return np.column_stack([.5*np.cos(theta), .4+.4*np.sin(theta)])


def glyphs():
    th = np.linspace(np.pi, 0., 80)
    arch = np.column_stack([.5*np.cos(th), .35+.45*np.sin(th)])
    return {
        'n': [np.array([[-.5, 0], [-.5, .8]]), np.array([[-.5, .8], [-.5, .35]]), arch, np.array([[.5, .35], [.5, 0]])],
        'w': [np.array([a, b]) for a, b in zip([[-.5, .8], [-.25, 0], [0, .48], [.25, 0]], [[-.25, 0], [0, .48], [.25, 0], [.5, .8]])],
        'N': [np.array([[-.5, 0], [-.5, 1]]), np.array([[-.5, 1], [.5, 0]]), np.array([[.5, 0], [.5, 1]])],
        'e': [np.array([[-.5, .4], [.5, .4]]), arc(0., 1.8*np.pi)],
        'o': [arc(np.pi/2, 2.5*np.pi)],
        'W': [np.array([a, b]) for a, b in zip([[-.5, 1], [-.25, 0], [0, .6], [.25, 0]], [[-.25, 0], [0, .6], [.25, 0], [.5, 1]])],
        'a': [arc(0., 2*np.pi), np.array([[.5, .4], [.5, .8]]), np.array([[.5, .8], [.5, 0]])],
    }


def sample_poly(points, u):
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    distance = float(smooth(u))*lengths[-1]
    return np.array([np.interp(distance, lengths, points[:, j]) for j in range(3)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit-case', choices=('baseline','no_friction','no_damping','no_passive','usd_arm_mass','usd_arm_mass_no_passive','no_self_collision','end_effector_1kg','usd_arm_mass_tip','usd_arm_mass_tip_no_passive'), default='baseline')
    ap.add_argument('--physics-audit', type=Path)
    ap.add_argument('--robot', choices=('a2w_z1', 'b2_z1'), default='a2w_z1')
    ap.add_argument('--plane-x', type=float, help='Fixed world X; use the same value for cross-robot comparisons.')
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--xml', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--width', type=float, default=.20)
    ap.add_argument('--height', type=float, default=.22)
    ap.add_argument('--baseline-z', type=float, default=.50)
    ap.add_argument('--spacing', type=float, default=.32)
    ap.add_argument('--forward', type=float, default=.60)
    ap.add_argument('--retract-forward', type=float, default=.55)
    ap.add_argument('--draw-speed', type=float, default=.10)
    ap.add_argument('--vy', type=float, default=.14)
    ap.add_argument('--shift-timeout', type=float, default=10.)
    ap.add_argument('--placement', choices=('fixed_world', 'after_shift'), default='fixed_world')
    ap.add_argument('--shift-mode', choices=('waypoint', 'pulse'), default='waypoint')
    ap.add_argument('--shift-seconds', type=float, default=3.)
    ap.add_argument('--text', choices=('NeoWa', 'neowa'), default='NeoWa')
    ap.add_argument('--stationary', action='store_true', help='One fixed world plane, centered word, all base commands zero; no side steps or re-anchoring.')
    ap.add_argument('--no-video', action='store_true')
    a = ap.parse_args()
    if a.stationary:
        a.placement = 'fixed_world'
        a.vy = 0.
    a.output.mkdir(parents=True, exist_ok=True)
    wheeled = a.robot == 'a2w_z1'
    robot_label = 'A2W + Z1' if wheeled else 'B2 + Z1'
    tree = ET.parse(a.xml)
    compiler = tree.getroot().find('compiler')
    if compiler is not None:
        compiler.set('meshdir', str((a.xml.resolve().parent/compiler.get('meshdir', '')).resolve()))
    for body in tree.findall('.//body'):
        if body.get('name') in ('wrist_up_mount', 'wrist_down_mount'):
            for geom in body.findall('geom'):
                if geom.get('group') == '1':
                    geom.set('quat', '0.7071067811865476 0.7071067811865476 0 0')
    variant = a.output/'scene_mount_visual_aligned.xml'
    if wheeled:
        tree.write(variant, encoding='utf-8', xml_declaration=True)
    else:
        # Read-only USD frame audit: connector_to_arm_joint and ee_gripper.
        # Modify only an output-private MJCF; original official files stay untouched.
        robot_xml = a.xml.resolve().parent/'b2_z1.xml'
        robot_tree = ET.parse(robot_xml)
        rc = robot_tree.getroot().find('compiler')
        rc.set('meshdir', str((robot_xml.parent/rc.get('meshdir')).resolve()))
        robot_tree.find('.//body[@name="link00"]').set('pos', '0.15000000596046448 0 0.22500000894069672')
        ET.SubElement(robot_tree.find('.//body[@name="link06"]'), 'body',
                      name='end_effector', pos='0.20100000500679016 0 -0.009999999776482582')
        robot_variant = (a.output/'b2_training_frames.xml').resolve()
        robot_tree.write(robot_variant, encoding='utf-8', xml_declaration=True)
        for inc in tree.findall('.//include'):
            inc.set('file', str(robot_variant))
        variant = a.output/'scene_training_frames.xml'
        tree.write(variant, encoding='utf-8', xml_declaration=True)
    m = mujoco.MjModel.from_xml_path(str(variant))
    # In-memory diagnostic ablations only; never write changed dynamics to source assets.
    if a.audit_case in ('no_friction', 'no_passive', 'usd_arm_mass_no_passive', 'usd_arm_mass_tip_no_passive'):
        m.dof_frictionloss[:] = 0.
    if a.audit_case in ('no_damping', 'no_passive', 'usd_arm_mass_no_passive', 'usd_arm_mass_tip_no_passive'):
        m.dof_damping[:] = 0.
    if a.audit_case in ('usd_arm_mass', 'usd_arm_mass_no_passive', 'usd_arm_mass_tip', 'usd_arm_mass_tip_no_passive'):
        data = json.loads(a.physics_audit.read_text())['bodies']
        for mj_name, usd_name in [(f'link0{i}', f'link{i}') for i in range(7)]+[('end_effector','end_effector')]:
            body = m.body(mj_name).id
            spec = data[usd_name]
            if spec['mass'] == 0:
                # USD zero-mass marker has automatic COM/axis sentinels; keep it massless.
                continue
            m.body_mass[body] = spec['mass']
            m.body_ipos[body] = spec['com']
            m.body_inertia[body] = spec['inertia'] if spec['mass'] > 0 else 0.
            m.body_iquat[body] = spec['principal_axes']
        b = m.body('gripperMover').id
        m.body_mass[b] = 0.
        m.body_inertia[b] = 0.
    if a.audit_case in ('end_effector_1kg', 'usd_arm_mass_tip', 'usd_arm_mass_tip_no_passive'):
        # Measured from the live PhysX articulation, NOT inferred from USD mass=0.
        # USD's automatic mass fallback gives this empty rigid body 1 kg.
        b = m.body('end_effector').id
        m.body_mass[b] = 1.
        m.body_ipos[b] = 0.
        m.body_iquat[b] = [1., 0., 0., 0.]
        m.body_inertia[b] = 9.999999747378752e-5
    if a.audit_case == 'no_self_collision':
        for gid in range(m.ngeom):
            if not (m.geom_contype[gid] or m.geom_conaffinity[gid]):
                continue
            robot_geom = m.geom_bodyid[gid] != 0
            m.geom_contype[gid] = 2 if robot_geom else 1
            m.geom_conaffinity[gid] = 1 if robot_geom else 2
    mujoco.mj_setConst(m, mujoco.MjData(m))
    m.opt.timestep = .005
    d = mujoco.MjData(m)
    names = [f'{leg}_{j}_joint' for leg in ('FR', 'FL', 'RR', 'RL') for j in ('hip', 'thigh', 'calf')]
    names += [f'joint{i}' for i in range(1, 7)]
    if wheeled:
        names += [f'{leg}_wheel_joint' for leg in ('FR', 'FL', 'RR', 'RL')]
    ids = np.array([m.joint(n).id for n in names])
    qids, vids = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
    motors = [int(np.flatnonzero(m.actuator_trnid[:, 0] == j).item()) for j in ids]
    default = np.array(([0., .8, -1.5]*4 if wheeled else [-.1,.8,-1.5,.1,.8,-1.5,-.1,1.,-1.5,.1,1.,-1.5])+[0.]*(len(names)-12))
    kp = np.array(([120.]*12 if wheeled else [250.]*12)+[50.,50.,80.,30.,30.,20.]+([0.]*4 if wheeled else []))
    kd = np.array(([3.]*12 if wheeled else [5.]*12)+[3.,2.,3.,3.,2.5,1.]+([2.]*4 if wheeled else []))
    d.qpos[2], d.qpos[3:7], d.qpos[qids] = .55, [1., 0., 0., 0.], default
    m.dof_armature[vids] = .01
    mujoco.mj_forward(m, d)
    link, ee = m.body('link0' if wheeled else 'link00').id, m.body('end_effector').id
    plane_x = float(d.xpos[link, 0]+a.forward) if a.plane_x is None else a.plane_x
    initial_link0 = d.xpos[link].copy()
    letters = list(a.text)
    centers = [(i-2 if a.stationary else i)*a.spacing for i in range(5)]
    paths = []
    plane_origins, plane_laterals = [], []
    for i, letter in enumerate(letters):
        strokes = []
        for xy in glyphs()[letter]:
            points = np.zeros((len(xy), 3))
            points[:, 0] = plane_x
            points[:, 1] = centers[i]+xy[:, 0]*a.width
            points[:, 2] = a.baseline_z+xy[:, 1]*a.height
            strokes.append(points)
        paths.append(strokes)
        plane_origins.append(np.array([plane_x, centers[i], a.baseline_z]))
        plane_laterals.append(np.array([0., 1., 0.]))
    policy = torch.jit.load(str(a.policy), map_location='cpu').eval()
    torch.set_num_threads(2)
    history = [np.zeros((3, n), np.float32) for n in (3, 3, 18, len(names), len(names), 3, 7)]
    action = np.zeros(len(names), np.float32)
    target, wheel_target = default.copy(), np.zeros(4 if wheeled else 0)
    phase, phase_start, index, stroke = 'APPROACH', 0., 0, 0
    segment_start = paths[0][0][0].copy()
    goal = segment_start.copy()
    phase_duration = 3.
    rows, events, transitions = [], [], []
    trails = [[[] for _ in strokes] for strokes in paths]
    retract_start, retract_end = goal.copy(), goal.copy()
    move_start = d.qpos[:3].copy()
    move_target_y = 0.
    last_target = goal.copy()
    renderer, writer = None, None
    option = mujoco.MjvOption()
    if wheeled:
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid] and (m.geom_contype[gid] or m.geom_conaffinity[gid]):
                m.geom_group[gid] = 3
    # B2 leg meshes serve both physics and appearance. Preserve their groups;
    # hiding every collidable mesh would also hide all four visible legs.
    option.geomgroup[3] = 0
    if not a.no_video:
        m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
        renderer = mujoco.Renderer(m, height=720, width=1920, max_geom=12000)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [.28, 0., .5]
        cam.distance, cam.azimuth, cam.elevation = 2.8, 145., -16.
        writer = imageio.get_writer(a.output/(a.text+'.mp4'), fps=25, codec='libx264', quality=8, macro_block_size=1)
    cyan, orange = (0., .82, 1., 1.), (1., .42, .08, 1.)
    outcome = 'incomplete'
    try:
        for step in range(60000):
            t = step*.005
            mujoco.mj_forward(m, d)
            elapsed = t-phase_start
            R = d.xmat[link].reshape(3, 3)
            local_retract = d.xpos[link]+R@np.array([a.retract_forward, 0., 0.])
            local_retract[2] = .68
            vel = np.zeros(3, np.float32)
            # State transitions never reset simulation, actions, history, or base pose.
            if phase in ('APPROACH', 'WRITE', 'HOLD', 'RETRACT', 'SETTLE', 'FINAL') and elapsed >= phase_duration:
                if phase in ('APPROACH', 'SETTLE'):
                    phase, stroke = 'WRITE', 0
                    phase_duration = max(.6 if a.stationary else 1.2, np.linalg.norm(np.diff(paths[index][stroke], axis=0), axis=1).sum()/a.draw_speed)
                elif phase == 'WRITE':
                    if stroke+1 < len(paths[index]):
                        stroke += 1
                        phase_duration = max(.6 if a.stationary else 1.2, np.linalg.norm(np.diff(paths[index][stroke], axis=0), axis=1).sum()/a.draw_speed)
                    else:
                        phase, phase_duration = 'HOLD', .25 if a.stationary else .6
                elif phase == 'HOLD':
                    phase, phase_duration = 'RETRACT', .7 if a.stationary else 2.
                    retract_start, retract_end = last_target.copy(), local_retract.copy()
                    if a.stationary:
                        retract_end = last_target+np.array([-.04, 0., 0.])
                elif phase == 'RETRACT':
                    if index == 4:
                        phase, phase_duration = 'FINAL', 2.
                    elif a.stationary:
                        index += 1
                        phase, phase_duration = 'SETTLE', 1.
                        segment_start = retract_end.copy()
                    else:
                        phase = 'SIDE STEP'
                        move_start = d.qpos[:3].copy()
                        move_target_y = (index+1)*a.spacing
                elif phase == 'FINAL':
                    outcome = 'completed_sequence'
                    break
                phase_start, elapsed = t, 0.
                events.append(dict(t=t, letter=letters[index], phase=phase, stroke=stroke))
            if phase == 'SIDE STEP':
                error_y = move_target_y-d.qpos[1]
                shift_done = ((elapsed >= 1. and abs(error_y) <= .025) or elapsed >= a.shift_timeout) if a.shift_mode == 'waypoint' else elapsed >= a.shift_seconds
                if shift_done:
                    transitions.append(dict(letter_from=letters[index], letter_to=letters[index+1],
                        duration=elapsed, start_base=move_start.tolist(), end_base=d.qpos[:3].tolist(),
                        target_y=move_target_y if a.shift_mode=='waypoint' else None,
                        displacement_xy=(d.qpos[:2]-move_start[:2]).tolist(),
                        error_y=float(error_y) if a.shift_mode=='waypoint' else None,
                        timed_out=bool(a.shift_mode=='waypoint' and abs(error_y)>.025), mode=a.shift_mode))
                    index += 1
                    if a.placement == 'after_shift':
                        # Latch once per letter, then keep its plane fixed in the world while writing.
                        heading = np.arctan2(R[1, 0], R[0, 0])
                        forward = np.array([np.cos(heading), np.sin(heading), 0.])
                        lateral = np.array([-np.sin(heading), np.cos(heading), 0.])
                        center = d.xpos[link]+a.forward*forward
                        center[2] = a.baseline_z
                        plane_origins[index], plane_laterals[index] = center, lateral
                        paths[index] = [center[None, :]+xy[:, :1]*a.width*lateral[None, :]+xy[:, 1:]*a.height*np.array([[0., 0., 1.]]) for xy in glyphs()[letters[index]]]
                    phase, phase_start, phase_duration, elapsed = 'SETTLE', t, 2.5, 0.
                    segment_start = local_retract.copy()
                    events.append(dict(t=t, letter=letters[index], phase=phase, stroke=0))
                else:
                    if a.shift_mode == 'pulse':
                        vel[1] = a.vy*smooth(elapsed/.6)*smooth((a.shift_seconds-elapsed)/.6)
                    else:
                        vel[1] = np.clip(.9*error_y, -a.vy, a.vy)*smooth(elapsed/.7)
            if phase == 'APPROACH':
                goal = paths[index][0][0].copy()
            elif phase == 'SETTLE':
                goal = (1-smooth(elapsed/phase_duration))*segment_start+smooth(elapsed/phase_duration)*paths[index][0][0]
            elif phase == 'WRITE':
                goal = sample_poly(paths[index][stroke], elapsed/phase_duration)
            elif phase == 'HOLD':
                goal = paths[index][-1][-1].copy()
            elif phase == 'RETRACT':
                goal = (1-smooth(elapsed/phase_duration))*retract_start+smooth(elapsed/phase_duration)*retract_end
            else:
                goal = retract_end.copy() if a.stationary else local_retract.copy()
            last_target = goal.copy()
            if step % 4 == 0:
                delta = R.T@(goal-d.xpos[link])
                pitch = np.clip(-np.arctan2(delta[2], np.linalg.norm(delta[:2])), -3.14/4, 3.14/3)
                yaw = np.clip(np.arctan2(delta[1], delta[0]), -3.14/4, 3.14/3)
                cp, sp, cy, sy = np.cos(pitch/2), np.sin(pitch/2), np.cos(yaw/2), np.sin(yaw/2)
                pose = np.array([delta[0], delta[1], goal[2], cp*cy, -sp*sy, sp*cy, cp*sy], np.float32)
                rotation = np.zeros(9)
                mujoco.mju_quat2Mat(rotation, d.qpos[3:7])
                gravity = rotation.reshape(3, 3).T@np.array([0., 0., -1.])
                obs = [d.qvel[3:6]*.2, gravity, d.qpos[qids[:18]]-default[:18], d.qvel[vids]*.05, action, vel, pose]
                for h, v in zip(history, obs):
                    h[:-1], h[-1] = h[1:], v
                with torch.inference_mode():
                    action = policy(torch.from_numpy(np.concatenate([h.ravel() for h in history])).unsqueeze(0)).squeeze(0).numpy()
                if action.shape != (len(names),) or not np.isfinite(action).all():
                    raise RuntimeError('Invalid policy output')
                target[:18] = np.clip(default[:18]+.25*action[:18], -10., 10.)
                if wheeled:
                    wheel_target = np.clip(10.*action[18:], -3., 3.)
                actual = d.xpos[ee].copy()
                rows.append(dict(t=t, phase=phase, letter=letters[index], stroke=stroke,
                    goal=goal.tolist(), actual=actual.tolist(), error_m=float(np.linalg.norm(actual-goal)),
                    base=d.qpos[:3].tolist(), base_quaternion=d.qpos[3:7].tolist(), command=vel.tolist(),
                    wheel_target=wheel_target.tolist(), wheel_actual=d.qvel[vids[18:]].tolist(), upright=float(-gravity[2]),
                    joint_position=d.qpos[qids].tolist(), joint_velocity=d.qvel[vids].tolist(),
                    joint_target=target.tolist(), raw_action=action.tolist(), pose_command=pose.tolist(),
                    ee_quaternion=d.xquat[ee].tolist(), link_position=d.xpos[link].tolist(), link_quaternion=d.xquat[link].tolist(),
                    self_contacts=[(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name) for c in d.contact if m.geom_bodyid[c.geom1] and m.geom_bodyid[c.geom2]],
                    ctrl_previous=d.ctrl[motors].tolist(), qfrc_passive=d.qfrc_passive[vids].tolist(), qfrc_bias=d.qfrc_bias[vids].tolist()))
                if phase == 'WRITE' and step % 8 == 0:
                    trails[index][stroke].append(actual)
                if d.qpos[2] < .20 or -gravity[2] < .3:
                    outcome = 'unstable_robot'
                    break
            if renderer is not None and step % 8 == 0:
                if not a.stationary:
                    cam.lookat[0] += .04*(.28+d.qpos[0]-cam.lookat[0])
                    cam.lookat[1] += .04*(d.qpos[1]-cam.lookat[1])
                renderer.update_scene(d, camera=cam, scene_option=option)
                for strokes in ([paths[index]] if a.placement=='after_shift' else paths):
                    for pts in strokes:
                        for p0, p1 in zip(pts[:-1], pts[1:]):
                            line(renderer.scene, p0, p1, cyan, .0018)
                for strokes in ([trails[index]] if a.placement=='after_shift' else trails):
                    for trail in strokes:
                        pts = trail[::2]
                        for p0, p1 in zip(pts[:-1], pts[1:]):
                            line(renderer.scene, p0, p1, orange, .0025)
                ball(renderer.scene, goal, cyan, .009)
                ball(renderer.scene, d.xpos[ee], orange, .008)
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                frame[:720] = renderer.render()
                cv2.rectangle(frame, (0, 0), (1920, 104), (20, 24, 30), -1)
                cv2.putText(frame, a.text+' | '+robot_label+' | '+('One fixed writing plane | Zero base commands' if a.stationary else 'Learned WBC in MuJoCo'), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1., (245, 245, 245), 2, cv2.LINE_AA)
                cv2.putText(frame, f'{index+1}/5  {letters[index]}: {phase}  |  vx=0  vy={vel[1]:+.2f} m/s  yaw rate=0  |  {t:.1f}s', (30, 84), cv2.FONT_HERSHEY_SIMPLEX, .76, (235, 235, 235), 2, cv2.LINE_AA)
                # Fixed-world overview, not normalized separately for each measured letter.
                panel = frame[720:1080, :]
                panel[:] = (24, 28, 34)
                measured = [p for strokes in trails for trail in strokes for p in trail]
                zlow = min(a.baseline_z-.025, min((p[2] for p in measured), default=a.baseline_z)-.025)
                zhigh = max(a.baseline_z+a.height+.025, max((p[2] for p in measured), default=a.baseline_z+a.height)+.025)
                scale = min(1800. if a.stationary else 1040., 260./(zhigh-zlow), 1650./(4*a.spacing+a.width))
                def pixel(p, i):
                    if a.placement=='fixed_world':
                        xpos = 960+p[1]*scale if a.stationary else 105+(p[1]+a.width/2)*scale
                    else:
                        xpos = 200+i*375+float((p-plane_origins[i])@plane_laterals[i])*scale
                    return (int(xpos), int(335-(p[2]-zlow)*scale))
                overview_label = 'World Y-Z trace' if a.placement=='fixed_world' else 'Per-letter local planes (NOT world spacing)'
                cv2.putText(panel, overview_label+' | Cyan: reference | Orange: measured | No IK / no resets', (30, 32), cv2.FONT_HERSHEY_SIMPLEX, .66, (220, 230, 240), 1, cv2.LINE_AA)
                for i, strokes in enumerate(paths):
                    if a.placement == 'after_shift':
                        cv2.putText(panel, letters[i], (188+i*375, 62), cv2.FONT_HERSHEY_SIMPLEX, .70, (220, 230, 240), 1, cv2.LINE_AA)
                    for pts in strokes:
                        cv2.polylines(panel, [np.array([pixel(v, i) for v in pts], np.int32)], False, (0, 209, 255), 2, cv2.LINE_AA)
                    for pts in trails[i]:
                        if len(pts)>1:
                            cv2.polylines(panel, [np.array([pixel(v, i) for v in pts], np.int32)], False, (255, 107, 20), 2, cv2.LINE_AA)
                # Enlarged current-letter inset uses a shared target/measured world frame.
                small = frame[140:600, 1460:1880]
                small[:] = (24, 28, 34)
                cv2.putText(small, f'{letters[index]}  |  {phase}', (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .70, (245, 245, 245), 1, cv2.LINE_AA)
                current_measured = [p for trail in trails[index] for p in trail]
                local_zlow = min(a.baseline_z-.025, min((p[2] for p in current_measured), default=a.baseline_z)-.025)
                local_zhigh = max(a.baseline_z+a.height+.025, max((p[2] for p in current_measured), default=a.baseline_z+a.height)+.025)
                local_scale = min(1050., 260./(local_zhigh-local_zlow))
                def local_pixel(p):
                    horizontal = p[1]-centers[index] if a.placement=='fixed_world' else float((p-plane_origins[index])@plane_laterals[index])
                    return (int(210+horizontal*local_scale), int(350-(p[2]-local_zlow)*local_scale))
                for pts in paths[index]:
                    cv2.polylines(small, [np.array([local_pixel(v) for v in pts], np.int32)], False, (0, 209, 255), 2, cv2.LINE_AA)
                for pts in trails[index]:
                    if len(pts)>1:
                        cv2.polylines(small, [np.array([local_pixel(v) for v in pts], np.int32)], False, (255, 107, 20), 2, cv2.LINE_AA)
                cv2.putText(small, f'3D error: {rows[-1]["error_m"]*100:.1f} cm', (16, 410), cv2.FONT_HERSHEY_SIMPLEX, .68, (230, 230, 230), 1, cv2.LINE_AA)
                writer.append_data(frame)
                if phase == 'HOLD' and elapsed < .045:
                    imageio.imwrite(a.output/f'{index+1}_{letters[index]}_finished.png', frame)
                if phase == 'FINAL':
                    final_frame = frame.copy()
            tau = kp*(target-d.qpos[qids])-kd*d.qvel[vids]
            if wheeled:
                tau[18:] = kd[18:]*(wheel_target-d.qvel[vids[18:]])
            for j, aid in enumerate(motors):
                if m.actuator_ctrllimited[aid]:
                    tau[j] = np.clip(tau[j], *m.actuator_ctrlrange[aid])
            d.ctrl[motors] = tau
            mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all() or d.warning.number.sum():
                raise RuntimeError('MuJoCo numerical failure')
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    if not a.no_video and 'final_frame' in locals():
        imageio.imwrite(a.output/'final.png', final_frame)
    metrics = {}
    for letter in letters:
        r = [v for v in rows if v['phase']=='WRITE' and v['letter']==letter]
        if r:
            err = np.array([v['error_m'] for v in r])
            metrics[letter] = dict(mean_error_cm=float(err.mean()*100), p95_error_cm=float(np.percentile(err,95)*100), seconds=len(r)*.02)
    summary = dict(robot=a.robot, initial_link0_world=initial_link0.tolist(),
        frame_adaptation='A2W existing frames' if wheeled else 'USD mount xyz and fixed ee_gripper frame in output-private MJCF; original untouched',
        actor_input_dim=sum(h.size for h in history), action_dim=len(names), joint_order=names,
        outcome=outcome, simulation_seconds=t, glyphs=a.text, letters=metrics, shifts=transitions,
        policy_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arguments={k:str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        world_paths=[[p.tolist() for p in strokes] for strokes in paths],
        plane_origins=[v.tolist() for v in plane_origins], plane_laterals=[v.tolist() for v in plane_laterals],
        fixed_base=False, ik_correction=False, simulation_resets=0, velocity_control='vx=vy=yaw_rate=0 throughout' if a.stationary else 'vx=0, yaw_rate=0; vy only in SIDE STEP',
        max_base_xy_displacement_m=max(float(np.linalg.norm(v['base'][:2])) for v in rows),
        stationary_writing=a.stationary,
        action_processing='affine_then_clip; wheel target limited to +/-3rad/s' if wheeled else 'affine_then_clip; 18 position targets, no wheels',
        note='Completing the sequence does not imply accurate letters; measured traces and movement timeouts are retained.')
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (a.output/'trace.json').write_text(json.dumps(rows)+'\n')
    (a.output/'events.json').write_text(json.dumps(events, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='world_paths'}, indent=2))


if __name__ == '__main__':
    main()
