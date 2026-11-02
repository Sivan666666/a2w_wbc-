"""Isolated WheelV2 bridge: native actor, mixed-frame EE commands, no IK/base locking."""
import argparse
from collections import deque
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

NAMES = [f'{leg}_{j}_joint' for leg in ('FR','FL','RR','RL') for j in ('hip','thigh','calf')]
NAMES += [f'joint{i}' for i in range(1,7)] + [f'{leg}_wheel_joint' for leg in ('FR','FL','RR','RL')]
DEFAULT = np.array([0.,.8,-1.5]*4+[0.]*10)
KP = np.array([120.]*12+[50.,50.,80.,30.,30.,20.]+[0.]*4)
KD = np.array([3.]*12+[3.,2.,3.,3.,2.5,1.]+[.6]*4)
LIMITS = np.array([120.,120.,180.]*4+[30.,60.,30.,30.,30.,30.]+[20.]*4)
SPEED_LIMITS = np.array([22.,22.,14.6667]*4+[10.]*6+[50.]*4)
DIMS = [3,3,18,22,22,3,7]
CASES = {'vx1': [1,0,0], 'vy_pos': [0,.2,0], 'vy_neg': [0,-.2,0],
    'yaw_pos':[0,0,.5], 'yaw_neg':[0,0,-.5],
    'mixed_pos':[.6,.15,.35], 'mixed_neg':[.6,-.15,-.35]}

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def scene(source, output):
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find('compiler')
    compiler.set('meshdir', str((source.parent/compiler.get('meshdir','')).resolve()))
    for body in root.findall('.//body'):
        if body.get('name') in ('wrist_up_mount','wrist_down_mount'):
            for geom in body.findall('geom'):
                if geom.get('group') == '1':
                    geom.set('quat','0.7071067811865476 0.7071067811865476 0 0')
    for joint in root.findall('.//joint'):
        if joint.get('name') in NAMES:
            limit = LIMITS[NAMES.index(joint.get('name'))]
            joint.set('actuatorfrcrange', f'{-limit} {limit}')
    for motor in root.findall('./actuator/motor'):
        if motor.get('joint') in NAMES:
            limit = LIMITS[NAMES.index(motor.get('joint'))]
            motor.set('ctrlrange', f'{-limit} {limit}')
    asset = root.find('asset')
    ET.SubElement(asset,'texture',name='mjcheck',type='2d',builtin='checker',rgb1='.26 .30 .34',rgb2='.38 .42 .46',width='256',height='256')
    ET.SubElement(asset,'material',name='mjfloor',texture='mjcheck',texrepeat='100 100',reflectance='.05')
    floor = root.find('.//geom[@name="floor"]')
    floor.set('material','mjfloor')
    floor.set('rgba','1 1 1 1')
    visual = root.find('visual')
    if visual is None: visual = ET.SubElement(root,'visual')
    headlight = visual.find('headlight')
    if headlight is None: headlight = ET.SubElement(visual,'headlight')
    headlight.set('ambient','.45 .45 .45')
    headlight.set('diffuse','.8 .8 .8')
    headlight.set('specular','.1 .1 .1')
    for light in root.findall('./worldbody/light'):
        light.set('directional','true')
        light.set('castshadow','false')
        light.set('dir','-1 -1 -2')
    tree.write(output, encoding='utf-8', xml_declaration=True)
    return mujoco.MjModel.from_xml_path(str(output))

def mapping(m):
    ids = np.array([m.joint(n).id for n in NAMES])
    motors = np.array([np.flatnonzero(m.actuator_trnid[:,0]==j).item() for j in ids])
    return m.jnt_qposadr[ids], m.jnt_dofadr[ids], motors

def parity(m, export, output):
    z = np.load(export/'native_parity_states.npz')
    qids, vids, _ = mapping(m)
    d = mujoco.MjData(m)
    bodies = [m.body(n).id for n in ('base','link0','end_effector')]
    max_p, max_r, max_obs = np.zeros(3), np.zeros(3), np.zeros(7)
    for i in range(len(z['q'])):
        for j in range(z['q'].shape[1]):
            d.qpos[:7] = z['root_pose'][i,j]
            d.qpos[qids] = z['q'][i,j]
            d.qvel[vids] = z['dq'][i,j]
            mujoco.mj_forward(m,d)
            r = d.xmat[bodies[0]].reshape(3,3)
            gravity = r.T@np.array([0.,0.,-1.])
            observed = np.split(z['obs'][i,j], np.cumsum(np.array(DIMS)*3)[:-1])
            latest = [x.reshape(3,n)[-1] for x,n in zip(observed,DIMS)]
            reconstructed = [z['root_ang_vel_b'][i,j]*.2, gravity,
                d.qpos[qids[:18]]-DEFAULT[:18],d.qvel[vids]*.05,latest[4],
                z['vel_command'][i,j],z['ee_command'][i,j]]
            max_obs = np.maximum(max_obs, [np.max(np.abs(x-y)) for x,y in zip(latest,reconstructed)])
            reference = z['body_pose'][i,j]
            max_p = np.maximum(max_p,np.linalg.norm(d.xpos[bodies]-reference[:,:3],axis=-1))
            for b,bid in enumerate(bodies):
                q = reference[b,3:].astype(float); q /= np.linalg.norm(q)
                dot = abs(np.dot(d.xquat[bid],q))
                max_r[b] = max(max_r[b],2*np.arccos(np.clip(dot,0,1)))
    result = dict(samples=400,bodies=['base','link0','end_effector'],
        max_position_error_m=max_p.tolist(), max_orientation_error_rad=max_r.tolist(),
        latest_obs_max_abs_error=max_obs.tolist(),
        passed=bool(max_p.max()<2e-4 and max_r.max()<2e-3 and max_obs.max()<2e-4))
    save(output/'bridge_parity.json',result)
    assert result['passed'], result
    return result

def sample_pose(rng, d, link):
    # Exact upstream construction: mixed-frame XYZ, then pointing pitch/yaw + random Euler offsets.
    p = rng.uniform([.55,-.35,.1],[.85,.35,1.])
    R = d.xmat[link].reshape(3,3)
    world = d.xpos[link]+R@np.r_[p[:2],0.]
    world[2] = p[2]
    delta = R.T@(world-d.xpos[link])
    euler = rng.uniform([-3.14/3,-3.14/4,-3.14/6],[3.14/3,3.14/4,3.14/6])
    euler[1] += -np.arctan2(delta[2],np.linalg.norm(delta[:2]))
    euler[2] += np.arctan2(delta[1],delta[0])
    roll,pitch,yaw = np.clip(euler,-3.14/4,3.14/3)
    cr,sr,cp,sp,cy,sy = np.cos(roll/2),np.sin(roll/2),np.cos(pitch/2),np.sin(pitch/2),np.cos(yaw/2),np.sin(yaw/2)
    return np.r_[p,[cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy]].astype(np.float32)

def ball(scene, p, color, radius):
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.ones(3)*radius,p,np.eye(3).ravel(),np.array(color,np.float32))
    scene.ngeom += 1

def run(m, policy, out, case, seed, seconds, video=False, delay_mode='random'):
    out.mkdir(parents=True,exist_ok=False)
    rng = np.random.default_rng(seed)
    delay_rng = np.random.default_rng(seed+1000000)
    delays = delay_rng.integers(0,5,4) if delay_mode=='random' else np.zeros(4,dtype=int)
    groups = [np.array([i for i in range(12) if i%3!=2]),np.arange(2,12,3),np.arange(18,22),np.arange(12,18)]
    joint_delay = np.zeros(22,dtype=int)
    for ids, delay in zip(groups,delays): joint_delay[ids] = delay
    qids,vids,motors = mapping(m)
    d = mujoco.MjData(m)
    d.qpos[2],d.qpos[3:7],d.qpos[qids] = .55,[1.,0.,0.,0.],DEFAULT
    mujoco.mj_forward(m,d)
    base,link,ee = [m.body(n).id for n in ('base','link0','end_effector')]
    history = [np.zeros((3,n),np.float32) for n in DIMS]
    action = np.zeros(22,np.float32)
    target = DEFAULT.copy()
    wheel = np.zeros(4)
    buffer = deque(maxlen=5)
    pose = None
    resample_at = 0.
    last_resample = 0.
    rows = []
    samples = []
    max_speed = np.zeros(22)
    speed_exceed_counts = np.zeros(22,dtype=int)
    renderer = writer = None
    cam = mujoco.MjvCamera()
    cam.distance,cam.azimuth,cam.elevation = 2.6,135.,-18.
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    if video:
        renderer = mujoco.Renderer(m,height=720,width=1280)
        writer = imageio.get_writer(out/'play.mp4',fps=25,codec='libx264',quality=8,macro_block_size=1)
    failure = None
    schedule = list(CASES)[1:]
    try:
        for tick in range(round(seconds/.02)):
            t = tick*.02
            phase = schedule[min(int(t//8),5)] if case=='schedule' else case
            vel = np.array(CASES[phase],np.float32)
            if t+1e-9>=resample_at:
                pose = sample_pose(rng,d,link)
                last_resample = t
                resample_at = t+rng.uniform(8,10)
                samples.append(dict(t=t,pose=pose.tolist()))
            R = d.xmat[base].reshape(3,3)
            obs = [d.qvel[3:6]*.2,R.T@np.array([0,0,-1]),
                d.qpos[qids[:18]]-DEFAULT[:18],d.qvel[vids]*.05,action,vel,pose]
            for h,x in zip(history,obs):
                if tick==0: h[:]=x
                else: h[:-1],h[-1]=h[1:],x
            with torch.inference_mode():
                action = policy(torch.from_numpy(np.concatenate([h.ravel() for h in history])).unsqueeze(0)).squeeze(0).numpy()
            assert action.shape==(22,) and np.isfinite(action).all()
            target[:18] = np.clip(DEFAULT[:18]+.25*action[:18],-10,10)
            wheel = np.clip(5*action[18:],-100,100)
            targets = np.r_[target[:18],wheel]
            for sub in range(4):
                buffer.append(targets.copy())
                delayed = np.array([buffer[max(0,len(buffer)-1-joint_delay[j])][j] for j in range(22)])
                tau = KP*(np.r_[delayed[:18],np.zeros(4)]-d.qpos[qids])-KD*d.qvel[vids]
                tau[18:] = .6*(delayed[18:]-d.qvel[vids[18:]])
                d.ctrl[motors] = np.clip(tau,-LIMITS,LIMITS)
                mujoco.mj_step(m,d)
                speeds = np.abs(d.qvel[vids])
                max_speed = np.maximum(max_speed,speeds)
                speed_exceed_counts += speeds>SPEED_LIMITS
                if not np.isfinite(d.qpos).all() or d.warning.number.sum():
                    failure = 'numerical_warning'
                    break
            mujoco.mj_forward(m,d)
            R = d.xmat[base].reshape(3,3)
            L = d.xmat[link].reshape(3,3)
            actual = np.r_[(R.T@d.qvel[:3])[:2],d.qvel[5]]
            local = L.T@(d.xpos[ee]-d.xpos[link])
            native = np.linalg.norm(np.r_[local[:2]-pose[:2],d.xpos[ee,2]-pose[2]])
            goal = d.xpos[link]+L@np.r_[pose[:2],0.]
            goal[2] = pose[2]
            world = np.linalg.norm(d.xpos[ee]-goal)
            desired_q = np.zeros(4)
            mujoco.mju_mulQuat(desired_q,d.xquat[link],pose[3:].astype(float))
            orientation = 2*np.arccos(np.clip(abs(np.dot(desired_q,d.xquat[ee])),0,1))
            tilt = np.degrees(np.arccos(np.clip(R[2,2],-1,1)))
            rows.append(dict(t=t+.02,phase=phase,vel_command=vel.copy(),actual_velocity=actual,
                ee_native_m=native,ee_world_m=world,orientation_rad=orientation,
                velocity_xy_error=np.linalg.norm(actual[:2]-vel[:2]),yaw_error=abs(actual[2]-vel[2]),
                height=d.qpos[2],tilt=tilt,command_age=t-last_resample,
                ee_command=pose.copy(),ee=d.xpos[ee].copy(),goal=goal.copy(),
                root_pose=d.qpos[:7].copy(),wheel_target=wheel.copy(),wheel_speed=d.qvel[vids[18:]].copy(),
                wheel_torque=d.ctrl[motors[18:]].copy(),action=action.copy()))
            if d.qpos[2]<.2 or tilt>60: failure = failure or 'fall_height_or_tilt'
            if video and tick%2==0:
                cam.lookat[:] = d.qpos[:3]+np.array([.25,0,.1])
                renderer.update_scene(d,camera=cam,scene_option=opt)
                ball(renderer.scene,goal,[0,.85,1,1],.04)
                ball(renderer.scene,d.xpos[ee],[1,.42,.08,1],.03)
                frame = renderer.render()
                cv2.rectangle(frame,(0,0),(1280,124),(20,24,30),-1)
                text = [f'A2W + Z1 | WheelV2 20K | MuJoCo | {phase} | t={t+.02:.2f}s',
                    f'CMD vx/vy/wz: {vel[0]:+.2f} / {vel[1]:+.2f} / {vel[2]:+.2f}     ACT: {actual[0]:+.2f} / {actual[1]:+.2f} / {actual[2]:+.2f}',
                    f'EE native {native*100:.1f} cm | world {world*100:.1f} cm | orientation {np.degrees(orientation):.1f} deg']
                for j,line in enumerate(text): cv2.putText(frame,line,(24,32+36*j),cv2.FONT_HERSHEY_SIMPLEX,.69,(240,240,240),1,cv2.LINE_AA)
                cv2.rectangle(frame,(0,678),(1280,720),(20,24,30),-1)
                cv2.putText(frame,'CYAN: target position     ORANGE: measured EE     Random position + orientation | No IK',(24,705),cv2.FONT_HERSHEY_SIMPLEX,.64,(230,230,230),1,cv2.LINE_AA)
                writer.append_data(frame)
                if tick in (200,600,1000,1400,1800,2200): imageio.imwrite(out/f'preview_{tick:04d}.jpg',frame)
            if failure: break
    finally:
        if writer: writer.close()
        if renderer: renderer.close()
    z = {k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(out/'trace.npz',**z)
    def stats(mask):
        if not mask.any():
            return dict(samples=0)
        return dict(samples=int(mask.sum()),actual_velocity_mean=z['actual_velocity'][mask].mean(0).tolist(),
            **{k:dict(mean=float(z[k][mask].mean()),p95=float(np.quantile(z[k][mask],.95))) for k in
            ['ee_native_m','ee_world_m','orientation_rad','velocity_xy_error','yaw_error']})
    summary = dict(case=case,seed=seed,requested_seconds=seconds,completed_seconds=float(z['t'][-1]),
        completed=bool(failure is None and len(rows)==round(seconds/.02)),failure=failure,
        actuator_delay_steps=dict(zip(['hips_thighs','calves','wheels','arm'],delays.tolist())),
        min_height=float(z['height'].min()),max_tilt=float(z['tilt'].max()),
        max_abs_joint_speeds=max_speed.tolist(),native_speed_limit_exceed_physics_steps=speed_exceed_counts.tolist(),
        wheel_speed_limit_exceed_steps=int(speed_exceed_counts[18:].sum()),
        wheel_mean_abs_speed=float(np.abs(z['wheel_speed']).mean()),wheel_mean_abs_target=float(np.abs(z['wheel_target']).mean()),
        all_frames=stats(np.ones(len(rows),bool)),after_initial_2s=stats(z['t']>2),
        target_age_over_2s=stats(z['command_age']>=2),
        phases={phase:stats(z['phase']==phase) for phase in set(z['phase'])})
    save(out/'summary.json',summary)
    save(out/'ee_commands.json',samples)
    print('RESULT',json.dumps(summary),flush=True)
    return summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xml',type=Path,required=True)
    p.add_argument('--export',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=list(CASES)+['schedule','all','parity'],default='all')
    p.add_argument('--seconds',type=float,default=60.)
    p.add_argument('--seeds',type=int,nargs='+',default=[20260916,20260917,20260918])
    p.add_argument('--video',action='store_true')
    p.add_argument('--delay-mode',choices=['random','zero'],default='random')
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    m = scene(a.xml.resolve(),a.output/'scene_wheelv2.xml')
    m.opt.timestep = .005
    qids,vids,motors = mapping(m)
    m.dof_armature[vids] = .01
    m.vis.global_.offwidth,m.vis.global_.offheight = 1280,720
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] and (m.geom_contype[gid] or m.geom_conaffinity[gid]): m.geom_group[gid]=3
    result = parity(m,a.export,a.output)
    manifest = dict(source_xml=str(a.xml),source_xml_sha256=hashlib.sha256(a.xml.read_bytes()).hexdigest(),
        export_manifest=json.loads((a.export/'manifest.json').read_text()),bridge_parity=result,
        bridge_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),mujoco_version=mujoco.__version__,
        joint_order=NAMES,kp=KP.tolist(),kd=KD.tolist(),torque_limits=LIMITS.tolist(),
        wheel_action_scale=5,wheel_target_clip=[-100,100],native_speed_limits=SPEED_LIMITS.tolist(),
        speed_limit_note='MuJoCo torque motors do not reproduce PhysX maxJointVelocity constraints. No qvel overwrite or extra braking; all native limit exceedances explicitly counted at 200 Hz.',
        protocol='Nominal URDF mass/friction, free base, no IK/external pushes/resets. 200 Hz physics/50 Hz policy, term-major three-frame history first-fill repeated. Random 0..4 physics-step actuator-group delays per trial. Native EE distributions and 8..10s resampling, numpy RNG (not bitwise Isaac RNG). Test failures stop trial and remain in report.')
    save(a.output/'manifest.json',manifest)
    if a.case=='parity': return
    policy = torch.jit.load(str(a.export/'policy.pt'),map_location='cpu').eval()
    results=[]
    for seed in a.seeds:
        for case in (list(CASES) if a.case=='all' else [a.case]):
            results.append(run(m,policy,a.output/f'{case}_seed{seed}',case,seed,a.seconds,a.video,a.delay_mode))
            save(a.output/'status.json',dict(stage='RUNNING',finished=len(results)))
    save(a.output/'summaries.json',results)
    save(a.output/'status.json',dict(stage='COMPLETE',finished=len(results)))

if __name__=='__main__': main()
