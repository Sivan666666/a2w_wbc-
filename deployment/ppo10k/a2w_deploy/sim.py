import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from queue import SimpleQueue
import time
import numpy as np


def save_result(out, rows, backend, demo):
    out.mkdir(parents=True, exist_ok=True)
    trace = {k: np.asarray([row[k] for row in rows]) for k in rows[0]}
    np.savez_compressed(out/'trace.npz', **trace)
    if demo:
        (out/'inputs.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in demo.events))
    def stats(mask):
        return {k: {'mean':float(np.mean(trace[k][mask])), 'p95':float(np.percentile(trace[k][mask],95)),
                    'max':float(np.max(trace[k][mask]))}
                for k in ('ee_cm','orientation_deg','xy_error','yaw_error','tilt_deg')}
    phases = {}
    for name in dict.fromkeys(trace['label']):
        mask=trace['label']==name
        settled=mask & (trace['phase_age']>=2.)
        phases[str(name)]={'frames':int(mask.sum()),'all_frames':stats(mask),
                           'after_first_2s':stats(settled) if settled.any() else None}
    low_or_tilt=(trace['base_height']<.2)|(trace['tilt_deg']>60.)
    # Crouching intentionally starts below the standing-policy height threshold.
    # Preserve all low-height samples, but attribute policy falls only to policy.
    policy=trace['phase']=='policy'
    fallen=low_or_tilt & policy
    report={'seconds':float(trace['time'][-1]),'control_frames':len(rows),
            'policy_frames':int(policy.sum()),
            'model':'Precision PPO10K seed42, checkpoint 15f1d7c295d42ee12...',
            'mujoco_version':__import__('mujoco').__version__,
            'physics_hz':1/backend.model.opt.timestep,'policy_hz':50,'resets':0,
            'fall_frames_height_lt_0_2_or_tilt_gt_60':int(fallen.sum()),
            'fall_metric_scope':'policy phase only; passive crouch/preparation has separate raw low-height statistics',
            'preparation_low_height_or_tilt_frames':int((low_or_tilt & ~policy).sum()),
            'all_phase_low_height_or_tilt_frames':int(low_or_tilt.sum()),
            'first_fall_time':float(trace['time'][fallen][0]) if fallen.any() else None,
            'numerical_warnings':backend.data.warning.number.tolist(),
            'min_base_height':float(trace['base_height'].min()),
            'all_frames':stats(np.ones(len(rows),dtype=bool)), 'phases':phases,
            'policy_frames_only':stats(policy) if policy.any() else None,
            'vr_input':('synthetic PXREA Controller packets over loopback UDP'
                        if any('vr' in event for event in demo.events)
                        else 'none; keyboard replay only') if demo else 'live input',
            'metrics':'Post-physics; native mixed-frame position and root-relative orientation; no resets or removed frames. Settled subset is reported separately.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('seconds','resets','fall_frames_height_lt_0_2_or_tilt_gt_60','numerical_warnings','all_frames')},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'deploy.json')
    parser.add_argument('--mode',choices=['keyboard','vr'],default='keyboard')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--demo',action='store_true',help='60 s keyboard + synthetic VR input replay')
    parser.add_argument('--initial',choices=['crouch','stand'],default='crouch')
    parser.add_argument('--seconds',type=float,default=60.)
    parser.add_argument('--video',type=Path)
    parser.add_argument('--output',type=Path,default=Path('sim_results'))
    args=parser.parse_args()
    if args.headless:
        os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    from .mujoco_backend import MuJoCo
    from .controller import Controller
    from .teleop import VRReceiver
    from .sim_demo import Demo,description
    from .sim_video import markers, Video
    cfg=json.loads(args.config.read_text())
    backend=MuJoCo(initial=args.initial,config=cfg)
    controller=Controller(cfg,args.config.resolve().parent/cfg['actor'])
    receiver=VRReceiver('127.0.0.1',0) if args.demo else (
        VRReceiver(cfg['vr_bind'],cfg['vr_port']) if args.mode=='vr' else None)
    demo=Demo(receiver) if args.demo else None
    video=Video(backend,args.video) if args.video else None
    rows=[]
    keys=SimpleQueue()
    def on_key(code):
        if 32 <= code <= 126:
            keys.put(chr(code).lower())
    print('MuJoCo: z stand | c PPO | r stop PPO / return ready | x damping | wasd/qe base | ijkluo XYZ | 1..6 orientation | h home',flush=True)
    try:
        with ExitStack() as stack:
            viewer=None
            if not args.headless:
                import mujoco.viewer
                viewer=stack.enter_context(mujoco.viewer.launch_passive(backend.model,backend.data,key_callback=on_key))
                viewer.opt.geomgroup[3]=0
                viewer.cam.distance,viewer.cam.azimuth,viewer.cam.elevation=2.5,135.,-19.
            deadline=time.monotonic()
            for tick in range(round(args.seconds/.02)):
                t=tick*.02
                if viewer and not viewer.is_running():break
                if demo:demo.step(tick,controller,backend)
                while not keys.empty():controller.key(keys.get(),backend,t)
                if receiver:
                    for packet in receiver.drain():controller.vr(packet)
                controller.tick(backend,t)
                backend.advance()
                metric=backend.metrics(controller.command)
                label,source,begin=description(t) if demo else (controller.phase,args.mode,0.)
                rows.append(dict(metric,label=label,phase_age=t-begin,phase=controller.phase,
                                 target_q=backend.target_q.copy(),target_dq=backend.target_dq.copy(),
                                 kp=backend.kp.copy(),kd=backend.kd.copy(),action=controller.last_action.copy()))
                if video and tick%2==0:
                    frame=video.frame(metric,label,source)
                    if tick in (200,800,1200,2000,2400):
                        args.output.mkdir(parents=True,exist_ok=True)
                        frame.save(args.output/f'frame_{tick:04d}.jpg')
                if viewer:
                    with viewer.lock():
                        viewer.user_scn.ngeom=0
                        markers(viewer.user_scn,metric)
                        viewer.cam.lookat[:]=backend.data.xpos[backend.base]+[.2,0,.1]
                    viewer.sync()
                    deadline+=.02
                    time.sleep(max(0.,deadline-time.monotonic()))
                if tick%500==0:
                    print(f'{t:5.1f}s {label}: EE {metric["ee_cm"]:.2f} cm / {metric["orientation_deg"]:.2f} deg',flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if video:video.close()
        if rows:save_result(args.output,rows,backend,demo)
        if demo:demo.close()
        if receiver:receiver.close()
        backend.stop()


if __name__=='__main__':
    main()
