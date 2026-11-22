"""Explicit real A2W stand/crouch cycle through PC2; no Z1 or PPO instance."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .controller import stand_leg_target
from .gains import phase_gains
from ._hardware import BodyControl


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=root/'deploy.nx.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stand-hold-seconds',type=float,default=2.)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text())
    body=BodyControl(config['body_host'],config['body_peer_port'],config['body_local_port'])
    body.wait_state()
    before=body.status()
    initial=body.state()
    crouch=np.asarray(initial['q'][:12])
    pre=np.asarray(config['stand_pre_leg_q'])
    stand=np.asarray(config['stand_final_leg_q'])
    duration=float(config['stand_seconds'])
    half=duration/2
    crouch_duration=float(config['crouch_seconds'])
    crouch_pre=np.asarray(config['crouch_pre_leg_q'])
    hold=args.stand_hold_seconds
    down_begin=duration+hold
    down_end=down_begin+crouch_duration
    fade_begin=down_end+1.
    end=fade_begin+1.5
    stand_kp,stand_kd=phase_gains(config,'stand')
    body_indices=np.r_[0:12,18:22]
    stand_kp,stand_kd=stand_kp[body_indices],stand_kd[body_indices]
    # Keep preparation gains through descent and crouch hold, with no gain
    # step at the stand-to-crouch transition. Only the final release fades Kp.
    crouch_kp,crouch_kd=stand_kp.copy(),stand_kd.copy()
    report=dict(initial_state=initial,initial_bridge=before,
                stand_leg_target=stand.tolist(),crouch_leg_target=crouch.tolist(),
                stand_seconds=duration,stand_hold_seconds=hold,crouch_seconds=crouch_duration,
                stand_pre_leg_target=pre.tolist(),crouch_pre_leg_target=crouch_pre.tolist(),
                stand_interpolation='two_stage_linear_position',stand_target_dq=0.,
                crouch_interpolation='two_stage_linear_position',crouch_target_dq=0.,
                stand_gain_ramp=False,
                stand_kp=stand_kp.tolist(),stand_kd=stand_kd.tolist(),
                crouch_kp=crouch_kp.tolist(),crouch_kd=crouch_kd.tolist(),
                z1_used=False,ppo_used=False,completed=False,
                completion_scope='trajectory sequence only; measured stance errors are reported separately')
    report_path=args.output/'summary.json'
    def save():report_path.write_text(json.dumps(report,indent=2)+'\n')
    save()
    rows=[]
    released=False
    try:
        print('Releasing native A2W motion through PC2.',flush=True)
        body.release_native()
        released=True
        report['native_release_acknowledged']=True
        state=body.state()
        start_q=np.asarray(state['q'][:12])
        start=time.monotonic(); deadline=start
        previous_phase=None
        down_start=None
        while True:
            t=time.monotonic()-start
            if t>=end:break
            target=np.zeros(16); velocity=np.zeros(16)
            gain=1.
            if t<duration:
                phase='rise_pre' if t<half else 'rise_stand'
                target[:12]=stand_leg_target(start_q,pre,stand,t,duration)
            elif t<down_begin:
                phase='stand_hold'; target[:12]=stand
            elif t<down_end:
                phase='crouch'
                if down_start is None:
                    state=body.state()
                    down_start=np.asarray(state['q'][:12])
                    report['standing_state']=state
                    report['standing_max_joint_error_rad']=float(np.abs(down_start-stand).max())
                    quat=state['quaternion']
                    report['standing_tilt_deg']=float(np.degrees(np.arccos(np.clip(
                        1-2*(quat[1]**2+quat[2]**2),-1,1))))
                    save()
                elapsed=t-down_begin
                target[:12]=stand_leg_target(down_start,crouch_pre,crouch,elapsed,crouch_duration)
            elif t<fade_begin:
                phase='crouch_hold'; target[:12]=crouch
            else:
                phase='crouch_release'; target[:12]=crouch
                gain=.5*(1+np.cos(np.pi*min(t-fade_begin,1.)))
            if phase!=previous_phase:
                print(f'{t:.3f}s {phase}',flush=True)
                previous_phase=phase
            # Full gains throughout ascent/hold/descent; only final release
            # fades Kp while retaining zero-velocity damping.
            kp,kd=(stand_kp,stand_kd) if t<down_begin else (crouch_kp,crouch_kd)
            active_kp=kp*gain
            active_kd=kd
            body.command(target,velocity,active_kp,active_kd)
            state=body.state()
            rows.append((t,phase,state,target.copy(),velocity.copy(),gain,
                         active_kp.copy(),active_kd.copy()))
            deadline+=.002
            delay=deadline-time.monotonic()
            if delay>0:time.sleep(delay)
            else:deadline=time.monotonic()
        report['completed']=True
    finally:
        if released:
            # Send damping before stopping the stream. The robot may then enter
            # STOP via its own timeout; native motion remains released.
            for _ in range(100):
                body.command(np.zeros(16),np.zeros(16),np.zeros(16),crouch_kd)
                time.sleep(.002)
        report['final_state']=body.state()
        report['final_bridge']=body.status()
        if rows:
            np.savez_compressed(args.output/'trace.npz',
                time=np.array([r[0] for r in rows]),phase=np.array([r[1] for r in rows]),
                q=np.array([r[2]['q'] for r in rows]),dq=np.array([r[2]['dq'] for r in rows]),
                quaternion=np.array([r[2]['quaternion'] for r in rows]),
                tick=np.array([r[2]['tick'] for r in rows]),
                target_q=np.array([r[3] for r in rows]),target_dq=np.array([r[4] for r in rows]),
                gain=np.array([r[5] for r in rows]),
                kp=np.array([r[6] for r in rows]),kd=np.array([r[7] for r in rows]))
        save()
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
