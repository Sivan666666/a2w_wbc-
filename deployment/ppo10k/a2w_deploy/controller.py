"""The same state machine and policy path for hardware and MuJoCo."""
import numpy as np
from .policy import Actor, Observation, KD, action_targets
from .gains import phase_gains
from .teleop import Command
from .arm_preparation import quintic


def stand_leg_target(start, pre, final, elapsed, duration):
    """Two linear position segments; the caller keeps leg dq targets at zero."""
    half = duration / 2
    if elapsed < half:
        u = np.clip(elapsed / half, 0., 1.)
        return (1-u)*start + u*pre
    u = np.clip((elapsed-half) / half, 0., 1.)
    return (1-u)*pre + u*final



class Controller:
    def __init__(self, config, actor_path):
        self.config = config
        self.stand_kp, self.stand_kd = phase_gains(config, 'stand')
        self.policy_kp, self.policy_kd = phase_gains(config, 'policy')
        self.actor = Actor(actor_path)
        self.command = Command(config)
        self.observation = Observation()
        self.phase = "idle"
        self.policy_requested = False
        self.last_action = np.zeros(22, dtype=np.float32)
        self.policy_takeover_seconds = config.get('policy_takeover_seconds', 1.0)
        # Hardware can retain ACT bridge servoing for PPO joint targets. The
        # simulation reference config still selects training PD / no feedforward.
        if config['prepare_arm_feedforward'] != 'inverse_dynamics' or config['policy_arm_feedforward'] not in ('none', 'inverse_dynamics'):
            raise ValueError('Supported policy feedforward: none or inverse_dynamics')
        self.policy_arm_ff_weight = float(config['policy_arm_feedforward'] == 'inverse_dynamics')

    def key(self, key, backend, now):
        key = key.lower()
        if key == 't':
            if self.phase in ('home_crouch', 'crouched'):
                return
            self.phase = 'home_crouch'
            self.policy_requested = False
            self.last_action[:] = 0
            self.observation = Observation()
            self.command.home()
            q, _, _, _ = backend.state()
            self.crouch_from = np.asarray(q).copy()
            # Hold the measured body pose while official SDK homing runs.
            backend.command(self.crouch_from, np.zeros(22), self.stand_kp,
                            self.stand_kd, self.command.gripper, arm_ff_weight=1.)
            self.home_arm_q = np.asarray(backend.home_arm())
            self.crouch_start = None
            return
        if key in ("z", "r"):
            q, dq, _, _ = backend.state()
            self.start_q, self.start_dq = np.asarray(q).copy(), np.asarray(dq).copy()
            self.pre_q = np.asarray(self.config['stand_pre_leg_q'])
            self.final_q = np.asarray(self.config['stand_final_leg_q'])
            requested = np.clip(self.config['arm_ready_q'],
                                self.config['arm_prepare_joint_min'], self.config['arm_prepare_joint_max'])
            self.arm_ready = np.asarray(backend.begin_arm_preparation(
                self.start_q[12:18], self.start_dq[12:18], requested,
                self.config['arm_ready_seconds']))
            self.stand_start = now
            self.returning_to_ready = key == 'r'
            self.phase = "return_ready" if self.returning_to_ready else "stand"
            self.policy_requested = False
            self.last_action = np.zeros(22, dtype=np.float32)
            self.observation = Observation()
            self.command.home()
        elif key == "c":
            if self.phase == 'ready':
                self.enter_policy(backend)
            elif self.phase == 'stand':
                self.policy_requested = True
        elif key == "x":
            self.phase = "damping"
            self.policy_requested = False
            self.command.velocity[:] = 0
            if hasattr(backend, 'arm_damping'):
                backend.arm_damping()
        else:
            self.command.key(key)

    def vr(self, packet):
        self.command.vr(packet)

    def enter_policy(self, backend):
        # Capture feedback at takeover, including static PD tracking error.
        # Clear the VR clutch so the next Grip anchors to this new target.
        self.command.capture_home(backend.capture_ee())
        self.phase = 'policy'
        self.policy_requested = False
        self.observation = Observation()
        # Keep the commanded preparation endpoint, not the sagged feedback.
        self.takeover_from = tuple(v.copy() for v in self.preparation_command)
        # Anchor at the first policy output, including a queued c transition.
        self.takeover_start = None

    def tick(self, backend, now):
        q, dq, quaternion, gyro = backend.state()
        command = self.command
        if self.phase == "policy":
            obs = self.observation.build(q, dq, quaternion, gyro, command.velocity, command.ee)
            self.last_action = self.actor(obs)
            self.observation.previous_action = self.last_action.copy()
            target_q, target_dq = action_targets(self.last_action)
            kp, kd = self.policy_kp, self.policy_kd
            ff_weight = self.policy_arm_ff_weight
            if self.takeover_start is None:
                self.takeover_start = now
            elapsed = now - self.takeover_start
            if self.policy_takeover_seconds > 0 and elapsed < self.policy_takeover_seconds:
                u = np.clip(elapsed / self.policy_takeover_seconds, 0., 1.)
                # Quintic blend: zero first/second derivatives at both ends.
                weight = u**3 * (10. + u * (-15. + 6. * u))
                q0, dq0, kp0, kd0 = self.takeover_from
                target_q = (1-weight)*q0 + weight*target_q
                target_dq = (1-weight)*dq0 + weight*target_dq
                kp = (1-weight)*kp0 + weight*kp
                kd = (1-weight)*kd0 + weight*kd
                ff_weight = (1.-weight) + weight*self.policy_arm_ff_weight
            backend.command(target_q, target_dq, kp, kd, command.gripper,
                            arm_ff_weight=ff_weight)
        elif self.phase in ('stand', 'return_ready', 'ready'):
            elapsed = max(now-self.stand_start, 0.)
            duration = (self.config['return_stand_seconds'] if self.returning_to_ready
                        else self.config['stand_seconds'])
            arm_duration = self.config['arm_ready_seconds']
            target_q, target_dq = np.zeros(22), np.zeros(22)
            if self.returning_to_ready:
                # Stop the actor and return directly to the final standing pose.
                # Do not replay Z's crouched intermediate pose from a standing robot.
                target_q[:12], target_dq[:12], _ = quintic(
                    self.start_q[:12], self.start_dq[:12], self.final_q, elapsed, duration)
            else:
                target_q[:12] = stand_leg_target(
                    self.start_q[:12], self.pre_q, self.final_q, elapsed, duration)
            target_q[12:18], target_dq[12:18], arm_ddq = quintic(
                self.start_q[12:18], self.start_dq[12:18], self.arm_ready, elapsed, arm_duration)
            # ACT preparation gains are applied at the measured starting pose.
            # Native backend samples this trajectory at 500 Hz, independently
            # of the 50 Hz policy/input loop.
            kp, kd = self.stand_kp.copy(), self.stand_kd.copy()
            backend.command(target_q, target_dq, kp, kd, command.gripper,
                            arm_ddq=arm_ddq, arm_ff_weight=1., arm_preparation=True)
            self.preparation_command = tuple(v.copy() for v in (target_q, target_dq, kp, kd))
            if elapsed >= max(duration, arm_duration) and self.phase in ('stand', 'return_ready'):
                self.phase = 'ready'
                self.command.capture_home(backend.capture_ee())
                if self.policy_requested:
                    self.enter_policy(backend)
        elif self.phase in ('home_crouch', 'crouched'):
            if self.crouch_start is None:
                self.crouch_start = now
            elapsed = now-self.crouch_start
            target = np.zeros(22)
            target[:12] = stand_leg_target(
                self.crouch_from[:12], np.asarray(self.config['crouch_pre_leg_q']),
                np.asarray(self.config['crouch_final_leg_q']), elapsed,
                self.config['crouch_seconds'])
            target[12:18] = self.home_arm_q
            backend.command(target, np.zeros(22), self.stand_kp, self.stand_kd,
                            command.gripper, arm_ff_weight=1.)
            if elapsed >= self.config['crouch_seconds']:
                self.phase = 'crouched'
        elif self.phase == "damping":
            backend.command(q, np.zeros(22), np.zeros(22), KD, command.gripper)
