"""Floating-base, full-dynamics MuJoCo implementation of the Hardware interface."""
from pathlib import Path
import json
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .policy import JOINTS, DEFAULT_Q, KP, KD
from .kinematics import ee_command
from .arm_preparation import quintic

SCENE = Path(__file__).resolve().parents[1] / "assets/a2w_z1/scene.xml"


class MuJoCo:
    def __init__(self, scene=SCENE, initial='crouch', config=None):
        config = config or json.loads((Path(__file__).resolve().parents[1]/'deploy.json').read_text())
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        # Separate, contact-free inverse dynamics workspace. This affects only
        # preparation/transition feedforward, never PPO's physics or gains.
        self.dynamics_model = mujoco.MjModel.from_xml_path(str(scene))
        self.dynamics_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.dynamics_data = mujoco.MjData(self.dynamics_model)
        joints = np.array([self.model.joint(name).id for name in JOINTS])
        self.qids = self.model.jnt_qposadr[joints]
        self.vids = self.model.jnt_dofadr[joints]
        self.arm_min = np.maximum(self.model.jnt_range[joints[12:18], 0], config['arm_prepare_joint_min'])
        self.arm_max = np.minimum(self.model.jnt_range[joints[12:18], 1], config['arm_prepare_joint_max'])
        self.motors = np.array([np.flatnonzero(self.model.actuator_trnid[:, 0] == j).item()
                                for j in joints])
        self.base, self.link, self.ee = [self.model.body(n).id for n in ("base", "link0", "end_effector")]
        self.data.qpos[:7] = [0., 0., .55, 1., 0., 0., 0.]
        self.data.qpos[self.qids] = DEFAULT_Q
        if initial == 'crouch':
            self.data.qpos[2] = config['sim_crouch_height']
            self.data.qpos[self.qids[:12]] = config['sim_crouch_leg_q']
        mujoco.mj_forward(self.model, self.data)
        # Crouch represents the operator's manual pre-start action; no automated
        # crouch motion is commanded. Idle is passive, matching released SDK mode.
        self.command(self.data.qpos[self.qids], np.zeros(22),
                     np.zeros(22), np.zeros(22), 0.)

    def start(self):
        pass

    def state(self):
        # XBODY is the URDF link frame. BODY uses the rotated principal-inertia
        # frame (ximat), which is NOT the robot IMU frame on this A2W asset.
        # flg_local=1 returns angular then linear velocity in the chosen frame.
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                self.base, velocity, 1)
        return (self.data.qpos[self.qids].copy(), self.data.qvel[self.vids].copy(),
                self.data.xquat[self.base].copy(), velocity[:3].copy())

    def begin_arm_preparation(self, start, velocity, target, duration):
        target = np.clip(target, self.arm_min, self.arm_max)
        self.arm_trajectory = (np.asarray(start).copy(), np.asarray(velocity).copy(), target.copy(), duration)
        self.arm_start_time = self.data.time
        return target.copy()

    def home_arm(self):
        # Simulation equivalent of SDK home; keep body targets while the arm
        # follows a smooth 3 s zero-joint trajectory. No encoder calibration.
        q, dq, _, _ = self.state()
        home = self.begin_arm_preparation(q[12:18], dq[12:18], np.zeros(6), 3.)
        self.arm_preparing = True
        for _ in range(150):
            self.advance()
        self.arm_preparing = False
        self.target_q[12:18] = home
        self.target_dq[12:18] = 0.
        self.arm_ddq[:] = 0.
        return home

    def command(self, q, dq, kp, kd, gripper, arm_ddq=None, arm_ff_weight=0., arm_preparation=False):
        self.target_q, self.target_dq = np.asarray(q).copy(), np.asarray(dq).copy()
        self.kp, self.kd = np.asarray(kp).copy(), np.asarray(kd).copy()
        self.gripper = gripper  # Training MJCF has a fixed gripper, not a 23rd action.
        self.arm_ddq = np.zeros(6) if arm_ddq is None else np.asarray(arm_ddq).copy()
        self.arm_ff_weight = arm_ff_weight
        self.arm_preparing = arm_preparation

    def arm_feedforward(self, target_q, target_dq, arm_ddq):
        d = self.dynamics_data
        d.qpos[:] = self.data.qpos
        d.qpos[self.qids[12:18]] = target_q[12:18]
        d.qvel[:] = 0.
        d.qvel[self.vids[12:18]] = target_dq[12:18]
        d.qacc[:] = 0.
        d.qacc[self.vids[12:18]] = arm_ddq
        mujoco.mj_inverse(self.dynamics_model, d)
        return d.qfrc_inverse[self.vids[12:18]].copy()

    def capture_ee(self):
        d = self.data
        return ee_command(d.xmat[self.base].reshape(3, 3), d.xpos[self.link],
                          d.xmat[self.link].reshape(3, 3), d.xpos[self.ee],
                          d.xmat[self.ee].reshape(3, 3))

    def advance(self):
        # Same 200 Hz physics / 50 Hz policy ratio as PPO training.
        for _ in range(round(.02 / self.model.opt.timestep)):
            target_q, target_dq, arm_ddq = self.target_q.copy(), self.target_dq.copy(), self.arm_ddq
            if self.arm_preparing:
                q0, v0, q1, duration = self.arm_trajectory
                target_q[12:18], target_dq[12:18], arm_ddq = quintic(
                    q0, v0, q1, self.data.time-self.arm_start_time, duration)
                target_q[12:18] = np.clip(target_q[12:18], self.arm_min, self.arm_max)
            torque = (self.kp * (target_q - self.data.qpos[self.qids]) +
                      self.kd * (target_dq - self.data.qvel[self.vids]))
            self.last_arm_feedforward = np.zeros(6)
            if self.arm_ff_weight > 0.:
                self.last_arm_feedforward = self.arm_ff_weight * self.arm_feedforward(target_q, target_dq, arm_ddq)
                torque[12:18] += self.last_arm_feedforward
            # Actuator ctrlrange/actuatorfrcrange implement training motor effort
            # limits, including WheelV2's 20 Nm wheel motor limit.
            self.data.ctrl[self.motors] = torque
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def metrics(self, command):
        d = self.data
        root_r = d.xmat[self.base].reshape(3, 3)
        link_r = d.xmat[self.link].reshape(3, 3)
        actual_r = d.xmat[self.ee].reshape(3, 3)
        target_r = root_r @ Rotation.from_quat(command.ee[[4, 5, 6, 3]]).as_matrix()
        # Training's mixed-frame position metric, not a fabricated world XYZ loss.
        local = link_r.T @ (d.xpos[self.ee] - d.xpos[self.link])
        native = np.linalg.norm(np.r_[local[:2] - command.ee[:2],
                                     d.xpos[self.ee, 2] - command.ee[2]])
        target_p = d.xpos[self.link] + link_r @ np.r_[command.ee[:2], 0.]
        # Solve the remaining link-local Z from the commanded world height.
        # Overwriting world Z alone would shift local XY on a tilted base.
        target_p += link_r[:, 2] * ((command.ee[2]-target_p[2]) / link_r[2, 2])
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_XBODY, self.base, velocity, 1)
        tilt = np.degrees(np.arccos(np.clip(root_r[2, 2], -1, 1)))
        return dict(time=float(d.time), ee_cm=float(native * 100),
                    orientation_deg=float(np.degrees(Rotation.from_matrix(target_r.T @ actual_r).magnitude())),
                    xy_error=float(np.linalg.norm(velocity[3:5] - command.velocity[:2])),
                    yaw_error=float(abs(velocity[2] - command.velocity[2])),
                    base_height=float(d.xpos[self.base, 2]), tilt_deg=float(tilt),
                    actual_velocity=np.r_[velocity[3:5], velocity[2]],
                    target_p=target_p, actual_p=d.xpos[self.ee].copy(),
                    target_r=target_r.copy(), actual_r=actual_r.copy(),
                    root_pose=d.qpos[:7].copy(), q=d.qpos[self.qids].copy(),
                    ee_command=command.ee.copy(), velocity_command=command.velocity.copy())

    def stop(self):
        pass
