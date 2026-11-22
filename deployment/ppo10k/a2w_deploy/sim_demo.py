"""Deterministic input demo: keyboard events then synthetic PXREA packets.

Synthetic VR is explicitly labelled; it exercises the same UDP receiver and
command mapping as a real Pico bridge, without claiming a physical headset run.
"""
import json
import socket
import numpy as np
from scipy.spatial.transform import Rotation


EVENTS = {
    0: 'z',
    200: 'c',
    250: '111111',                        # +0.30 rad roll
    500: 'h44444',                        # -0.25 rad pitch
    750: 'h55555jjjjjjjjuuuuuu',          # yaw + position
    1000: 'h' + 'k'*30 + 'j'*10 + 'u'*5, # captured-home offset: dx=-.30, dy=.10, dz=.05
    1250: 'hwwwwwwqqq',                   # vx=.30, yaw=.15
    1500: 'h',
    1750: 'h',
}


def description(t):
    phases = [(0, 'Z: two-stage leg + arm preparation', 'keyboard replay'),
              (3, 'Ready: hold joint targets', 'keyboard replay'),
              (4, 'C: policy entry', 'keyboard replay'),
              (5, 'Roll +0.30 rad', 'keyboard replay'),
              (10, 'Pitch -0.25 rad', 'keyboard replay'),
              (15, 'Yaw +0.25 rad + XYZ target', 'keyboard replay'),
              (20, 'Home-relative reach: dx=-0.30 m', 'keyboard replay'),
              (25, 'Whole-body forward + turn', 'keyboard replay'),
              (30, 'Return to home / settle', 'keyboard replay'),
              (35, 'Continuous 6D end-effector motion', 'synthetic VR / UDP'),
              (45, '6D target + mobile base', 'synthetic VR / UDP'),
              (55, 'Grip release / B home', 'synthetic VR / UDP')]
    return next((label, source, begin) for begin, label, source in reversed(phases) if t>=begin)


class Demo:
    def __init__(self, receiver):
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = receiver.socket.getsockname()
        self.events = []

    def step(self, tick, controller, backend):
        t = tick*.02
        for key in EVENTS.get(tick, ''):
            controller.key(key, backend, t)
            self.events.append({'time':t, 'key':key})
        if t >= 35:
            u = t-35
            if t < 55:
                position = np.array([.12*np.sin(.55*u), .09*np.sin(.7*u), .08*np.sin(.4*u)])
                rpy = [.30*np.sin(.65*u), -.25*np.sin(.5*u), .20*np.sin(.4*u)]
                basis = np.array(controller.config['vr_basis'])
                vr_position = basis.T @ position
                vr_rotation = basis.T @ Rotation.from_euler('xyz',rpy).as_matrix() @ basis
                pose = np.r_[vr_position, Rotation.from_matrix(vr_rotation).as_quat()]
                right = dict(pose=','.join(map(str,pose)), grip=1., trigger=0., secondaryButton=False)
                left = dict(axisY=.30 if t>=45 else 0., axisX=.18*np.sin(.7*(t-45)) if t>=45 else 0.)
            else:
                right = dict(pose='0,0,0,0,0,0,1', grip=0., trigger=0., secondaryButton=t<55.1)
                left = dict(axisY=0.,axisX=0.)
            packet = {'right':right,'left':left}
            self.sender.sendto(json.dumps(packet).encode(), self.address)
            self.events.append({'time':t,'vr':packet})

    def close(self):
        self.sender.close()
