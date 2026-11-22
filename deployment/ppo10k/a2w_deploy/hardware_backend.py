"""Native transport plus startup EE capture from measured joints and IMU."""
from .kinematics import Kinematics


class Hardware:
    def __init__(self, *args):
        from ._hardware import Hardware as NativeHardware
        self.kinematics = Kinematics()
        self.native = NativeHardware(*args)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def capture_ee(self):
        q, _, quaternion, _ = self.native.state()
        return self.kinematics.command(q, quaternion)
