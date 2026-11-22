"""Run on the PXREA RoboticsService host; relay raw 6D controller JSON via UDP."""
import argparse
import ctypes
import json
import signal
import socket


class DeviceState(ctypes.Structure):
    _fields_ = [("devID", ctypes.c_char * 32), ("stateJson", ctypes.c_char * 16352)]


Callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", required=True, help="host running a2w_deploy.run --mode vr")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--sdk-lib", required=True, help="libPXREARobotSDK.so from installed RoboticsService")
    args = parser.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sdk = ctypes.CDLL(args.sdk_lib, mode=ctypes.RTLD_GLOBAL)
    sdk.PXREAInit.argtypes = [ctypes.c_void_p, Callback, ctypes.c_uint]
    sdk.PXREAInit.restype = ctypes.c_int
    sdk.PXREADeinit.argtypes = []
    sdk.PXREADeinit.restype = ctypes.c_int

    @Callback
    def callback(_context, event, _status, data):
        if event != 1 << 25 or not data:
            return
        # Keep Python exceptions inside the C callback boundary.
        try:
            packet = ctypes.cast(data, ctypes.POINTER(DeviceState)).contents
            outer = json.loads(packet.stateJson.decode())
            value = outer["value"]
            if isinstance(value, str):
                value = json.loads(value)
            if "Controller" in value and "right" in value["Controller"]:
                sock.sendto(json.dumps(value["Controller"]).encode(), (args.robot, args.port))
        except (KeyError, ValueError, OSError) as error:
            print(f"PXREA callback: {error}", flush=True)

    result = sdk.PXREAInit(None, callback, 0xffffffff)
    if result:
        raise RuntimeError(f"PXREAInit returned {result}")
    print(f"Pico Controller JSON -> {args.robot}:{args.port}", flush=True)
    try:
        while True:
            signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        sdk.PXREADeinit()
        sock.close()


if __name__ == "__main__":
    main()
