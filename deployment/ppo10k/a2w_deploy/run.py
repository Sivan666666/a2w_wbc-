import argparse
import json
from pathlib import Path
import time
from .controller import Controller
from .teleop import Keyboard, VRReceiver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "deploy.json")
    parser.add_argument("--mode", choices=["keyboard", "vr"], default="keyboard")
    parser.add_argument("--interface")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    controller = Controller(cfg, args.config.resolve().parent / cfg["actor"])
    # Importing the extension is inert; constructing Hardware opens robot channels.
    from .hardware_backend import Hardware
    hardware = Hardware(args.interface or cfg["interface"], cfg["domain"], cfg["z1_host"],
                        cfg["z1_peer_port"], cfg["z1_local_port"], cfg["has_gripper"],
                        cfg["z1_kp_factor"], cfg["z1_kd_factor"],
                        cfg["gripper_kp_si"], cfg["gripper_kd_si"],
                        cfg.get("body_host", ""), cfg.get("body_peer_port",16840),
                        cfg.get("body_local_port",16841))
    vr = VRReceiver(cfg["vr_bind"], cfg["vr_port"]) if args.mode == "vr" else None
    print("z:起立准备 c:策略 r:停止推理并恢复准备姿态 t:机械臂回零后回蹲 x:阻尼")
    print("wasd/qe:底盘 Space:清零 | ijkluo:末端XYZ")
    print("1/2 3/4 5/6:末端roll/pitch/yaw | h:目标复位 n/m:夹爪 | Ctrl+C:退出")
    if vr:
        print(f"VR UDP {cfg['vr_bind']}:{cfg['vr_port']}; Grip=6D末端, B=目标复位, Trigger=夹爪")
    try:
        print(f"PPO机械臂伺服：Kp={cfg['policy_arm_kp']}，Kd={cfg['policy_arm_kd']}，前馈={cfg['policy_arm_feedforward']}。", flush=True)
        print("启动流程：退出狗的原生运控 → Z1 backToStart 归零 → 等待按 z。", flush=True)
        hardware.start()
        print("Z1 backToStart 已完成；当前 idle，按 z 腿臂一起准备，完成后按 c 启动 PPO。", flush=True)
        with Keyboard() as keyboard:
            deadline = time.monotonic()
            while True:
                previous_phase = controller.phase
                key = keyboard.read()
                if key:
                    key = key.lower()
                if key:
                    controller.key(key, hardware, time.monotonic())
                    if key in "zcrtx":
                        if key == 't':
                            print('Z1回零完成，正在2秒回蹲；完成后保持240/6。', flush=True)
                        if key == 'r':
                            print('PPO已停止；轮速目标归零，腿臂直接恢复准备姿态，完成后保持。', flush=True)
                        if key == 'c' and controller.phase == 'return_ready':
                            print('正在恢复准备姿态；完成后再次按 c 才启动 PPO。', flush=True)
                        if key == 'c' and controller.phase == 'idle':
                            print('先按 z 完成腿臂准备。', flush=True)
                        if controller.policy_requested:
                            print('c 已收到；准备轨迹完成后进入 PPO。', flush=True)
                if vr:
                    for controllers in vr.drain():
                        controller.vr(controllers)
                controller.tick(hardware, time.monotonic())
                if controller.phase != previous_phase:
                    print(f'{previous_phase} -> {controller.phase}', flush=True)
                    if previous_phase == 'return_ready' and controller.phase == 'ready':
                        print('已恢复站立准备姿态并保持；按 c 可重新启动 PPO。', flush=True)
                deadline += .02
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    deadline = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        hardware.stop()
        if vr:
            vr.close()


if __name__ == "__main__":
    main()
