# 现有 NX / PC2 安装启动说明

本仓库快照对应安装目录 `/home/anx/wbc/ppo10k_20261122/a2w_z1_ppo10k_deploy`。该目录的父目录包含 `.venv/`、`vendor/z1_sdk`、`vendor/z1_controller`、`vendor/unitree_sdk2`。`scripts/nx_env.sh` 会配置这些路径。新 clone 的 `deployment/ppo10k` 不是已安装的 NX 环境，通用编译入口见 README。

## 分工

- NX：PPO、键盘/VR、Z1 SDK；机器人网卡 eno1，地址 192.168.124.25。
- PC2：A2W DDS 和原生运控切换；地址 192.168.124.162。
- Z1 MCU：192.168.124.110:8880；控制器本地 UDP 8881。
- NX 上策略与 z1_ctrl 使用 127.0.0.1 的 8072/8071；PC2/NX body bridge 使用 16840/16841。
- 局域网地址是本机安装示例，其他机器需按自己的网络修改。

## 已安装机器

先让狗趴稳、机械臂上电。NX 登录（此 alias/HostKeyAlias 是现有工作站配置）：

```bash
ssh -t -o HostName=192.168.1.140 -o HostKeyAlias=192.168.1.173 nx
cd /home/anx/wbc/ppo10k_20261122/a2w_z1_ppo10k_deploy
```

在 PC2 上只启动一个 body bridge；已运行就沿用。PC2 已编译二进制启动命令：

```bash
/home/unitree/ppo10k_20261122/build/pc2_body_bridge eth0 192.168.124.25 16840 16841 0
```

NX 终端一，只启动一个控制器：

```bash
bash run_nx_z1_controller.sh
```

NX 终端二，键盘和 VR 二选一：

```bash
cd /home/anx/wbc/ppo10k_20261122/a2w_z1_ppo10k_deploy
bash run_nx_keyboard.sh
# 或 bash run_nx_vr.sh
```

启动完整入口会退出原生运控并让机械臂 backToStart；等待完成后按 Z，ready 后按 C。当前机械臂策略 Kp `[50,50,80,30,30,20]`、Kd `[3,2,3,3,2.5,1]`，动力学前馈开启，准备仍为高增益。按键含义见 README。

修改配置必须退出旧部署进程再启动才能生效。正常结束应先按 T 完成回零/回蹲，再在部署终端 Ctrl+C 退出；如果还要重启 z1_ctrl，先确保发送客户端已退出，再在控制器终端 Ctrl+C。不要为了加载配置启动第二个发送进程。T 最新修复仍只有离线报文验证记录，未在本次发布中补做真机复测。

单独读取 Z1（只在没有其他 SDK 客户端占用 8072 时运行）：

```bash
bash run_nx_read_z1.sh
```

`run_nx_arm_prepare.sh`、`a2w_deploy.body_cycle` 是会产生运动的独立动作程序，不是常规键盘启动的前置步骤。

## 在 PC2 从本仓库编译 bridge

```bash
cd deployment/ppo10k
cmake -S . -B build -DBODY_BRIDGE_ONLY=ON \
  -DUNITREE_SDK2_ROOT=/path/to/unitree_sdk2
cmake --build build --target pc2_body_bridge -j4
./build/pc2_body_bridge eth0 192.168.124.25 16840 16841 0
```

无需在 PC2 安装 PyTorch/Z1 SDK；只读转发启动后，收到 NX 显式命令才下发 A2W 指令。
