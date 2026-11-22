# A2W + Z1：PPO10K 部署版本（2026-11-23）

本目录保存用户于 2026-11-23 确认真机运行效果的部署配置：**机械臂准备阶段高增益，PPO 阶段训练低 Kp/Kd＋SDK 逆动力学前馈**。包含 NX/PC2 通信、键盘、Pico/PXREA VR、MuJoCo、PPO10K 权重和对应训练源码快照。它使用历史 Precision seed42 的第 10,000 轮模型，不是仓库主文档中的最终 50K 模型，也不是 SuperMimic actor。

## 当前参数

以下是 SI 增益，机械臂顺序为 J1–J6：

| 阶段 | 机械臂 Kp | 机械臂 Kd | 前馈 |
|---|---|---|---|
| 准备和准备保持 | `[512,768,768,512,384,256]` | `[25.6]*6` | SDK inverseDynamics |
| PPO 策略 | `[50,50,80,30,30,20]` | `[3,2,3,3,2.5,1]` | SDK inverseDynamics |

- 腿：起立/恢复/回蹲 `240/6`，策略 `120/3`；轮子 `Kp=0, Kd=0.6`。
- `deploy.nx.json` 是本次真机配置；`deploy.json` 保留训练低 PD、**零前馈**的仿真参考，两者不能混称为同一执行模型。
- 策略 50 Hz 更新关节位置目标，500 Hz 原生线程重复下发。策略阶段 `dq=0, ddq=0`，逆动力学此时主要提供重力项；没有新增持续五次插值或 IK。
- 按 C 后保留 1 秒五次权重接管，平滑混合目标与增益，机械臂前馈权重保持 1。
- 经典 Z1 SDK 的线缆增益为 `Kp/25.6`、`Kd/0.0128`；不要把这些系数与新版无倍率接口混用。

## 启动和操作

现有 NX/PC2 安装使用 [NX_RUN.md](NX_RUN.md) 的命令。启动完整部署程序会退出狗的原生运控并让 Z1 执行官方 `backToStart`，不是只读程序。

1. 操作者先让狗趴稳；PC2 body bridge 和 NX 上唯一的 `z1_ctrl` 已启动。
2. 运行键盘或 VR 入口，等待自动回零完成。
3. 按 `z`：腿用 2 秒、两段线性插值，经 `[0,1.36,-2.65]×4` 到 `[0,0.8,-1.5]×4`，从首帧使用 240/6；机械臂同时用 3 秒五次轨迹到准备角，使用高增益和动力学前馈。
4. `ready` 后按 `c`：捕获当前实际 EE 为目标、清零底盘指令，经过 1 秒接管进入 PPO。`ready` 表示计划轨迹结束，不表示关节误差经过阈值验收。

机械臂当前准备角为 `[0.01284927,2.17967153,-0.74254072,-1.5,-0.02659849,0.01928292] rad`。配置是实际控制依据，`startup_pose.json` 只记录历史候选姿态的来源。

| 按键 | 功能 |
|---|---|
| `z` | 腿臂准备 |
| `c` | 启动 PPO；在 Z 准备中按下会排队，idle 不直接启动 |
| `r` | 停止策略，腿 2 秒直接恢复站姿、机械臂 3 秒恢复准备姿态，保持高增益；需重新按 C 才推理 |
| `t` | 停止策略并清零轮速；腿保持支撑，Z1 官方回零后保留原生 JOINTCTRL，腿再用 2 秒回蹲并保持 240/6 |
| `x` | 进入阻尼；T 的原生零位保持状态下，Z1 转 PASSIVE |
| `w/s`, `a/d`, `q/e` | 底盘前后、左右、yaw 速度增减 |
| 空格 | 底盘速度清零 |
| `i/k`, `j/l`, `u/o` | EE 目标 X、Y、Z 增减 |
| `1/2`, `3/4`, `5/6` | EE roll/pitch/yaw 增减 |
| `h` | 复位到本次 C 捕获的 EE 目标，清零底盘命令 |
| `n/m` | 打开/闭合夹爪 |
| `Ctrl+C` | 退出，狗阻尼、Z1 PASSIVE；不自动恢复原生运控 |

按键增量会保留，松键不会自动清除底盘速度。`h` 是目标复位，不是机械臂回零。T 回零后屏蔽 LOWCMD 机械臂写入、清除旧轨迹及发送缓存；重复 T 不重放动作，仅 Z/R 会重新准备。这个 T 修复做过真实 SDK 的本机模拟端口验证，未在本次发布过程中重新做真机复测。

## 编译

Linux x86_64/aarch64、Python 3.10+、C++17、CMake、Eigen3、Python 开发头文件和 nlohmann-json 开发头文件。扩展必须与运行 Python 和目标 CPU 架构匹配。使用独立环境中的现有 CPU PyTorch，然后安装其余依赖：

```bash
cd deployment/ppo10k
python -m pip install -r requirements.txt
bash scripts/fetch_sdks.sh
bash scripts/build.sh
```

固定 SDK 版本记录于 `provenance.json`；Z1 控制器和 PXREA 厂家库不随仓库分发。可使用已经部署成功的 SDK：

```bash
DEPLOY_PYTHON=/path/to/python bash scripts/build.sh \
  -DUNITREE_SDK2_ROOT=/path/to/unitree_sdk2 \
  -DZ1_SDK_ROOT=/path/to/z1_sdk
```

如果使用通用入口和自有环境：

```bash
DEPLOY_PYTHON=/path/to/python bash run_keyboard.sh --config deploy.nx.json
# VR 二选一
DEPLOY_PYTHON=/path/to/python bash run_vr.sh --config deploy.nx.json
```

`run_nx_*.sh` 是现有安装布局的入口，使用父目录 `.venv` 和 `vendor/`，详见 NX_RUN.md。`z1_host` 是 `z1_ctrl` 地址（同机为 127.0.0.1），不是机械臂 MCU IP。

## VR

键盘和 VR 共用 Controller、命令坐标和 actor。VR 入口仍从终端接受 Z/C/R/T/X；左摇杆控制 vx/yaw，右 Grip 控制 EE 六维增量，右 B 复位目标，Trigger 控制夹爪。

在安装 PXREA RoboticsService 的机器启动厂商服务后：

```bash
python a2w_deploy/pxrea_bridge.py \
  --robot 192.168.124.25 \
  --sdk-lib /opt/apps/roboticsservice/SDK/arm64/libPXREARobotSDK.so
```

`--robot` 为策略主机地址，端口默认 9090；x86_64 使用 `SDK/x64/`。Pico 中连接 RoboticsService 并发送 Controller 数据。默认位移轴映射 `[-vr_z,-vr_x,vr_y]`；姿态处理保留完整四元数。

## 模型与接口

包含 `models/model_10000.pt` 和 `models/ppo10k_actor.jit`。checkpoint SHA256 为 `15f1d7c295d42ee12d75e1a7c241612c2a73471000da1beb3bc4a22a083fe87c`；actor SHA256 为 `08d8c1ed4494b7e9b07549ff05c7a762b9c36bcb1c71380948f742a7c6429cbc`。

- ELU MLP `234→512→256→128→22`，Identity 观测归一化，确定性输出。
- 每项观测分别保留三帧，从旧到新拼接：角速度×0.2、投影重力、18维相对关节位置、22维速度×0.05、22维上一原始动作、3维底盘命令、7维末端命令。
- 策略关节顺序：FR/FL/RR/RL 各 hip/thigh/calf，Z1 J1–J6，四轮。A2W 物理轮槽12–15对应策略18–21。
- 腿臂目标 `clip(default + 0.25*action,-10,10)`；轮速 `clip(5*action,-100,100)`，沿用训练动作处理。Z1 下发前使用 SDK jointProtect。
- EE 的 XY 为 link0 局部坐标、Z 为世界高度，姿态目标相对机身。真机高度由四轮同处水平地面的几何假设估计，不是外部定位；按 C 捕获 EE 不等于锁定机身世界位置。

`training_reference/` 保存 PPO10K 评测对应的训练源码快照，不是另一套可直接安装的完整训练环境。模型、参考 SDK 和资产来源见 `provenance.json`、`assets/a2w_z1/provenance.json` 与 `RELEASE.json`。

## 验证范围和已知问题

- 用户反馈本次“策略低 PD＋前馈”真机效果良好；没有把主观反馈当作正式精度统计。
- 低 PD＋前馈改变了训练时零前馈的执行关系。已有 MuJoCo 对照中虽然关节稳定，末端任务误差仍可能大于训练配置；不能以不抖动等同于精度已解决。
- 此快照保留现有 100 μs SDK 接收等待配置；出现过 UDP timeout warning，反馈新鲜度和收发队列问题尚未完成修复，本次不悄悄修改已确认的运行参数。
- 离线状态机、输入接口及 SDK 本机报文验证与真机运动验证分开记录。打包时没有重新运行机器人。

```bash
python -m unittest discover -s tests -v
python -m a2w_deploy.benchmark
```

[MuJoCo 与录制说明](SIMULATION.md)。`verification.json` 是初版离线验证存档，不能代表所有后续修改。参考 S10/Piper 的输入与分层架构、Unitree SDK2/Z1 SDK 和 rl_sar 的部署方式；相关来源固定在 provenance.json。SDK 及厂家库仍按各自上游授权获取，训练参考保留版权头和 LICENSE。
