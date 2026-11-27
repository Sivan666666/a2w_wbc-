# A + SDK 软件限位，无重力前馈：力估计与力控迁移

本目录是独立、显式启用的实现。基础为提交 `5d1555f` 的 hip A：无相机支架、SDK **软件模型**限位、原生延迟 PD；物理重力保留，不启用 SDK 重力前馈。不修改历史 A 的入口、奖励文件、模型或已有训练。

来源：[learning-compliance](https://github.com/Improbable-AI/learning-compliance/tree/c760e1d74ad165d3c069d4f57ab5d066f6a41eb6)，MIT（见 `vendor/LICENSE`）；[Learning Force Control for Legged Manipulation](https://arxiv.org/abs/2405.01402)。源码散列及映射见 `provenance.json`。

## 已迁移的链路

1. **位置／力模式**：每 10 秒重新采样模式，默认各 50%。力模式屏蔽 actor 的 EE 位置命令，保留姿态及底盘速度命令。
2. **模拟外力场**：每个物理子步在末端施加 `Kp*(anchor-position)-Kd*velocity`，Kp 随机 25–400 N/m，Kd=0.1*Kp，每轴限幅 ±70 N。转换为刚体局部坐标后施加，不直接改关节位置或把力命令转换成特权关节控制。
3. **力命令**：三轴 ±70 N；1–3 秒渐增、1 秒保持、渐减；20% freed 周期同时关闭力场和力命令。这是公开代码的行为，不能说成“零力目标时仍保留外力扰动”。
4. **并行状态估计**：10 帧本体观测历史 → 256/128 ELU MLP → 机身线速度、末端外力、末端位置，共 9 维。actor 只读历史和估计值；真实外力只用于 critic、监督损失和统计。估计器随 PPO 并行更新。
5. **力奖励**：逐轴 `3*exp(-abs(F_measured-F_command)/50)`，仅力模式启用，三轴相加。原仓库可选的 x/y 力幅值惩罚保留为 `core.force_magnitude_penalty`，默认不启用，和其发布训练配方一致。
6. **导出与恢复**：TorchScript 同时封装 actor 和 estimator；checkpoint 包含 actor/critic/estimator、两个 optimizer、学习率、更新计数、随机数状态。恢复时重置仿真 episode，**不声称恢复了 PhysX 的完整状态**。新增观测和网络结构不兼容旧 A checkpoint，不能直接载入旧 A 当作已训练力控策略。
7. **阻抗接口**：`inference.ImpedanceCommand` 用估计位置及差分速度产生力命令；不依赖仿真真值，不包含真机通信或电机指令。位置必须在连续一致的参考系中，不能把移动／旋转机身坐标下的位置差直接当作世界速度。

## 哪些保持我们的版本

髋约束 A（−0.3/−1/−0.2）、轮速惩罚（−0.05/−0.01，前 500 更新渐增）、低速重采样配方、底盘跟踪、姿态奖励、关节／力矩／动作等其余惩罚、22 维动作与原执行器均保留。40% 是低速重采样概率，不是帧占比。

**唯一需要说明的非力奖励调度变化**：原有粗／中／细三项 EE 位置奖励权重、公式及参数不变，但只在位置模式计分，在力模式置零，以免“顶住外力”和“必须停在原位置”互相冲突；EE 姿态奖励在两种模式都保留。运行会生成 `reward_audit.json`，逐项验证原奖励参数和权重。没有加入 2 cm 奖励、困难采样或重力前馈，也没有导入上游 B1 的步态／足端接触奖励来替换我们的轮足奖励。

## 力方向和真值含义

接口里的 F 是**环境施加在机器人上的末端外力**，以 base yaw 坐标表达；机器人施加给环境的准静态反作用力符号相反。不要把两者混用。当前监督真值是注入的弹簧阻尼外力，不是六维力传感器，不包含所有未知接触力、碰撞和外力矩。真实接触物体或真实机械臂上的估计精度需后续单独验证。

仿真重力是 `[0,0,-9.81]`；“无重力前馈”不等于无重力。

## 与论文／公开实现的明确差异

以公开训练脚本为参数依据：它实际用 10 帧历史、逐轴三项权重各 3、误差尺度 50 N；论文描述的 30 帧历史及范数奖励不能与此混称一致。源码中 Kd 最后被覆盖为 0.1*Kp。

- 保留我们的 A 仿真、奖励、22 维轮足＋六关节臂，未移植上游 17 维 B1 动作和执行器参数。
- 上游估计器输出还包含关节动力学、摩擦与 EE 目标等 16 维特权状态；这里迁移力控所需的速度／力／EE 位置 9 维，并保留我们的 critic 观测，不引入其他域随机化奖励。
- 历史使用 IsaacLab 的 **term-major、每项时间由旧到新**布局，reset 重复首帧；上游 wrapper 为 frame-major、reset 清零。这是观测 ABI 适配，导出必须使用本目录 `PolicyHistory`，不能直接套上游 wrapper。
- 默认外力锚点从我们的 EE 位置命令得到（link0 的 XY＋世界 Z），并处理 link0 Z 轴接近水平时的奇异分母；可选 `entry_pose` 为入段末端位置。公开源码默认是位置命令锚点，不能冒充论文描述的全部实现完全相同。
- 用经过时间构造连续力命令脉冲，修正上游结束时刻绝对时间混用；周期不得短于渐增＋保持＋渐减总长度。
- PPO 使用我们的 A 超参数并接入 CSE 双优化器；critic 为历史＋我们的特权观测。保留 A 的 timeout bootstrap 约定。既有 RSL checkpoint 不能直接在此入口恢复。

## 使用与验证

在能运行本仓库的 IsaacLab 环境中，从仓库根执行（`PYTHONPATH` 必须包含 `source/LeggedManip_Lab`）：

```bash
python reproduction/force_control/test_core.py
python reproduction/force_control/run.py --mode smoke --output /path/to/new-smoke --num_envs 16 --steps 300 --headless --device cuda:0
```

测试需在相邻目录保留上述 upstream checkout，用其真实奖励函数作独立数值对照。`smoke` 为 300 控制步＋**仅 2 次小规模 PPO 更新**，用于检查梯度、优化器、导出和恢复；不代表训练好的力控精度。输出路径必须新建，避免覆盖旧结果。

初次迁移时提供的正式训练入口（2026-11-27 已完成从零 30K，见下方发布说明）：

```bash
python reproduction/force_control/run.py --mode train --iterations TOTAL_UPDATES --output /path/to/new-run --num_envs 4096 --headless --device cuda:0
```

`--iterations` 是总更新数。`--resume CHECKPOINT` 恢复学习器，不能改变观测／动作 ABI 或 PPO 配置。评测入口 `--mode evaluate --checkpoint CHECKPOINT --steps N` 保存力估计与力跟踪的逐轴误差；没有训练完成前，不应将其数字解释为力控性能。

本次结果记录在工作区 `outputs/learning_compliance_migration_20261126`。最终验证摘要见 `VALIDATION.json`。该段描述初次迁移验证阶段；之后已完成正式 30K 训练，未启动真机。

## PS2 从零 30K 的阶段评测（2026-11-27）

新增 `--eval-interval 10000 --eval-steps 1500`：每达到 10K／20K／30K 保存 checkpoint 后，在同一卡上启动一个独立评测子进程。父学习器保留在内存中等待，评测完成再继续；不加载评测后的环境状态、不回退更新数。评测失败会留下 checkpoint 和错误日志并结束该训练进程，不静默跳过。

`stage_eval.py` 固定 19 工况 × 4 环境 × 30 秒：7 项历史 EE 工况；12 项力模式工况（零力、XYZ 各 ±20 N、正负对角力、带前进／倒车／转向）。外力场 Kp=100 N/m、Kd=10 Ns/m；0–8 秒初始姿态，8–12 秒渐变，然后保持。零力工况保留弹簧阻尼场，用于检查顺应响应。所有训练阶段使用相同 seed=20261127。该 seed 和新力控观测不同于旧 A 的历史评测，不能把差异归因于单一因素。

保存 `trace.npz`、`summary.json`、模型哈希、阶段状态与退出码。报告全程及末 5 秒的 EE cm／deg，力跟踪／力估计逐轴 N 和向量范数 N 的均值、P95、RMSE，底盘速度误差，臂／狗腿各自关节目标误差，以及失败事件／原因。EE 的 link0 姿态参考和 root 姿态参考分别记录，避免与历史计算混淆。位置模式 EE 是位置跟踪成绩；力模式下 EE 位置仅作诊断，不当成必须同时满足的位置目标。

力估计使用 actor 实际读取的行动前历史，其误差与同一时刻的真值配对；力跟踪与 EE 状态在物理步之后、reset 之前采集。每帧 finite 检查，失败帧不会悄悄丢弃。真实外力仍只代表注入场，不等于全部物体接触合力。

## 已完成的 30K 发布

本分支附带 `models/learning_compliance/checkpoint_0030000.pt` 和 `actor_0030000.jit`。训练来源提交 `6552923`，seed42、4096环境、30000更新；10/20/30K阶段评测全部完成。详细复现命令、哈希与指标见 [30K发布说明](../../docs/LEARNING_COMPLIANCE_30K.md)。新增 `demos/impedance_rope.py` 是推理演示，不修改训练代码或权重。
