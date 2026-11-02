# Precision 50K 相对官方 LeggedManip_Lab 的完整修改说明

整理日期：2026-11-02。对象是 **2026-09-26 在 A100 启动、2026-09-28 完成的 A2W+Z1 Precision 50K**，不是后续实机桥接、机械臂平滑或写字实验的代码。

## 1. 来源、版本与审计边界

| 项目 | 实际版本 / 来源 |
|---|---|
| 上游仓库 | https://github.com/zzzJie-Robot/LeggedManip_Lab |
| 官方基准 commit | `4e12109d1eccb1d9da2e4a35f189fabea3f13677` |
| 实际训练代码 | PS1工作区的A2W/WheelV2适配快照 + 新增Precision配置，独立复制到A100 |
| Isaac Lab | commit `47c9f95eeab90dc3611981d894d59f163191f5b1`，安装元数据isaaclab 0.54.3 |
| Isaac Sim | 5.1.0.0 |
| RSL-RL | 5.0.1，PPO |
| Python / PyTorch | 3.11.15 / 2.7.0+cu128 |
| Gymnasium / TensorDict | 1.2.1 / 0.14.2 |
| 轮足设计参考 | robot_lab commit `500399ed75f510aeaff28705a8ce736c514dbec3`；仅参考设计，不是用该仓库训练 |
| 精确来源清单 | [原始训练manifest](provenance/training_manifest.json)、[源码哈希审计](provenance/training_hash_audit.json) |

本次发布前，原训练manifest覆盖的 **1,056个文件（含Isaac Lab）逐一核对一致**。完整新增/修改文件、原始与发布SHA256见 [逐文件清单](provenance/FILE_CHANGES.md) 与 [机器可读inventory](provenance/file_diff_inventory.json)。[文本补丁](provenance/training_changes.patch)包含新增文本代码及已有文件改动；二进制mesh通过inventory记录并直接包含在仓库。

**不能只checkout官方commit就得到本模型。** 继承路径是：

```text
官方 B2Z1WBCEnvCfg
  → A2WZ1WBCEnvCfg              新机器人、22维动作/观测、轮接触与侧移范围
  → A2WZ1WheelV2EnvCfg          轮执行器参数和分组正则奖励
  → A2WZ1WheelV2PrecisionEnvCfg 扩展EE x + 两档精细位置奖励
```

相同路径的官方共享环境、奖励函数、命令生成、课程和PPO配置/训练入口，在此快照中与官方基准逐字节一致。核心算法差异通过**新增子类配置**实现，不是重写PPO。已有 `scripts/rsl_rl/play.py` 仅增加4行checkpoint模式解析兼容分支，见第8节。

## 2. 机器人适配：B2+Z1 → A2W+Z1

相关代码：`scripts/prepare_a2w_asset.py`、`source/LeggedManip_Lab/LeggedManip_Lab/assets/a2w_z1/`，以及 `config/a2w_z1/`。

### 2.1 URDF与坐标命名

- 使用用户推门项目的完整A2W+Z1 URDF，不是float-IK里拆分底盘/机械臂的运动学代理。
- 原始URDF SHA256：`0a857505af3f0004957c3523b5139e70c7c704dea025cf3d81a8a3ed742142d1`。
- 删除源文件的 `world` link 和 `floating_base_joint`；仿真器创建自由根，`fix_base=False`。
- 重命名参考帧，以对接官方WBC命令和观测：

| 原帧 | WBC帧 |
|---|---|
| `base_link` | `base`（根机身） |
| `base`（机械臂安装参考） | `z1_mount` |
| `link00` | `link0`（机械臂命令参考） |
| `ee_gripper_link` | `end_effector` |

- 四个轮关节改为continuous，删除无穷lower/upper，保留URDF力矩/速度字段；实际执行器限制由配置覆盖。
- `jointGripper` 固定在源模型零位，移除其axis/limit/dynamics。**策略不控制夹爪开闭，不是抓取策略。**
- 唯一无惯量帧 `z1_mount` 补质量 `1e-6 kg`（1mg）、对角惯量 `1e-9 kg·m²`，避免导入器补入未知质量。
- 不修改已有几何、原有质量惯量、非轮关节限位或安装关系。适配后URDF总质量约 **54.61697 kg**。
- 本训练快照保留 `wrist_up_mount` 与 `wrist_down_mount`；不能把其他可视化实验的“去掉水平mount”操作混称为本次训练资产修改。
- 固定安装链：`base→z1_mount`为单位变换，`z1_mount→link0`平移`[0.174,0,0.142]m`且rpy为0。因此本资产中根机身与link0姿态相同，只是原点不同。
- `link06→gripperStator`平移x=0.051m，`gripperStator→end_effector`平移x=0.135m；末端测量点不是简单的link06原点。

导入：`merge_fixed_joints=False`、`make_instanceable=True`、接触传感器开启、重力开启、自碰撞关闭、位置/速度求解迭代4/0、最大去穿透速度1m/s、软关节位置限位系数0.9。初始根高度0.55m，腿默认hip=0、thigh=0.8、calf=-1.5rad，臂六关节与轮子初始角均为0。

URDF中mesh路径曾从PS1迁移为A100绝对路径，仅路径变化。历史 `asset_manifest.json` 的output hash属于更早PS1输出，不应拿它当A100迁移后的hash。A100训练URDF的hash为 `acc213be86c7fd9d10c50ec5f5eee716c866125a4973f879ccfe25fda8db1ef7`；原样副本见 `provenance/a2wz1_wbc.training.urdf`。

### 2.2 动作顺序与执行器

输出由官方B2的18维位置动作扩为 **22维**：前12维腿位置、随后6维臂位置、末4维轮速度。

```text
腿顺序：FR, FL, RR, RL；每腿 hip, thigh, calf
臂顺序：joint1 … joint6
轮顺序：FR_wheel_joint, FL_wheel_joint, RR_wheel_joint, RL_wheel_joint
```

腿/臂使用默认关节角偏置，`q_target = clip(q_default + 0.25 * action, -10, 10)`；轮子无位置偏置，`dq_target = clip(5 * action, -100, 100)` rad/s。**在固定Isaac Lab版本里，动作先scale/offset，再clip**，不能改成先clip再乘scale。动作层限幅与关节/执行器物理限制是不同层次。

| 执行器组 | Kp | Kd | 力矩限制Nm | 仿真速度限制rad/s |
|---|---|---|---|---|
| hip / thigh | 120 | 3 | 120 | 22 |
| calf | 120 | 3 | 180 | 14.6667 |
| wheels（WheelV2最终值） | 0 | 0.6 | 20 | 50 |
| arm joint1…6 | 50,50,80,30,30,20 | 3,2,3,3,2.5,1 | joint2=60，其余30 | 10 |

全部使用 `DelayedPDActuatorCfg`，armature=0.01、延迟0–4物理步（200Hz下0–20ms），并保留增益随机化。实际gain会随随机化变化，不始终等于表中标称值。轮子基类曾为scale10、clip±3、Kd2、24.5Nm、114.4rad/s；**Precision最终继承WheelV2覆盖值**，不能误读基类参数。

### 2.3 观测适配

Actor不含世界位置、不含图像、也不直接输入末端实际位姿。输入是本体状态和命令：

| 单帧Actor项 | 维度 | scale / 噪声 |
|---|---:|---|
| 机身角速度 | 3 | scale0.2；均匀噪声±0.2 |
| 投影重力 | 3 | 噪声±0.05 |
| 腿/臂相对默认关节角 | 18 | 噪声±0.01；**不包含轮累计角** |
| 全部活动关节相对速度 | 22 | scale0.05；噪声±1.5 |
| 上一步动作 | 22 | 无额外噪声 |
| 底盘vx、vy、vyaw命令 | 3 | 无额外噪声 |
| EE位置3＋四元数4命令 | 7 | 无额外噪声 |

单帧78维，历史长度3，总 **Actor234维**；不要按“整帧拼接”猜测历史存储布局，部署必须复用Isaac Lab逐term历史展开约定。训练启用观测噪声，PLAY关闭。

Critic114维，包括根线速度/角速度、重力、命令、18位置、22速度、22上一动作、22力矩、4轮接触、EE相对link0位姿7维。接触观测由`.*_foot`改为`.*_wheel`。官方B2 Actor210维/18动作不能直接替换A2W Actor234维/22动作权重。

## 3. EE / 底盘命令与课程

### 3.1 Precision相对WheelV2的范围修改

| 命令 | 原WheelV2最终范围 | Precision最终范围 | 坐标/说明 |
|---|---|---|---|
| EE x | [0.55,0.85]m | **[0.10,0.85]m** | link0 |
| EE y | [-0.35,0.35]m | 不变 | link0 |
| EE z | [0.10,1.00]m | 不变 | 世界绝对高度 |
| vx | [-1,1]m/s | 不变 | 机身系 |
| vy | [-0.2,0.2]m/s | 不变 | A2W适配已从官方B2 ±0.6缩到±0.2 |
| vyaw | [-1,1]rad/s | 不变 | 机身系 |

Precision同时修改 `ranges.pos_x` 与 `limit_ranges.pos_x`。EE每8–10秒重采样，底盘每10秒，站立比例0.1、heading command关闭。

姿态生成**未修改**：先按目标方向估算pitch/yaw，再加入roll±π/3、pitch±π/4、yaw±π/6扰动，最后三个Euler分量整体clamp到[-3.14/4,3.14/3]，转成wxyz四元数（按配置规范符号）。这不是直接均匀采样三个最终Euler角，也不是固定姿态。代码没有新增IK可达性筛选；x扩展到近身可能增加困难目标。

### 3.2 课程并未新增或改写

在固定官方commit中，EE与底盘的 `curriculum_enabled=False` 已是默认。课程term虽然仍注册，但reset时强制 `cfg.ranges=limit_ranges`，指标返回-1。Precision继续沿用这一行为，不采用“先易后难到1K后突然切换”的课程。

因此基类中的初始小范围（例如EE x=0.4–0.45）不代表正式训练持续使用该范围。审计实测6400个目标，Precision x范围0.1000108–0.8498309m。base height奖励目标 **固定0.48m**，不是随机高度命令；初始根高度0.55m也不是奖励目标。

## 4. 奖励修改：完整有效权重

对比对象是**官方B2-Z1 WBC最终配置**，不是泛用环境基类的默认值。表中权重在RewardManager中还会乘环境步长dt=0.02s。共享奖励函数源码未改。

| 奖励项 | 官方B2 WBC | Precision最终设置 | 差异 |
|---|---|---|---|
| EE位置exp | 4.5，std=√0.1m | 相同 | 保留宽范围奖励 |
| `ee_position_medium` | 无 | +1.0，std=0.10m | 新增 |
| `ee_position_fine` | 无 | +1.5，std=0.05m | 新增 |
| EE姿态角误差 | −4.0 | −4.0 | 不变 |
| 底盘XY速度exp | +3.5，std=0.5 | 相同 | 不变 |
| 底盘yaw速度exp | +2.5，std=0.5 | 相同 | 不变 |
| 底盘高度exp | +0.35，目标0.48m，std=0.02 | 相同 | 不变 |
| 根z速度平方 | −2.0 | −2.0 | 不变 |
| 根roll/pitch角速度平方 | −0.05 | −0.05 | 不变 |
| 平坦姿态 | −0.5 | −0.5 | 不变 |
| `dof_torques_l2` | −2e−5，全18关节 | −2e−5，仅12腿 | 分组 |
| `arm_torques_l2` | 含在上项 | −2e−5，仅6臂 | 拆出；无轮力矩平方惩罚 |
| hip最大绝对力矩平方 | −8e−5 | 相同 | 保留 |
| thigh最大绝对力矩平方 | −8e−5 | 相同 | 保留 |
| calf最大绝对力矩平方 | −5e−5 | 相同 | 保留 |
| `dof_acc_l2` | −2.5e−7，全18关节 | −2.5e−7，仅腿 | 分组 |
| `arm_acc_l2` | 含在上项 | −2.5e−7，仅臂 | 保留臂原强度 |
| `wheel_acc_l2` | 无轮 | −2.5e−9，仅轮 | 比腿/臂弱100倍 |
| 动作变化率平方 | −0.01 | −0.01，全22动作 | 权重保留、维度适配 |
| 机械功率Σabs(τ×dq) | −4e−5，全关节 | −2e−5，仅腿 | 排除轮与臂并减小权重 |
| hip默认角偏离L1 | −0.1 | 相同 | 保留 |
| calf/thigh默认角偏离L1 | −0.02 | 相同 | 保留 |
| arm默认角偏离L1 | −0.1 | 相同 | 保留 |
| 软关节位置限位 | −1.0 | −1.0 | 保留 |
| feet_slide | −0.1 | None | 关闭 |
| feet_air_time | +0.5 | None | 关闭 |
| feet_long_air | −1.0 | None | 关闭 |
| air_time_variance | −1.0 | None | 关闭 |
| joint_mirror | −0.2 | None | 关闭对角步态约束 |

没有添加“轮速越大越好”奖励，也没有轮速平方惩罚。通过避免把正常滚动视为滑脚、减弱轮加速度惩罚，让轮子能承担移动；不等同于已经验证无滑移滚动。

### 4.1 精细位置奖励公式

令 `p_l = R_link0^T (p_EE_world - p_link0_world)`，定义

```text
e = norm([p_l.x - cmd.x, p_l.y - cmd.y, p_EE_world.z - cmd.z])

官方B2 WBC / WheelV2位置部分：
R_pos = 4.5 exp(-e²/0.1)

Precision位置部分：
R_pos = 4.5 exp(-e²/0.1)
      + 1.0 exp(-e²/0.01)
      + 1.5 exp(-e²/0.0025)
```

保留31.62cm宽尺度，引入10cm与5cm尺度来区分厘米级误差。不是把官方std直接改小，也不是重写奖励函数。这里的e是混合坐标距离，不是单一世界坐标欧氏距离。

配置实现使用 `deepcopy(self.commands/rewards)`：`configclass.copy()`基于`dataclasses.replace`可能丢掉WheelV2动态追加的轮/臂正则项。审计发现后，Precision配置改为深拷贝，并确认原版配置不被污染、轮/臂惩罚保留。这是本次Precision配置修复，不是官方共享代码修改。

### 4.2 姿态与位置的已有坐标口径限制

- 姿态奖励把命令用根机身姿态转换；原生姿态metric用link0姿态转换。此A2W资产固定安装链没有旋转，二者旋转等价；其他机器人必须另查，不能泛化。
- 原项目可视化目标：`p_target= p_link0 + R_link0*[cmd.x,cmd.y,0]`，然后覆盖世界z为cmd.z。link0倾斜时，它不一定等于混合坐标位置约束的精确目标。
- 因此独立评测同时报告原生误差和到可视化目标的世界距离，**本训练没有修复/统一这两种位置定义**。
- 高度奖励为 `exp(-abs(z_base - 0.48)/0.02)`，不是平方高斯核；不要与EE位置公式混用。

## 5. PPO与正式训练配置

| 项目 | Precision 50K实际值 |
|---|---|
| Gym任务 | `A2W-Z1-WBC-WheelV2-Precision` |
| experiment_name | `a2w_z1_wbc_wheel_v2_precision` |
| 初始化 | 从头；`agent.resume=false`，不加载20K或smoke权重 |
| seed / 并行环境 | 42 / 4096 |
| PPO iterations | 50,000（不是50,000物理步） |
| rollout长度 | 每环境24步，即每轮98,304个transition |
| Actor / Critic | MLP [512,256,128]，ELU；Actor234输入/22动作，Critic114输入 |
| Gaussian探索 | init_std=1.0，std_type=log |
| 初始学习率 / schedule | 1e−3 / adaptive（训练中会变） |
| epochs / minibatches | 5 / 4 |
| clip / value coefficient | 0.2 / 1.0，clipped value loss开启 |
| entropy coefficient | 0.005 |
| gamma / GAE lambda | 0.99 / 0.95 |
| desired KL / max grad norm | 0.01 / 1.0 |
| NaN检查 | 开启 |
| 保存 / 日志 | 每1000 iterations；TensorBoard；WANDB_MODE=disabled |
| physics / policy | dt=0.005s（200Hz），decimation4（50Hz） |
| episode / env spacing | 20秒 / 2.5m |
| 场景 | 平地；headless，关闭地板远端视觉材质、天空HDR、命令debug标记 |
| 最终checkpoint | `model_49999.pt`（零起算） |

官方B2 PPO类默认max_iterations=5000；本次用CLI覆盖为50000。隐藏层、学习率等PPO超参未另调。50K总采样理论量约49.152亿env-step；模拟速度与真实墙钟耗时不同。没有为训练加入五次插值、在线关节速度/加速度轨迹整形，也没有实机LowCmd/Z1 bridge闭环。

原始启动命令（A100环境包装之后执行）：

```bash
python scripts/rsl_rl/train.py \
  --task A2W-Z1-WBC-WheelV2-Precision --num_envs 4096 \
  --headless --device cuda:0 --max_iterations 50000 --seed 42 \
  --run_name precision_train_seed42_20260926 --logger tensorboard \
  agent.resume=false agent.save_interval=1000 \
  env.scene.terrain.visual_material=null env.scene.sky_light.spawn.texture_file=null \
  env.commands.ee_pose.debug_vis=false env.commands.base_velocity.debug_vis=false
```

GPU1通过CUDA_VISIBLE_DEVICES映射为进程内cuda:0。基线在GPU0，用同seed/4096env/50K，任务改为`A2W-Z1-WBC-WheelV2`。启动脚本另设Kit日志文件位置，属于日志配置，不影响控制。

## 6. 随机化、终止和物理配置（沿用，不是Precision新增）

| 项目 | 设置 |
|---|---|
| 地面物理材质 | static/dynamic friction=1；restitution=1；combine=multiply |
| 机器人摩擦 | 启动时static/dynamic [0.5,1.2]，restitution[0,0.1]，64 buckets，make_consistent |
| 质量 | 启动时所有body乘[0.9,1.1] |
| 惯量 | 启动时乘[0.8,1.2] |
| 根COM | 启动时x/y±0.05m、z±0.01m |
| 执行器gain | reset时Kp/Kd乘[0.8,1.2] |
| 根reset | x/y±0.5m、yaw±3.14rad；初始速度0 |
| 关节reset | 默认位置乘[0.5,1.5]、速度0；默认值0的关节仍为0 |
| 随机推力 | B2 WBC已关闭push_robot；reset外力/力矩范围均0 |
| 超时 | 20秒 |
| 根接触终止 | base接触阈值0.5N |
| 姿态终止 | bad_orientation limit_angle=1rad |

Precision没有改这些终止条件，也没有通过删去失败轨迹改善评测数字。资产自碰撞关闭是A2W导入配置，不能据此保证真实机器人无自碰撞。

## 7. 评测、checkpoint与已知取舍

最终Precision checkpoint SHA256：`378e5fd30611d5de9a1c61db56bfbfdc2d054128a80a2f1b04c7ece94ec15fee`。基线SHA256：`f8c576a5a8a297d3c43e3b350e9d58af630e217722ee818d0b7930ef1c82d633`。**本仓库此次发布代码、资产与记录，不上传checkpoint、视频或大规模trace。**

每个模型×范围：3 seeds（20260916/17/18）×64env×200秒。测量在物理步后、reset和命令重采样前，包含所有帧和切换瞬态。PLAY关闭策略观测噪声，保留原生随机化/执行器延迟。两种模型都用同一WheelV2 PLAY任务，只覆盖测试x范围，严格加载同形状网络。

| 测试x范围 | 模型 | 原生EE cm↓ | 可视化世界距离cm↓ | 姿态°↓ | XY速度误差m/s↓ | yaw误差rad/s↓ | 非超时reset |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.55–0.85 | WheelV2 50K | 4.39 | 6.52 | 1.76 | 0.0630 | 0.0562 | 3 |
| 0.55–0.85 | Precision 50K | 2.81 | 5.58 | 3.63 | 0.1119 | 0.0866 | 5 |
| 0.10–0.85 | WheelV2 50K | 8.27 | 13.33 | 2.61 | 0.0771 | 0.0652 | 63 |
| 0.10–0.85 | Precision 50K | 2.58 | 10.32 | 3.78 | 0.1186 | 0.0889 | 7 |

完整逐seed统计在 [comparison.json](evaluation/comparison.json)。Precision提升位置精度，但姿态和底盘速度误差变大；不是全面优于原版。扩展范围对基线属于分布外；本实验同时改范围和奖励，**不是单因素奖励消融**。不是静态稳态到点、连续写字或真机成功率测试。

## 8. 工程兼容、辅助代码与发布边界

### 8.1 训练快照已有的工程变化

- `scripts/rsl_rl/play.py` 增加 `checkpoint == "model_.*.pt"` 分支：在指定run内解析默认匹配模式，不把它当字面文件路径。只有play路径改变，`train.py`没有修改。
- 新增资产准备、smoke、配置审计、离线评测、视频、导出、MuJoCo验证、20K续训等脚本。它们全部列入逐文件inventory，但只有训练入口、配置/资产参与本次50K训练。
- `resume_wheel_v2_20k.py` / `chain_wheel_v2_20k.py` 属于更早5K→20K实验，**Precision50K没有执行它们**。
- B2固定偏差、FK、MuJoCo历史观测和写字诊断脚本是历史辅助工具，不是对Precision训练任务的额外修改。
- `eval_wheelv2_a100_domains.py`、自动等待评测脚本、2026-10-01录视频脚本属于训练启动后的测评工具；原训练manifest主要覆盖source，并未为所有历史scripts记录启动时hash。发布inventory另行记录其当前快照，不伪称全部是训练启动时冻结的脚本。
- A100使用独立用户目录/环境、缓存、日志，不修改共享系统CUDA/Python；PS1需要独立容器绕过Ubuntu20.04 glibc不足，宿主环境不升级。
- Isaac Lab5.1配套URDF importer需要2.4.31。A100不能提供RTX渲染，验证的是headless CUDA物理/PPO；视频是在PS1 RTX环境录制。
- 两条已知pip元数据冲突：IsaacSim kernel钉死FastAPI0.115.7/typing-extensions4.12.2，实际使用PS1对齐的0.141.1/4.16.0。实际训练与评测通过，但不能写成pip依赖完全无冲突。完整记录见 `provenance/pip_check.txt` 和 `pip_freeze.txt`。

### 8.2 此次GitHub发布才新增的整理（不算训练改动）

- README、本说明、NOTICE、逐文件差异/哈希/patch和静态校验工具。
- `reproduction/a100_original/` 原样存档历史启动、安装、审计和评测工具；它们含原机器绝对路径，不应直接当成通用安装脚本运行。
- `reproduction/train.sh` 提供同超参的相对路径训练入口，显式加载本仓库包；没有改PPO或任务。
- `reproduction/relocate_assets.py` 显式替换URDF mesh目录，并检查所有非路径XML完全相同。运行它后工作树URDF会出现预期路径diff，原训练URDF在provenance中保留。
- 排除生成USD、缓存、虚拟环境和不相关机器人的大USD资产；A2W所需URDF和全部mesh包含。其他机器人任务仍有源码配置，但需要从官方commit取回对应资产才能运行。
- 依赖Isaac Lab用固定commit单独安装，不把整个Isaac Sim二进制或环境上传仓库。
- 保留官方Apache-2.0许可与版权；vendor机器人资产不因发布而自动重新授权，见NOTICE。

### 8.3 验证与复现入口

```bash
# 不需要GPU/Isaac即可检查发布源码、22活动关节、4轮和mesh存在性
python reproduction/audit_source.py

# 在新clone中执行；只改mesh目录，禁止复用旧生成USD
python reproduction/relocate_assets.py
python reproduction/audit_source.py --allow-relocated-urdf

# 激活你自己的隔离IsaacSim5.1 + 固定IsaacLab + RSL-RL5.0.1环境后
CUDA_VISIBLE_DEVICES=0 bash reproduction/train.sh precision

# 小规模管线测试，不冒充50K复现完成
NUM_ENVS=32 ITERATIONS=2 RUN_NAME=precision_publication_smoke \
  CUDA_VISIBLE_DEVICES=0 bash reproduction/train.sh precision
```

发布后完整训练仍需GPU时间；静态检查/短测不能证明新环境下50K逐bit复现或真机可直接部署。硬件、并行随机化和数值库差异会影响结果。

2026-11-02发布验收已执行：PS1独立目录/独立容器、空闲GPU3、32env、seed42、2个PPO iterations；容器退出码0，生成6,148,116字节的 `model_1.pt`。Actor234/Critic114/动作22，两个新增位置奖励与轮/臂正则项均实际注册并输出日志。原URDF路径迁移前后非路径XML一致。解析后的短测配置与结果见 [publication_smoke](provenance/publication_smoke/result.json)；该目录的env/agent.yaml明确是**2轮短测配置**，不是50K训练配置。
