# A2W + Z1 WBC — WheelV2 / Precision 50K

基于 [LeggedManip_Lab](https://github.com/zzzJie-Robot/LeggedManip_Lab) 的A2W轮足＋Z1全身控制适配与已训练实验代码。不是官方发行版，不是robot_lab任务，也不是ACT/图像策略。

**主文档：[Precision 50K相对官方的全部修改](docs/PRECISION_50K_CHANGES.md)**，包含机器人/关节/观测适配、全部奖励权重、命令和课程、训练超参、评测及已知限制。

## 内容

| 路径 | 用途 |
|---|---|
| `source/LeggedManip_Lab/` | 实际训练包；官方源码加A2W、WheelV2、Precision子类 |
| `scripts/rsl_rl/train.py` | 官方PPO训练入口，未修改 |
| `scripts/` | 训练快照中的资产、诊断、评测、play和MuJoCo辅助脚本 |
| `reproduction/train.sh` | 新clone可用的同超参训练入口 |
| `reproduction/a100_original/` | 原A100实验工具原样存档；路径为历史机器专用 |
| `docs/provenance/` | 源码哈希、完整差异清单、补丁和环境版本记录 |
| `docs/evaluation/` | 两个最终50K模型的正式评测统计 |

## 精确版本

- LeggedManip_Lab：`4e12109d1eccb1d9da2e4a35f189fabea3f13677`，加本仓库记录的修改。
- Isaac Lab：`47c9f95eeab90dc3611981d894d59f163191f5b1`。
- Isaac Sim5.1.0.0、Python3.11.15、PyTorch2.7.0+cu128、RSL-RL5.0.1、Gymnasium1.2.1。
- 两组均从头训练50,000 PPO iterations、4096env、seed42；不是20K续训。

## 安装与训练

建议使用独立venv/容器，**不要在服务器现有共享环境直接升级依赖**。安装Isaac Sim须遵循NVIDIA系统需求/EULA；PS1 Ubuntu20.04不能直接启动Sim5.1，需要兼容glibc的容器。

下面假设你已经拥有安装Isaac Sim5.1、匹配Torch/CUDA的隔离Python环境；本仓库不提供驱动或系统软件安装脚本。

```bash
git clone https://github.com/Sivan666666/a2w_wbc-.git
cd a2w_wbc-
git clone https://github.com/isaac-sim/IsaacLab.git .deps/IsaacLab
git -C .deps/IsaacLab checkout 47c9f95eeab90dc3611981d894d59f163191f5b1

# 用当前隔离环境的python；完整版本参考docs/provenance/pip_freeze.txt。
# 为防止pip自动替换Torch/Sim，先准备好固定版本依赖，再用--no-deps安装源码包。
python -m pip install --no-deps --no-build-isolation \
  -e .deps/IsaacLab/source/isaaclab \
  -e .deps/IsaacLab/source/isaaclab_assets \
  -e .deps/IsaacLab/source/isaaclab_tasks \
  -e .deps/IsaacLab/source/isaaclab_rl \
  -e source/LeggedManip_Lab

python reproduction/audit_source.py
python reproduction/relocate_assets.py
python reproduction/audit_source.py --allow-relocated-urdf

# 先短测（2 PPO iterations），通过后再正式训练
NUM_ENVS=32 ITERATIONS=2 RUN_NAME=precision_smoke \
  CUDA_VISIBLE_DEVICES=0 bash reproduction/train.sh precision

CUDA_VISIBLE_DEVICES=0 bash reproduction/train.sh precision
# 或原版对照
CUDA_VISIBLE_DEVICES=1 bash reproduction/train.sh baseline
```

`pip_freeze.txt` 是历史版本记录，包含历史editable绝对路径，**不要直接pip install -r它**。原环境保留两条已知IsaacSim依赖元数据冲突，见主文档；安装后必须做实际仿真与PPO短测，而非仅依赖pip check。

原始A100启动脚本和安装过程放在 `reproduction/a100_original/`，用于审计，不可在别人的机器上原样执行其硬编码路径。

## 已完成的50K评测（原范围 x=0.55–0.85m）

每行3seeds×64env×200秒；完整扩展范围结果见 [比较表](docs/evaluation/comparison.md)。

| 模型 | 原生EE误差cm↓ | 可视化世界目标距离cm↓ | 姿态误差°↓ | XY速度误差m/s↓ |
|---|---:|---:|---:|---:|
| WheelV2 50K | 4.39 | 6.52 | 1.76 | 0.0630 |
| Precision 50K | 2.81 | 5.58 | 3.63 | 0.1119 |

Precision提升位置精度但姿态/底盘速度跟踪变差；同时改了x范围和奖励，不能称为单因素消融。原生EE误差混用link0 XY与世界Z，不等于到可视化球的纯世界距离。policy接收本体状态与EE位姿/底盘速度命令，不接收depth/Plücker图像。

本次仅发布代码、A2W所需URDF/mesh及记录；不含权重、视频、原始大trace、IsaacSim二进制。历史脚本中的MuJoCo/20K/服务器路径需按用途配置，不是Precision50K默认执行流程。其他机器人USD资产未随此A2W发布包复制，可从固定官方commit获取。

发布验收：已在PS1独立目录完成32env×2轮PPO短训练并保存checkpoint，退出码0；[验收记录](docs/provenance/publication_smoke/result.json)。该短测验证代码/资产/训练入口可运行，不代表重新训练或评测了50K。

## 许可与安全

保留上游Apache-2.0 [LICENSE](LICENSE) 和版权声明；见 [NOTICE](NOTICE) 了解修改与第三方资产范围。这是仿真研究代码，夹爪固定，未提供已验证的A2W/Z1真机电机闭环。不能把模型直接发送给真机执行。
