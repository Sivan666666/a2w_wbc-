# 当前优选：髋关节方案 A（2026-11-25）

按用户选择，**A 标注为当前训练效果最好的一版，作为主方案**。这个标记是当前实验的综合选择，不代表每项指标都第一；历史 A/B/C/D 的平均 EE 误差与髋姿态各有取舍。A 自身阶段评测中，30K 平均 EE 误差最低，不能把最终 50K 当作最优 checkpoint。

## 完整设置

| 项目 | A 设置 |
|---|---|
| 髋角偏离默认值 L1 | −0.3 |
| 超出默认值 ±0.25 rad 的部分平方和 | −1.0 |
| 左右髋镜像平方和 | −0.2 |
| 实际／目标轮速平方和 | −0.05／−0.01 |
| 底盘指令重采样 | 20% 全零 + 20% 低速 + 60% 原范围常规运动 |
| 从零训练 | seed 42，4096 环境，24 steps/update，默认 10,000 updates |
| 机械臂控制 | 训练原生低 Kp/Kd，无动力学前馈 |

髋项仅包含 FR/FL/RR/RL 四个 hip，自首帧全权重，不附加姿态或命令门控。镜像项为 `(FR+FL)^2+(RR+RL)^2`。轮速项在 `abs(vx), abs(vy), abs(yaw_rate)` 全部低于 0.1 时生效，≤0.08 为全强度，0.08–0.1 平滑衰减，前 500 updates 从零渐增。低速分支 `vx∼Uniform(-0.1,0.1)`、vy=yaw_rate=0；常规分支排除该低速盒。40% 是重采样概率，不是保证的训练帧占比。

完整设置由 [训练入口](../reproduction/train_hip_a.sh) 和 [实验代码](../reproduction/hip_a/) 应用。直接调用通用 `scripts/rsl_rl/train.py` 不会自动添加这些轮速项和低速混合采样，不能称为完整方案 A。

## 当前源码的使用

使用项目对应的 Isaac Lab / Isaac Sim 5.1 和 RSL-RL 环境，先按主 README 安装依赖和项目，然后运行：

```bash
python reproduction/audit_source.py
python reproduction/relocate_assets.py
python reproduction/audit_source.py --allow-relocated-urdf
bash reproduction/train_hip_a.sh "$PWD/outputs/hip_a_sdk_limits_new"
```

最后一条命令会真正启动训练，仅在需要新训练时执行。输出目录必须不存在，默认从零 10K；可通过 `NUM_ENVS`、`ITERATIONS` 显式覆盖，覆盖后实验预算也随之改变。每次使用独立 USD 缓存，不能复用旧限位缓存。

`train_hips.py` 保存阶段 checkpoint 和 JIT actor，但不自动评测。训练完成后可使用 `python reproduction/hip_a/eval_stage.py --run <绝对输出目录> --step 10000 --headless --device cuda:0` 单独评测；评测目录同样必须不存在。

`train_resume.py` 保留 A 的专用 10K→50K 续训流程：只接受 A、起始 iteration=10000、追加40000；恢复 actor、critic、optimizer 和 normalizer，逐项核对恢复状态，使用训练步数偏移避免重复渐增，并在10/20/30/40/50K同步调用阶段评测。它需要另行提供完整10K checkpoint，本次未上传 A 的权重。当前模型用于所有新运行；若给它历史 checkpoint，也不构成旧模型的精确复现。

## 模型变更与结果边界

当前训练 URDF 已去掉两处相机支架（显示、碰撞、惯量及固定连接），保留22个主动关节。机械臂物理限位按 SDK 更新如下，单位 rad：

| 关节 | 下限 | 上限 |
|---|---:|---:|
| J1 | −2.61799387799 | 2.61799387799 |
| J2 | 0 | 2.87979326579 |
| J3 | −2.88559265359 | 0 |
| J4 | −1.51843644924 | 1.51843644924 |
| J5 | −1.34390352404 | 1.34390352404 |
| J6 | −2.79252680319 | 2.79252680319 |

新缓存为 `usd_no_camera_mounts_sdk_limits`。SDK 原始记录和当前 URDF 校验快照保存在 [provenance](provenance/hip_a_20261125/)。旧 provenance 文件继续保留，当前文件校验使用该目录的新 manifest。

历史 A 从零10K使用含支架模型；之后自身10K→50K续训及10K新模型基线使用无支架模型，**仍是旧的较宽关节限位**。本次 SDK 限位修改尚未重新训练或评测，不能把下列成绩归于当前新限位模型。历史 PPO10K 真机部署权重仍在独立目录，本次没有替换为 A，也没有启动真机。

## 历史 A 阶段评测

七工况各4环境×30秒、评测seed20261123；误差统计末5秒，各工况等权，失败统计全30秒。训练单seed42，结果不代表多seed统计结论。

| 总更新数 | EE位置均值 cm | EE姿态均值 ° | 失败reset |
|---|---:|---:|---:|
| 10K | 1.4429 | 2.8687 | 0 |
| 20K | 1.1796 | 1.3381 | 0 |
| **30K** | **0.9163** | **1.1884** | 0 |
| 40K | 1.2876 | 1.6814 | 0 |
| 50K | 1.4831 | 2.0377 | 0 |

50K倒车工况有退化。完整底盘速度、EE、髋角和接触指标见 [A 10K→50K](evaluation/hip_a_20261125/a_10k_to_50k.txt)，四方案对比见 [A/B/C/D 10K](evaluation/hip_a_20261125/abcd_10k.txt)。量化评测协议与用户60秒单环境录像协议独立。

本次发布验证包括 CPU 奖励检查、采样与门控检查、Python 语法、模型关节限位及文件哈希。没有在本次发布中启动 Isaac 训练或真机；脚本来自实际实验，发布适配后的完整运行仍需在对应 Isaac 环境验证。
