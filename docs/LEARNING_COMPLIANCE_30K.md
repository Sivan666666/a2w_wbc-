# Learning-compliance 30K 发布

本分支保留正式训练提交 `6552923a332b3c18a97c24162de01ebe8e5fbdfe` 的训练代码与模型资产，另加入已完成的权重、阻抗演示和发布说明。不是 UniFP，不包含 SDK 重力前馈，不是旧 A 30K 的观测结构。

## 模型文件

| 文件 | 用途 |
|---|---|
| `models/learning_compliance/checkpoint_0030000.pt` | 完整训练检查点，iteration=30000；actor/critic/estimator/log_std、PPO与estimator两个优化器、学习率、随机数状态 |
| `models/learning_compliance/actor_0030000.jit` | 推理模型，包含actor及估计器；`model(history)`输出22维动作，`model.estimate(history)`输出9维状态 |
| `models/learning_compliance/metadata.json` | 训练来源、维度、有限性验证与哈希 |
| `models/learning_compliance/SHA256SUMS` | 两个权重文件的完整SHA256 |

模型直接存储在 Git 中，不需要 Git LFS。完整 checkpoint 约19.8 MB，JIT约3.36 MB。

```bash
git clone --branch learning-compliance https://github.com/Sivan666666/a2w_wbc-.git
cd a2w_wbc-
(cd models/learning_compliance && sha256sum -c SHA256SUMS)
```

训练：随机初始化，seed42，4096环境，24步/更新，总30000更新；2,949,120,000个真实仿真交互样本，无想象样本。物理重力保留。地形为平地，不包含崎岖地形训练。模型限位来自SDK内部软件模型值，不等于已确认的真机物理硬限位。

## 观测与坐标

actor为10帧历史，单帧82维、总820维；布局为term-major，每项从旧到新。critic121维，动作22维。估计器9维依次为底盘线速度、末端外力、末端位置，各3维。

末端位置相对base原点、用base-yaw坐标表达；这个坐标架随base平移/yaw转动，不随pitch/roll倾斜。外力是环境作用在机器人上的注入外力，以base-yaw坐标表达，不代表全部碰撞接触合力。

不能直接将旧A的actor观测或上游仓库的frame-major历史送进本模型。离线历史构造见 `reproduction/force_control/inference.py`；IsaacLab运行使用本分支的环境适配器。

## 运行环境与阶段评测

已验证：Python3.11、Isaac Sim5.1、IsaacLab2.3兼容环境、PyTorch2.7+cu128。需要可运行本仓库任务的IsaacLab环境；环境准备参考根README中的安装说明，不要直接升级其他任务使用的共享环境。

在仓库根目录、已激活的IsaacLab环境中运行；`PYTHONPATH`应包含 `source/LeggedManip_Lab` 及IsaacLab相关包。以下命令只评测、不更新权重；输出目录必须新建。

```bash
python reproduction/force_control/run.py \
  --mode stage \
  --checkpoint models/learning_compliance/checkpoint_0030000.pt \
  --output outputs/lc30k_stage_eval \
  --steps 1500 --headless --device cuda:0
```

这会执行原19工况×4环境×30秒评测：7位置模式＋12力模式。力场Kp100 N/m、Kd10 Ns/m，seed20261127。30K已完成，不需要再以`--resume`启动训练来查看模型。

## 10K / 20K / 30K 结果

每阶段相同评测，以下为各模式末5秒汇总。EE位置为原评测的link0局部XY与世界Z误差范数，姿态为link0参考误差。力误差为三维向量误差范数；关节误差为实际角度与policy处理后目标之差，未对齐执行器延迟。

| 指标（均值） | 10K | 20K | 30K |
|---|---:|---:|---:|
| 位置模式EE位置，cm | 4.069 | 1.880 | 1.286 |
| 位置模式EE姿态，deg | 3.275 | 3.940 | 3.835 |
| 力模式力跟踪，N | 16.785 | 9.690 | 6.464 |
| 力模式力估计，N | 7.771 | 6.222 | 5.261 |
| 位置模式机械臂关节MAE，deg | 3.828 | 3.727 | 3.677 |
| 位置模式狗腿关节MAE，deg | 6.978 | 6.353 | 6.211 |
| 力模式机械臂关节MAE，deg | 8.061 | 7.028 | 6.434 |
| 力模式狗腿关节MAE，deg | 7.759 | 6.555 | 6.068 |

三阶段轨迹形状均1500×76，数值有限；训练和评测退出0。单训练seed；低速蠕行仍几乎不移动、转向速度偏低，不能仅凭EE指标称全任务优秀。力模式EE位置只作诊断，不能和位置模式成绩混算。

## 阻抗控制与绳索演示

实现见 `reproduction/force_control/demos/impedance_rope.py`，参考 [论文V-B](https://arxiv.org/pdf/2405.01402)。

- 固定30K推理，实际`num_envs=1`，50Hz。
- 进入力模式时，用估计的末端位置建立固定base-yaw相对参考点。
- 机器人对环境的期望力为 `-100*(x_est-x_ref)-10*v_est`；因policy定义环境对机器人的力，指令取反并逐轴限幅±70N。
- 速度由估计位置差分并低通得到：`v=.8*v_previous+.2*diff(x_est)/.02`；是相对坐标的变化率，不是世界绝对速度。与论文的坐标/估计器输出存在适配差异。
- 仿真真实EE状态不进入阻抗反馈，只用于环境绳力和记录。保留原姿态指令、底盘零速度指令。
- 5秒位置模式准备后，录像全程力模式。可视化绳无分布质量和碰撞，施加的是纯水平张力。

`demo`：60秒弹性绳轻拉、重拉、释放。绳刚度150N/m、阻尼8Ns/m、原长1m。

`load`：理想绞盘反馈实际末端位置以保持规定张力，每8秒增加15N（2秒渐增＋6秒保持），最高150N。环境加载装置使用真值，policy不使用真值。环境终止或倾角/高度保护触发即停止。

```bash
python reproduction/force_control/demos/impedance_rope.py \
  --repo "$PWD" --actor "$PWD/models/learning_compliance/actor_0030000.jit" \
  --output "$PWD/outputs/lc30k_rope_demo" --mode demo \
  --headless --enable_cameras --device cuda:0 --render-gpu 0 \
  --kit_args="--/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false"
```

极限测试将`--mode demo`改为`--mode load`并换一个新输出目录。相机GPU需选择机器实际的Vulkan物理GPU编号；`cuda:0`是CUDA可见设备编号，二者不一定相同。本次验证使用物理GPU1，运行时`--render-gpu 1`及`activeGpu=1`，且CUDA_VISIBLE_DEVICES绑定对应UUID。发布脚本仅将原固定渲染GPU参数化，控制与加载公式不变。

输出`raw.mp4`、`trace.npz`、`protocol.json`、`status.json`。终止时环境自动reset，原始视频最后一帧可能为reset画面；原始轨迹在reset前采集。正式展示视频应移除该帧并明确标记终止，不能把reset当成机器人自己恢复。

本次结果（seed20261127，世界+X水平拉力，初始base朝向随机）：

- 轻拉实际约8.8N，末端相对机身位移10.7cm；重拉约20.8N、位移18.6cm；最后释放回到距初始位置约1.6cm。
- 约44N起J2距限位不足3度，60N起基本贴限位。
- 90N底盘仍基本原地，但臂已贴限位，不代表良好柔顺力跟踪。
- 105N开始持续被拖走；150N时累计漂移14.45m、倾角58.74度，触发坏姿态终止，未触发机身触地。

这是当前闭环、姿态、地面和加载过程的结果，不是通用或真机安全负载。尚未进行这个力控模型的真机验证。

## 校验与许可证

30K检查点112个张量均有限，iteration及哈希已核验；发布前核对正式源码清单无差异。阻抗/历史布局/弹簧相关3项测试通过，两个录像进程exit0，轨迹有限，逐样本公式与水平张力回算通过。

完整CPU测试中的上游奖励独立对照需要相邻`learning-compliance`原仓库；PS2缺少该独立checkout时此项报错，其余8项通过。来源与MIT许可证见 `reproduction/force_control/provenance.json` 和 `vendor/LICENSE`。本仓库原有第三方许可证继续保留。
