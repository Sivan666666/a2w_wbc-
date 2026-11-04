# SuperMimic 25K基线与迁移记录

2026-11-04，用户选择 Go2 对齐 BPTT 25K 为当前版本，停止 PS1 原训练，
将代码与固定25K权重迁往 A100；最新授权为先迁移、暂不训练、先给优化方案。
本次不修改算法、奖励、模型结构或预算，不启动仿真采集。

## 当前基线

25K在已完成独立评测的10K至35K检查点中扩展域姿态误差最低，
并非所有指标均最优：30K原生EE位置误差更小。选择由用户明确确认。

扩展域为link0参考坐标 x∈[0.10,0.85]m，三种子各64环境×200秒，
noise off，全帧post-physics pre-reset。下表不是训练即时误差或世界模型预测误差。

| 模型 | 原生EE cm | EE姿态 deg | XY m/s | yaw rad/s | 失败reset | 训练时间 | 物理transitions |
|---|---:|---:|---:|---:|---:|---|---:|
| BPTT 25K | 2.30525 | 8.79693 | 0.0852595 | 0.0867852 | 0 | 18:50:03 | 102473728 |

想象transitions为6474956800。原训练为PS1 RTX4090、seed0，包含编译和共享干扰；
不能与历史A100 seed42 PPO作严格算法wall-time加速比较。
历史PPO10K扩展域姿态约4.792°，25K仍未达到整体同等效果。
物理步包括18个启动步；BPTT采集iteration不等于PPO iteration。

固定文件是 `checkpoint_0025000.pt` 和包含RunningNorm的 `actor_0025000.jit`。
哈希及完整计数见 `reproduction/supermimic_25k_baseline.json`。
不会用latest或其他检查点替代25K；权重独立存储，不提交Git。

## 相对分支起点3b67f1b的改动

1. 发布实际使用的 `scripts/go2_aligned/` 与 `train_go2_aligned.py`。
   固定Go2参考提交d9eee9b3f248e5300e636c433255db2c1c3e086a；
   4096env，1步采集/2次WM更新，WM2048×8，想象8192×32，
   300轮WM预热、1000轮Actor LR渐增、RunningNorm、编译、原collapse guard。
2. 保留A2W/Z1必要适配：22动作、234策略输入、86维物理状态、766维物理context，
   原EE奖励/命令/执行器延迟和轮足资产。并非完整复现Go2任务。
   细节见 `GO2_SUPERMIMIC_ALIGNMENT.md`。
3. 发布指定检查点阶段评测脚本，以及历史PPO参照评测和来源manifest显式参数。
   `eval_supermimic_stage.py` 是旧格式归档工具，不能用于新Go2检查点。
4. 记录旧PPO队列/旧BPTT已取消，保留PS1隔离运行脚本作历史记录，
   不把硬编码PS1路径当作A100启动配置。
5. 新增25K基线manifest、CPU权重核验工具、改动说明和待批准优化方案。
   本次没有合入失败的姿态WM pilot/AB/ABC试验，也没有更改已训练网络。

发布中的训练入口、Go2模块及复用的supermimic模块与PS1停止时源码SHA256一致。
机器人资产的绝对mesh路径仍需在目标目录中用既有relocate工具处理；
迁移本身不代表A100仿真可运行或已复验策略。

## PS1停机

2026-11-04 19:31:34北京时间停止容器 `a2w-go2-aligned-train-seed0-20261103`。
先发送停止信号，60秒未退出后Docker强制结束，exit137，OOMKilled=false。
最后完整日志36400轮，最后完整检查点36000；没有新建停机检查点，
没有完成原40K预算。原检查点及日志保留，原30分钟监控已删除。

## A100目标及隔离范围

新授权端点为端口42300的root会话；不等于旧端口42019的sivan账户。
个人根目录使用 `/data01/niuwa/txc`，代码放workspace、权重放models，
缓存/临时文件/日志/独立环境分别放 `.cache`、tmp、logs、envs。
不修改其他用户目录或系统Python/CUDA；密码不进入仓库或启动脚本。

物理GPU7核验为A100-SXM4-80GB，UUID为
`GPU-80244c44-c25d-2343-4813-831bed5dc35d`。当前不启动训练，也不占用其他GPU。
只读环境核验：Ubuntu22.04.5、驱动580.95.05；`/opt/conda/bin/python`
为Python3.12.13、Torch2.12.0+cu130、cuDNN9.20；系统python3是3.10.12。
该conda环境未安装Isaac Sim、Isaac Lab或Gymnasium。

原训练栈是Isaac Sim5.1、Python3.11、Torch2.7+cu128，不能仅因系统CUDA13
存在就宣称兼容。后续须单独准备固定版本的隔离环境并验证；本次不安装共享依赖。
`reproduction/verify_supermimic_25k.py` 只作CPU权重/Actor接口核验，不启动仿真。

检查点含Actor/Critic/WM与优化器状态，但没有完整replay或PhysX状态。
现有训练入口只支持从零；未来warm-start必须另建入口与run，明确初始化、
新采样/预填充、学习率/优化器处理和新增预算，不宣称精确续训。
