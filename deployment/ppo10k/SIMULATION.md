# MuJoCo：与部署共用键盘、VR 和 PPO

只使用 MuJoCo 后端，不构造真机 Hardware，不连接 DDS/Z1。浮动机身、重力和轮地接触开启，22个电机；夹爪几何固定。模型和网格为相对路径，来源记录在 assets/a2w_z1/provenance.json。

## 安装和启动

在 Python 3.10+ 部署环境中：

```bash
cd deployment/ppo10k
python -m pip install -r requirements-sim.txt
# 当前 NX 的低 PD＋前馈配置
bash run_sim.sh --config deploy.nx.json --mode keyboard --seconds 3600
# 训练 PD/零前馈参考配置
bash run_sim.sh --config deploy.json --mode keyboard --seconds 3600
```

焦点放在 MuJoCo 窗口：Z 起立准备、C 策略、R 恢复准备、T 回零回蹲、X 阻尼。其他键盘/VR 命令见 README。初始为手动蹲姿的模拟：腿 `[0,1.35,-2.7]×4`，根高度0.25m。准备控制为腿2秒线性轨迹、臂3秒五次轨迹和高增益。C 后1秒混合接管，随后策略目标帧间保持，无持续插值。

VR：

```bash
bash run_sim.sh --config deploy.nx.json --mode vr --seconds 3600
python a2w_deploy/pxrea_bridge.py --robot 127.0.0.1 \
  --sdk-lib /opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so
```

## 录制

```bash
bash run_sim.sh --config deploy.nx.json --headless --demo --seconds 60 \
  --video sim_results/demo.mp4 --output sim_results/demo
```

demo 回放键盘和合成 PXREA VR 数据，不能声称连接了实体头显。输出包含视频、输入和轨迹记录；无显示服务器时可按运行环境配置 EGL。

## 解释结果

默认物理步长5ms，策略20ms。机械臂前馈用MuJoCo模型逆动力学计算，真机则用SDK模型，不能视为完全相同的执行器/延迟模型。SDK限位、实际反馈延迟和关节伺服频率也需要单独核对。

此前30秒/60秒视频及约1cm量级误差属于历史训练PD、零前馈配置，不代表当前 deploy.nx.json。近期零输入对照显示低PD＋前馈稳定，但末端位置误差仍可能约7cm；不能将关节跟踪好等同于末端任务精度好。高PD策略闭环抖动在更小物理步长下也曾复现；本目录保留用户确认的低PD＋前馈候选，不包含再次调参。

```bash
python -m unittest discover -s tests -v
```
