"""Compare measured 50 Hz WRITE samples; never align away tracking error/drift."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
specs = [('original', '53.0–64.2', a.root/'a2w_neowa_stationary'),
         ('10cm', '63.0–74.2', a.root/'a2w_neowa_raised/10cm'),
         ('15cm', '68.0–79.2', a.root/'a2w_neowa_raised/15cm')]
report = []
for variant, height, reference in specs:
    ref_summary = json.loads((reference/'summary.json').read_text())
    ref_trace = json.loads((reference/'trace.json').read_text())
    other = a.output/variant
    if not (other/'summary.json').exists():
        continue
    for label, folder in [('A2W + Z1', reference), ('B2 + Z1', other)]:
        summary = json.loads((folder/'summary.json').read_text())
        trace = json.loads((folder/'trace.json').read_text())
        assert summary['outcome'] == 'completed_sequence', summary['outcome']
        assert summary['fixed_base'] is False and summary['ik_correction'] is False
        assert summary['simulation_resets'] == 0 and not summary['shifts']
        assert all(v['command'] == [0., 0., 0.] for v in trace)
        # Exact equality of every timed target, not just glyph shape.
        assert [(v['t'], v['phase'], v['letter'], v['stroke'], v['goal']) for v in trace] == [(v['t'], v['phase'], v['letter'], v['stroke'], v['goal']) for v in ref_trace]
        assert summary['world_paths'] == ref_summary['world_paths']
        rows = [v for v in trace if v['phase'] == 'WRITE']
        diff = np.array([v['actual'] for v in rows])-np.array([v['goal'] for v in rows])
        errors = np.linalg.norm(diff, axis=1)*100
        base = np.array([v['base'] for v in trace])
        base_write = np.array([v['base'] for v in rows])
        values = dict(robot=label, variant=variant, height_cm=height, samples=len(rows),
                      mean_cm=float(errors.mean()), rmse_cm=float(np.sqrt(np.mean(errors**2))),
                      p95_cm=float(np.percentile(errors, 95)), max_cm=float(errors.max()),
                      axis_mae_cm=(np.abs(diff).mean(0)*100).tolist(),
                      axis_signed_mean_cm=(diff.mean(0)*100).tolist(),
                      base_max_from_spawn_cm=float(np.linalg.norm(base[:, :2], axis=1).max()*100),
                      base_max_from_write_start_cm=float(np.linalg.norm(base_write[:, :2]-base_write[0, :2], axis=1).max()*100),
                      min_upright=min(v['upright'] for v in trace),
                      letters=summary['letters'], policy_sha256=summary['policy_sha256'])
        report.append(values)
a.output.mkdir(parents=True, exist_ok=True)
(a.output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
lines = ['# B2 + Z1 与 A2W + Z1：原地 neowa 书写对比', '',
         '使用各自已有的5000-iteration WBC checkpoint，不重新训练。全程底盘指令 vx=vy=vyaw=0；不锁底盘、不加IK、不重置仿真。', '',
         '## 对齐与误差定义', '',
         '- 同一世界平面 X=0.9239999999999999m；文字总宽42.4cm，每个小写字母高11.2cm，宽7.2cm；中心间距8.8cm。',
         '- 逐帧断言两种机器人目标位置、时刻、字母、笔画和阶段完全一致；控制频率50Hz，物理步长5ms。',
         '- 误差仅统计WRITE阶段：e(t)=||p_actual(t)-p_target(t)||₂，使用同期三维位置，不做最近曲线匹配或时间对齐。',
         '- Mean为逐帧误差平均；RMSE=sqrt(mean(e²))；P95为95百分位。单位均为cm。',
         '- 底盘最大位移为相对初始出生位置的水平距离，含初始站稳阶段；另在JSON保留相对首个书写帧的位移。',
         '- 这是一条确定目标轨迹的MuJoCo诊断，不是多seed基准，也不能单独归因于足式/轮式构型。机器人资产、PD和已训练策略不同。', '',
         '| 目标高度(cm) | 机器人 | Mean(cm) | RMSE(cm) | P95(cm) | 最大底盘位移(cm) |',
         '|---|---|---:|---:|---:|---:|']
for r in report:
    lines.append(f"| {r['height_cm']} | {r['robot']} | {r['mean_cm']:.2f} | {r['rmse_cm']:.2f} | {r['p95_cm']:.2f} | {r['base_max_from_spawn_cm']:.2f} |")
lines += ['', '## 各字母平均三维位置误差(cm)', '', '| 高度(cm) | 机器人 | n | e | o | w | a |', '|---|---|---:|---:|---:|---:|---:|']
for r in report:
    lines.append('| '+r['height_cm']+' | '+r['robot']+' | '+' | '.join(f"{r['letters'][v]['mean_error_cm']:.2f}" for v in 'neowa')+' |')
lines += ['', '## B2视频', '', '[原高度](original/neowa.mp4) · [上移10cm](10cm/neowa.mp4) · [上移15cm](15cm/neowa.mp4)', '',
          '青色为目标，橙色为实际末端轨迹；底部投影为统一世界Y–Z坐标，没有对各字母单独平移/旋转配准。数值误差包含X方向。', '',
          '每个目录保存trace.json、summary.json、events.json及视频/截图；comparison.json包含汇总和checkpoint SHA256。', '',
          '## 结果解读', '',
          '在本次同一条目标轨迹上，B2底盘水平位移更小，但末端位置误差明显更大，主要表现为整段文字偏高，而不是完全无法跟随字形。',
          '高位版本B2的平均有符号XYZ误差为约(+4.15, -2.99, +14.52)cm；没有通过平移目标、平移实测轨迹、IK补偿或重新训练消除偏差。',
          '偏差的完整原因尚未确定：这不是严格匹配所有动力学参数的跨模拟器验证，不能直接认定B2构型本身不如A2W，也不能作为实机部署精度。', '',
          '## 坐标与复现校验', '',
          '- B2原MJCF安装位置为(0.20,0,0.25)m，训练USD固定关节为约(0.15,0,0.225)m。本次仅在输出副本对齐安装位置。',
          '- 在link06上添加无质量、无碰撞的固定end_effector坐标体，平移约(0.201,0,-0.01)m，与训练USD的ee_gripper关节一致。详见[坐标审计](frame_audit.txt)。',
          '- B2腿部PD为250/5；Z1各关节PD沿用训练配置；原MJCF的质量、惯量、摩擦、碰撞和力矩限幅没有重新调参。没有声称这些动力学项已经与USD逐项一致。',
          '- 原始官方MJCF文件git diff为空；所有适配只写入各输出目录。',
          '- B2为210维输入/18维动作；A2W为234维输入/22维动作，均为三帧历史。',
          '- 已验证scale+offset后clip的实际代码表达式，两种构型各1000组随机输入通过。',
          '- 新脚本A2W高位分支回归：1537帧trace与旧视频记录逐元素完全一致，包括状态、目标、动作相关记录。',
          '- 三个视频均为1920×1080、25fps、769帧、30.76秒；完整解码无错误，已检查截图中四腿、机械臂与目标/实测轨迹均可见。', '',
          '## PS1重跑', '',
          '脚本位于`/mnt/sda/ps/leggedmanip-lab/LeggedManip_Lab/scripts/`，使用本项目隔离环境，选择空闲GPU。', '',
          '```bash',
          'bash /mnt/sda/ps/leggedmanip-lab/LeggedManip_Lab/scripts/repro_container.sh 1 leggedmanip-b2-neowa-rerun bash /mnt/sda/ps/leggedmanip-lab/LeggedManip_Lab/scripts/run_b2_neowa_comparison.sh',
          '```', '',
          '注意：重跑前修改run_b2_neowa_comparison.sh中的输出目录，避免覆盖本次记录。', '',
          '[推理与录制脚本](../scripts/repro_mujoco_neowa_compare.py) · [批量运行脚本](../scripts/run_b2_neowa_comparison.sh) · [统计脚本](../scripts/analyze_neowa_comparison.py)']
(a.output/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report, indent=2))
