# 2026-11-25 在线识别与码垛复验

场景使用 76 cm 宽托盘、30 cm 高墙和 15 cm 箱体。单个 Isaac Sim 4.5.0 实例按实测断点串行运行，复验覆盖下层第二排和上层第一排。以下是各段实测的汇总。 [验收数据](acceptance.json)

## 上层第 4 箱在线识别

在线观测器使用 D455、双腕 TCP 六维力和机器人编码器，并记录两臂实际发出的 TCP 位置指令。承载真值根据同期 PhysX 接触对象与冲量生成。评分从侧推开始，到撤离开始时结束。识别结果用于对照评分，控制器在箱体坐稳后开杯撤离。

| 状态 | PhysX 真值步数 | 正确步数 | 召回率 | 精确率 | F1 |
|---|---:|---:|---:|---:|---:|
| S5 单面承载 | 3404 | 1693 | 49.74% | 92.46% | 64.68% |
| S6 双面承载 | 138 | 89 | 64.49% | 100.00% | 78.41% |

S6 连续三步首次确认耗时 0.425 s。49 步漏检均发生在估计器尚未确认后墙承载时。侧载为 0.1–1 N 时，28 个真实 S6 步命中 8 步。数据来自一次正常码垛复验。 [详细指标和同轨对照](upper4/acceptance_metrics.json) · [逐步评分](upper4/u4_paired_scores.jsonl.gz) · [识别时间图](upper4/u4_s5_s6_timeline.png)

归档的 [传感器输入](upper4/contact_states/sensor_inputs.jsonl.gz)、[PhysX 参考流](upper4/contact_states/reference.jsonl.gz)和[实测在线输出](upper4/contact_states/u4_handover.jsonl.gz)按物理步和仿真时间配对。下列命令用当前观察器逐步复算，核对 S5/S6、边界对象及侧向和后向承载输出。命令在仓库根目录运行，需要 NumPy。

```bash
e=scenes/w760_300mm_two_layer/experiments
python3 "$e/u4_online_replay.py" "$e/results/online_acceptance_20261125/upper4"
```

## 码垛和撤离

上层第 1–4 箱稳定就位后，双臂使用同一条在线 RRTConnect 轨迹同步撤离。第 5 箱沿 Y 方向直推约 86 mm，最后由左臂撤出，右臂已停靠。各次撤离约 3 s，361 个轨迹采样点均通过碰撞检查，运行中 PhysX 未记录意外接触。腕力表用于复核开杯和撤离衔接，峰值取自实际撤离段。

| 上层箱号 | 撤离腕部合力峰值（N） | 复验起点 | 实测报告 |
|---|---:|---|---|
| 1 | 2.24 | A 阶段落稳后 | [报告](upper1/runner_report.json) |
| 2 | 1.78 | 第二次跨缝落稳后 | [报告](upper2/runner_report.json) |
| 3 | 2.23 | 跨缝落稳并恢复已记录的双臂位姿 | [报告](upper3/runner_report.json) |
| 4 | 0.14 | 跨缝落稳后，执行换臂贴靠 | [报告](upper4/runner_report.json) |
| 5 | 0.08 | A 阶段开杯后 | [报告](upper5/runner_report.json) |

第 3 箱重新接近后再横推的尝试曾顶到下层箱边缘，目标侧邻箱距离仍约 1 mm。恢复跨缝断点记录的双臂位姿后，PhysX 连续 30 步确认侧邻箱承载，随后箱体稳定就位。第 5 箱直推结束的位置误差约 0.99 mm。 [第 3 箱中止记录](upper3/pose_only_attempt_report.json) · [下层第二排记录](lower_front_withdrawal_metrics.json) · [第 5 箱直推详情](upper5/straight_push_metrics.json) · [整排完成断点](upper5/final_checkpoint.json)

各上层报告同目录下保存原始输入断点和撤离时段的腕力与关节速度。复跑时用 `two_row_physx.py --resume` 加载相应的 `input_checkpoint.json`，第 3 箱使用保留双臂位姿恢复逻辑的 `experiments/u3_seam_resume.py`。例如，复跑第 3 箱可用以下命令。

```bash
s=scenes/w760_300mm_two_layer
archive="$s/experiments/results/online_acceptance_20261125"
out="$(mktemp -d)/upper3"
/home/ubuntu2004/isaacsim-4.5.0/python.sh "$s/experiments/u3_seam_resume.py" "$out" \
  --resume "$archive/upper3/input_checkpoint.json" --headless --close-on-finish
```
