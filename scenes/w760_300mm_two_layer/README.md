# 76 cm 托盘、30 cm 高墙、双层 15 箱场景

托盘内宽 760 mm，四面墙高 300 mm。箱体边长 150 mm、质量 1.25 kg，双臂滑轨沿 Y 轴布置，中心位于 X = ±600 mm。后置 D455 仿真相机高度为 2.45 m，RGB 分辨率为 1280 × 960，以 15 Hz 采样。PhysX 和控制均为 120 Hz，界面按 30 Hz 刷新。

场景从已就位的第一层后排五箱开始，先码放第一层前排五箱，再码放第二层靠后墙的一排五箱。搬运使用侧吸，箱体保持直立。上层第 1、2 箱分别贴左、右墙，第 3、4 箱分别贴第 1、2 箱，第 5 箱按相机复核的中间槽位沿 +Y 直推。上层从左到右为 1、3、5、4、2。

新增前排参与上层落地承重。上层第 1–4 箱在接近支撑接缝时用侧吸短抬横移，落稳后再完成贴靠。第 4 箱放下后的撤离轨迹向前偏移 3 mm，以留出右腕绕过第 2 箱的空间。运行器用 3 个 CPU 进程检查碰撞，轨迹根据观测和导引路点重新规划。最终验收运行的轨迹缓存读取数为 0。

## 可视化运行

在仓库根目录运行，令 `ISAAC_SIM_DIR` 指向 Isaac Sim 4.5.0 的安装目录。官方 FR3 USD 和 D455 资产需能通过 Isaac Sim 的资产源解析。每次选一条命令，输出目录使用新路径。

```bash
# 加载已码好的第一层后排，先码第一层前排，再码第二层
"$ISAAC_SIM_DIR/python.sh" scenes/w760_300mm_two_layer/two_row_physx.py /tmp/fr3-two-layer

# 加载已码好的第一层两排，从第二层第 1 箱开始
"$ISAAC_SIM_DIR/python.sh" scenes/w760_300mm_two_layer/two_row_physx.py /tmp/fr3-upper-row \
  --resume scenes/w760_300mm_two_layer/validation/lower_two_rows_checkpoint.json

# 加载第二层第 5 箱的落地断点，复验相机对准和直推
"$ISAAC_SIM_DIR/python.sh" scenes/w760_300mm_two_layer/two_row_physx.py /tmp/fr3-upper-5 \
  --resume scenes/w760_300mm_two_layer/validation/upper5_landing_checkpoint.json
```

可用系统 Python 和 `--check-only` 检查启动参数、断点和生成的运行器。`upper4_after_transfer_checkpoint.json` 是最终验收段的起点，保留第 4 箱跨缝落稳的实测状态。

## 验收记录

本轮以单实例、断点续跑方式完成 Isaac Sim 可视化和 PhysX 验收。上层五箱均连续 60 个物理步稳定承重，最大就位中心误差为 2.12 mm。最终验收段记录到零次非预期机械臂接触。 [运行报告](validation/runner_report.json)

第 5 箱沿 +Y 推进 85.98 mm，最大横向漂移 0.76 mm，实际最小右侧间隙 0.93 mm。 [直推审计](validation/upper5_straight_audit.json)

[最终截图](validation/upper_cube_final.png) · [15 箱最终位姿](validation/final_15_box_poses.json)

最终位姿文件用于查看和分析。验收材料中的运行路径使用本地实验编号表示。整排一次无中断运行和实体 FR3 验证仍待进行。

## 第 5 箱接触识别实验

以下命令从上层第 5 箱的实测落地断点开始，在仿真创建时给箱体绕世界 Z 轴叠加偏航角。用 `0` 跑对照，再逐次换成其他角度。每次使用新的输出目录。

```bash
scene=scenes/w760_300mm_two_layer
run=$(mktemp -d)/upper5
"$ISAAC_SIM_DIR/python.sh" "$scene/two_row_physx.py" "$run" \
  --resume "$scene/validation/upper5_landing_checkpoint.json" \
  --upper5-yaw-deg 0 --close-on-finish
python3 "$scene/experiments/score_upper5.py" "$run" \
  --labels "$run/upper5_truth.jsonl" > "$run/upper5_score.json"
python3 "$scene/experiments/fusion_upper5.py" \
  --case "$run" "$run/upper5_truth.jsonl" --delay-ms 0 --delay-ms 100 --delay-ms 200 \
  --out "$run/fusion_replay.json"
```

评分使用 PhysX 的当前步接触冲量和箱体位姿。入槽检查拒绝、运动中腕力超限、邻箱持续受阻分别计数。回放使用相机时间戳、ArUco、腕部力觉和 TCP 运动学，真值只用于事后对照。延迟是回放中额外加入的图像传输时间。回放结果列出各运行缺失的状态。

## 技能 3 → 4 的接触判定

用 `experiments/fusion_skill_boundary.py` 比较视觉＋运动学、腕力＋运动学和三者融合。针对第 4 箱贴邻箱与换臂贴后墙、第 5 箱直推，以及第 2 箱跨缝后尝试贴后墙，任务预先指定执行臂、方向和预期目标。第 4 箱前推时与已贴邻箱的接触可以保留。每个工况由原始运行报告限定回放窗口，并按实际接触对象、接触冲量和箱体推进量逐步评分。

```bash
python3 scenes/w760_300mm_two_layer/experiments/fusion_skill_boundary.py \
  --case "$upper4_run" u4_side --case "$upper4_run" u4_rear \
  --case "$upper5_run" u5_rear --case "$upper2_reseat_run" u2_reseat \
  --out comparison.json --trace comparison_trace.jsonl
```

每条回放记录的 `predictions` 是可用于技能选择的观测判定，`reference` 是独立物理标签。`TARGET_CANDIDATE` 表示接近指定边界且推进很慢，可进入技能 4 的确认流程。`TARGET_CONTACT` 表示 PhysX 记录到箱体和预期目标实际承载。力增量辅助判据也作为候选记录，评估其命中、漏判和触发时间。

本组使用已有模拟日志进行离线对照。缺少可用于评分的真实同层邻箱卡滞样本，因此卡滞标签不会进入本组的识别率。腕力超限标签由原运行器事件和独立箱体推进量共同确定。

在仿真命令中加 `--upper5-z-unload --upper5-blur-ms 200`，可沿用第 5 箱竖向卸载并同期记录清晰与模糊的 ArUco 结果。模糊观测按真实采样时间对连续 D455 图像积分，以曝光中点为时间戳，重新检测 ArUco 后写入 `vision/blurred_camera.jsonl`。200 ms 是 15 Hz 多帧合成的压力测试。机械臂控制继续使用原始 D455 观测。对同一运行加 `--blurred-camera` 再做一次融合回放，使用独立的输出 JSON。

## 技能 4 的指令与受力响应回放

回放使用原始腕力日志里的实际关节位置指令，经相同的 FR3 Lula 模型转换为 TCP 指令。用 0.5 s 滑动窗口比较两端各 8 帧的平均值。窗口内沿推进方向的指令增量须为 40–130 μm，编码器测得的 TCP 位移幅度最多为指令增量的 35%，轴向力增量至少为 0.30 N。融合位姿给出预期目标的间隙。两个版本都要求间隙绝对值不超过 2 mm，探索版再应用原控制器的正间隙上限，后墙 0.45 mm、邻箱 1.8 mm。

在原始日志上可直接回放，无需启动 Isaac Sim。

```bash
scene=scenes/w760_300mm_two_layer
python3 "$scene/experiments/active_probe_replay.py" --isaac-sim-dir "$ISAAC_SIM_DIR" \
  --case "$upper4_run" u4_side --case "$upper4_run" u4_rear \
  --case "$upper5_run" u5_rear --case "$upper2_reseat_run" u2_reseat \
  --out active_probe_archive.json
```

将末尾 0.25 s 内至少 80% 的物理步同时记录到预期目标与推进吸盘承载，作为独立评分参考。12 段档案中，2 mm 判据产生 649 个候选窗口，其中 416 个符合独立承载参考。应用正间隙上限后保留了这 416 个窗口。第 2 箱受支撑摩擦影响、尚未抵住后墙的一段有 36 个宽松候选，探索版全部剔除。第 3 箱已有真实邻箱接触和 95 μm 主动探测，统一判据仍未检出。滑动窗口相互重叠，上述计数只描述这批档案。

对贴邻箱另做一次几何消融。保持力和指令响应判据、−2 至 1.8 mm 间隙范围，使用两个箱体观测轮廓的分离轴间隙。第 3 箱从 0 个候选增至 136 个，全部符合独立承载参考。第 4 箱贴邻箱时仍有 34 个符合承载的候选。第 3 箱首个检出出现在显式主动探测前的低速贴靠期间。

第 5 箱可从同一落地断点对比原有终点控制和主动探测。两次用相同偏航角、配置和独立输出目录。第二次在命令末尾加 `--upper5-active-probe`，复用现有的后墙再贴靠探测。

```bash
scene=scenes/w760_300mm_two_layer
run=$(mktemp -d)/upper5
"$ISAAC_SIM_DIR/python.sh" "$scene/two_row_physx.py" "$run" \
  --resume "$scene/validation/upper5_landing_checkpoint.json" \
  --upper5-yaw-deg 0.35 --upper5-z-unload --close-on-finish
```

启用主动探测时，`upper5_active_probe.jsonl` 记录 `physics_step`、`time_s`、`arm`、`target_path`、`command_y_m`、`measured_tcp_y_m`、`encoder_center_y_m`、`estimated_target_gap_m`、`axial_force_N`、`resultant_force_N`、`camera_frame_id`、`capture_time_s` 和 `received_time_s`。回放用 `--case "$run" u5_rear` 连接该记录与原始腕力、相机和 PhysX 真值。输出的 `logged_active_probe` 包含记录步数、最后窗口的承载步数和探测响应。若运行器先发出 `SENSOR_PROBE_CANDIDATE`，回放单独审计该时刻的实际承载。

## 从实测末端断点对照

两组终点对照可从同一实测断点恢复。断点原始记录的后墙间隙为 1.002 mm，轴向腕力为 5.701 N。此入口恢复箱体、导轨和关节状态，从贴靠段开始。以下两条命令依次运行。

```bash
scene=scenes/w760_300mm_two_layer
checkpoint="$scene/experiments/results/probe_terminal_checkpoint.json"

"$ISAAC_SIM_DIR/python.sh" "$scene/two_row_physx.py" "$(mktemp -d)/terminal-passive" \
  --resume "$checkpoint" --upper5-yaw-deg 0 --close-on-finish
"$ISAAC_SIM_DIR/python.sh" "$scene/two_row_physx.py" "$(mktemp -d)/terminal-active" \
  --resume "$checkpoint" --upper5-yaw-deg 0 --upper5-active-probe --close-on-finish
```

[多源状态判断实验记录](experiments/results/README.md)
