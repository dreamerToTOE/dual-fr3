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

在仿真命令上加 `--upper5-z-unload --upper5-blur-ms 200`，可启用竖向卸载并同时记录原始相机和模糊相机的观测。模糊观测由 15 Hz 连续实渲染帧按曝光窗口加权合成，再重新检测 ArUco。它近似了 200 ms 的长曝光，用于压力测试。将回放命令的输出换成新文件并加 `--blurred-camera`，可在同一条 PhysX 记录上比较原始与模糊观测。

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
