# 76 cm 托盘、30 cm 高墙、双层 15 箱场景

托盘内宽 760 mm，四面墙高 300 mm。箱体边长 150 mm、质量 1.25 kg，双臂滑轨沿 Y 轴布置，中心位于 X = ±600 mm。后置 D455 仿真相机高度为 2.45 m，RGB 分辨率为 1280 × 960，以 15 Hz 采样。PhysX 和控制均为 120 Hz，界面按 30 Hz 刷新。

场景从已就位的第一层后排五箱开始，先码放第一层前排五箱，再码放第二层靠后墙的一排五箱。搬运使用侧吸，箱体保持直立。上层第 1、2 箱分别贴左、右墙，第 3、4 箱分别贴第 1、2 箱，第 5 箱按相机复核的中间槽位沿 +Y 直推。上层从左到右为 1、3、5、4、2。

新增前排参与上层落地承重。上层第 1–4 箱在接近支撑接缝时用侧吸短抬横移，落稳后再完成贴靠。第 4 箱放下后的撤离轨迹向前偏移 3 mm，以留出右腕绕过第 2 箱的空间。运行器用 3 个 CPU 进程检查碰撞，轨迹根据观测和导引路点重新规划。最终验收运行的轨迹缓存读取数为 0。

## 视觉基线

视觉拟合法使用后墙 D455 的 RGB-D 图像拟合箱体位置，并有独立启动入口和实测断点。 [视觉拟合法场景](markerless.md)

## 技能与状态改造

本分支的设计用状态事件结束技能。允许的安全运动范围内，任何位置的正确识别均视为同等成功。 [新状态表、技能库与改造顺序](long_skill_short_state.md) · [15 箱逐箱运行线](long_skill_short_state_routes.md) · [试验运行记录](observed_skills_validation.md)

[15 箱整线验收](dynamic_full_15_20261128.md) · [新技能与状态观测实验方案](experiments/long_skill_observer_plan.md)

## ArUco 版可视化运行

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
