# 观测驱动技能运行记录

新入口沿用 760 mm 托盘、300 mm 高墙、150 mm 箱体和后墙 D455 无标记拟合。场景、导引文件和历史实测断点保留在原分支及本分支的基线历史中。

## 已接入的运行路径

- A 段从相机重新取得供料位置。使用现场观察注册搬运的起点、landing 和导轨行程，关节轨迹从实测关节角和携箱关系重新求解。导引中的相对空间走廊和关节分支初值仍可用于求解。
- B1 依据相机或已建立接触后的估计箱位生成预接触位。从关节实测和当前导轨位置计算 TCP，查验 IK 目标与场景障碍，再调用现有 RRTConnect。接触前最后一段沿目标面的法向低速搜索腕力。
- 主要撤离动作根据腕力卸载及实测 TCP、箱面之间的估计间隙结束。每次执行前检查下一小段，行程上限只限定搜索范围。完整就位的仿真审计仍使用独立 PhysX 承托和位置记录。
- 前排靠墙箱的 B2 目标使用标定侧墙和实测后排箱面。中途让开相机视野前，用动态卸力确认两杯分离，再从实测关节状态规划 RRTConnect。由新相机观测生成下一次接近目标。
- 从空托盘开始的入口先供料并处理第一层后排，再衔接现有前排与上层流程。逐箱的目标对象、执行臂和切换条件见[15 箱运行线](long_skill_short_state_routes.md)。

本轮保留了关节归位姿势、若干临时高位避障走廊、计算安全范围所需的静态场景标定，以及部分既有 PhysX 互锁。它们仍需要结合新技能逐步迁移和实跑。静态编译只检查新入口的装配与断点识别，不能证明运动可达和接触任务成功。

## 单实例试验进度

15 个箱位及其断点入口均通过静态装配检查。以下记录按箱位列出单实例 Isaac Sim 验证范围。

### R1 后排第 1 箱

| 环节 | 记录 |
| --- | --- |
| A 段落稳后撤离 | S0 在累计指令撤离 3.325 mm 时触发。此时两杯估计间隙为 3.110 mm 与 3.139 mm。PhysX 对照显示两杯均已分离，地面承载约 12.26 N。试验搜索上限为 12 mm。 |
| B1 转入接触控制 | 首次轨迹按指令结束时，实测 TCP 仍距预接触位约 18 mm。修正后等待实测到位再切换。后墙前断点续试中，左右两臂分别在误差 0.894 mm、0.976 mm 时进入腕力贴面。两段 RRTConnect 共检查 2,126 个轨迹采样点。 |
| B2 后墙承载 | 旧腕力门槛在 6.07 N 时触发，PhysX 后墙承载仍为 0 N。加入编码器位移和腕力增量后，在第 6,780 步确认，腕力 7.32 N，PhysX 后墙法向力 4.39 N。首次持续超过 0.1 N 的 PhysX 承载在此之前约 0.33 s。右臂进场时后墙承载多次降为零，箱体仍靠近后墙。 |
| B2 横推 | 原横推走了 0.26 mm，右腕合力到 9.19 N 时触及 9 N 上限。该步右腕竖直反力 3.91 N，地面支撑增至 16.18 N。同一断点仅加入竖向顺应后，箱横移 2.36 mm，右腕合力峰值 9.21 N。再接入既有双腕接触同步后，两腕在约 1.99 N 和 0.97 N 的预载下确认稳定，横推仍在 0.33 mm 时触及力限。三次 PhysX 均未记录意外碰撞。 |
| 横推臂构型对照 | 将历史可行构型临时用作优先 IK 初值，保持当前视觉目标和 RRTConnect 实测起点。右臂到达预接触位，但触碰时法向力峰值 10.35 N，未进入横推。该 IK 初值选择已撤回。 |

运行证据保存了识别事件、同一物理步的 PhysX 对照、失败点和实际测得的恢复断点。箱位的通关以本分支的实跑结果为准。 [R1 运行证据](validation/observed/rear1_progress.json)

### F1 前排第 1 箱

从 B1 结束的实测断点开始，右臂横推至左墙。左墙有载荷的时间窗为 24/24 帧。右腕实测撤离 3.56 mm 时确认该工具分离。随后左腕实测撤离 1.66 mm，触发两腕分离的 S0。PhysX 同步记录显示两杯均无接触或吸附，地板承载约 12.26 N。RRTConnect 以当时实测关节为起点，清开相机视野，替代了原固定 65 mm 撤退。

首次再次接近未通过 IK 门槛。记录并恢复实测侧墙断点后，使用撤离时实测的近箱腕姿作为备用 IK 初值，从 D455 重新拟合下一个接近目标。只让左臂清出相机视野时，前排箱最终抵达 `Cube_1`，但右杯仍靠近箱侧，估计间隙 2.50 mm，最终 S0 未确认。PhysX 对照显示两杯当时已分离。

改为中途 S0 后双臂一起用 RRTConnect 清场，再由左臂从实测关节位置规划接近。最后一次从侧墙断点续跑通过，向后推入约 24.01 mm，连续 24 帧记录到指定后排箱承载。最终 S0 在左腕实测撤离 2.13 mm 时确认，两杯均无接触，地板承载约 12.26 N。最终归位及就位验收完成，PhysX 非预期接触为零。 [F1 运行证据](validation/observed/front1_progress.json) · [F1 侧墙断点](validation/observed/front1_side_seat_checkpoint.json) · [F1 就位断点](validation/observed/front1_complete_checkpoint.json)

### F2 前排第 2 箱

从 F1 就位断点开始。D455 拟合供料箱位，左腕吸后面、右腕吸前面。整段导轨搬运的 9,324 个物理采样点均保持两杯吸附。落稳开杯后，两腕分别实测外撤 2.784 mm、2.786 mm，触发 S0。PhysX 在该时刻确认两杯分离，地板承载 12.26 N。B1 用落箱后的相机观测和右腕实测关节位置生成接近路径，实测 TCP 误差小于 1 mm 后进入贴面，随后沿 +Y 推至横推区。 [F2 运行证据](validation/observed/front2_progress.json) · [F2 landing 断点](validation/observed/front2_landing_checkpoint.json) · [F2 B1 完成断点](validation/observed/front2_b1_checkpoint.json)

首次 B2 在横推前被避障检查拦截。计划样本中右臂第 6 连杆距右墙仅 1.879 mm，低于既有 2 mm 规划余量。PhysX 未记录非预期接触。复验从 landing 断点出发，在 B1 优先采用历史可行构型作 IK 初值。仍用当前相机拟合箱面生成目标，用实测关节位置开始 RRTConnect。横推的 1,801 个采样点通过避障检查。

F2 由左臂横推至右墙，确认两杯分离后双臂 RRTConnect 清场，重新观测箱位，再由右臂沿 +Y 推至 `Cube_2`。右墙和目标后排箱均各有 24/24 帧的对应承载记录。最终 S0 在右腕实测外撤 3.137 mm 时确认，PhysX 同步确认两杯无接触、地面承载 12.26 N。F2 从 landing 到就位验收通过，非预期接触为零，可从生成的断点选中 F3 `Cube_13`。 [F2 就位断点](validation/observed/front2_complete_checkpoint.json)

### U5 上层第 5 箱

从带 14 个已码箱的实测 landing 断点出发。相机拟合的两侧槽隙为 4.54 mm 和 3.92 mm，左臂根据观测和实测关节生成 RRTConnect 路径，实测 TCP 误差达到 0.966 mm 后开始贴面。沿 +Y 推进 86.65 mm，横向位移全幅 1.54 mm。确认后墙承载时，PhysX 同步参考为 6.02 N。

最终 S0 在左腕实测外撤 2.37 mm 时确认。两杯均无接触、无吸附，PhysX 测得底部支撑约 11.98 N。空载返回的 RRTConnect 路径通过碰撞检查，实跑记录零次非预期接触。最终完成报告为 `PASS_W760_UPPER_ROW`，其中本轮主动运行的箱体只有 U5，范围为 B1 至就位，A 段及其他 14 个箱位来自恢复断点。 [U5 运行证据](validation/observed/upper5_acceptance.json)

## 运行和续跑

在仓库根目录用一个 Isaac Sim 4.5 实例运行。每次使用新的输出目录。

```bash
scene=scenes/w760_300mm_two_layer
sim=/home/ubuntu2004/isaacsim-4.5.0
"$sim/python.sh" "$scene/markerless_rows.py" /tmp/fr3-observed-full \
  --observed-skills --start-empty --close-on-finish
```

可用 `--stop-after-rear 1` 在第一个后排箱就位后结束。成功落到 landing 后保存 `rear_landing_checkpoint.json`，后排每箱就位后保存 `rear_row_checkpoint.json`。续跑时用 `--resume <断点路径>` 代替 `--start-empty`。历史入口按原参数运行。无界面实跑可加 `--headless`，只需生成脚本并校验参数时用系统 Python 和 `--check-only`。

复现 F1 最终推入的试验入口。

```bash
"$sim/python.sh" "$scene/markerless_rows.py" /tmp/fr3-observed-front1-terminal \
  --observed-skills --resume "$scene/validation/observed/front1_side_seat_checkpoint.json" \
  --stop-after-front 1 --close-on-finish
```

复现上层第 5 箱时直接恢复它的 landing 断点。

```bash
"$sim/python.sh" "$scene/markerless_rows.py" /tmp/fr3-observed-upper5 \
  --observed-skills --resume "$scene/validation/markerless/upper5_landing_checkpoint.json" \
  --close-on-finish
```
