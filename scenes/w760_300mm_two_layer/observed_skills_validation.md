# 观测驱动技能运行记录

新入口沿用 760 mm 托盘、300 mm 高墙、150 mm 箱体和后墙 D455 无标记拟合。场景、导引文件和历史实测断点保留在原分支及本分支的基线历史中。

## 已接入的运行路径

- A 段从相机重新取得供料位置。使用现场观察注册搬运的起点、landing 和导轨行程，关节轨迹从实测关节角和携箱关系重新求解。导引中的相对空间走廊和关节分支初值仍可用于求解。
- B1 依据相机或已建立接触后的估计箱位生成预接触位。从关节实测和当前导轨位置计算 TCP，查验 IK 目标与场景障碍，再调用现有 RRTConnect。接触前最后一段沿目标面的法向低速搜索腕力。
- 主要撤离动作根据腕力卸载及实测 TCP、箱面之间的估计间隙结束。每次执行前检查下一小段，行程上限只限定搜索范围。完整就位的仿真审计仍使用独立 PhysX 承托和位置记录。
- 从空托盘开始的入口先供料并处理第一层后排，再衔接现有前排与上层流程。逐箱的目标对象、执行臂和切换条件见[15 箱运行线](long_skill_short_state_routes.md)。

本轮保留了关节归位姿势、若干临时高位避障走廊、计算安全范围所需的静态场景标定，以及部分既有 PhysX 互锁。它们仍需要结合新技能逐步迁移和实跑。静态编译只检查新入口的装配与断点识别，不能证明运动可达和接触任务成功。

## 单实例试验进度

15 个箱位及其断点入口均通过静态装配检查。已在 Isaac Sim 运行 R1 的上料、搬运、落地、观测重定位和推送，以及 U5 从落箱断点开始的接近、直推和撤离。下面的数据来自不同次运行。

### R1 后排第 1 箱

| 环节 | 记录 |
| --- | --- |
| A 段落稳后撤离 | S0 在累计指令撤离 3.325 mm 时触发。此时两杯估计间隙为 3.110 mm 与 3.139 mm。PhysX 对照显示两杯均已分离，地面承载约 12.26 N。试验搜索上限为 12 mm。 |
| B1 转入接触控制 | 首次轨迹按指令结束时，实测 TCP 仍距预接触位约 18 mm。修正后等待实测到位再切换。后墙前断点续试中，左右两臂分别在误差 0.894 mm、0.976 mm 时进入腕力贴面。两段 RRTConnect 共检查 2,126 个轨迹采样点。 |
| B2 后墙承载 | 旧腕力门槛在 6.07 N 时触发，PhysX 后墙承载仍为 0 N。加入编码器位移和腕力增量后，在第 6,780 步确认，腕力 7.32 N，PhysX 后墙法向力 4.39 N。首次持续超过 0.1 N 的 PhysX 承载在此之前约 0.33 s。右臂进场时后墙承载多次降为零，箱体仍靠近后墙。 |
| B2 横推 | 原横推走了 0.26 mm，右腕合力到 9.19 N 时触及 9 N 上限。该步右腕竖直反力 3.91 N，地面支撑增至 16.18 N。同一断点仅加入竖向顺应后，箱横移 2.36 mm，右腕合力峰值 9.21 N，仍触及上限。两次 PhysX 均未记录意外碰撞。 |

运行证据保存了识别事件、同一物理步的 PhysX 对照、失败点和实际测得的恢复断点。箱位的通关以本分支的实跑结果为准。 [R1 运行证据](validation/observed/rear1_progress.json)

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

复现上层第 5 箱时直接恢复它的 landing 断点。

```bash
"$sim/python.sh" "$scene/markerless_rows.py" /tmp/fr3-observed-upper5 \
  --observed-skills --resume "$scene/validation/markerless/upper5_landing_checkpoint.json" \
  --close-on-finish
```
