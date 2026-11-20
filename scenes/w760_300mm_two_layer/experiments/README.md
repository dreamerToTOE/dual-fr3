# 八个技能的切换实验

[技能定义](../packing_skills.md)规定了八个技能和各自的完成判据。这里先研究两个切换位置。

- **入槽检查与侧吸纠偏**使用 D455 观测到的两侧间隙和箱体投影宽度。若当前间隙不够，但转正后预计有足够间隙，从技能 1 进入技能 8。纠偏后通过原入槽检查，回到技能 1。其他类型的间隙不足沿用原流程的拒绝结果。
- **受支撑推移与目标贴靠**用创新点二导出的状态回放，对照 7 N 轴向力阈值。技能 4 的候选状态表示可以开始或继续贴靠确认。退出技能 4 仍需要独立的到位证据。各方法使用相同的 9 N 腕力上限。

```bash
scene=scenes/w760_300mm_two_layer
"$ISAAC_SIM_DIR/python.sh" "$scene/two_row_physx.py" "$(mktemp -d)/upper5" \
  --resume "$scene/validation/upper5_landing_checkpoint.json" \
  --upper5-yaw-deg 1.25 --upper5-z-unload --upper5-realign --close-on-finish

python3 "$scene/experiments/replay_skill_selection.py" \
  /path/to/fusion_skill_boundary_trace.jsonl /path/to/skill_selection_replay.json
```

回放可加 `--entry-run <run_directory>`，从该次运行的 D455 测量重建入槽选择。模块同时记录技能 4 候选和实际目标承载的重合情况。在同一条物理记录上回放只测选择结果。任务完成率需要让选择真实驱动动作后另行统计。

## 入槽纠偏对照

各方法沿用 1 mm 入槽间隙和 9 N 腕力上限。固定流程在入槽检查通过后沿原路线推进，拒绝时停止。间隙阈值状态机在任意一侧间隙不足时选择一次技能 8。任务状态策略只有在预计转正能恢复间隙时选择技能 8。后两者的纠偏动作和复测条件相同。

从同一落地断点恢复，固定偏航扰动和竖向力反馈。运行固定流程时，省略上方 Isaac Sim 命令中的 `--upper5-realign`。为任务状态策略保留该参数。报告既列出真实执行的动作与物理结果，也从同一初始相机观测独立重放三个策略的首个选择。

```bash
python3 "$scene/experiments/compare_entry_policies.py" \
  --run fixed /path/to/fixed_run \
  --run task_aware /path/to/adaptive_run \
  --out /path/to/entry_policy_comparison.json
```

可重复添加 `--run`。早于任务状态策略接入的纠偏实验标为 `historical_realign`。只有实际由间隙阈值状态机控制的运行才用 `gap_fsm` 标签。程序按场景、断点摘要、偏航扰动、以 0.1 mm 为单位四舍五入的初始 X、竖向力反馈和两项共用阈值整理实跑结果。偏航单变量工况可能让两种可纠偏策略选择同一动作，此时结果用于验证纠偏的作用。

纠偏使箱体重新通过入槽检查属于阶段恢复。只有最终稳定就位才计入整段完成。仅回放选择的策略没有可据此评估的物理完成率。

要比较纠偏中的横移方式，在任务状态策略的原命令中设定 `--upper5-yaw-deg=-0.65`，再加 `--upper5-realign-center simultaneous` 或 `--upper5-realign-center sequential`。两者均根据 D455 测得的邻箱轮廓确定槽口中点。前者在 3 s 内同时转正和横移，后者先转正、再横移，每段 3 s，过程中保持同一次侧吸。比较程序分别使用 `--run task_aware_centered <run_directory>` 和 `--run task_aware_sequential <run_directory>`。

[入槽策略对照结果](results/entry_policy_comparison.md)
