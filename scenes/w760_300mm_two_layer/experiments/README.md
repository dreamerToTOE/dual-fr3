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
