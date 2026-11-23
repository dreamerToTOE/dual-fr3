# U4 low-friction compliant-pad weak-reload validation

- Trials: 4 independent runs at 0.35, 0.70, 0.70 and 1.00 N.
- Single Isaac Sim process per run, headless, from the upper-4 checkpoint.
- Pad: 1 mm × 120 mm × 120 mm, μs=0.005, μd=0.003, compliant contact k=1500 N/m, damping=15 N/(m/s).

| Target force | Approach displacement | Residual to target | Weak episodes | S6 transition hits | False S6 ticks |
|---:|---:|---:|---:|---:|---:|
| 0.35 N | 0.155 mm | 6.920 mm | 0 | 0 | 0 |
| 0.70 N | -0.769 mm | 6.027 mm | 0 | 0 | 0 |
| 0.70 N | -0.717 mm | 6.065 mm | 0 | 0 | 0 |
| 1.00 N | -0.724 mm | 6.067 mm | 0 | 0 | 0 |

## Conclusion

四次同一低摩擦/合规垫试验均未在解除侧载后形成连续 3 步以上、0.1–1.0 N 的有效侧向 PhysX 接触。Cube_8 最终仍距离目标约 0.72–0.99 mm，因而没有把这些试验计入 S5→S6 的正例召回率。未观察到误报 S6。

原因是弱驱动力仍不足以克服当前上层箱体与下层支撑堆叠的实际约束。低摩擦垫本身已注册并被评分器识别，当前结果说明还需要改变支撑几何或使用更低阻力的独立滑移支撑，不能仅靠继续调观察阈值解决。
