# U4 independent sliding-support validation

- Support: original Cube_8–Cube_15/Cube_14 contacts filtered, an independent pose-controlled prismatic rail and visual sled, plus the 1 mm compliant side pad.
- Pad friction: μs=0.005, μd=0.003. Compliant contact: 1500 N/m and 15 N/(m/s).
- Three single-Isaac-Sim trials were run from the same upper-4 checkpoint.

| Target | Result | Weak window | Side force | S6 transition | False S6 |
|---:|---|---:|---:|---:|---:|
| 0.10 N | strong contact before weak onset | none | none | 0 | 0 |
| 0.35 N | valid weak contact | 312 ticks | 0.211–0.920 N (median 0.476) | 0 | 0 |
| 0.70 N | strong contact before weak onset | none | none | 0 | 0 |

## Conclusion

独立低阻力滑移支撑能够形成真实弱接触。0.35 N 试验得到连续 312 步的有效弱回载，侧向真值范围 0.211–0.920 N，中央値 0.476 N。0.1 N 与 0.7 N 在首次接触时先产生强冲击，按当前“先失载、再连续弱载”的冻结判据不计为正例。三次试验均无误报 S6。

0.35 N 的弱接触已证明支撑结构方向正确，但融合观察器本次没有在该弱窗口内切到 S6，说明下一步需要单独处理弱载下的状态确认条件，不应把强接触后的弱振荡当成独立弱回载。
