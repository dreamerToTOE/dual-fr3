# A 阶段 S0 观测对照

从下层两排已码好的断点启动上层第 1 箱。三个正常接近试次的初始偏航为 0°、+0.75° 和 −0.75°。每次用一个 Isaac Sim 4.5 实例跑到侧吸闭合后 24 个物理步。各方案回放同一试次的编码器、腕力和带时间戳的相机观测，承托与杯面接触用独立 PhysX 真值评分。

## 先测深度，再测供料相机

在后置 D455 等效针孔相机的同一渲染帧上，取 ArUco 标记内的深度采样拟合平面。三个试次箱心高度绝对误差的中位数从原 RGB 的 26.4–42.9 mm 降到约 0.50 mm，平面位置误差的中位数为 1.43–2.29 mm。供料相机用标定料台平面恢复标记位姿，平面位置误差中位数为 0.64–0.85 mm。供料相机的 Z 坐标由料台标定给定。

| 观测与分离间隙阈值 | S0 正确步／真值步 | S0 识别率 | 小间隙 S0 正确步／真值步 |
|---|---:|---:|---:|
| 原后置 RGB，5 mm | 3484／11817 | 29.5% | 64／3139 |
| 后置 RGB＋深度，5 mm | 6362／11817 | 53.8% | 0／3139 |
| 后置 RGB＋深度，1.5 mm | 8487／11817 | 71.8% | 494／3139 |
| 供料 RGB＋料台平面，5 mm | 8482／11817 | 71.8% | 0／3139 |
| 供料 RGB＋料台平面，1.5 mm | 11005／11817 | 93.1% | 2396／3139 |

小间隙指 PhysX 记录有正间隙且最小值不超过 5 mm 的 S0 步。各组均用同一套编码器和腕力判据，5 mm 和 1.5 mm 是该判据中的法向分离阈值。原 RGB 在小间隙子集识别的 64 步可能来自位姿偏差对几何距离的高估。深度和供料相机修正箱体位置后，5 mm 阈值仍会把小间隙留为待确认。

## 虚拟短距测距对照

在 −0.75° 试次给两个吸盘组各采一条被动射线，相对杯面的量程为 50 mm，距离以 0.5 mm 量化。保持供料相机的 1.5 mm 版本，在有回波且满足分离条件时用测距补充判据。同一试次的 S0 识别率从 3827／4062（94.2%）升至 3892／4062（95.8%），多识别 65 步，小间隙识别由 834／1046 升至 899／1046。短距测距是实验开关，量化射线结果代表理想测距，实际光学产品误差不在本次测量范围。

三个正常试次的 1562 个 PhysX 接触步均未被判成 S0，其中总法向载荷低于 0.3 N 的样本只有 1 步。报告的识别率针对 A 阶段接近到闭合这段，按三个试次的真值步合并。深度是仿真后置相机渲染的对齐深度。

## 低力轻触对照

另做一次 −0.75° 接近，吸盘接触后用腕力控制尝试保持低压。运行中未达到双腕连续 120 步稳压的结束条件，因而在闭合吸盘前停止。有效记录含 2731 个 PhysX 轻触步，总法向载荷低于 0.3 N，最长连续 127 步（1.06 s）。该试次单独评分。

| 观测 | S0 识别率 | 把轻触判成 S0 |
|---|---:|---:|
| 原后置 RGB，5 mm | 19.9% | 53／2731 |
| 后置 RGB＋深度，1.5 mm | 70.9% | 0／2731 |
| 供料 RGB＋料台平面，1.5 mm | 91.2% | 0／2731 |
| 供料 RGB＋虚拟短距测距，1.5 mm | 92.9% | 0／2731 |

现有传感器中，供料相机与编码器、腕力的组合识别率最高。后置深度可用于核对承托高度。加入理想短距测距后，两条同轨对照各提高约 1.6 个百分点，论文主方案可先保留现有传感器。这些结果覆盖当前料台接近和小幅偏航，整体 S0 识别率仍需按全流程评分。

## 复算和采集

保存了 [0°](nominal.jsonl.gz)、[+0.75°](yaw_plus.jsonl.gz)、[−0.75°](yaw_minus.jsonl.gz) 及 [低力轻触](light_touch.jsonl.gz) 的配对数据和 [逐项评分](scores.json)。从仓库根目录运行以下命令可复算相同片段。

```bash
scene=scenes/w760_300mm_two_layer
experiments="$scene/experiments"
data="$experiments/results/s0_optical_approach_20261125"
OPENBLAS_NUM_THREADS=1 python3 "$experiments/s0_optical_score.py" \
  "$data/yaw_minus.jsonl.gz" --observations supply_rgb --separation-margin-mm 1.5
OPENBLAS_NUM_THREADS=1 python3 "$experiments/s0_optical_score.py" \
  "$data/yaw_minus.jsonl.gz" --observations short_range --separation-margin-mm 1.5
```

采集时加载下层两排断点，例如下面的 +0.75° 试次。用 `--s0-short-range` 采集同轨迹的虚拟测距。低力对照增加 `--s0-light-touch`，若运行因稳压超时停止，提取时使用 `--allow-incomplete` 保留到最后一个物理步。数据提取把传感器与真值按物理步和时刻配对。

```bash
out="$(mktemp -d)/s0-approach"
/home/ubuntu2004/isaacsim-4.5.0/python.sh "$scene/two_row_physx.py" "$out" \
  --resume "$scene/validation/lower_two_rows_checkpoint.json" \
  --s0-supply-yaw-deg 0.75 --headless --close-on-finish
python3 "$experiments/seven_dataset.py" "$out" s0 "$out/paired_s0.jsonl.gz"
```
