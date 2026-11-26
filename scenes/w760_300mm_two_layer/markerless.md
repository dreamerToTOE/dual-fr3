# 76 cm 双层 15 箱码垛场景

**视觉基线为视觉拟合法。** 单台后墙 D455 仿真相机以 15 Hz 采集 1280 × 960 的 RGB-D 图像。运行器根据已知的 150 mm 箱体边长和支撑面高度拟合箱顶边缘，估计箱心、偏航角和插入槽位。场景中的箱体无 ArUco 标记。相邻箱体同高且贴紧时，沿用原场景配色辅助分辨边界。已定位的邻箱被遮挡时，使用最近一次有效测量。相机流在运行日志中标记为 `rear_rgbd_square`。

托盘内宽 760 mm，墙高 300 mm。流程加载已就位的第一层后排五箱，依次完成第一层前排和第二层靠后墙各五箱。搬运使用侧吸，推入时使用腕力和运动学反馈。PhysX 记录对照位姿，并核验仿真碰撞、接触承载和稳定性。

## 可视化运行

使用 Isaac Sim 4.5.0。将 `ISAAC_SIM_DIR` 指向安装目录，从仓库根目录选一条命令运行，每次使用新的输出目录。

```bash
# 加载已码好的第一层后排，从前排第 1 箱上料开始
"$ISAAC_SIM_DIR/python.sh" scenes/w760_300mm_two_layer/markerless_rows.py /tmp/fr3-fitting-full

# 加载实测的第一层两排，从第二层第 1 箱上料开始
"$ISAAC_SIM_DIR/python.sh" scenes/w760_300mm_two_layer/markerless_rows.py /tmp/fr3-fitting-upper \
  --resume scenes/w760_300mm_two_layer/validation/markerless/lower_two_rows_checkpoint.json

# 从第二层第 5 箱的实测 landing 复验拟合对准与直推
"$ISAAC_SIM_DIR/python.sh" scenes/w760_300mm_two_layer/markerless_rows.py /tmp/fr3-fitting-upper5 \
  --resume scenes/w760_300mm_two_layer/validation/markerless/upper5_landing_checkpoint.json
```

## 分段验收

单个 Isaac Sim 实例逐段续跑，覆盖第一层前排和第二层的全部十箱。十箱放置和撤离后均稳定承重 60 个物理步，最终相机与 PhysX 对照的箱心差最大为 1.42 mm。第一层后排五箱以已就位的实测布置加载。整段一次无中断运行仍待验证。 [逐箱验收](validation/markerless/acceptance_summary.json) · [最终 15 箱位姿](validation/markerless/final_15_box_poses.json)

前排和上层的第 5 箱分别沿 +Y 直推 191.74 mm 和 86.64 mm，记录到的两侧邻箱接触冲量均为零。 [前排直推数据](validation/markerless/front5_straight_audit.json) · [上层直推数据](validation/markerless/upper5_straight_audit.json) · [上层最终相机画面](validation/markerless/upper_cube_final.png)

另对三个静态布置的 17 个可见箱位进行拟合，水平箱心与 PhysX 对照的最大差值为 2.65 mm。 [静态拟合对照](validation/markerless/static_fit_depth_priority.json)
