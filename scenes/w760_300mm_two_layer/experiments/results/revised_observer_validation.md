# 创新点二 · 四类误判修正后的回放验收

本次按 S1、S2/S3、S0、S5/S6 顺序修改观察器。旧模式保持不变，新增模式通过 SevenObserver(revised=True) 启用。TCP 指令由原始关节目标按同一时刻的 FK 换算到世界坐标，和编码器 TCP 位姿、腕力、D455 观测一起回放。

## 1. S1 正常推压

确认贴靠前，先检查 0.5 秒内的指令位移、编码器位移和法向力增量。TCP 能跟随指令时不确认贴靠。两个独立第 5 箱片段中，S1 由 3218/4411、2954/4399 提升到 4407/4411、4396/4399。原来把 S1 误报为墙或邻箱的 1304 个边界判断全部消失。后墙确认从提前 608 步、830 步改为实际接触后 26 步、39 步。

## 2. S2 刚吸住与 S3 起落交接

S2 在已确认双侧吸附且相向时，用 TCP 面心复核支撑高度。A 片段的 S2 从 57/97 提升到 88/97。S3 在 S4 快捷判断之前检查支撑交接。起吊和落箱交接从 28/53 提升到 48/53，悬空搬运 S4 仍为 9331/9331。

## 3. S0 工具分离

工具已经离开箱面和工具已经退到安全停车位分开记录。新版用有符号箱面间隙、连续低腕力和观测稳定性判断前者，保留 25 mm 作为后者。A 的 S0 从 6770/10346 提升到 7296/10346。两个第 5 箱片段从 2119/4622、2153/4615 提升到 3001/4622、3463/4615。已知非 S0 状态没有新增 S0 误报。

## 4. S5 与 S6

后墙和邻箱分别确认。第 5 箱两个片段的 S1 阶段墙或邻箱误报降为零，后墙目标确认从提前 5–7 秒改为真实接触后约 0.22 秒、0.33 秒。逐步 S5 标签仍受接触间歇影响，两个片段为 0/19、1/32，说明目标确认和逐步主状态仍需分开评价。上层第 4 箱三个独立换臂片段的 S6 识别为 0/187、95/185、101/179。后两条识别到侧向反力或力矩，第一条没有建立可复用的侧向见证，因此仍然漏检。

## 复算

从仓库根目录执行以下命令。

    e=scenes/w760_300mm_two_layer/experiments
    python3 "$e/seven_commands.py" RUN_DIR paired.jsonl.gz command_paired.jsonl.gz
    python3 "$e/seven_replay.py" command_paired.jsonl.gz output --revised
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 python3 -m pytest -q tests

完整计数在 [revised_observer_validation.json](revised_observer_validation.json)。原有七状态评分在 [seven/README.md](seven/README.md)。
