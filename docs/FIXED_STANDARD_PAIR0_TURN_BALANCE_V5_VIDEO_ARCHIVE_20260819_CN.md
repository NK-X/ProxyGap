# PAIR0 转向平衡 V5：正式结果后的只读视频封存协议

## 冻结结论

V5 正式 C0/C1 训练及 130 个预声明评测 episode 已完成。两条最终分支均通过标准坡面连续性门，但均未通过转向门：

- C0 `C0_STRAIGHT_CONTINUE`：turn FAIL，slope PASS；
- C1 `C1_BALANCED_TURN`：turn FAIL，slope PASS；
- final decision：`both_fail_turning_HOLD_retain_source_PAIR0`；
- Stage B：`HOLD`；fixed-map evaluation 未获授权。

视频只能在上述数值结果、checkpoint 和 gate 全部冻结后制作。视频不参与 gate、checkpoint 或 seed 选择，也不能改变正式结论。

## 四个预声明 episode

唯一允许的 seed 为 `96131`，条件为 `curve_left_020` 与 `curve_right_020`，分别重放 C0 和 C1 最终 checkpoint，共四段：

1. C0 / `curve_left_020` / seed 96131；
2. C0 / `curve_right_020` / seed 96131；
3. C1 / `curve_left_020` / seed 96131；
4. C1 / `curve_right_020` / seed 96131。

每段必须完整运行 600 个 control steps、3,000 个 physics substeps，使用 deterministic policy。重放产生的完整 episode row 必须与正式 `turn_episode_metrics.csv` 保持相同字段顺序并逐字段精确相等；任何一个字段不等即 fail closed。

## 视觉与 QA 合同

每段视频为 1,280 × 720、20 fps、每个 control step 渲染一帧的双视角合成：左侧为机器人跟随视角，右侧为固定轨迹总览。所有帧必须明确显示：

`TURN GATE: FAIL | SLOPE CONTINUITY: PASS | FIXED-MAP: NOT AUTHORISED`

每个 episode 保存视频、600 行 trace、重放 metrics、逐字段 comparison、final frame、四时点 contact sheet 和 episode manifest。根目录另保存四段 final frame 的总 contact sheet、冻结 config、renderer snapshot、报告及 manifest。

每个 MP4 必须逐帧完整解码，并验证帧数、分辨率和帧率。只读取 checkpoint；渲染前后必须再次核验 checkpoint、正式输入和 runtime hashes。

## Once-only 与失败边界

Canonical root：`artifacts/dev/tb_v5_video_archive_20260819/a0`。

该 root 必须在运行前不存在。一旦创建，无论成功或失败均不得覆盖或在同一 root 重试。创建后的任何异常必须写入 `FAILURE_RECORD.json`，并保留 `training_performed=false`、`checkpoint_write_performed=false` 和 `retry_same_root_permitted=false`。

## 证据边界

该封存仅提供四个既定 flat-scene episode 的定性可视化和精确重放证据。它不支持 fixed-map readiness、路线或能耗最优、随机地图泛化、自然步态或电池能耗改善等结论。坡面 PASS 来自已冻结的标准坡面数值矩阵，不由这些 flat-turn 视频重新测试。
