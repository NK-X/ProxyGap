# PAIR0 最终续训 V4：一次性短路径工程修复

V2/V3 formal 在训练开始前的 `prepare_standard_slope_scenes` 阶段失败。失败原因是 Windows 路径长度边界：目标 XML 的绝对路径为260字符；不是策略、环境、门槛或训练结果。旧 root 永久保留，`FAILURE_RECORD.json` SHA-256 为 `21ccdebc692af2f32ec96a2e33795cd0eac45ea4aac852eface8a02f26709d23`，其中没有 checkpoint ZIP，也没有任何 `training_record.json`，因此没有部分权重可复用。

V4 唯一变化是把 canonical root 缩短为 `artifacts/dev/tb_v4_20260819/a0`。V2 config/runner、V3 smoke gate、PAIR0 source checkpoint、63806训练 seed、8 workers、C0/C1命令曝光、每分支65,536步、最终2,793,472 checkpoint、奖励、摩擦、能耗 measurement-only、held-out seeds、转向门、坡面连续性门与 hard stop 全部保持不变。

正式运行前，V4 对 runtime snapshot、平地训练场景、四个标准坡面场景的两套PAIR0变体、两分支最终checkpoint、转向/坡面评估输出和顶层证据文件进行绝对路径预算检查；所有列举路径必须不超过239字符。旧失败路径的同一相对后缀及尚未生成的 bowl-exit 变体均在检查集合中。

V4 只能运行一次。如果再次失败，不再修复或启动其他结构、奖励、自由度或训练轮次。无论最终 PASS、FAIL 或不可评价，运动优化均 hard stop；之后只按预声明 seed 96131、左右 `0.20 m^-1` 条件制作只读验收视频，视频不参与数值门。
