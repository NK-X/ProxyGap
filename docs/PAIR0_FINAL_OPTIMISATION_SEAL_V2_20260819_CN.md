# PAIR0 运动优化最终封存 V2（2026-08-19）

## 1. 最终状态

本项目的运动优化现已 hard stop，不再训练、调参、选择 checkpoint 或追加完整地图实验。

- 保留模型：原 PAIR0 source `checkpoint_2727936.zip`，SHA-256 `5121abeff92859205e1537f123f0df1e97edb5ea1fa80be1a72959a5931fac1c`。
- 最后一轮正式比较：C0 与 C1 各续训 65,536 步；两者均 `slope PASS / turn FAIL`。
- 最终决定：`both_fail_turning_HOLD_retain_source_PAIR0`。
- C0 final checkpoint 与 C1 final checkpoint 均归档但不晋级；它们不是“最终成功模型”。
- A*／waypoint Stage B 的前置条件要求 C1 转向与坡面同时通过。C1 转向失败，因此 `fixed_map_authorised=false`，三次地图测试未启动。
- 能耗只作 measurement-only 机械代理；没有进入奖励或 gate，也不代表电池能耗。

## 2. 为什么 V5 是可评价结果

V2 formal 因 Windows 路径长度在训练前失败；V4 通过短路径修复后，又因 runner 读取不存在的 `audit["passed"]` 字段而把四个实际合格接触场景误判为失败。V5 没有更改科学协议，只把该工程判断改为对完整编译接触合同逐字段验证。

V4 与 V5 engineering smoke 的 24 个 XML／NPY 科学资产哈希逐文件一致。独立 pre-formal 审计确认 P0=0、P1=0 后，唯一 formal run 正常完成。正式根共 139 个文件；manifest 排除自身后的 138／138 项 inventory 在路径、大小和 SHA-256 上均完全一致。两分支各有一个 final checkpoint 和一个 training record，无中间 checkpoint、无 failure record、两个 stderr 均为空。

## 3. 最后一轮科学结果

每分支评价 45 个转向 episode 和 20 个标准坡面 episode，合计 130／130 个 episode。C1 的核心数值如下：

| 指标 | C1 结果 | 冻结门槛 | 判断 |
|---|---:|---:|---|
| 直行累计 yaw | −0.658 rad | `|yaw| ≤ 0.5` | FAIL |
| 左 `0.10 m⁻¹` yaw ratio／同号 | −0.288／0/5 | `0.7–1.3`／5/5 | FAIL |
| 右 `0.10 m⁻¹` yaw ratio／同号 | 1.206／5/5 | `0.7–1.3`／5/5 | PASS |
| 左 `0.20 m⁻¹` yaw ratio | 0.171 | `0.7–1.3` | FAIL |
| 右 `0.20 m⁻¹` yaw ratio | 0.658 | `0.7–1.3` | FAIL |
| 左／右 `0.35 m⁻¹` yaw ratio | −0.039／0.286 | `|ratio| ≥ 0.5` | FAIL／FAIL |
| 坡面 overall／uphill／downhill progress | 8.751／8.039／9.993 m | 7.193／6.186／8.811 m | PASS |
| 坡面 zero-foot | 3.325% | `≤5.806%` | PASS |
| fall／torso／持续非足端／持续滑动事件 | 0／0／0／0 | 全为0 | PASS |

平衡命令曝光在本训练 seed 和冻结预算下没有修复左转弱响应。该结果不能外推为方法在所有 seed、预算或结构下必然无效；它只支持本次预声明干预不晋级。

## 4. 最终视频归档

视频选择在训练前冻结为 evaluation seed `96131`、C0/C1 各自的 `curve_left_020` 和 `curve_right_020`，共四段。视频不参与 gate，不允许结果后换 seed。

| 视频 | SHA-256 |
|---|---|
| C0 left 0.20 | `e696d7800eb8056bd8c65664dbca7d8bc9aa60e4918388cb207258f29ace67e0` |
| C0 right 0.20 | `3585c2e584d7957988c44a9f2b061ad7ccea1c5da396c5d5d37c9452d2fd6cff` |
| C1 left 0.20 | `0bdbec4a0252146d540d24b13492b9431bcd2d09e4cd874b04a0edbd53e7cb59` |
| C1 right 0.20 | `e96d421bb6d17d4960289fbd9204b70fee5b7061667e1239f8aeb6c3600f3c0b` |

每段 640 帧、1280×720、20 fps、32.0 s；全部帧完整解码。每段重放 600 个控制步／3,000 个物理子步，与正式 CSV 的 55 个字段在字段顺序和值上完全一致。视频 archive manifest SHA-256 为 `e25d4ce1f2426c24093211befaf6bff65955a55f595d1d4060cf33ec221d596a`，32／32 项 inventory 复核无差异。画面明确标记 `TURN GATE: FAIL | SLOPE CONTINUITY: PASS | FIXED-MAP: NOT AUTHORISED` 和 `STAGE B: HOLD`。

## 5. 证据索引

| 证据 | SHA-256 | 结论边界 |
|---|---|---|
| PAIR0 source checkpoint | `5121abeff92859205e1537f123f0df1e97edb5ea1fa80be1a72959a5931fac1c` | 当前已知最佳坡面候选，不是转向合格模型 |
| 标准坡度边界 manifest | `fd1995b28b1cbb24c1e371d3a1f6b24833eb52efb0cc1191440eccb3f73360e3` | 上坡连续被测下界12°；下坡结果非单调 |
| 平地转向诊断 manifest | `d0beda1fb2bf3f7bb948b918ff81de81023fd3ea8f229e0e302e66c92ce5f8d6` | 旧 source 闭环左右不对称 |
| V2 formal failure | `21ccdebc692af2f32ec96a2e33795cd0eac45ea4aac852eface8a02f26709d23` | 训练前路径故障，不是策略结果 |
| V4 formal failure | `9695bd3b5d628907053a2f785ec874efa18b2fc47a317c452a88566c0d624812` | 训练前字段接口故障，不是策略结果 |
| V5 formal manifest | `f107543dae1a19dc972df46fe5472cdaaa469506c6178ede2a9a4f289af2be7b` | 130-episode final paired case study，单训练 seed |
| V5 video archive manifest | `e25d4ce1f2426c24093211befaf6bff65955a55f595d1d4060cf33ec221d596a` | 只读视觉复放，不改变 gate |
| post-seal direct-goal map manifest | `b3d37af1a8f57e4bd89fb2e5dbbb6c16bde24bced77610e7fa163651e08f6038` | 单 fresh seed 未到达；没有全局 planner |

## 6. Report 与 Presentation 边界

最终汇报可以声称：PAIR0 在冻结标准场景中显著减少 zero-foot exposure 并保留推进；坡面测试给出了条件性能力边界；最后的左右平衡续训完成且产生了可评价负结果；完整地图 direct-goal episode 未到达。

最终汇报不得声称：自然步态已验证、多训练 seed 稳健性、左右转向已解决、A* 地图导航已执行、随机／未见地图泛化、最优路线、最优时限、最优能耗或真实电池效率。Presentation 应把 V2/V4 工程阻断与 V5 科学负结果分开，并明确 Stage B 因前置门失败而 withheld。

封存后只允许复核、报告、演示和发布，不再进行模型优化。
