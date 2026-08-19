# PAIR0 左右转向平衡 V5 正式结果（2026-08-19）

## 1. 结论先行

本轮是封存前最后一次运动优化实验。`C0_STRAIGHT_CONTINUE` 与 `C1_BALANCED_TURN` 均从同一冻结 PAIR0 source checkpoint 开始，各续训 65,536 步，并且只评价预声明的最终 checkpoint（绝对 timestep 2,793,472）。两条分支的标准坡面门均通过，但转向跟踪门均失败，因此组合门均失败。正式决策为：

`both_fail_turning_HOLD_retain_source_PAIR0`

由此，C1 没有被晋级；原 PAIR0 source 仍仅作为当前已知最佳坡面候选保留。按照实验前冻结的条件规则，C1 转向门失败后不得进入 A*／waypoint 完整地图 Stage B。本轮 hard stop 已生效，不再继续模型优化。

## 2. 实验合同

- Source checkpoint：`checkpoint_2727936.zip`，SHA-256 `5121abeff92859205e1537f123f0df1e97edb5ea1fa80be1a72959a5931fac1c`。
- 分支：C0 仅延续直行命令；C1 对 `0`、`±0.10`、`±0.20`、`±0.35 m⁻¹` 曲率命令进行严格左右平衡曝光。
- 训练：每分支 8 个并行环境、master seed `63806`、额外 65,536 步；两分支从 source 独立加载，未复用对方权重。
- 评测：每分支 45 个转向 episode（9 条件 × 5 held-out seeds）和 20 个标准坡面 episode（4 场景 × 5 seeds），共 130 个 episode。
- 选择：不保存、评价或选择中间 checkpoint；只评价最终预算 checkpoint。
- 冻结项：135D observation、8D action、PAIR0 四个显式足端接触 pair、摩擦、奖励、能耗公式、终止语义、滑动门控和评价阈值。
- 能耗：只记录动作平方、力矩时间积分及机械功代理；不进入奖励、checkpoint 选择或 gate，不能解释为电池能耗。

V5 只修复 V4 训练前工程门中的字段接口错误：旧代码读取并不存在的 `audit["passed"]`；V5 改为对完整编译接触合同逐字段验证。V4 与 V5 smoke 的 24 个 XML／NPY 科学资产逐文件哈希一致，因此本轮没有借修复工程门改变场景或接触物理。

## 3. 正式结果

| 指标 | C0 直行续训 | C1 左右平衡续训 | 冻结门槛／解释 |
|---|---:|---:|---|
| 转向门 | FAIL | FAIL | 组合门必须同时通过转向与坡面 |
| 坡面门 | PASS | PASS | 两分支标准坡面均保留 |
| 直行累计 yaw 均值 | −0.5443 rad | −0.6581 rad | `|yaw| ≤ 0.5 rad`；两者均失败 |
| 左 `0.10 m⁻¹` yaw ratio | −0.2322（0/5 同号） | −0.2880（0/5 同号） | 目标区间 `0.7–1.3` 且 5/5 同号 |
| 右 `0.10 m⁻¹` yaw ratio | 1.1336（5/5） | 1.2055（5/5） | 两者通过该单项 |
| 左 `0.20 m⁻¹` yaw ratio | 0.2969（5/5） | 0.1709（5/5） | 低于 `0.7` |
| 右 `0.20 m⁻¹` yaw ratio | 0.8995（5/5） | 0.6584（5/5） | C1 低于 `0.7` |
| 左 `0.35 m⁻¹` yaw ratio | 0.1329（3/5） | −0.0392（1/5） | 要求绝对 ratio `≥0.5` 且 5/5 同号 |
| 右 `0.35 m⁻¹` yaw ratio | 0.3147（4/5） | 0.2862（5/5） | 绝对 ratio 均低于 `0.5` |
| 坡面平均最佳推进 | 10.4940 m | 8.7508 m | 最低 7.1934 m |
| 上坡 8° 平均最佳推进 | 8.7415 m | 8.0388 m | 最低 6.1858 m |
| 下坡 8° 平均最佳推进 | 11.6383 m | 9.9931 m | 最低 8.8114 m |
| 坡面完整控制区间 zero-foot | 3.8667% | 3.3250% | 上限 5.8056% |
| 坡面 fall／torso／持续非足端／持续滑动事件 | 0／0／0／0 | 0／0／0／0 | 安全子门通过 |

C1 在支撑与坡面安全方面仍合格，但没有解决关键的左转响应；在 `+0.10 m⁻¹` 与 `+0.35 m⁻¹` 条件下，累计 yaw 的均值甚至保持负号。该结果支持“对称命令曝光在此单一训练 seed 与固定预算下不足以消除闭环方向偏置”，但不能证明该方法在其他训练 seed、超参数或结构下必然无效。

## 4. 证据与可复现性

- V5 config SHA-256：`5aaf05d346c2b19c9c2714d7ce5ad033f9cd575836be33d8572c47aaa87be908`。
- V5 runner SHA-256：`bb255f76212105ed8ad17f52fa8b955e6d9d0a31358daa9e1d8751838c407980`。
- V5 engineering-smoke manifest SHA-256：`b1f9a794a83cd1ce8e3d898ba76f6a38f57b645dcbc18d0da39e539e64492c53`；工程 smoke 未训练、未写 checkpoint。
- Formal manifest SHA-256：`f107543dae1a19dc972df46fe5472cdaaa469506c6178ede2a9a4f289af2be7b`；138 个 inventory 工件，不含 manifest 本身。
- C0 final checkpoint SHA-256：`b46240e3264af57a4ac2cf500962684348ce819029bcda97df4f9672a225b494`。
- C1 final checkpoint SHA-256：`ce258e289fe3a782b1d5e582240508a94e1c468f72357cce65dfc2f057a41d14`。
- 完整正式耗时约 462.8 s；两分支各完成 65,536／65,536 步。
- 130／130 个 episode 行完整，运行时 live／snapshot before／after 哈希一致，未产生 failure record。

## 5. 解释边界与最终决策

本实验只使用一个训练 master seed；held-out evaluation seeds 不能替代独立训练 seed。因此结果是一个严格配对、可审计的单训练-seed case study，而不是训练稳健性证明。坡面 gate 通过也不代表自然步态、随机地图泛化、可靠终点到达或时间—能耗最优。

本轮没有启动完整地图 Stage B。这不是遗漏：Stage B 的预声明前置条件明确要求 C1 同时通过转向与坡面门；C1 转向门失败，所以 `fixed_map_authorised=false`。任何绕过该规则而运行地图、回退 checkpoint 或更换 seed 的做法都会构成结果后选择。

最终封存：保留 source PAIR0 作为坡面候选；C0、C1 final checkpoints 与全部负结果证据均归档，但不晋级；停止后续模型优化，转入 report、presentation 与 GitHub 交付。
