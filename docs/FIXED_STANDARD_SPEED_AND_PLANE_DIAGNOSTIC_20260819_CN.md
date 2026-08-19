# 固定标准场景速度消融与原生 Plane 对照

日期：2026-08-19<br>
状态：只读诊断完成；不建议启动下一轮速度训练或奖励扫描

## 1. 设计边界

本轮没有训练。135 维 source 与 W0 配对对照分别在 0.20、0.30、0.40、0.55 m/s 下运行，场景为数值平地、8° 上坡和 8° 下坡，seeds 77801–77803，每轮最多 600 个 0.05 s 控制步。机器人仍为同一 8-DOF Ant，摩擦固定 `[1.0, 0.5, 0.5]`、`condim=3`；奖励、能耗公式和 checkpoint 均未改变。W1 不参与本轮，也未晋级；原始 source 始终是 incumbent。

低速非停滞门槛为：平均最佳进度至少 1.5 m，且相对该速度 30 s 指令距离的进度比例至少 0.35。候选还必须不增加摔倒、控制步末端滑动率增量不超过 2 个百分点，并满足“末端四足端无接触率至少下降 5 个百分点”或“平均支撑足数至少增加 0.1”之一。最后还必须在完整 5 个物理子步指标上把零足端接触率降低至少 5 个百分点。

## 2. 速度消融结果

| 速度 | episodes | 摔倒 | 成功 | 平均最佳进度 | 指令距离进度比 | 末端零足端接触 | 平均支撑足数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 m/s | 18 | 2 | 0 | 3.811 m | 0.635 | 68.114% | 0.372 |
| 0.30 m/s | 18 | 1 | 0 | 4.842 m | 0.538 | 68.216% | 0.369 |
| 0.40 m/s | 18 | 1 | 0 | 6.185 m | 0.515 | 67.943% | 0.370 |
| 0.55 m/s | 18 | 0 | 0 | 6.946 m | 0.579 | 69.306% | 0.356 |

三个低速都不属于单纯停滞，但均增加摔倒，且末端无接触改善只有 1.09–1.36 个百分点、支撑足数增加只有 0.012–0.015，远低于门槛。确定性排序选择 0.40 m/s 仅用于进一步审计，不代表推荐。

0.40 m/s 与 0.55 m/s 在 source/W0、平地/上坡、seed 77802 的完整子步配对结果为：完整周期零足端接触率由 27.625% 降至 23.063%，绝对改善 4.5625 个百分点，仍低于预声明的 5 个百分点。因此 endpoint gate 和 full-substep gate 均失败。降低速度不值得进入下一轮训练。

这里的“持续滑动”仍只在控制步末端读取；报告将其解释为 endpoint-sampled sustained slip，而不是物理子步修正后的真实连续滑动。

## 3. 原生 Plane 与数值 Flat-Heightfield 对照

为隔离接触后端，本轮从同一个数值平地 XML 仅把地面碰撞类型由 heightfield 改为原生 plane；机器人编译签名、初始姿态、wrapper 高度输入、checkpoint、速度 0.55 m/s、seeds、horizon、摩擦和命令完全配对。该对照不参与训练和速度排序。

两个 checkpoint、六个配对 episode 的 pooled 结果：

| 指标 | 数值 flat-heightfield | 原生 plane | plane − heightfield |
|---|---:|---:|---:|
| 平均最佳进度 | 5.424 m | 9.078 m | +3.654 m |
| 控制步末端零足端接触率 | 77.944% | 19.611% | −58.333 个百分点 |
| 平均支撑足数 | 0.251 | 1.305 | +1.053 |
| 成功 | 0/6 | 1/6 | +1 |
| endpoint-sampled 持续滑动 | 0.222% | 6.083% | +5.861 个百分点 |

物理子步结果同样显示巨大差异：source 的完整周期零足端接触由 32% 降至 5%，W0 由 33% 降至 3%，pooled 改善 28.375 个百分点。末端零足端接触分别由 78%→24% 和 80%→15%。这些变化超过所有预声明 backend-difference 门槛，并在两个 checkpoint 上方向一致。

原生 plane 下 source 有 1/3 到达终点，但这不等于已经解决坡面或复杂地形问题：plane 没有坡度、曲率和凹坑，也没有证明随机地图泛化。它只说明当前大量支撑失败主要与从原生 plane 迁移到即使近乎完全平坦的 MuJoCo heightfield 有关；继续扫描奖励或降低速度会把接触后端问题误归因为神经网络步态。

plane 的末端滑动指标更高不能直接解释为“更差”：plane 上足端接触时间大幅增加，而旧滑动指标只在有末端接触时才有机会记录。下一步需在每个 0.01 s 子步同时记录每足切向速度后再比较。

## 4. 决策

1. 不晋级 W0 或 W1；继续保留 source checkpoint。
2. 不进行速度课程训练：0.40 m/s 最接近门槛但仍失败，并增加摔倒。
3. 下一轮应先做 heightfield 接触迁移诊断，而不是继续调奖励：检查 heightfield 接触数量、法向、penetration/contact distance、solver 参数和足端 geom 与三角网格尺度；保持机器人、摩擦、命令和奖励不变，一次只改一个接触表示因素。
4. 在 heightfield 能复现 plane 的基本支撑之前，不应把本轮问题归因于关节不足，也不应增加关节或吸附力。

本结果来自两个既有 checkpoint、三个 seeds 和一个近零起伏 heightfield，仅作为强辨别性开发证据，不构成复杂地形总体性能结论。

## 5. 证据入口

- 配置：`configs/fixed_standard_speed_ablation_v1_20260819.json`（SHA-256 `2b4c784efef8910ee445489d7ba49c7f90e1400fbd5ffbdcb070710ecc27c048`）
- 脚本：`scripts/run_fixed_standard_speed_ablation.py`（formal-run SHA-256 `5b2a219fe38e0ea434ee1227b684cff55af93f1ac4eb59bafa4cb05970a81e16`）
- 测试：`tests/test_fixed_standard_speed_ablation.py`
- 正式根目录：`artifacts/dev/fixed_standard_speed_ablation_v1_20260819/seed_matrix_77801`
- 速度摘要：`speed_ablation_summary.json`（SHA-256 `77db48477bba5a62696bb1187ad8baaa2cadb2ae1c477ec75650a7bf405f61d2`）
- plane 对照：`flat_plane_comparator/comparison_summary.json`（SHA-256 `23e6a4bb15d047be0e0c956ce409bad425574a1c5ae289ecf672e071c98ed373`）
- 完整子步摘要：`high_frequency_contact/paired_summary.json`（SHA-256 `ef73735046876e3a12a294dc6b828973cacb248bcb7c7f712c44c36da629aeb6`）
- manifest：`manifest.json`（SHA-256 `90c63bbe761dc2f68fa7646563e3c5e09c2e30ef54267902cc9db0c417db1b34`）
