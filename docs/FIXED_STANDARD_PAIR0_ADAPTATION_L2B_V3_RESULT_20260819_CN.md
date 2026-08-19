# 固定标准场景 PAIR0 适配 L2b V3：正式运行结果

日期：2026-08-19（Europe/London）<br>
状态：`FINAL VERIFIED`；单训练 seed、固定标准场景的 diagnostic gate 通过。该状态不等于随机地图泛化、固定地图到达、路径最优、节能或候选模型晋级。

## 1. 结论

唯一 canonical V3 formal `attempt_0` 完整执行，两条件均从各自原始 L2 `checkpoint_2662400` 加载 optimiser state，各增加 65,536 timesteps，并只在固定最终 checkpoint `2,727,936` 作最终比较。held-out 与 continuity 两套预声明 gate 均可评价且全部通过。

V1 与 V2 的部分运行均未复用。V2 因协议校验不一致中断；V3 重新从两套原始 L2 endpoint 开始。

## 2. 执行与安全审计

| 项目 | DEFAULT_CONTINUE | PAIR0_ADAPT |
|---|---:|---:|
| process ID | 45116 | 50492 |
| Torch threads | 2 | 2 |
| 完成训练预算 | 65,536 | 65,536 |
| checkpoint 数 | 4 | 4 |
| intermediate early stop | 否 | 否 |
| final held-out non-finite episode | 0 | 0 |
| final continuity non-finite episode | 0 | 0 |
| final 两集合 energy components finite | 是 | 是 |
| final 两集合 force-qualified denominator 可评价 | 是 | 是 |
| final fall / torso / sustained non-foot | 0 / 0 / 0 | 0 / 0 / 0 |
| final sustained slip / slip event rate | 0 / 0 | 0 / 0 |

四个中期 checkpoint 只用于灾难审计。下表中的 progress 是诊断量，不用于早停或择优：

| 额外 timesteps | DEFAULT progress (m) | DEFAULT zero-foot | PAIR0 progress (m) | PAIR0 zero-foot |
|---:|---:|---:|---:|---:|
| 16,384 | 7.7966 | 0.23736 | 6.7341 | 0.02750 |
| 32,768 | 7.9497 | 0.23708 | 6.5932 | 0.02792 |
| 49,152 | 8.3662 | 0.24597 | 9.1771 | 0.03306 |
| 65,536 | 8.6664 | 0.24486 | 9.3040 | 0.03361 |

PAIR0 的学习曲线并非单调；最终结论来自预声明 final checkpoint，而不是事后选择 49,152 或其他 checkpoint。

## 3. Held-out gate：seeds 83801–83805

每条件包含 4 场景 × 5 seeds = 20 个 evaluation episodes。

| 指标 | DEFAULT | PAIR0 | 预声明比较 | 结果 |
|---|---:|---:|---:|---|
| mean best progress (m) | 8.9203 | 9.3773 | PAIR0/DEFAULT ≥ 0.90；实测 1.0512 | 通过 |
| uphill progress (m) | 6.7931 | 7.7891 | ratio ≥ 0.85；实测 1.1466 | 通过 |
| downhill progress (m) | 10.3354 | 9.1341 | ratio ≥ 0.85；实测 0.8838 | 通过 |
| pooled zero-foot fraction | 0.24042 | 0.03208 | reduction ≥ 0.10；实测 0.20833 | 通过 |
| mean support count | 0.35335 | 1.34362 | increase ≥ 0.20；实测 0.99027 | 通过 |
| success count | 0 | 2 | PAIR0−DEFAULT ≥ 0 | 通过 |
| PAIR0 worsening vs frozen unadapted PAIR0 | — | 0.00403 | ≤ 0.03 | 通过 |

PAIR0 的 fall、torso-ground、sustained non-foot、corrected sustained slip 与 corrected slip-event rate 检查均通过；held-out gate 共 12/12 checks 为 true。

## 4. Continuity gate：seeds 82801–82803

每条件包含 4 场景 × 3 seeds = 12 个 evaluation episodes。

| PAIR0 指标 | 实测 | 预声明下限或上限 | 结果 |
|---|---:|---:|---|
| overall mean best progress (m) | 9.3040 | ≥ 7.19339076746881 | 通过 |
| uphill mean best progress (m) | 8.3950 | ≥ 6.18579923623122 | 通过 |
| downhill mean best progress (m) | 10.2048 | ≥ 8.81135708033187 | 通过 |
| pooled zero-foot fraction | 0.03361 | ≤ 0.058055555555555555 | 通过 |
| fall / torso / sustained non-foot | 0 / 0 / 0 | 均为 0 | 通过 |
| corrected sustained slip / events | 0 / 0 | ≤ 0.02 / ≤ 0.20 | 通过 |

continuity gate 共 9/9 checks 为 true。两条件在该集合中的 non-finite、energy-finite 与 force-qualified denominator 也均通过 fail-closed 检查。

## 5. 能耗代理：仅作描述，不作 gate

以下均为相同 600-step episodes 的算术平均；它们是相对任务能耗代理或机械量，不是电池电能。

| 集合 | 条件 | squared action | ∫|τ|dt (N·m·s) | positive mechanical work (J) | absolute mechanical work (J) |
|---|---|---:|---:|---:|---:|
| held-out | DEFAULT | 155.1712 | 5,246.0506 | 3,737.5097 | 4,388.8832 |
| held-out | PAIR0 | 171.6734 | 5,600.2279 | 3,749.8500 | 4,333.5913 |
| continuity | DEFAULT | 156.8662 | 5,273.2622 | 3,771.2354 | 4,419.7146 |
| continuity | PAIR0 | 171.9509 | 5,609.3666 | 3,794.5664 | 4,376.1270 |

PAIR0/DEFAULT 比值为：held-out `1.1063 / 1.0675 / 1.0033 / 0.9874`，continuity `1.0962 / 1.0637 / 1.0062 / 0.9901`，顺序对应表中四个代理。因此，PAIR0 明显改善支撑并保留进展，但 squared-action 与 torque-time 代理上升，positive work 基本持平，absolute work 略低。当前证据不支持“PAIR0 更节能”的单一结论，也不能把这些数值解释为电池焦耳。

## 6. Provenance 与边界

- formal artifact root：`artifacts/dev/pair0_l2b_v3_20260819/attempt_0`；
- manifest SHA-256：`d9d6088ad80e152d6c8c10a7cedadd1695ba8fcbfbc0d40734a0029cec4e77ac`；
- frozen config SHA-256：`6ad69370cc868e0fd84c3561cd336fe7540d471dad92129eea3f94a3420ae5da`；
- frozen/live runner SHA-256：`6dfd283aaa7661e3806e9ca26f874a6088ebf0a39985d3ef3bd0c7edb6e493aa`；
- artifact inventory：144/144，零缺失、零额外、零 bytes 或 SHA-256 mismatch；
- runtime snapshot：19/19 文件；20/20 parent/worker dependency maps 一致；
- 8/8 checkpoint ZIP 的 SHA-256、`num_timesteps` 与 optimiser entry 已独立复核；
- 无 `FAILURE_RECORD`；未运行 fixed map、未渲染视频、未晋级 candidate；
- `hard_stop_further_contact_budget_extension = true`。

| checkpoint | DEFAULT SHA-256 | PAIR0 SHA-256 |
|---:|---|---|
| 2,678,784 | `0c6a2c41…ca09bc` | `83f30734…cd6d42` |
| 2,695,168 | `a203aa0b…f4ff9f` | `c1297a08…6309a3` |
| 2,711,552 | `5bf15259…9e07f9` | `092cbf8a…155517` |
| 2,727,936 | `1c084b69…3914c9` | `5121abef…1fac1c` |

完整 64 位 checkpoint SHA-256 保存在 manifest 的 `condition_training.*.checkpoint_records` 及 artifact inventory 中；表内缩写仅用于阅读。最终 gate 只读取每条件最后一行 checkpoint，未选择前三个中间模型。

- DEFAULT final `checkpoint_2727936.zip`：`1c084b69575a4984c28b417d1b73b9d8ad6cfcc93e2e484462a082f8553914c9`；
- PAIR0 final `checkpoint_2727936.zip`：`5121abeff92859205e1537f123f0df1e97edb5ea1fa80be1a72959a5931fac1c`。

本实验只有一对继续训练的策略，evaluation episodes 嵌套于该策略，不能当作独立训练重复。结果支持“在本次固定标准场景与冻结 seeds 下，V3 PAIR0 达到预声明诊断门槛”，不支持可靠到达随机地图终点、泛化、最优路径、自然步态或能耗优化的结论。进入下一阶段必须另立协议，不能继续追加 contact-budget 训练或事后改变本次 gate。
