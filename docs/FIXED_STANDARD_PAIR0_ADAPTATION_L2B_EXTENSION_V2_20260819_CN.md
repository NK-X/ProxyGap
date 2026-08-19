# 固定标准场景 PAIR0 适配 L2b V2：审计修正版一次性协议

日期：2026-08-19（Europe/London）<br>
状态：**已被 V3 取代，不得执行或解释**。V2 formal `attempt_0` 因审计发现 gate/contract validation mismatch 而中断；本文仅保留为历史协议记录。

## 1. 历史说明：V2 已失效，V3 是唯一可执行 L2b 协议

当前唯一可执行配置为 `fixed_standard_pair0_adaptation_l2b_extension_v3_20260819`。V2 `attempt_0` 的部分 `DEFAULT_CONTINUE` 权重与指标不得复用；精确中断证据见该 attempt 根目录内的 `INVALID_PROTOCOL_MISMATCH_DO_NOT_INTERPRET.json`。

V2 配置 `fixed_standard_pair0_adaptation_l2b_extension_v2_20260819` 取代全部 V1 smoke 与 development/formal 尝试。V1 artifacts 必须原地保留以供审计，但不得再用于科学解释、checkpoint 来源、择优、续训或 V2 指标合并。

| V1 范围 | 已观察事实 | 当前状态 |
|---|---|---|
| 全部 V1 smoke | 发生在 final-gate non-finite 检查及完整传递式 runtime dependency freeze 之前 | `superseded_after_audit_no_scientific_result`；只保留为历史工程证据 |
| V1 formal `attempt_0` | 仅启动 `DEFAULT_CONTINUE`；`PAIR0_ADAPT` 未启动 | 已中断且 superseded；没有 matched comparison |
| V1 formal 的已完成 monitor 行 | 24 个 episode，每个 600 步 | 至少 `24 × 600 = 14,400` 个已观察 training transitions；这是 monitor 下界，不是完整 PPO 训练步数 |
| V1 formal checkpoint | 0 个 ZIP；training record 0 个 | 没有可评价、可复用或可解释的 checkpoint |

因此，V1 formal 不是“在训练前中断”，而是“在第一个 `DEFAULT_CONTINUE` 训练 chunk 内、首个 checkpoint 之前中断”。这 24 个 episode 既没有配对条件，也没有冻结 endpoint，不得计算条件差、趋势或成功/失败结论。V2 明确不复用 V1 的 weights 或 metrics。

旧中断记录中曾写入 V1 `attempt_1` 的历史修复设想；后续 supersession 决定已经取代该设想。**不得启动 V1 `attempt_1`。** V2 使用新的短根目录和自己的 canonical `attempt_0`，不是 V1 retry。

## 2. V2 没有改变科学阈值、seed 或预算

V2 是 provenance 与 fail-closed 语义的审计修复，不是根据 V1 结果进行的调参。V1 没有产生可解释科学结果；V2 保持原预声明的训练预算、随机流、checkpoint 集、评估 seeds 和最终阈值不变。

两组仍从各自冻结的 L2 endpoint（绝对 timestep `2,662,400`）出发：

| 条件 | 显式足端 pair 数 | checkpoint SHA-256 |
|---|---:|---|
| `DEFAULT_CONTINUE` | 0 | `6549c279ca5795636d3b1d6f61c36782f4f843a32107276adf0630c39871cb6f` |
| `PAIR0_ADAPT` | 4 | `9eb1268352aeb90024f681b70ca3b42cb036f6e5ea882e56dbb85262bd8c500e` |

训练仍只有一个新的 master seed `62806`。四个并行场景 worker 的有效随机流依次为 `62806`（flat）、`62807`（uphill 8°）、`62808`（downhill 8°）和 `62809`（bowl exit）。这四个 worker seed 属于同一个 vectorised training run，不能作为四个独立训练重复。

每个条件只追加 `65,536` timesteps，并在新增 `16,384`、`32,768`、`49,152` 和 `65,536` 步处保存 checkpoint；对应绝对 timesteps 为 `2,678,784`、`2,695,168`、`2,711,552` 和 `2,727,936`。前三个 checkpoint 只允许 safety audit，只有最后一个可进入最终科学 gate。performance futility、intermediate selection 和 intermediate promotion 均保持禁用。

## 3. 135D observation 已包含 13D local-terrain preview

现有 135D source observation 已包含 13D local-terrain preview。本轮不得把 downhill 或 terrain-feature utilisation 问题解释为局部地形输入缺失，也不得把同一 13D 信息再次包装后声称为全新 terrain observation。

V2 仍冻结 135D observation、8D action、PPO 参数、reward、energy formula、friction、机器人结构与 20 Hz 控制频率。Relative Mission Energy 继续仅作 measurement-only 诊断，reward weight 为 `0`；任何 non-finite energy component 都是运行或最终 gate 的 fail-closed 失败。

## 4. 完整 19 文件 runtime dependency snapshot

只冻结 V2 wrapper 或单独复制 imported L2 runner 都不足以建立 runtime provenance。V2 在每个 attempt 内创建 `runtime_snapshot/`，按仓库相对路径复制并哈希核验完整的 19 文件传递式依赖闭包：

1. `scripts/run_fixed_standard_pair0_adaptation_l2b_extension.py`
2. `scripts/run_fixed_standard_pair0_adaptation_l2_pilot.py`
3. `scripts/evaluate_fixed_standard_distal_margin0_paired.py`
4. `scripts/evaluate_local_preview_final_paired_direct_goal.py`
5. `scripts/run_fixed_goal_support_priority_pilot.py`
6. `scripts/run_fixed_standard_support_curriculum.py`
7. `scripts/run_fixed_goal_terrain_training.py`
8. `scripts/run_curved_gait_training.py`
9. `src/proxygap/__init__.py`
10. `src/proxygap/ant_wrapper.py`
11. `src/proxygap/curved_gait.py`
12. `src/proxygap/fixed_goal_terrain.py`
13. `src/proxygap/metrics.py`
14. `src/proxygap/planar_transition.py`
15. `src/proxygap/experiment.py`
16. `src/proxygap/divergence.py`
17. `src/proxygap/protocol.py`
18. `src/proxygap/selection.py`
19. `src/proxygap/two_experiment_protocol.py`

精确路径次序与 SHA-256 map 以 V2 config 的 `runtime_dependency_contract.exact_relative_path_sha256` 为唯一合同。V2 wrapper `scripts/run_fixed_standard_pair0_adaptation_l2b_extension.py` 的冻结 SHA-256 为 `1a57cdbd3b69520438442c07225f9565f67d63514279825b7aa8002d916900da`；实际 imported L2 runner `scripts/run_fixed_standard_pair0_adaptation_l2_pilot.py` 的冻结 SHA-256 为 `1c426d7a78cd73bd7e9448e2ecd7f6ab5688871894281d607ca61c61fdd7e7dd`。

parent process 必须记录全体 workers 前后 live 与 snapshot map，并在每个 condition worker 前后分别复核。每个 worker 还必须在创建 environment/model 之前及训练之后记录 live/snapshot map。路径缺失、额外路径、顺序改变、哈希不符或 snapshot 变化均必须 fail closed。manifest 的 `runtime_dependency_closure` 和每个 training record 的 `runtime_dependency_verification` 保存这些证据。

## 5. 最终 gate 对 non-finite 与 energy 数据 fail closed

最终评估仍分为：

- held-out：seeds `83801`–`83805`；
- continuity：seeds `82801`–`82803`。

两套评估都要求 `DEFAULT_CONTINUE` 与 `PAIR0_ADAPT` 两个 condition 完整存在。无论问题出现在 held-out 或 continuity，也无论出现在任一 condition，只要 `nonfinite_episode_count > 0` 或 `energy_components_finite=false`，对应 gate 就必须返回 `evaluable=false`、`passed=false`。force-qualified 主 denominator 为零同样不可评价并 fail closed。

只有 held-out 与 continuity 的全部预声明检查同时通过，overall gate 才能记录 diagnostic pass。即使通过，也不授权 fixed-map evaluation、video rendering、promotion 或替换 incumbent。

## 6. canonical attempt_0 only

V2 的 `maximum_protocol_retry_index=0`，所以 smoke 与 formal 的有效协议编号都只有 `attempt_0`：

- canonical smoke root：`artifacts/smoke/pair0_l2b_v2_20260819/attempt_0`；
- canonical formal root：`artifacts/dev/pair0_l2b_v2_20260819/attempt_0`。

正式 V2 禁止 custom output root；不得用另一个目录伪装 retry 或绕过 once-only 语义。任何 canonical attempt root 已存在时，runner 必须拒绝覆盖。V2 smoke 通过只表示工程合同通过，不应用科学 gate；formal 若安全提前停止，则运行记录可以完整收口，但科学 gate 必须是 `evaluable=false`、`passed=false`，且不得 dereference 缺失的 final aggregates。

## 7. 失败后的预声明去向

V2 无论因安全停止、non-finite、energy、零 denominator 或任一最终阈值失败，都硬停止进一步 contact-budget extension。下一干预不是“首次增加 terrain preview”，而是保留现有 13D preview，另立预声明结构实验，重新设计或强化 terrain-feature utilisation 与 terrain-normal/downhill controller；reward 在该新实验初始阶段保持不变。

该下一干预需要独立 config、消融、测试与 artifact root。本 V2 协议不授权自动启动，也不允许用它解释尚未产生的 L2b V2 结果。

## 8. 可审计入口

- V2 config：`configs/fixed_standard_pair0_adaptation_l2b_extension_v2_20260819.json`
- runner：`scripts/run_fixed_standard_pair0_adaptation_l2b_extension.py`
- tests：`tests/test_fixed_standard_pair0_adaptation_l2b_extension.py`
- V2 runtime snapshot：每个 canonical attempt 下的 `runtime_snapshot/`
- V1 smoke supersession：`artifacts/smoke/pair0_l2b_v1_20260819/SUPERSEDED_AFTER_AUDIT_NO_SCIENTIFIC_RESULT.json`
- V1 development supersession：`artifacts/dev/pair0_l2b_v1_20260819/SUPERSEDED_AFTER_AUDIT_NO_SCIENTIFIC_RESULT.json`
- V1 interruption record：`artifacts/dev/pair0_l2b_v1_20260819/attempt_0/INTERRUPTED_DURING_FIRST_DEFAULT_TRAINING_CHUNK_NO_CHECKPOINT_DO_NOT_INTERPRET.json`（SHA-256 `fb27fea3c8581e95e9be8e73327d0a43f459681c73ac543d8e12e0677208fa4d`）

当前交付状态只支持“V2 protocol/engineering verification ready”。在 canonical V2 formal `attempt_0` 完成、所有预声明 artifacts 与 hashes 核验通过之前，不存在 L2b V2 科学结果。
