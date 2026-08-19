# 固定标准场景 PAIR0 适配 L2b V3：Gate-only 审计修复协议

日期：2026-08-19（Europe/London）<br>
状态：协议冻结候选；本文不是训练结果报告，独立审计通过前不得启动 V3 smoke 或 formal。

## 1. 为什么需要 V3

V3 只修复 gate 与合同校验，不改变科学设计。V2 formal `attempt_0` 启动后，第二次独立审计发现两项不一致：

1. continuity gate 只检查 `PAIR0_ADAPT` 的 force-qualified primary denominator，而协议要求任一 condition 的 denominator 为零均不可评价；
2. 若干会影响动力学、评估或 gate 的配置叶字段未被精确冻结，另存配置可能放宽合同。

V2 因此立即停止并标为 `INVALID_PROTOCOL_MISMATCH_DO_NOT_INTERPRET`。V2 部分运行的静态证据为：仅 `DEFAULT_CONTINUE` 启动；完成 76 个 600-step monitor episode，即至少 45,600 个已完成 monitor transitions；生成 `2,678,784` 与 `2,695,168` 两个 checkpoint 并完成对应中期评估；第三训练 chunk 已开始但无第三 checkpoint；`PAIR0_ADAPT` 未启动；无 training record、final evaluation 或 manifest。上述任何权重或指标均不得解释、续训或并入 V3。

## 2. 科学合同完全不变

V3 从两套原始 L2 endpoint `checkpoint_2662400` 重新开始，不复用 V1/V2 状态：

| 项目 | 冻结值 |
|---|---|
| 条件顺序 | `DEFAULT_CONTINUE`，然后 `PAIR0_ADAPT` |
| source SHA-256 | DEFAULT `6549c279…71cb6f`；PAIR0 `9eb12683…c500e` |
| 训练预算 | 每条件额外 65,536 timesteps |
| checkpoint | `2,678,784`、`2,695,168`、`2,711,552`、`2,727,936` |
| master / worker seeds | `62806`；四场景依次 `62806–62809` |
| 场景 | flat、uphill 8°、downhill 8°、bowl exit |
| 巡航速度 | 0.55 m/s |
| observation / action | 135D（已含 13D local-terrain preview）/ 8D |
| reward / PPO / control | 与 L2 完全一致；20 Hz；每 worker Torch threads = 2 |
| 摩擦 | geom `[1.0, 0.5, 0.5]`；PAIR0 explicit pair `[1.0, 1.0, 0.5, 0.5, 0.5]` |
| 能耗 | 仅测量；reward weight = 0；公式不变；不作电池电能声称 |

中期 seeds `82801–82803` 只用于灾难审计，不允许 performance futility early stop、checkpoint 选择或中期晋级。只有 non-finite/contract mismatch、fall、torso ground、持续 non-foot contact、PAIR0 full-zero > 0.08、force-qualified sustained slip > 0.02 或 events > 0.20 才可早停。

## 3. 唯一最终判定

只能使用固定最终 checkpoint `2,727,936`，并且以下两套 gate 必须同时通过：

- held-out `83801–83805`：执行预声明的 PAIR0-versus-DEFAULT progress、support、zero-foot、slip 与安全检查；
- continuity `82801–82803`：PAIR0 overall ≥ 7.19339076746881 m、uphill ≥ 6.18579923623122 m、downhill ≥ 8.81135708033187 m，同时满足接触与安全上限。

两套集合中的 `DEFAULT_CONTINUE` 与 `PAIR0_ADAPT` 都必须满足：`nonfinite_episode_count = 0`、`energy_components_finite = true`，且 force-qualified primary denominator 可评价。任一 condition 缺失、值非有限、能耗分量非有限或 denominator 为零，整个 final gate 均 fail closed。

primary denominator 固定为“至少一个 distal foot 接触且法向力 ≥ 1 N 的 physics substeps”；any-contact denominator 只作 secondary 诊断。

## 4. 配置与运行时 provenance

V3 对 14 个非 runtime 科学 section 的 230 个叶字段保存 canonical JSON SHA-256；测试逐叶修改并要求全部 fail closed。`runtime_dependency_contract` 的 24 个叶字段也逐一测试。

运行时闭包固定为 19 个文件。为避免 wrapper 自哈希循环，合同摘要先把 wrapper self-entry 规范化为 `<RUNNER_SELF_SHA256>`，其硬编码 canonical SHA-256 为 `ed499890cc12a63016c48f0ab1abf71fce8315b44b8f47a0b8252d2c24cf490a`；wrapper 实际哈希再由配置、live 文件和 attempt-local snapshot 三方核对。snapshot 必须恰好包含这 19 个文件，缺失、增加或哈希变化均 fail closed。

parent 与每个 clean condition worker 都在环境/模型建立前及训练后核对 live 与 snapshot。manifest 必须保存所有核对映射、独立 PID、实际 Torch threads、完整 artifact inventory（不含 manifest 自身）及 SHA-256。

## 5. 一次性执行边界

- smoke canonical root：`artifacts/smoke/pair0_l2b_v3_20260819/attempt_0`；
- formal canonical root：`artifacts/dev/pair0_l2b_v3_20260819/attempt_0`；
- 最大 retry index = 0；已存在 attempt root 必须拒绝覆盖；
- formal parent 只能读取 canonical V3 config 路径；condition worker 只能使用 parent 复制的 frozen config；
- 不评估 fixed map，不渲染视频，不晋级 candidate；
- 无论 gate 通过或失败，本次之后都 hard-stop contact-budget extension。

若 final gate 失败，下一项工作不是继续加训练预算，也不是声称缺少地形预瞄。135D observation 已含 13D preview；应另立预声明实验，保留 reward 不变，重新设计或强化 terrain-feature utilisation 及 terrain-normal/downhill controller。

## 6. 当前门槛

只有以下条件全部满足后才可运行 V3 smoke：

1. runner、config、tests 与 V2 invalid record 的最终 SHA-256 已回填；
2. validate-only 通过；
3. exhaustive leaf-mutation、denominator、snapshot membership、canonical path 与 early-stop 测试全部通过；
4. 独立只读审计给出 GO。

smoke 只证明工程合同可执行，不构成科学结果。只有 smoke 通过并完成独立 artifact inventory/provenance 复核后，才允许启动唯一 formal `attempt_0`。
