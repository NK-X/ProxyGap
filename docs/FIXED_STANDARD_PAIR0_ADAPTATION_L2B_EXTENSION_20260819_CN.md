# 固定标准场景 PAIR0 适配 L2b：一次且仅一次的同预算延长协议

日期：2026-08-19（Europe/London）<br>
状态：协议已冻结、尚未据此产生 L2b 科学结果；本文件不是训练完成报告。

## 1. 唯一问题与证据边界

本轮只回答一个受限问题：在 L2 的冻结终点继续给予两组完全相同的额外训练预算后，`PAIR0_ADAPT` 的标准场景表现是否仍在改善，并且是否同时保持新的 held-out 表现与原 L2 评估分布上的 continuity（连续性）。

这是 **L2 之后一次且仅一次** 的 matched-budget（同预算）诊断延长。它不是 checkpoint 搜索、超参数搜索、正式多 seed 试验或候选晋级。无论结果通过或失败，均不得再增加第二段 contact-budget extension。单次通过也不能证明稳健性、随机地图泛化、转向能力、固定地图可达性、路径最优性或电池能耗改善。

## 2. 冻结起点

两组都从绝对 timestep `2,662,400` 的各自 L2 最终 checkpoint 继续：

| 条件 | 显式足端 pair 数 | 冻结 checkpoint | SHA-256 |
|---|---:|---|---|
| `DEFAULT_CONTINUE` | 0 | `artifacts/dev/fixed_standard_pair0_adaptation_l2_pilot_v1_20260819/attempt_0/default_continue/models/checkpoint_2662400.zip` | `6549c279ca5795636d3b1d6f61c36782f4f843a32107276adf0630c39871cb6f` |
| `PAIR0_ADAPT` | 4 | `artifacts/dev/fixed_standard_pair0_adaptation_l2_pilot_v1_20260819/attempt_0/pair0_adapt/models/checkpoint_2662400.zip` | `9eb1268352aeb90024f681b70ca3b42cb036f6e5ea882e56dbb85262bd8c500e` |

runner 必须在创建输出目录之前逐文件复核上述 SHA-256，并同时复核冻结的 L2 config、manifest、final-gate、标准协议及奖励配置。任一文件缺失、哈希不符、维度不符或合同不符都必须 fail closed，不得以当前磁盘中的近似文件替代。

其中三项直接 L2 provenance 的冻结 SHA-256 为：

- frozen config：`3fde34618a02ce0fb7134f8b852eb5b8ed0b4c72f041b83da70fae47dd931be2`；
- manifest：`89b7075e737b21e7ecda5c54cae12b133f56190efa715180da330004e2578568`；
- prospective final gate：`3713063e9825c3b622ea1e88fca84f9dbe28c04624c415a8a4c3d3e69b39bad2`。

这里的 manifest 哈希是当前冻结文件的实测值；不得使用早期记录中的陈旧哈希。

`DEFAULT_CONTINUE` 与 `PAIR0_ADAPT` 使用各自 checkpoint 中的 optimiser state，`reset_num_timesteps=false`。两组的 PPO 参数、训练预算、场景次序、随机流、评估方式和停止规则相同；唯一保留的接触差异是 `PAIR0_ADAPT` 的四个显式 `floor × distal ankle` pairs。全局 geom margin、摩擦、`condim`、机器人结构、135D observation、8D action 和 20 Hz 控制频率保持冻结。

## 3. 一次且仅一次，以及 single-seed 的准确含义

本轮只有一个新的训练 master seed：`62806`。四个并行场景 worker 按 `scene_order` 顺序映射的有效 seed 固定为：

| 场景次序 | 场景 | 有效 worker seed |
|---:|---|---:|
| 1 | `flat` | 62806 |
| 2 | `uphill_8deg` | 62807 |
| 3 | `downhill_8deg` | 62808 |
| 4 | `bowl_exit` | 62809 |

这四个 worker seed 是同一个 vectorised training run 内由 master seed 派生的场景随机流，**不是四个独立训练 seed，也不能记作 `n=4`**。两组必须使用相同映射，并按 `DEFAULT_CONTINUE`、`PAIR0_ADAPT` 的固定次序，在彼此独立的干净进程中串行执行；不得让两组同时训练，也不得交换次序后挑选较有利结果。

`maximum_protocol_retry_index=1` 只允许在预声明的工程性失败后，以完全相同的 config 和 seed 在新的 `attempt_1` 目录保留式重试。它不构成第二次科学延长、第二个 seed 或可供择优的重复。`attempt_0` 与 `attempt_1` 都存在时，不得选取较好者作为结果；失败记录必须保留并解释。

## 4. 同预算与 checkpoint 冻结

每个条件只追加 `65,536` timesteps，每 `16,384` timesteps 保存一次：

| 新增 timesteps | 绝对 timesteps | 中途用途 |
|---:|---:|---|
| 16,384 | 2,678,784 | safety audit only |
| 32,768 | 2,695,168 | safety audit only |
| 49,152 | 2,711,552 | safety audit only |
| 65,536 | 2,727,936 | 唯一允许进入最终科学 gate 的 checkpoint |

中间 checkpoint 使用 seeds `82801`–`82803` 仅做安全审计；不得应用科学 gate、选择 checkpoint、提前晋级或把中间最优值替代最终值。只有绝对 timestep `2,727,936` 可进入最终判定。

## 5. 提前停止只允许由安全失败触发

`performance_futility_stopping_enabled=false`，因此表现差、进度低、改善缓慢或曲线非单调都不得触发提前停止。允许触发停止的只有预声明安全或合同失败，包括：

- 任一 non-finite 值或编译/运行合同不匹配；
- 任一条件出现摔倒、torso-ground episode 或持续非足端接地；
- `PAIR0_ADAPT` 的 pooled full-interval zero-foot fraction 超过 `0.08`；
- 任一条件的 force-qualified corrected sustained-slip fraction 超过 `0.02`；
- 任一条件每 100 个 force-qualified supported substeps 的 corrected-slip events 超过 `0.20`；
- force-qualified supported denominator 为零，使主滑动指标不可评价。

qualified transient 指标只能产生 warning，不能单独停止训练。安全停止后不得用另一个 seed 替换该条件，也不得把未完成预算的结果送入最终 gate。

## 6. 最终 gate：held-out 与 continuity 缺一不可

最终 gate 必须同时接收以下两套互相区分的评估：

- **held-out**：seeds `83801`–`83805`，此前不得用于中间 checkpoint；
- **continuity**：seeds `82801`–`82803`，用于检验最终策略是否保持原 L2 分布上的能力。

两套评估都使用四个冻结标准场景、每轮最多 600 个控制步、deterministic policy 和每控制步全部五个 physics substeps。held-out 必须满足相对于 `DEFAULT_CONTINUE` 的接触改善、支撑数增加、整体及上下坡进度保留、安全和滑动上限；continuity 必须同时满足 `PAIR0_ADAPT` 的绝对整体/上下坡进度下限、零摔倒/零 torso-ground/零持续非足端接地、zero-foot 上限和 force-qualified 滑动上限。

判定采用 `required_all_checks=true`：

1. held-out 通过但 continuity 失败，整体失败；
2. continuity 通过但 held-out 失败，整体失败；
3. 任一必要条件缺失，整体不可评价并按失败处理；
4. 任一主 force-qualified denominator 为零，整体不可评价并 fail closed；
5. 只有两套 gate 的全部预声明检查同时通过，才可记录“L2b 标准场景诊断 gate 通过”。

即使第 5 项成立，`promotion=false` 仍然有效；该记录不授权 fixed-map、视频或候选晋级。

## 7. 能耗与禁止事项

Relative Mission Energy 仍是 measurement-only：公式保持不变、reward weight 固定为 `0.0`，且四项原始分量都必须保存：累积动作平方、绝对力矩时间积分、正机械功代理和绝对机械功代理。任一能耗分量 non-finite 均视为 run failure。不得把这些机械/控制代理改称电池或电气能耗，也不得依据本轮单 seed 结果声称能耗改善。

本轮明确禁止：

- fixed-map evaluation；
- video rendering；
- promotion 或替换 incumbent；
- 中间 checkpoint 选择；
- 奖励、能耗公式、摩擦、观察维度、控制频率或机器人结构改动；
- 覆盖 L2 旧 artifact 或任何已存在的 L2b attempt root。

## 8. 失败后的唯一预声明去向

若最终 gate 失败、不可评价或因安全规则提前停止，应立即硬停止进一步 contact-budget extension。当前 135D observation 已包含 13D local-terrain preview；任何失败解释都必须承认该输入已经存在。配置冻结的下一去向是：**保留现有 13D preview**，另立预声明的结构实验，重新设计或强化 terrain-feature utilisation，以及 terrain-normal/downhill controller；reward 在该实验初始阶段保持不变。下一协议必须逐字段声明特征通路、控制器、编码、消融与观测维度，不能重复计入已有 preview，也不能把本轮中较弱的 learned influence 误写成输入缺失。该去向仍需新 config、测试和 artifact root；本协议本身不授权启动该干预。

## 9. 执行与可审计入口

预期入口如下：

- 配置：`configs/fixed_standard_pair0_adaptation_l2b_extension_v1_20260819.json`
- runner：`scripts/run_fixed_standard_pair0_adaptation_l2b_extension.py`
- 定向测试：`tests/test_fixed_standard_pair0_adaptation_l2b_extension.py`
- smoke 根：`artifacts/smoke/fixed_standard_pair0_adaptation_l2b_extension_v1_20260819/attempt_{attempt}`
- development 根：`artifacts/dev/fixed_standard_pair0_adaptation_l2b_extension_v1_20260819/attempt_{attempt}`

运行顺序应为 config validation、定向测试、工程 smoke，最后才是获准的唯一一次 `--run`。任何目标 attempt root 已存在时都必须拒绝启动；不得删除、清空或覆盖目录来绕过这一保护。
