# PAIR0 左右转向平衡配对续训协议（设计冻结前版本）

## 1. 结论与研究边界

当前冻结策略在平地九条件诊断中全部通过安全门，但闭环行为呈明显负 yaw 偏移：左转 `0.10 m⁻¹` 的五轮实际累计 yaw 全部与目标异号，而右转 `0.10 m⁻¹` 明显跟随目标。现有证据只能说明“冻结策略、Ant 结构、PAIR0 接触、初始状态和控制接口组成的闭环系统”存在左右不对称，不能把原因单独归因于神经网络权重。

因此，本轮采用一个最小配对实验：

- C0：通用直行续训对照；
- C1：严格平衡的左右转向命令续训；
- 两者使用同一源 checkpoint、优化器状态、训练种子、worker 顺序、平地、PAIR0 接触、摩擦、奖励、能耗测量、PPO 总预算及 episode 长度；
- 唯一实验差异是每个 episode 的外部曲率命令表；
- 不向低层策略加入全局地图。135 维观测已经包含 `122D + 13D` 局部地形预瞄，本实验保留它而不是重复新增地形输入。

这也是用户限定的最后一轮运动优化。无论最终 PASS 或 FAIL，完成预声明的转向、标准坡面及安全复测后均停止新的结构、奖励、接触、自由度或训练干预。

## 2. 训练设计

源模型固定为 `PAIR0 checkpoint_2727936`，SHA-256 为：

```text
5121abeff92859205e1537f123f0df1e97edb5ea1fa80be1a72959a5931fac1c
```

两分支共同使用：

| 项目 | 冻结值 |
|---|---:|
| 并行环境 | 8 |
| 每个 episode | 512 control steps |
| 每个 worker | 16 episodes = 8,192 steps |
| 每分支总预算 | 65,536 steps |
| PPO `n_steps` | 256 |
| 每个 rollout | `8 × 256 = 2,048` transitions |
| `batch_size` | 1,024 |
| 每分支 rollout 数 | 32 |
| 最终绝对 timestep | 2,793,472 |
| 指令速度 | 0.55 m s⁻¹ |
| 训练地形 | 40 m × 40 m、513 × 513 数值平地 heightfield |
| 接触 | 四个 PAIR0 显式足端—地形 pair |
| 摩擦 | 固定，不随机化 |
| 唯一训练 master seed | 63806（worker 首次 reset 为63806–63813） |
| PyTorch CPU threads | 2 |

`n_steps` 从源 checkpoint 的512改为256，但总 rollout 仍为2,048 transitions。这样 C0 与 C1 的更新规模、minibatch 数和训练预算完全匹配。该改变仍会改变单个环境的 GAE 切段，因此 C0 是必要的共同变化对照；不能声称这与旧的4环境续训过程完全相同。

## 3. 精确命令暴露

C0 的8个 worker 均为直行。

C1 的 worker 分配为：

| Worker | 曲率幅值 | Episode 0 | Episode 1 | 后续规则 |
|---:|---:|---:|---:|---|
| 0, 1 | 0 | 0 | 0 | 始终直行 |
| 2 | 0.10 | +0.10 | −0.10 | 逐 episode 交替 |
| 3 | 0.10 | −0.10 | +0.10 | 与 worker 2 反相 |
| 4 | 0.20 | +0.20 | −0.20 | 逐 episode 交替 |
| 5 | 0.20 | −0.20 | +0.20 | 与 worker 4 反相 |
| 6 | 0.35 | +0.35 | −0.35 | 逐 episode 交替 |
| 7 | 0.35 | −0.35 | +0.35 | 与 worker 6 反相 |

C1 的完整65,536步暴露量为：

| 命令 | Steps |
|---|---:|
| 直行 | 16,384 |
| 左/右 `0.10 m⁻¹` | 各8,192 |
| 左/右 `0.20 m⁻¹` | 各8,192 |
| 左/右 `0.35 m⁻¹` | 各8,192 |

因此四个“曲率幅值族”各16,384步，所有非零命令的左右暴露严格相等。七个有符号命令不是均匀采样：直行暴露是任一单独有符号曲率的两倍。协议已明确记录该区别，避免把“幅值族等权”误写成“七命令等权”。

完整 worker—episode 表的规范化 SHA-256 为：

```text
b393134d5255567df048cf974a90a49957ac7a9240fa4074790b13feb36bfffe
```

## 4. 命令、奖励和 terminal observation 时序

独立 wrapper 在 `reset` 后、策略看到第一份观测前安装本 episode 的命令。对曲率 `κ`：

\[
\omega=0.55\kappa,
\qquad
\psi^*_t=\psi^*_0+\omega t\Delta t,
\qquad \Delta t=0.05\;\mathrm{s}.
\]

每个动作的 reward 使用动作前已经冻结的本拍目标朝向和 yaw-rate；物理步结束后，wrapper 才推进下一拍目标，再生成下一份122D命令观测和13D局部预瞄。因此不会发生“动作按命令A执行，却用命令B计算 reward”的错位。

第512步 TimeLimit 截断时也会把目标朝向推进一个 `ωΔt`，重写 terminal 122D 命令并追加对应的13D预瞄，再交给 Stable-Baselines3 做价值 bootstrap。VecEnv 随后自动 reset 产生的零执行步 episode 不计入暴露量；只有实际 transition 会进入整数曝光计数。

任一以下情况立即使整次 attempt 失败并保留失败证据：

- 早于第512步 termination 或 truncation；
- 非有限观测、reward、`qpos` 或 `qvel`；
- reward 实际消费的 heading/yaw-rate 与预期命令不一致；
- 观测或动作空间改变；
- worker 相位、episode 数或逐步曝光表不一致。

## 5. Checkpoint 与优化器续接

源 checkpoint 保存的是 `n_steps=512, n_envs=4`。不能在加载后直接写 `model.n_steps=256`，因为 rollout buffer 仍会保持旧尺寸；也不能调用 `set_env` 把4环境改成8环境。

预声明加载方式为：

```python
PPO.load(
    source_checkpoint,
    env=eight_environment_vec,
    device="cpu",
    force_reset=True,
    n_steps=256,
    seed=63806,
)
```

`force_reset=True` 删除 checkpoint 中旧的4环境 `_last_obs`。`seed=63806` 显式覆盖 checkpoint 保存的旧 seed 62802；每个独立分支在模型加载前固定 PyTorch 为2线程，并在第一次真实 reset 后核验8个 worker 实际消费的 seed 是63806–63813。随后以 `reset_num_timesteps=False` 设置续训，目标绝对计数为2,793,472。C0 与 C1 必须在独立进程中分别从同一源 checkpoint 重新加载，C1 不得接着 C0 的模型训练。

只读工程预检已经验证：

- policy `state_dict` 逐 tensor 完全一致；
- Adam optimiser state 完全一致，共13项；
- 同一观测的确定性动作逐位一致；
- `num_timesteps=2,727,936` 保持不变；
- rollout buffer 正确重建为 `256 × 8`；
- `batch_size=1,024`、`n_epochs=10` 保持不变；
- 模型 RNG seed 为63806、待消费首次 reset seeds 为63806–63813，PyTorch 为2线程；
- `_setup_learn(..., reset_num_timesteps=False)` 后 `_last_obs` 为 `(8,135)`、episode-start 为 `(8,)`，且尚未进行 rollout 或梯度更新。

## 6. 预声明最终复测与门槛

新 held-out seeds 固定为：

```text
96131, 96137, 96149, 96153, 96177
```

这些 seeds 不用于训练、中间检查或 checkpoint 选择。C0 与 C1 只评估各自最终2,793,472 checkpoint。

### 6.1 九条件转向与安全

条件为：直行；左右 `0.10/0.20/0.35 m⁻¹`；以及两个 `0.10 m s⁻¹, ±0.10 rad s⁻¹` 的正速度低速探针。低速探针不是原地旋转，也不设转向有效性门，但必须通过安全门。

九个条件分别必须：

- 摔倒、躯干触地、持续非足端触地均为0；
- 每个 seed 的 force-qualified 支撑分母均大于0；
- corrected sustained slip substeps 与 slip events 均为0；
- pooled full-interval zero-foot fraction 不超过 `0.0580555556`。

转向有效性门为：

- 直行 `|mean yaw change| ≤ 0.5 rad`；
- 每个非零正式转向条件均为 `5/5` 与目标同号；
- `|κ|=0.10,0.20`：左右各自 mean yaw ratio 均在 `[0.7,1.3]`，且左右 ratio 差不超过0.3；
- `|κ|=0.35`：左右各自 ratio 绝对值至少0.5，且左右绝对 ratio 差不超过0.5。

### 6.2 标准坡面连续性

使用相同新 held-out seeds 测试 flat、uphill 8°、downhill 8° 和 bowl exit，保持600步、0.55 m s⁻¹及五个0.01秒物理子步。必须同时满足：

- overall mean best progress ≥ `7.19339076746881 m`；
- uphill mean best progress ≥ `6.18579923623122 m`；
- downhill mean best progress ≥ `8.81135708033187 m`；
- 摔倒、躯干触地、持续非足端触地均为0；
- pooled zero-foot fraction ≤ `0.058055555555555555`；
- 每个 seed 的 force-qualified 支撑分母大于0；
- corrected sustained slip fraction ≤0.02；
- corrected slip events per 100 force-qualified supported substeps ≤0.2。

若 C1 未通过转向门，最终结论是“转向 HOLD”；原 source PAIR0 checkpoint 仍保留为目前已知最佳坡面候选，不再启动下一轮优化。若 C1 通过，也只封存结果，不再启动新一轮优化。C0 是否也通过将用于判断改善是否可能来自通用续训，而不是自动把所有改善归因于平衡课程。

## 7. 能耗边界

能耗继续为 measurement-only，不进入 reward、checkpoint 选择或最终门槛。必须记录且保持有限：

- cumulative squared action；
- actuator absolute torque-time integral；
- actuator positive mechanical work；
- actuator absolute mechanical work。

本实验不修改能耗公式，也不声称得到真实电池电能。当前不加入能耗奖励，是为了避免策略通过少转、少推进或停滞来获得表面上的“节能”优势，从而混淆转向修复。

## 8. 结果后视频和封存

训练阶段仍不渲染视频，避免视频流程影响训练和数值门。视频选择在训练前已经固定为 held-out seed `96131`，并固定正式转向条件为 `curve_left_020` 与 `curve_right_020`；全部最终数值和 gate 冻结后，再按此规则另立只读交付视频合同：

- 至少保留 C0/C1 在同一预声明 seed 下的左转和右转对照，可用清晰四格或等价双视角合成；
- C0/C1 必须使用同一 seed 96131，且不得依据“最好看”或最终结果改换 seed/条件；
- 保存逐步 trace、manifest、SHA-256，并完成全帧 decode 验证；
- 若 C1 通过转向门，另提供代表上坡和下坡复放；若未通过，可明确标注并复用或重渲染冻结 source PAIR0 的坡面交付视频；
- 视频不参与任何科学门槛，也不能改变 PASS/FAIL。

C0、C1 最终 checkpoint、完整训练证据、全部 held-out 原始数据、失败记录和视频交付均须保留。不得只保留“表现较好”的分支。

## 9. 当前工程状态

已落盘：

- 配置：`configs/fixed_standard_pair0_turn_balance_continuation_v1_20260819.json`；
- wrapper：`src/proxygap/paired_turn_balance.py`；
- validate/preflight runner：`scripts/run_fixed_standard_pair0_turn_balance_continuation.py`；
- 定向测试：`tests/test_fixed_standard_pair0_turn_balance_continuation.py`。

当前结果：

- `validate-only`：通过；
- 定向测试：9 passed（新增 C0 wrapper 与既有固定目标 wrapper 的首份观测及首步语义等价检查）；
- 临时目录工程预检：通过；
- 实际 PAIR0 环境：`npair=4`、135D观测、8D动作、首拍 worker 2 曲率 `+0.10 m⁻¹`、yaw-rate `+0.055 rad s⁻¹`；
- 未调用 `learn`，未执行梯度更新，未写 checkpoint；
- 未创建 smoke 或 formal artifact root；
- 未执行固定地图、视频或 promotion。

当前 V1 配置明确禁止训练。下一步只能是独立只读审计；若审计通过，再另行冻结训练授权版本，且科学阈值、种子、预算和命令表不得改变。

## 10. 残余风险

- worker 幅值与 worker rank 固定绑定；相反相位和同 worker 的左右交替消除了方向暴露失衡，但不能完全消除幅值与 seed stream 的混杂。
- 两个分支共享同一个 master training seed；配对比较可控制该 seed，但不能声称训练结果具有多 seed 稳健性。
- 平地训练使13D地形预瞄接近常数，可能遗忘坡面能力，因此坡面连续性门是硬门。
- episode 从600改为512、单环境 `n_steps` 从512改为256，均是续训分布变化；C0 只能控制这些共同变化，不能使其等同于旧训练。
- 即使 C1 通过，也只能支持“完整平衡命令干预有效”，不能单独证明负 yaw 来源于策略权重、接触不对称、动作顺序或初始状态。
- 本协议不测试真实原地旋转、固定大地图导航、路径最优性、随机地图泛化或自然步态美观性。
