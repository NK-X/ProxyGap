# PAIR0 左右转向平衡：最终一次可执行协议 V2

## 1. 当前状态

V2 是 V1 配对设计的可执行封装，但在独立审计给出 GO 之前不得运行 smoke 或 formal。本文落盘时尚未调用 `learn`、尚未创建任何 V2 artifact root，也没有生成新 checkpoint。

本轮是项目的最后一次运动优化。无论最终结果为 PASS、FAIL，还是因既定分母/数据完整性条件而不可评价，formal 完成后均写入 hard-stop 记录，不再开展新的结构、奖励、接触、自由度或训练干预。

## 2. 两个配对分支

- C0 `C0_STRAIGHT_CONTINUE`：8个 worker 全部接受零曲率命令；
- C1 `C1_BALANCED_TURN`：worker 0/1直行，2/3、4/5、6/7分别接受逐 episode 反相交替的 `±0.10/±0.20/±0.35 m⁻¹` 命令；
- 两分支分别在独立干净进程中，从同一个 PAIR0 `checkpoint_2727936` 重新加载 policy 和 Adam optimiser；
- 唯一实验差异是每 episode 曲率表。135D 输入仍为原有 `122D + 13D` 局部地形预瞄，不加入全局地图；
- 能耗仍为 measurement-only，不进入 reward、checkpoint 选择或任何门槛。

## 3. 随机种子、线程和 PPO 续接

唯一训练 master seed 为63806，8个 worker 第一次真实 reset 必须分别消费63806–63813。加载必须显式采用：

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

每个分支进程在模型加载前调用并核验 `torch.set_num_threads(2)`。加载后必须证明 policy tensor、optimizer state 和确定性动作与源 checkpoint 完全一致，同时证明 rollout buffer 已重建为 `256 × 8`。首次 reset 后还必须从每个 wrapper 的只读状态中核验实际消费的 seed；仅记录 `vec_env.seed()` 返回值不算充分证据。

本协议只有一个训练 master seed。C0/C1 的配对比较可以控制该 seed，但不能声称训练结果具有多 seed 稳健性。

## 4. 工程 smoke

唯一 smoke root 为：

```text
artifacts/smoke/pair0_turn_balance_v2_20260819/attempt_0
```

每分支8,192步，即4个2,048-transition rollout、每 worker 2个完整512步 episode。这个长度使每个 C1 转向 worker 都实际经历一次左命令和一次右命令；任何 worker 的 seed、相位、正负暴露、episode 数、非有限 transition 或提前结束不符都会使 smoke 失败。

Smoke 只验证工程执行链，不保存 checkpoint、不运行 held-out 评估，也不产生科学结论。Formal 不能只凭 manifest 中的一行“成功”状态放行；必须重新核验同一 V2 配置哈希下唯一 canonical smoke 的完整证据：两个分支精确 membership、每分支8,192步和最终计数2,736,128、8个 worker 各2个 episode、实际首次 reset seeds、C1 每个转向 worker 的左右各512步、C0直行1,024步、无 checkpoint/held-out/video、25文件 runtime live/snapshot before/after、frozen config/source hashes、summary、environment/git，以及 inventory 与现场每个文件的 membership/size/SHA-256。任一不符或存在 `FAILURE_RECORD` 均禁止 formal。

## 5. 唯一 formal attempt

唯一 formal root 为：

```text
artifacts/dev/pair0_turn_balance_v2_20260819/attempt_0
```

每分支固定65,536步：32个 rollout、每 worker 16个完整 episode，最终绝对 timestep 为2,793,472。不得保存、评估或选择中间 checkpoint；仅保存并保留：

```text
C0_STRAIGHT_CONTINUE/models/checkpoint_2793472.zip
C1_BALANCED_TURN/models/checkpoint_2793472.zip
```

两个分支训练结束后，才可使用新 held-out seeds `96131, 96137, 96149, 96153, 96177` 对两个最终 checkpoint 做相同的最终复测。

## 6. 最终复测和不可评价边界

每个分支必须完成：

- 9个平地转向条件 × 5 seeds = 45轮；
- flat、uphill 8°、downhill 8°、bowl exit × 5 seeds = 20轮；
- 每轮600 control steps、每步5个物理子步；
- 所有能耗代理分量有限；
- 转向、坡面、安全、corrected slip 及 force-qualified 分母按 V1 预声明门执行。

若任一正式条件或 seed 的 force-qualified 支撑分母为0，或所需比值/指标缺失或非有限，相应整套门必须标为 non-evaluable；不能把它当作普通数值 FAIL 后继续作科学选择。只有 C0/C1 的 turn 与 slope 两套门均可评价时，才能给出预声明的配对决策。无论可评价与否，项目优化均 hard-stop，固定地图仍为 HOLD。

## 7. Fail-closed 与 provenance

V2 冻结精确传递运行依赖闭包，并在父进程和两个分支进程中核验：

- live runtime before/after；
- preserving-relative-path runtime snapshot before/after；
- canonical V2 configuration、V1 design、源 checkpoint 和两个最终 checkpoint 的 SHA-256；
- scene/contact audit、训练暴露、软件版本、Git 状态和 artifact inventory。

Canonical smoke/formal root 只允许 `attempt_0`，已存在即拒绝覆盖。根目录创建后的任意异常必须写 `FAILURE_RECORD.json`，包括阶段、异常、traceback、不可评价、所有科学决策撤回、禁止重试、源 checkpoint 预期/实测哈希及已取得的 runtime evidence；部分 root 永久保留。

## 8. 结果后只读视频

训练和数值门阶段不渲染视频。为避免事后挑选“最好看”的样本，视频规则已在训练前固定：

- seed：`96131`；
- 正式条件：`curve_left_020` 和 `curve_right_020`；
- C0 与 C1 使用同一 seed/条件，制作清晰的左右转对照，可采用四格或等价双视角；
- 保存逐步 trace、manifest、SHA-256，并做全帧 decode 验证；
- 若 C1 通过转向门，再按独立只读合同提供代表上坡/下坡复放；否则明确标注并复用或重渲染冻结 source PAIR0 坡面交付；
- 视频不进入科学门，不得在看到数值结果后更换 seed 或转向条件。

Formal 无论结果如何，均保留 C0/C1 最终 checkpoint 和全部数值证据，随后只触发上述只读视频封存，不再触发训练。

## 9. 结论边界

该实验最多回答：严格平衡的局部转向命令续训是否相对于匹配直行续训，修复已观察到的闭环左右不对称，同时保持既定坡面与安全下限。它不能隔离偏置究竟来自 policy、接触、动作顺序或初态，也不能证明原地转向、固定地图可用、随机地图泛化、路线最优、自然步态或真实电池能耗。
