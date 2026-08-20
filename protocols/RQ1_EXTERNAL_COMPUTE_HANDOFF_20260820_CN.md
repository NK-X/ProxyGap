# RQ1 匹配奖励消融：外部高性能电脑执行交接

## 任务边界

该任务只执行已经冻结的 Stage 1 匹配消融：

- `D0_DEFAULT_REWARD`：Ant-v5 默认 forward reward；
- `S1_STAGE1_SHAPED`：冻结的 Stage 1 shaped-reward package；
- 两组共享相同的 113-D observation、Ant-v5、PPO、网络、训练预算、checkpoint 规则和评测 seed；
- 三个独立 training seeds：`62401, 62402, 62403`；
- 每个策略训练 `1,000,000` steps；
- 每个最终策略使用相同的十个 nested evaluation seeds：`72401–72410`。

分析单位是独立 training seed（`n=3`），不是 60 个 evaluation episodes。结果只能作为资源受限的描述性证据；不得声称“自然步态已被证明”。

## 冻结文件身份

执行前必须逐项核验 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `configs/rq1_matched_baseline_v1_20260820.json` | `036bce3b2b676792f3f4fb321d8aedd942214a60e9b7b1d704d93b95ffc01b15` |
| `scripts/run_rq1_matched_baseline_v1_20260820.py` | `0b7c0139f6254a3cbfbf41caaeb18474ceb64251434def02086a884036242271` |
| `scripts/analyse_rq1_matched_baseline_v2_20260820.py` | `e6843d4ee3703914e4fcfc6067dc5487053d374da03eb89e92ce436b1630c5e4` |
| `tests/test_rq1_matched_baseline_v1_20260820.py` | `b863c2a322ca75e7c293d91d848123b85d3d606198a363ef38724e3a64301ee2` |
| `tests/test_analyse_rq1_matched_baseline_v2_20260820.py` | `4a77c05505be974586a477f53c0bf42164e514e60816b498d0fe4c1f9a4a2661` |
| `protocols/RQ1_MATCHED_BASELINE_PROTOCOL_V1_20260820.md` | `31d20da845abff43a05b9509d4b217260f90fd52fc033bd7182e45d8602610da` |

任何哈希不匹配都必须停止，不得“顺手修复”后继续跑。

## 建议环境

- Windows 11 x64；
- Python `3.12.13`；
- PyTorch `2.13.0+cpu`；
- Gymnasium `1.3.0`；
- Stable-Baselines3 `2.9.0`；
- MuJoCo `3.11.0`；
- NumPy `2.5.1`；
- pytest `9.1.1`。

CUDA不是必需的；该协议以 CPU PPO 为冻结参考。不得为了提速修改 `n_steps`、batch、网络、seed、budget 或 reward。

## 唯一执行顺序

在全新的 GitHub clone 中执行，正式 root 必须不存在：

```powershell
git fetch origin
git checkout agent/slope-support-relief-v1
git pull --ff-only origin agent/slope-support-relief-v1
git status --short

python scripts/run_rq1_matched_baseline_v1_20260820.py --validate-only
python -m pytest tests/test_rq1_matched_baseline_v1_20260820.py tests/test_analyse_rq1_matched_baseline_v2_20260820.py -q -p no:cacheprovider

python scripts/run_rq1_matched_baseline_v1_20260820.py --smoke
python scripts/run_rq1_matched_baseline_v1_20260820.py

python scripts/analyse_rq1_matched_baseline_v2_20260820.py --validate-only
python scripts/analyse_rq1_matched_baseline_v2_20260820.py
```

不要传 `--max-workers` 覆盖冻结配置；正式配置已固定最多四个并行任务。不要复用本机已中断的权重或目录。

## 完成验收

必须同时存在并通过：

- `artifacts/formal/rq1_matched_baseline_v1_20260820/attempt_0/execution_record.json`：`status=complete`、`failures=[]`；
- `logs/evaluation_metrics_full.csv`：精确 `240` 行唯一设计单元；
- 6 个最终 `1,000,000`-step policy；
- `analysis_v2_boolean_repair/policy_level_metrics.csv`：6 行（2 conditions × 3 training seeds）；
- `analysis_v2_boolean_repair/paired_training_seed_effects.csv`：3 行；
- `analysis_v2_boolean_repair/result.json`；
- source sibling snapshot 与其 manifest 完整闭合；
- 所有 manifest 的成员、字节数和 SHA-256 与现场文件一致。

如果运行异常，应保留失败目录和记录，不得删除后换 seed 重跑。

## GitHub 回传

完成后先检查单文件大小，再精确上传 smoke/formal roots（这些目录通常被 `.gitignore` 忽略，因此需要 `-f`）：

```powershell
git add -f -- artifacts/smoke/rq1_matched_baseline_v1_20260820/attempt_0
git add -f -- artifacts/formal/rq1_matched_baseline_v1_20260820/attempt_0
git commit -m "Add formal RQ1 matched reward ablation evidence"
git push origin agent/slope-support-relief-v1
```

不要使用 `git add -A` 或 `git add .`。回传时请同时给出：Git commit SHA、formal manifest SHA-256、`analysis_v2_boolean_repair/result.json` SHA-256，以及运行环境版本。
