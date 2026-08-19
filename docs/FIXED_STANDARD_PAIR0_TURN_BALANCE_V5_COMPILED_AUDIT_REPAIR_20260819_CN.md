# PAIR0 转向平衡 V5：compiled audit 假失败修复与训练前封存

## 结论

V4 没有开始训练。它在四个标准坡面场景均完成 PAIR0 XML 编译审计后，因为父 runner 检查了不存在的 `audit["passed"]` 语义而被误判为失败。`audit_compiled_pair` 的真实契约是“任一字段不匹配即抛出异常；成功时返回完整字段记录”，返回值本身没有 `passed` 字段。

V5 只替换这一项工程判定：逐字段核验完整 compiled contract，包括 XML 仅增加四个 floor–distal pair、pair 数量和目标集合、geom/pair margin、gap、condim、friction、solref、solreffriction、solimp、adhesion、非 distal geom、root joint 以及 physics timestep。V5 不补造 `passed` 字段，任何缺失、额外字段或数值漂移均 fail closed。

## 不变的科学协议

- 同一个源 checkpoint：`checkpoint_2727936.zip`，SHA-256 `5121abeff92859205e1537f123f0df1e97edb5ea1fa80be1a72959a5931fac1c`。
- C0 `C0_STRAIGHT_CONTINUE` 与 C1 `C1_BALANCED_TURN` 均从源 checkpoint 独立 fresh load。
- master seed `63806`、8 workers，每分支增加 `65,536` timesteps，最终 timestep `2,793,472`。
- held-out seeds 保持 `[96131, 96137, 96149, 96153, 96177]`。
- reward、contact/friction、command schedule、energy measurement、turn/slope gates 均不变。
- 不保存、评估或选择 intermediate checkpoint；不运行 fixed-map，不在数值门中制作视频，不 promotion。
- 正式结果无论 PASS、FAIL 或 non-evaluable 均 hard stop，不再授权下一轮 locomotion optimisation。

## 失败证据与 once-only 根目录

旧 root 永久只读保留：

- V2 路径长度失败：`artifacts/dev/pair0_turn_balance_v2_20260819/attempt_0/FAILURE_RECORD.json`，SHA-256 `21ccdebc692af2f32ec96a2e33795cd0eac45ea4aac852eface8a02f26709d23`。
- V4 compiled-audit 假失败：`artifacts/dev/tb_v4_20260819/a0/FAILURE_RECORD.json`，SHA-256 `9695bd3b5d628907053a2f785ec874efa18b2fc47a317c452a88566c0d624812`。

两者均无 checkpoint ZIP、无 `training_record.json`、无 success manifest，不能复用部分权重或在相同 root 重试。

V5 使用新的短根目录：

- 工程 smoke：`artifacts/smoke/tb_v5_20260819/a0`；只生成并核验四个标准场景，不训练。
- 正式运行：`artifacts/dev/tb_v5_20260819/a0`；只能在 canonical smoke 为 GO 后运行一次。

所有预声明绝对路径均必须不超过 239 字符。每个 root 一旦创建，无论成功或失败均不可覆盖；失败记录必须保留。

## 运行边界

```powershell
python scripts/run_fixed_standard_pair0_turn_balance_continuation_v5_compiled_audit_repair.py --validate-only
python scripts/run_fixed_standard_pair0_turn_balance_continuation_v5_compiled_audit_repair.py --engineering-smoke
```

本轮只允许运行以上验证与工程 smoke，并在正式训练前停止。`--formal` 仅在 smoke manifest 完整、自身 inventory 和 runtime snapshot 验证通过且 pre-run decision 为 `GO` 后才可进入；它不是本轮预检的一部分。

## 证据边界

工程 smoke 的 GO 只表示 V5 判定逻辑、四场景生成、compiled contact contract、文件哈希与运行时快照通过工程验证。它没有训练策略，不能证明左右转向改善、坡面性能保持、全局导航成功、路线/能耗最优或自然步态。上述结论必须等待一次性正式 C0/C1 训练及预声明 held-out evaluation。

## 训练前封存记录

- V5 config SHA-256：`5aaf05d346c2b19c9c2714d7ce5ad033f9cd575836be33d8572c47aaa87be908`
- V5 runner SHA-256：`bb255f76212105ed8ad17f52fa8b955e6d9d0a31358daa9e1d8751838c407980`
- canonical smoke manifest SHA-256：`b1f9a794a83cd1ce8e3d898ba76f6a38f57b645dcbc18d0da39e539e64492c53`
- canonical smoke：`engineering_smoke_passed_no_training`
- pre-run decision：`GO`
- formal training started：`false`

该 GO 只解除正式运行的工程门；它没有产生或预示任何正式 C0/C1 科学结果。
