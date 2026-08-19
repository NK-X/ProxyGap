# PAIR0 最终续训 V3：仅修复 smoke 前置门

## 1. 修复原因

V2 canonical engineering smoke 已完整运行，两分支各8,192步，实际首次 reset seeds、命令曝光、线程、runtime snapshot 和 inventory 均通过原严格验证。Smoke 未保存 checkpoint，也没有运行 held-out 评估。

V2 验证器随后在最后一项禁止路径检查中，把地形生成器合法输出的两个 provenance JSON 也当成了伪造的嵌套 manifest：

| 相对路径 | Bytes | SHA-256 |
|---|---:|---|
| `training_scene_assets/scene_source/standard_scene_manifest.json` | 7,224 | `a68ae4276365cf818d1f98d2c52dc810c59215ba9346e7834292b51802e79efd` |
| `training_scene_assets/scene_source/standard_scenes/flat/scene_manifest.json` | 1,876 | `da17bbebb58f9d18ed3b2dc3d97e26a9b53028dcf05228fd5cebaba29cb4af89` |

这是验证规则过宽，不是训练、环境或科学门失败。V2 smoke root 因一次性合同保持原样，不删除、不修改、不重跑。

## 2. V3 的唯一变化

V3 是 pre-formal gate-only 修复，不包含新的 `learn` 或 `model.save` 调用。它先执行 V2 原有完整 smoke 验证，且只允许该验证到达并抛出“恰好上述两个路径”的最终错误；随后对两个文件再次核验精确路径、大小、SHA-256、合法 JSON 和 smoke inventory 条目。任何第三个 nested manifest，或任何 zip/model/final-evaluation/video/fixed-map/promotion artifact，仍会失败。

V3 还冻结并核验：

- V2 config SHA-256：`9004de242c2724dcbf128d78fb6f5d951f5826e5e2b5db80d4940e185e8f582c`；
- V2 runner SHA-256：`7154424b7d944bbed415b9c814dffa075ed1570843471c84456ce3f56ce051ce`；
- V2 smoke manifest：240,781 bytes，SHA-256 `a02b8dad18c94f75d50c8d41dbc9600884bb735726c9301c7967d1e8238d32be`；
- V2 smoke inventory：51项，逐文件 membership/size/hash 与现场一致；
- V1 config、PAIR0 source checkpoint 及 V2 的25文件传递 runtime 闭包。

## 3. 科学协议完全不变

V3 不改变 V2 的训练 seeds、两个分支、命令表、episode 长度、PPO、65,536步正式预算、最终2,793,472 checkpoint、奖励、摩擦、PAIR0 接触、135D观测、能耗 measurement-only 边界、held-out seeds、九条件转向门、标准坡面连续性门或 hard-stop 规则。

Formal 仍写入唯一原路径：

```text
artifacts/dev/pair0_turn_balance_v2_20260819/attempt_0
```

V3 放行证据会作为 `validated_smoke_prerequisite` 写入 V2 formal manifest；正式训练和评估仍由冻结的 V2 runner 执行，其25文件 runtime 会在 formal root 内重新 snapshot 并做 before/after 核验。

## 4. 终止边界

V3 validate 不创建 formal root。只有独立审计 GO 后才可运行唯一 formal attempt。Formal 无论 PASS、FAIL 或 non-evaluable 都停止全部后续运动优化，保留 C0/C1 最终 checkpoint 和完整证据，再按已冻结 seed 96131、左右 `0.20 m⁻¹` 条件执行只读视频封存；视频不参与科学门。
