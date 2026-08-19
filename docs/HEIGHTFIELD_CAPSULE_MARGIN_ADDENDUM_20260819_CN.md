# Heightfield–capsule margin 单因素诊断附录

## 1. 决策结论

在本次受控条件下，**双方默认 `0.01 m` margin 的组合是 heightfield 支撑异常的主要机制贡献者**，而分辨率相关的多三角形接触是剩余机制。不能据此称 MuJoCo 存在软件缺陷；这是当前长 ankle capsule、heightfield 碰撞表示和接触 margin 组合下的模型行为。

当 floor 和四个 ankle collider 的 margin 同时由 `0.01 m` 改为 0 时：

- 129 heightfield 的 8 s 开环四足端均无接触比例由 73.625% 降到 21.500%；
- 257 heightfield 由 77.500% 降到 21.875%；
- 对应的 native plane margin=0 为 22.125%；
- 129/257 heightfield 的平均受支撑足数分别为 1.24375/1.22250，而 plane margin=0 为 1.24375；
- 三种表面的零控制落体最后 1 s 均不再出现四足端同时无接触。

这足以把下一步优先级从“立即更换足端结构或降低网格分辨率”调整为“先在标准斜坡上 paired 复验 margin=0”。它尚不能证明复杂地形正式地图已经稳定，也不能直接把正式 XML 的 margin 改为 0。

## 2. 严格单因素矩阵

表面包括 native plane、精确平坦 129×129 heightfield 和精确平坦 257×257 heightfield。每个表面测试四个条件：

| 条件 | floor margin | 四个 ankle collider margin | pair inclusion margin（有接触时） |
|---|---:|---:|---:|
| default | 0.01 m | 0.01 m | 0.02 m |
| floor-only zero | 0 | 0.01 m | 预期 0.01 m |
| foot-only zero | 0.01 m | 0 | 预期 0.01 m |
| both zero | 0 | 0 | 0 |

除指定 margin 外，以下量保持一致并经过编译后校验：

- `friction=[1.0,0.5,0.5]`、`condim=3`；
- `solref=[0.02,1]`、`solimp=[0.9,0.95,0.001,0.5,2]`；
- robot `qpos0`、质量、惯量、执行器和动作范围；
- 同姿态探针 `qpos`；
- 同一初始 `qpos/qvel` 与同一封存 160-step 动作数组；
- 静态落体高度、零控制、步长和仿真时长；
- 奖励、能耗和正式地图均未修改，未执行训练。

floor-only zero 与 foot-only zero 的所有确定性结果完全一致，符合双方 margin 对称进入该接触对的观察；both-zero 才提供完整修复。

## 3. 同姿态碰撞探针

在完全相同 `qpos`、零 `qvel` 且不推进时间时：

| 表面 | margin 条件 | 足端接触点 | 接触距离范围 | 合法向力和 |
|---|---|---:|---:|---:|
| plane | default | 4 | +0.0197369 m | 8.573 N |
| hfield 129 | default | 10 | -0.0102631 至 -0.0002302 m | 106.373 N |
| hfield 257 | default | 23 | -0.0102631 至 -0.0002302 m | 155.361 N |
| 任一表面 | floor-only zero | 0 | 无接触返回 | 0 N |
| 任一表面 | foot-only zero | 0 | 无接触返回 | 0 N |
| 任一表面 | both zero | 0 | 无接触返回 | 0 N |

`contact.dist` 是 MuJoCo 窄相位碰撞计算返回的接触量，不能把 plane 与 heightfield 的 0.03 m 差值解释为真实地面高度差。可支持的结论是：默认 inclusion margin 为 0.02 m 时，该相同姿态在 hfield–capsule 对上生成了大量接触和很大的约束力；把任意一侧 margin 置零使 inclusion 范围缩到约 0.01 m，并在该姿态下不再生成接触。该结果明确要求继续检查 margin，而不是把差异全部归因于网格分辨率。

## 4. 零控制静态落体（最后 1 s）

| 表面 | margin | 四足端均无接触 | 平均受支撑足数 | 每个受支撑足的接触点 | 最终 |vz| |
|---|---|---:|---:|---:|---:|
| plane | default | 0% | 4.00 | 1.00 | 0.0125 m/s |
| plane | both zero | 0% | 4.00 | 1.00 | 0.0156 m/s |
| hfield 129 | default | 77% | 0.23 | 2.78 | 0.1551 m/s |
| hfield 129 | single-side zero | 20% | 0.91 | 1.81 | 0.0384 m/s |
| hfield 129 | both zero | 0% | 3.00 | 1.67 | 0.0169 m/s |
| hfield 257 | default | 73% | 0.30 | 6.17 | 0.1091 m/s |
| hfield 257 | single-side zero | 23% | 0.91 | 3.67 | 0.0754 m/s |
| hfield 257 | both zero | 0% | 3.87 | 1.49 | 0.0032 m/s |

single-side zero 有明显但不充分的改善；both-zero 才使所有表面的四足端同时无接触比例降为 0。257 heightfield 在 both-zero 下仍有每足 1.49 个接触点，说明三角形多接触尚未消失，但不再造成这一静态失稳结果。

## 5. 同状态同动作开环重放（8 s）

| 表面 | margin | 四足端均无接触 | 平均受支撑足数 | 每个受支撑足的接触点 | 净位移 |
|---|---|---:|---:|---:|---:|
| plane | default | 19.875% | 1.2750 | 1.000 | 3.004 m |
| plane | both zero | 22.125% | 1.2438 | 1.000 | 2.828 m |
| hfield 129 | default | 73.625% | 0.2925 | 3.231 | 3.064 m |
| hfield 129 | single-side zero | 56.625% | 0.5075 | 2.845 | 3.152 m |
| hfield 129 | both zero | 21.500% | 1.2438 | 1.577 | 2.709 m |
| hfield 257 | default | 77.500% | 0.2425 | 6.840 | 2.693 m |
| hfield 257 | single-side zero | 59.875% | 0.4800 | 4.958 | 2.692 m |
| hfield 257 | both zero | 21.875% | 1.2225 | 2.408 | 2.948 m |

与相同 plane margin=0 相比，both-zero 下 129 heightfield 的四足端均无接触比例低 0.625 个百分点、平均支撑足数相同；257 heightfield 低 0.250 个百分点、平均支撑足数低 0.02125。对于这个确定性开环诊断，主要支撑差异已经消失。

净位移不是本实验的主要结果，因为动作是从 plane/default 轨迹封存的开环序列，并未根据每个 margin 条件重新闭环计算。其作用是确认修改没有让机器人简单停止；不能用于宣布任务性能提升。

## 6. 更新后的行动顺序

1. 另建标准场景 XML 副本，仅把 floor 与四个 ankle collider 的 margin 同时设为 0；保留正式地图和原 XML 不变。
2. 对 plane、精确平坦 heightfield、+8°、-8° 和 bowl 做 paired、多 seed、闭环 checkpoint 评估；先不训练，确认现有策略没有因接触高度变化而失效。
3. 若评估通过，再用相同 margin=0 表示做小预算 continuation training；训练与测试不得混用不同 margin。
4. 只有 slope paired 结果仍出现明显多接触、滑动或失稳，才依次测试 distal-foot collider 和较低碰撞分辨率。不得把这两个修改与 margin 同时改变。
5. 能耗继续作为测量，不加入本轮奖励；任何正式能耗比较必须固定 margin，因为 margin 会改变接触力、动作轨迹和任务时长。

## 7. 边界与风险

- 这是确定性、单 checkpoint、单状态/动作序列的机制附加实验；没有训练 seed 方差。
- margin=0 会改变接触开始的高度和碰撞容差。在陡坡、快速落脚及真实复杂地图上可能出现漏碰撞或更深穿透，必须经过标准坡面和最大坡度边界检查。
- 不能把 `margin=0` 称为物理上唯一正确设置；它只是当前对照中消除主要差异的候选。
- 不能把该结果称为 MuJoCo bug。它可能是 documented contact-margin semantics 与 heightfield/capsule 接触生成共同导致的预期数值行为。

## 8. 工件

- 配置：`configs/heightfield_capsule_margin_diagnostic_v1_20260819.json`
- 脚本：`scripts/diagnose_heightfield_capsule_margin.py`
- 测试：`tests/test_heightfield_capsule_margin_diagnostic.py`
- 最终工件：`artifacts/dev/heightfield_capsule_margin_diagnostic_v1_final_20260819`
- manifest SHA-256：`7ce58d4c7f3d972a80413edb4f37a94d18f5e11e1927a3ae2e7875b875f61941`
- 逐子步日志：`logs/matched_open_loop_margin_substeps.csv` 与 `logs/static_drop_margin_substeps.csv`

复现命令：

```powershell
& 'C:\Users\18522\Documents\nn\drlv\Scripts\python.exe' `
  scripts\diagnose_heightfield_capsule_margin.py `
  --output-root artifacts\dev\heightfield_capsule_margin_diagnostic_reproduction
```
