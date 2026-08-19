# 固定复杂地形：terrain/target-frame reward 坐标修正 pilot

## 结论

本轮 grouped ablation **不通过预先声明的保留门槛，不保留新 checkpoint**。继续保留
`SOURCE_STAGE1_WORLD_FRAME` 作为后续开发起点；`TERRAIN_FRAME_REWARD_INTERVENTION`
仅作为负结果证据归档，不能替换现有策略。

这不是说地形坐标修正“物理上错误”，而是说：在固定的 131,072 步 continuation
预算、一个训练 seed 和三个验证 seed 下，同时修正六组已激活 shaping 信号，并未转化为
更好的足端支撑或任务推进。该结果也不支持继续单纯增加相同训练步数。

## 唯一干预与冻结边界

三组比较为：

1. `SOURCE_STAGE1_WORLD_FRAME`：135D preview stage-1 源 checkpoint；
2. `W4_ARCHIVED_WORLD_FRAME_CONTROL`：复用此前从同一源 checkpoint、相同 seed 和预算训练的 W4 continuation；
3. `TERRAIN_FRAME_REWARD_INTERVENTION`：仅启用 terrain/target-frame shaping context，并从同一源 checkpoint 继续 131,072 步。

冻结项目包括：80 m × 80 m 地图、height/XML 哈希、摩擦 `[1.0, 0.5, 0.5]`、
`condim=3`、8-DOF Ant、135D observation、8D action、高层 direct-to-goal controller、
所有 reward 权重、`ctrl_cost_weight=0.5`、PPO 参数、训练 seed `62803`、四个 spawn
fraction、训练速度 `0.55 m/s`、评估速度 `0.70 m/s` 和验证 seeds
`73801–73803`。V2 相对任务能耗仍是 measurement-only，未进入 reward。

## 实现定义

可选开关默认为 `false`；关闭时继续走原来的 world-frame 代码路径。启用时，
`FixedGoalTerrainWrapper` 在 reset 后、首个 policy step 前以及每次更新下一步 command 时安装
context。非法、非有限、非单位或向下的 normal 会 fail closed；没有有效 context 时禁止执行
physics step。

启用后的六组量为：

- distal-foot clearance：足端最低点世界高度减去该足端 XY 的 heightfield 高度；
- foot lateral velocity：投影到“目标左侧”的局部地形切向轴；
- foot vertical velocity：投影到各足端 XY 的局部地形法向；
- root vertical velocity：投影到 root XY 的局部地形法向；
- torso orientation：躯干完整 up-axis 与 root 局部地形法向的夹角，同时包含坡向与横坡分量；
- root roll/pitch angular speed：先把 free-joint 的局部角速度转到 world，再去除绕地形法向的分量。

MuJoCo 官方说明 free-joint 的前三个速度分量位于 global frame，而旋转速度位于 local
body frame。因此实现使用 `data.xmat[torso] @ qvel[3:6]` 后再作 world-normal 投影；测试还将
该结果与 `mj_objectVelocity(..., flg_local=0)` 对齐到 `1e-12`。
[MuJoCo floating objects](https://mujoco.readthedocs.io/en/3.3.1/overview.html#floating-objects)

在 canonical flat normal `[0,0,1]` 且 target heading `+x` 时，启用分支与原 world-z/world-y
shaping 数值完全一致；一般 target heading 会按定义旋转 lateral 轴，因此不应与固定 world-y
相同。默认关闭状态由既有回归测试保证行为不变。

## 预先声明门槛与正式结果

门槛全部相对于复用的 W4 control：

- airborne fraction 至少绝对下降 `0.05`；
- best-progress ratio 至少 `0.80`；
- 不增加 fall。

| 指标（3 seeds 均值） | W4 control | Terrain-frame | 变化 |
|---|---:|---:|---:|
| airborne fraction | 0.6656 | 0.6792 | +0.0136（变差） |
| best progress (m) | 5.5002 | 2.3573 | −3.1429；ratio=0.4286 |
| falls / 3 | 0 | 1 | +1 |
| mean support count | 0.3836 | 0.3637 | −0.0200 |
| sustained contact-speed transient fraction | 0.00787 | 0.00878 | +0.00091 |
| cumulative squared action | 980.50 | 924.29 | −5.73% |
| absolute torque-time integral (N m s) | 32323.33 | 30341.70 | −6.13% |
| positive mechanical work (J) | 14545.24 | 17698.83 | +21.68% |
| absolute mechanical work (J) | 16540.27 | 20138.67 | +21.76% |

因此三项门槛全部失败。Seed `73801` 在 3,042 步因 terrain-relative torso tilt
终止；代表 seed `73802` 未摔倒，但 3,600 步只取得 `0.8955 m` best progress，airborne
fraction 为 `0.6975`。

动作平方和 torque-time 下降而机械功上升，进一步说明这些量不能互相替代，也不能把动作平方
直接声称为焦耳。这里仅报告现有机械量，不对尚未实现的 V2 相对任务能耗作排名。

## 解释边界

证据支持的直接解释是：本轮 grouped coordinate correction 没有改善有效支撑，反而减少推进并
增加一次摔倒。可能机制包括 reward landscape 在一次短 continuation 中发生较大变化、六个信号
同时切换造成 credit assignment 困难，以及底层 8-DOF 支撑能力/接触时序仍是主瓶颈。这些是
待检验假设，不是本 pilot 单独证明的因果结论。

该实验只有一个训练 seed、三个固定地图验证 seed；禁止声称统计优势、随机地图泛化或结构自由度
已被充分排除。代表视频无论结果正负均保留，并经过完整帧解码验证。

## 建议

不继续使用当前 terrain-frame checkpoint。下一步若仍研究坐标修正，应拆成更小的预声明消融，
优先只修正 foot clearance（避免一次改变六种 shaping），同时保留 source/W4 paired control；
若仍不能降低 airborne，则应把工作重心转回接触时序/支撑控制，而不是再扩大 shaping 权重。

## 可审计材料

- 冻结配置：`configs/fixed_quad_terrain_v2_terrain_frame_reward_pilot_v1_20260819.json`
- runner：`scripts/run_fixed_goal_terrain_frame_reward_pilot.py`
- 正式根目录：`artifacts/dev/fixed_quad_terrain_v2_terrain_frame_reward_pilot_v1_20260819/seed_62803`
- 逐 episode 指标：`logs/evaluation_episodes.csv`
- paired gate：`comparison_summary.json`
- checkpoint、trace 与哈希：`manifest.json`
- 代表 trace：`traces/terrain_frame_reward_intervention_seed_73802_trace.csv`
- 代表视频合同：`configs/fixed_quad_terrain_v2_terrain_frame_video_contract_v1_20260819.json`
- canonical 视频：`videos/tf73802_v1/fixed_map_final_policy_seed_73802_dual_view_v1.mp4`
- 独立视频 QA：`videos/terrain_frame_video_qa.json`

视频 manifest 记录的是本次运行目录中的绝对 scene 路径；该 scene 与仓库已跟踪的可移植副本
`artifacts/dev/fixed_quad_terrain_v2_training_20260818/seed_62801/task_scenes/spawn_0_0.000.xml`
逐文件 SHA-256 相同。发布包不重复纳入新的 `task_scenes` 目录，复验时应使用上述已跟踪等价副本。

Canonical 视频为 1280 × 720、20 fps、241 帧、12.05 s，完整解码 241/241 帧；它将
180 s 物理 rollout 以 20× 速度播放。视频 trace 与正式 seed `73802` 的 3,600 步 trace
在位置、地形高度、躯干高度、终点距离、support count、接触切向速度、reward 及终止标志上
逐值一致。最终帧在 `t=180/180 s` 以红色状态条显示 `TIME LIMIT / SAFETY FAIL`，与
未到达且该 episode 曾出现四足端悬空和接触速度超限的记录一致。视频 SHA-256 为
`f9bd900e2bfd7cdeedd18a9d2038130e5ec6d084d283dd0d0d0ff26f88a97959`；视频 manifest
SHA-256 为 `8a332ad2c7f0e797f35a5e4abbc8e5d183aeef4803046c5d47dc3643ff87bb11`；独立 QA
SHA-256 为 `8dac91d95ca1720607093f44efdaae687fcba1cb13bdaddd97bcbd861648480d`。

首次渲染已经产生 MP4 和 trace，但在 Windows 长路径下写 contact sheet 时失败。该不完整目录
`videos/terrain_frame_seed_73802_dual_view_v1` 必须标记为
`INTERMEDIATE_DO_NOT_PUBLISH`；它不属于发布 allowlist，也不得替代短路径 canonical 视频。

`artifacts/smoke/fixed_quad_terrain_v2_terrain_frame_reward_pilot_v1_20260819` 仅是 512 步
工程 smoke，不参与正式性能结论，也不应作为交付 canonical evidence。
