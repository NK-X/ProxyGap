# 固定复杂地形导航阶段进展报告（2026-08-19 07:30）

> 项目：ProxyGap V3，Ant-v5 四足机器人在连续随机地形上的起点—终点导航
> 本轮范围：封存固定地图上的低层运动、局部地形感知、支撑、路线控制、时间—能耗协议和视频证据
> 证据性质：开发诊断与 pilot；不能外推为未见地图泛化、真实机器人性能或统计显著性
> 约束：封存地图、全图摩擦、8 自由度 Ant XML 和既有原始证据均未被覆盖

## 1. 结论先行

截至本报告证据截点，机器人在封存地图上已经证明**存在空间可达轨迹**，但尚未证明能够稳定、
安全、可重复地完成任务：固定的五个评价 seed 中有 `1/5` 进入终点并完成位置滞回，`2/5`
摔倒，而按支撑、腾空和稳定站立门槛计算的**合格完成仍为 `0/5`**。因此，本阶段不能声称
“机器人已经解决该地图”，也没有任何轨迹可以进入正式时间—能耗最优排序。

当前最有证据支持的主要瓶颈不是全图摩擦不足，也不是已经证实的机械自由度不足，而是：

1. 低层策略长期缺少可靠足端支撑，约 `65%--70%` 的控制步四个指定足端均未接触地面；
2. 现有低层策略并未学习全局选路：它不观察完整地图或终点坐标，固定地图训练也没有完整到达奖励；
3. 现有转向接口把偏航速度限制为“前进速度 × 曲率”，无法表达真正的零速原地旋转；
4. 局部地形预瞄、单纯增大腾空惩罚和几何航点路线均未同时恢复支撑、推进和稳定完成；
5. 原先称为“滑动率”的逐步指标主要捕捉落脚瞬态。加入落脚宽限、法向力和持续时间后，
   本轮十个配对 episode 中没有形成校正后的持续足端滑移事件；这削弱了“摩擦不足是主因”解释，
   但不证明不存在小腿、躯干接触滑移或指标未覆盖的整体滑动。

当前最合理的工程方向是继续把**低层支撑/地形相对运动**与**高层路线/人的时间—能耗意图**分开：
先使同一 8 自由度机器人在已声明坡度范围内通过稳定安全门槛，再标定相对任务能耗 V2，最后才
允许高层规划器在合格路线间学习或选择时间—能耗 Pareto 折中。现在增加关节、六足或吸附力会
同时改变动力学、能耗基线和研究问题，证据尚不足以启动这些结构改造。

## 2. 固定实验对象与判定口径

本轮共同输入为：

- 封存 `80 m × 80 m` heightfield；起点 `(-34,-34)`，终点 `(+34,+34)`；名义直线距离
  `96.167 m`；
- 高度数组 SHA-256：
  `59e60ddd91d799f44f84aa74a2ecff122ac01b1c7c7ea13fe14b032bc176eb9c`；
- 全图地面摩擦固定为 `[1.0,0.5,0.5]`，`condim=3`；没有摩擦随机化；
- 原 Ant XML、质量、8 个执行器、gear、动作范围和控制间隔不变；
- 基线 checkpoint `checkpoint_2465792.zip`，SHA-256：
  `d69ae7a45efdad14a72426ea49c300a670b8c4b92558a97577b3b4130d649abb`；
- 评价使用确定性策略。配对 episode seed 是同一训练策略下的嵌套评价，不能冒充独立训练 run。

复现环境仍有一项边界：局部预瞄正式训练记录为 Python 3.12.13 / NumPy 2.5.1，正式五 seed
配对回放记录为 Python 3.11.15 / NumPy 2.4.6（两者 MuJoCo 3.11.0）。同一配对回放内部的
候选比较仍使用共同环境，但尚未证明从训练到评价在单一 Python 3.11 环境中的端到端精确复训。

必须使用三层结果语义：

| 层次 | 定义 | 当前固定五 seed 结果 |
|---|---|---:|
| 空间到达 | 先进入 `1.5 m` 圈，再在 `2.0 m` 滞回圈内连续保持 `2 s` | `1/5` |
| 稳定站立完成 | 上述位置条件内，同时满足低速、健康姿态和足端支撑等连续门控 | `0/5` |
| 安全合格完成 | 稳定完成，且全轮无摔倒、四足同时腾空和失格滑移 | `0/5` |

seed 74803 在 `473.75 s` 后空间到达，最终距离 `1.373 m`，但终点位置保持的 40 个控制步中
有 24 步四足端同时无接触，支撑足数量最大为 1，最后一步仍为 `airborne=True`。因此这是一条
“空间可达、安全失败”的诊断轨迹，不是任务成功样本。

## 3. 最优路径和人的意图如何定义

### 3.1 不存在脱离任务偏好的唯一最优路径

“时间最短”和“能耗最低”通常冲突；在未明确人的任务偏好时，不能只给一个加权分数并称为
普适最优。V1 协议采用词典序约束：

1. 先满足到达、两秒稳定站立和全部安全硬约束；
2. 只有合格轨迹才比较任务时间与相对任务能耗；
3. 保存并报告原始时间、原始能耗和完整 Pareto 前沿；
4. 再根据用户显式选择的 `mission_profile` 从 Pareto 候选中排序。

对合格轨迹定义：

```math
J(w_T,w_E)
=w_T\frac{T_{\mathrm{mission}}}{T_{\mathrm{ref}}}
+w_E\frac{E_{\mathrm{mission,rel}}}{E_{\mathrm{ref}}},
\qquad w_T+w_E=1.
```

建议预声明三个可解释的人的意图：

| 人的意图档案 | `wT` | `wE` | 语义 |
|---|---:|---:|---|
| `time_prioritised` | 0.70 | 0.30 | 安全合格后偏向更快到达 |
| `balanced_demo` | 0.50 | 0.50 | 演示用折中，不代表普适最优 |
| `energy_prioritised` | 0.30 | 0.70 | 在成功时限内偏向较低任务能耗 |

这些权重是**高层路线偏好输入**，不是当前 PPO 奖励系数。若以后训练权重条件化高层策略，
`(wT,wE)` 必须作为显式输入；测试时不能暗中改变权重。`Tref` 和 `Eref` 只能用训练/验证地图
的合格基准轨迹确定，不能用测试地图重新标定。预声明 Pareto 扫描为
`wE=[0,0.25,0.50,0.75,1.00]`。

### 3.2 当前轨迹不是最优路径

seed 74803 的累计平面路径约 `219.82 m`，而名义起终点直线距离为 `96.17 m`；仅按二者比值，
几何路径效率约为 `43.75%`，且轨迹有明显摆动和绕行。这直接排除了“当前空间到达轨迹已经是
距离最优”的说法。它也不能参加时间—能耗排序，因为安全合格条件为 0。

独立重建的 16° 几何候选路线长 `106.150 m`，累计爬升/下降 `5.711/3.481 m`，走廊最大
坡度 `15.968°`，累计转向 `129.073°`，包含 214 个 0.5 m 间隔 waypoint，点列 SHA-256
为 `fbb764566f1ab5d40714c6fe420b454154fe3cae69720e69c33ce64201f37b53`。它只是冻结离散化和
几何约束下的可审查候选，没有全局最优保证，也未通过动力学安全门槛。

## 4. 为什么训练后仍会爬不上、退回或看似打滑

### 4.1 策略并没有学过完整的全局选路问题

现有 checkpoint 的低层观察不包含完整高度图、全局位置或终点坐标。高层控制器每一步只重新
计算“当前位置指向终点”的航向并发出前进/偏航命令，没有不可通行区、A*、航点记忆或重规划。
固定地图训练的 `additional_task_reward=0.0`，每个训练 episode 只有 900 步，即 `45 s`。
即使始终达到 `0.70 m/s` 指令速度，理论路程也只有 `31.5 m`，远小于约 `96.17 m` 的完整
对角线任务。因此，“已经训练足够久，所以应当知道换一条坡”不是现有训练目标能支持的推论。

### 4.2 主要故障先发生在支撑，而不只是坡顶选路

五个基线评价 seed 中，除 74803 外的失败轨迹最佳推进均不足约 `3.24 m`；很多问题在起点附近
就出现。成功轨迹全程四足端同时无接触率为 `68.07%`，最长连续 `1.05 s`。这说明机器人在
很大比例的时间没有由四个指定足端形成可用支撑。摩擦只能在存在足端法向力时提供牵引；足端
没有接触时，把摩擦系数调得更高也不能生成推进力。

### 4.3 摩擦与“滑动”的证据边界

封存地图最大坡度约 `34.156°`。简化静态无滑边界给出
`mu_min=tan(34.156°)≈0.678<1.0`；因此在静态近似下，当前滑动摩擦 `1.0` 并不偏低。
该计算不覆盖落脚冲击、法向力波动、错误落脚和 MuJoCo 接触求解，因此不能证明动态过程中绝不滑。

旧指标只要任一足端单步切向接触速度超过 `0.20 m/s` 就记一次“滑动”，把落脚冲击也混入其中。
本轮校正指标增加了 `0.10 s` 落脚宽限、至少 `1 N` 法向力、连续至少 `0.20 s` 超过阈值三项
条件。在 122D 基线与 135D 局部预瞄的五 seed 配对（共 10 个 episode）中，校正后持续足端
滑移事件总数均为 0；原始单步超限均值仍分别为 `22.77%` 与 `19.48%`。合理结论是：原始
“滑动率”主要是接触瞬态代理，现有证据不支持把摩擦不足列为主要根因。该指标仍不覆盖小腿或
躯干接触地面的整体滑移，也可能漏掉阈值以下的缓慢滑退，故不能反向声称“完全没有滑动”。

## 5. 已完成的可回退消融与负结果

### 5.1 局部地形预瞄：减少摔倒，但没有恢复推进

先完成 122→135 维 observation 零列迁移；迁移时策略和价值网络的旧列保持不变，初始动作、
动作分布参数和价值输出误差均为 0。随后加入身体前方 9 个相对高度、地形法向/坡度等 13 维
局部信息，新增训练 `262,144` 步。

三个开发评价 seed 中，中间 checkpoint `2596864` 的平均净推进从源模型的 `9.099 m` 提高到
`13.168 m`、摔倒从 1 降到 0，四足悬空率由 `69.79%` 降到 `65.26%`；继续训练到 final
`2727936` 后，平均净推进又降到 `3.291 m`，悬空率为 `66.00%`。这说明训练后期出现退化，
不能只因“训练步数更多”选 final。

在固定五 seed 的正式配对回放中：

| 条件 | 空间到达 | 稳定/合格 | 摔倒 | 平均净推进 | 平均悬空率 |
|---|---:|---:|---:|---:|---:|
| 122D 基线 `2465792` | 1/5 | 0/5 | 2/5 | `20.051 m` | `69.85%` |
| 135D final `2727936` | 0/5 | 0/5 | 0/5 | `3.468 m` | `66.88%` |

135D final 的新 13 列在策略第一层的权重范数远小于旧 122 列，且表现为“少摔但几乎不向终点
推进”。因此预瞄接口在工程上可用，但当前训练不支持“预瞄已经解决复杂地形运动”的结论；应保留
中间 checkpoint 并把训练后期退化作为正式诊断。

### 5.2 仅把四足腾空惩罚从 W4 提高到 W12：不保留

从同一 135D stage-1 checkpoint 出发，以相同 seed、PPO 预算和地图生成方式分别继续训练
W4 对照和 W12 干预，各 `131,072` 步。三个配对验证 seed 的结果为：

| 条件 | 平均最佳推进 | 平均净推进 | 平均悬空率 | 摔倒 |
|---|---:|---:|---:|---:|
| W4 等预算续训 | `5.500 m` | `5.166 m` | `66.556%` | 0/3 |
| W12 支撑优先 | `2.801 m` | `1.903 m` | `66.296%` | 0/3 |

悬空率只下降 `0.259` 个百分点，而最佳推进只保留 W4 的 `50.9%`。预声明保留门槛是至少降低
5 个百分点、推进保留至少 80%、不增加摔倒；W12 未通过。动作量和若干机械功诊断下降不能称为
节能，因为有效推进同时崩溃，V2 相对任务能耗也尚未实现。该结果不支持继续扫描 W8/W12 或单纯
延长相同奖励训练。

### 5.3 16° waypoint 路线：推进有所改善，支撑问题未解决

五 seed 配对中，较好的 `0.4 m/s、3 m 前视、|kappa|<=0.20 m^-1` 路线条件取得 `1/5`
位置保持、`0/5` 稳定 dwell、`0/5` 安全合格、`1/5` 摔倒，平均净推进 `37.402 m`，平均
悬空率 `67.00%`。其平均推进高于直接指向终点基线的 `18.001 m`，说明高层路线可能改善地图
穿越方向；但它没有修复低层支撑，也不能据此宣称路线更优。

单 seed 的 45° 初始朝向探索中，一个 `3 m` 前视条件达到路线进度 `100%`、最小终点距离
`2.901 m`，随后越界/摔倒；`5 m` 前视只达到 `14.13%` 路线进度。按坡度把速度从 0.5 m/s
降到最低 0.3 m/s 的单 seed 探索只达到 `16.02%` 路线进度。它们是故障定位样本，不能替代
五 seed 配对或作为成功路线选择证据。

### 5.4 terrain-frame reward pilot

该 grouped ablation 把足端高度/速度、躯干法向速度、角速度和姿态等六组平地世界坐标 shaping
统一改为局部地形法向与目标切向坐标。开关默认关闭，关闭路径与旧行为逐值一致；启用时在首个
physics step 前安装地形 context，非法法向或缺少 context 会 fail closed。地图、摩擦、8-DOF
结构、135D observation、全部奖励权重、PPO、训练 seed 和能耗边界均冻结。

干预从 135D stage-1 `checkpoint_2596864.zip` 出发继续训练 `131,072` 步，并与此前同源、同 seed、
同预算的 W4 world-frame continuation 配对比较。三个验证 seed 的结果为：

| 指标 | W4 world-frame | Terrain-frame | 结果 |
|---|---:|---:|---|
| 平均四足悬空率 | `66.556%` | `67.917%` | 反而增加 `1.361` 个百分点 |
| 平均最佳推进 | `5.500 m` | `2.357 m` | 只保留 `42.86%` |
| 平均支撑足数 | `0.3836` | `0.3637` | 下降 |
| 摔倒 | `0/3` | `1/3` | 新增一次 |
| 空间/合格完成 | `0/3` / `0/3` | `0/3` / `0/3` | 均未完成 |

预声明门槛要求悬空至少下降 5 个百分点、推进保留至少 80%、不增加摔倒；三项全部失败，因此不
保留 terrain-frame checkpoint，继续保留 135D stage-1 源模型。动作平方和转矩时间积分虽下降，
正/绝对机械功却分别上升约 `21.68%/21.76%`，再次说明动作代理、力矩积分和机械功不能互相替代，
也不能把其中任一个直接称为 V2 相对任务能耗。

这一负结果不证明地形相对坐标在物理上错误；它只说明一次同时切换六组 shaping、再做短 continuation
没有改善支撑。若继续研究，应先只修正 foot clearance 并保持等预算对照，而不再扩大同类权重或
同时改变更多模块。完整方法与证据见 `docs/FIXED_MAP_TERRAIN_FRAME_REWARD_PILOT_20260819_CN.md`。

## 6. 能耗机制与本轮运动优化的耦合边界

相对任务能耗 V2 的正确定位是**测量模块**，不是本轮低层奖励调参的代名词。其计划值为：

```math
C_{\mathrm{move},t}=\sum_{j=1}^{8}|\tau_{j,t}\Delta q_{j,t}|,
\qquad
C_{\mathrm{hold},t}=\Delta t\sum_{j=1}^{8}\tau_{j,t}^2,
```

```math
E_{\mathrm{rel}}
=w_m\frac{\sum_t C_{\mathrm{move},t}}{S_m}
+w_h\frac{\sum_t C_{\mathrm{hold},t}}{S_h}.
```

当前仅有正/负/绝对关节机械功、转矩积分和动作量诊断；`Sm`、`Sh`、权重、`Eref` 与 `Tref`
尚未标定，故不能把现有数值称为电池焦耳或正式相对任务能耗。能源供应不设上限仅表示不模拟
电池耗尽、不因能量用完终止；它不移除执行器、摩擦、稳定性和时间上限。

本轮每次控制改动都可能改变实际能耗，因而应继续记录原始关节量；但在低层消融期间不应同时修改
能耗公式或加入能耗奖励，否则无法判断推进退化来自支撑修改还是效率压力。只有合格完成的轨迹才
能进入时间—能耗排序；摔倒、不动或少推进导致的较低机械功不得被解释为节能。

若以后仅更换策略、路线或人的意图，而 Ant XML、质量、gear、动作范围、控制频率和终止规则不变，
可以沿用同一版 V2 公式并离线重算。若增加关节、可动躯干、两条腿或吸附机构，则机器人结构、
执行器数量和持力机制都改变，必须建立新的能耗基线和协议版本，不能直接与当前 8 自由度分数比较。

## 7. 8 自由度与原地旋转的当前证据

seed 74803 在不增加关节和吸附力的情况下实现过一次空间到达，因此“8 自由度机械上完全无法
穿越该地图”已被该反例否定。但 `0/5` 安全合格也意味着尚不能证明 8 自由度足以实现稳定、稳健
的完整任务。现阶段更改结构会掩盖仍未排除的软件坐标、观察和支撑控制问题，故额外自由度继续作为
后置、独立 XML 消融，而不是当前默认修复。

当前代码也没有真正测试“躯干中心位置不变的原地旋转”。`set_external_curve_command` 要求速度
严格为正，并计算 `curvature=yaw_rate/speed`；固定目标控制器又将最大偏航速度限制为
`speed*maximum_abs_curvature`。因此速度为 0 时无法发出非零偏航命令。这首先是**命令接口不支持**，
不能据此推断四条腿的机械结构不能原地旋转。若后续验证原地旋转，应另建零线速度/非零偏航接口，
分别记录质心平移、航向变化、支撑和能耗，再决定是否需要结构升级。

## 8. 泛化计划：先冻结低层能力，再学习路线代价

封存地图已被反复查看和调试，只能作为开发地图，不能再称为未见测试地图。建议按地图 seed 冻结：

- 训练地图 64 个；
- 验证地图 16 个，只用于 checkpoint、阈值、尺度和规划参数选择；
- 未见测试地图 16 个，在全部规则冻结后一次性运行。

地形 seed、起终点 seed、低层训练 seed 和评价 seed 必须分开记录。同一地图的多个起终点是嵌套
任务，不等于多个独立新地图。高层第一版优先使用可审查的 A*/Dijkstra 或 Hybrid A*：先剔除
超过已验证能力坡度的边，再以距离、升降、转向和失败风险产生候选。V2 标定后再用合格执行数据
拟合边的时间与能耗；若训练权重条件化高层策略，它必须与搜索规划器使用相同地图划分、低层策略、
预算和失败规则比较。

推荐顺序为：

1. 在平地、标准坡台和连续曲面上冻结低层支撑/停止门槛；
2. 验证地形相对奖励，必要时再测试连续每足接触间隔或支撑数量目标；
3. 在同一 8 自由度结构下实现真正原地转向接口的独立 pilot；
4. 获得合格 travel、原地转向和终点两秒 dwell 轨迹后标定 V2；
5. 固定地图比较直接控制、距离最短路线和时间—能耗路线；
6. 冻结协议后，在未见地图集合上使用多个独立训练 run 评价泛化；
7. 只有上述软件/控制干预系统性失败，才新建 12 自由度结构文件进行公平消融。

## 9. 视频证据与复验入口

### 9.1 双视角视频

所有正式双视角视频均在左侧提供防沟谷遮挡的机器人跟踪镜头，右侧提供从终点侧看向起点的全景，
在三维地表留下实际轨迹；右上角保留平面地图，底部面板记录条件、seed、物理时间和支撑状态。

- **基线空间到达、但安全失败**：
  [`fixed_map_final_policy_seed_74803_dual_view_v1.mp4`](../artifacts/dev/fixed_map_reach_a_corrected_replication_v2_20260819/videos/seed_74803_dual_view_v1/fixed_map_final_policy_seed_74803_dual_view_v1.mp4)
  H.264，1280×720，20 fps，535 帧，26.75 s，覆盖 473.75 s 物理轨迹（20 倍速）；
  SHA-256 `cf42efe00a67607b85f07f217414dac8991569957911c0ee3c6d90f8ce8ce9d4`，完整解码通过。
- **135D 局部预瞄 final、少摔但低推进**：
  [`fixed_map_final_policy_seed_74803_dual_view_v1.mp4`](../artifacts/dev/fixed_map_local_preview_final_paired_direct_goal_v1_20260819/videos/seed_74803_local_preview_135d_dual_view_v1/fixed_map_final_policy_seed_74803_dual_view_v1.mp4)
  H.264，1280×720，20 fps，221 帧，11.05 s；正式 trace 前 800 步逐值一致，完整解码通过；
  SHA-256 `bbec35f541ea3ae832094ac91c77524af1368e258d15e252eb7f39e6840cdc1e`。
- **W4 等预算对照**：
  [`w4 dual-view v3`](../artifacts/dev/fixed_quad_terrain_v2_support_priority_w12_pilot_v1_20260819/seed_62803/videos/w4_seed_73802_dual_view_v3/fixed_map_final_policy_seed_73802_dual_view_v1.mp4)，
  SHA-256 `847a18e21b17e08a48752e8c13dab2eef5a3b4676748f19cff9f767ff3fea4ab`。
- **W12 支撑惩罚干预**：
  [`w12 dual-view v3`](../artifacts/dev/fixed_quad_terrain_v2_support_priority_w12_pilot_v1_20260819/seed_62803/videos/w12_seed_73802_dual_view_v3/fixed_map_final_policy_seed_73802_dual_view_v1.mp4)，
  SHA-256 `10390742cf2aa418a68bfc81c00619b48baa4d7d712da28f58f96ff46ba4a636`。
- **Terrain-frame grouped ablation（负结果）**：
  [`terrain-frame seed 73802`](../artifacts/dev/fixed_quad_terrain_v2_terrain_frame_reward_pilot_v1_20260819/seed_62803/videos/tf73802_v1/fixed_map_final_policy_seed_73802_dual_view_v1.mp4)，
  SHA-256 `f9bd900e2bfd7cdeedd18a9d2038130e5ec6d084d283dd0d0d0ff26f88a97959`；
  241/241 帧完整解码，视频 trace 与正式 3,600-step seed 73802 trace 的预声明字段逐值一致；
  最终帧红色状态条为 `TIME LIMIT / SAFETY FAIL`。独立 QA SHA-256 为
  `8dac91d95ca1720607093f44efdaae687fcba1cb13bdaddd97bcbd861648480d`。

W4/W12 均为 H.264、1280×720、20 fps、241 帧、12.05 s，只展示正式 3,600-step trace 的
前 900 步（45 s 物理时间，5 倍速）。其位置、地形高度、支撑、接触速度、悬空、奖励和终止字段
与正式 trace 前缀逐值一致；QA 位于
[`paired_support_video_qa.json`](../artifacts/dev/fixed_quad_terrain_v2_support_priority_w12_pilot_v1_20260819/seed_62803/videos/paired_support_video_qa.json)。
早期 v1/v2 已标记为中间版本，交付时只使用 v3。

### 9.2 配置、脚本与原始结果

| 工作流 | 配置/脚本 | 主要机器可读结果 |
|---|---|---|
| 时间—能耗意图 | `configs/time_energy_path_objective_v1_20260819.json` | `docs/TIME_ENERGY_PATH_OBJECTIVE_V1_CN.md` |
| 122D/135D 正式配对 | `scripts/evaluate_local_preview_final_paired_direct_goal.py` | `artifacts/dev/fixed_map_local_preview_final_paired_direct_goal_v1_20260819/aggregate_results.json` |
| 局部预瞄训练 | `scripts/run_fixed_goal_local_preview_pilot.py` | `artifacts/dev/fixed_quad_terrain_v2_local_preview_pilot_v1_20260819/seed_62802/` |
| W4/W12 配对 | `scripts/run_fixed_goal_support_priority_pilot.py` | `artifacts/dev/fixed_quad_terrain_v2_support_priority_w12_pilot_v1_20260819/seed_62803/comparison_summary.json` |
| Terrain-frame 配对 | `scripts/run_fixed_goal_terrain_frame_reward_pilot.py` | `artifacts/dev/fixed_quad_terrain_v2_terrain_frame_reward_pilot_v1_20260819/seed_62803/comparison_summary.json` |
| 16°路线 | `scripts/evaluate_fixed_map_waypoint_route.py` | `artifacts/dev/fixed_map_waypoint_route_v1_20260819/paired/aggregate_summary.json` |
| 双视角渲染 | `scripts/render_fixed_goal_dual_view_video.py` | 各视频目录的 manifest、trace、接触表和最终帧 |

每项正式复验应使用其冻结配置，不覆盖已有输出目录，并保留失败 seed、checkpoint、配置、原始逐步
trace、运行环境和 SHA-256。视频是行为诊断证据，不能替代数值评价。

## 10. 事实、推断与下一决策台账

| 类型 | 内容 | 证据/边界 | 下一决策 |
|---|---|---|---|
| 已确认事实 | 固定五 seed 为 1/5 空间到达、0/5 合格，成功轮悬空 68.07% | 修正回放和完整 trace | 继续把空间到达与稳定完成分开报告 |
| 已确认事实 | 校正后持续足端滑移事件为 0/10 episode | 特定阈值、宽限和足端范围 | 保留原始指标，同时扩展身体/小腿滑移诊断 |
| 支持性推断 | 全图摩擦不足不是目前主要根因 | `mu=1` 静态边界、滑移校正、长期无足端支撑 | 正式摩擦保持不变，不用提高摩擦掩盖支撑问题 |
| 已确认事实 | 135D final 在本轮五 seed 观察中未摔倒，但平均推进观察值明显低于 122D | 相同固定地图/seed 配对；仅一个训练 run，不作统计显著性推断 | 不保留 final 为改进；调查 stage-1 与后期退化 |
| 已确认事实 | W12 未过预声明支撑/推进保留门槛 | 三个配对评价 seed、一个训练 seed | 停止单纯权重扫描 |
| 已确认事实 | waypoint 路线可提高部分条件推进，但所有条件仍 0/5 合格 | 固定地图探索；候选非全局最优 | 修复低层后再比较路线成本 |
| 已确认事实 | 当前接口不能发出零速非零偏航命令 | `speed>0` 与 `yaw_rate/speed` 约束 | 若需要，另建原地转向接口 pilot |
| 已确认事实 | terrain-frame grouped ablation 的悬空、推进和摔倒三项保留门槛全部失败 | 同源、同预算 W4 对照；三个验证 seed | 不保留新 checkpoint；若继续只做单一 foot-clearance 消融 |
| 尚未证实 | 8 自由度不足以完成稳定复杂地形运动 | 一次空间到达反驳“完全不可行”，但 0/5 合格 | 完成地形相对/支撑干预后再决定 12 自由度消融 |
| 尚未实现 | V2 合并相对任务能耗和时间—能耗最优路径 | 当前机械功只是诊断；无合格轨迹 | 获得合格 travel/turn/dwell 后标定并冻结 |

## 11. 当前工作流状态

```text
artifact: approved fixed terrain and dual-view evidence
engineering: reproducible / full-decode-validated for named videos
scientific: fixed-map diagnostic only
release: locally validated bundle; remote branch and commit are recorded in the final handoff

artifact: low-level fixed-map locomotion
engineering: runnable with traceable checkpoints and paired evaluations
scientific: safety and robustness unresolved; qualified completion 0/5
next_gate: isolate foot-clearance/contact-timing support mechanisms; do not continue grouped reward-weight scans

artifact: Time-Energy Path Objective V1
engineering: schema and decision protocol specified
scientific: V2 scale and route cost not calibrated
next_gate: obtain valid trajectories before Pareto evaluation

artifact: unseen-map generalisation
engineering: proposed split only
scientific: not tested
next_gate: freeze low-level policy, map seeds, planner and evaluation rules
```

terrain-frame 结果已经归档，全仓库回归测试与发布清单审计均已完成。本报告是固定地图开发阶段的正式进展记录，
但由于所有干预的安全合格完成率仍为零，它不是对最终算法性能或未见地图泛化能力的科学结论。
