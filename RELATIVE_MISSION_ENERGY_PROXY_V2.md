# Relative Mission Energy Proxy V2

> 项目：Ant-v5 四足机器人随机连续地形导航（V3）
> 状态：当前首选设计入口；接口可实施；数值系数尚未标定
> 日期：2026-08-18
> 取代：`RELATIVE_MISSION_ENERGY_PROXY_V1.md`
> 核心变化：取消有限电池预算，采用能源供应不设上限、成功与安全优先、时间—能耗二级优化

## 1. 项目目标与开发优先级

机器人只接收“前进速度＋身体旋转/偏航速度”意图，不主动横移、不后退、
不跳跃。最终任务是在随机连续地形上从起点到任意终点，并在满足完成、
稳定和安全条件后，比较时间与相对任务能耗的折中；正式测试应包含未见地图。

当前开发顺序：

1. 保证平地、斜坡和平滑曲面上的稳定、可控前进和转向；
2. 完成复杂地形上的起点到终点导航、可行坡度判断、时间与能耗记录；
3. 最后优化步态自然程度、对角腿交替和视觉协调性。

当前阶段不因为接触占空比不完全相等、没有标准生物学节律或动作视觉上不够
自然而单独否决模型。但是以下情况属于必须立即处理的严重不协调：

- 摔倒、翻转或持续大幅倾斜；
- 四足同时离地，即违反“不跳跃”约束；
- 足端或腿部持续拖地、明显自碰撞；
- 关节长期饱和或高频剧烈抖动；
- 某条腿长期失效并损害稳定、方向跟踪或地形通过能力；
- 在已验证可行坡度内持续滑退、失控加速或不能执行停止指令。

因此，当前保留最低协调性安全门槛，但暂不强制特定自然步态。

## 2. V2 的能源假设

V2 所称“能耗不受限制”准确表示：

- 仿真不设置电池容量 `B0`；
- 不维护剩余电量状态 `Bt`；
- 不因能量耗尽终止 episode；
- 路径规划器不需要观察剩余电量；
- 每一步和整轮任务的相对能耗仍持续累计并用于优化与评价。

它不表示机器人具有无限功率或无限机械能力。以下限制保持固定：

- actuator gear、控制范围和关节限制；
- 动作范围和动作限制器；
- 机器人质量、尺寸和刚体结构；
- 控制频率、MuJoCo 步长和 frame skip；
- episode 最大时长；
- 地形可通行范围和固定摩擦力；
- 摔倒、腾空、严重滑动等安全条件。

因此，即使能源供应不设上限，机器人也不可能仅靠“使用更多能量”爬上任意
陡坡。动力学、摩擦、足端几何和稳定性仍会形成实际能力上限。

## 3. 模块边界

```text
前进速度＋偏航速度
        ↓
运动策略输出 8 个关节控制量
        ↓
MuJoCo 执行并产生实际状态
        ↓
能耗模块读取关节、地形和接触数据
        ↓
输出逐步原始量、相对能耗和整轮摘要
        ↓
训练器或路径规划器决定如何使用能耗信号
```

能耗模块只负责测量和输出，不负责：

- 决定 PPO 或其他算法的完整奖励函数；
- 决定机器人是否绕行或掉头；
- 强制形成某一种自然步态；
- 把相对分数宣称为真实电池焦耳；
- 用低能耗覆盖任务失败或安全违规。

若未来把能耗接入训练，只将 `relative_energy_step` 交给上层训练器。到达、
时间、稳定和安全仍保持独立分量，日志中不得只保存合计回报。

## 4. 固定比较条件

相对能耗只在以下条件不变时具有直接可比性：

- Ant XML、质量、尺寸和刚体结构；
- 八个 actuator 的 gear、控制范围和关节限制；
- 动作范围、动作限制器和控制频率；
- MuJoCo 仿真步长和 frame skip；
- 地面摩擦系数与 MuJoCo 版本；
- episode 起止条件、最大时长和终点站立时长；
- 能耗归一化尺度与权重。

当前所有地形的 ground geom 摩擦固定为 `[1.0, 0.5, 0.5]`，`condim=3`。
正式比较还必须固定机器人 geom 和接触混合规则。该分数不用于直接比较结构、
质量或执行器参数不同的机器人。

## 5. 实际执行能耗代理

### 5.1 关节运动项

对执行关节 `j` 和控制步 `t`：

```math
\Delta q_{j,t}=q_{j,t+1}-q_{j,t}
```

```math
C_{\mathrm{move},t}
=
\sum_{j=1}^{8}|\tau_{j,t}\Delta q_{j,t}|
```

其中 `q` 是实际关节角，`tau` 是 MuJoCo 的 actuator joint torque。不能用
“前进、左转、右转”等高层标签代替，也不能把 `applied_action` 直接解释成
关节角度或焦耳。

同时保存正机械功和制动负功：

```math
C^+_t=\sum_j\max(\tau_{j,t}\dot q_{j,t},0)\Delta t
```

```math
C^-_t=\sum_j\max(-\tau_{j,t}\dot q_{j,t},0)\Delta t
```

V2 不模拟再生制动，因此负功只作为制动/耗散诊断，不能直接从总能耗中减去。

### 5.2 静态持力项

理想机械功在关节静止时为零，但真实机器人维持站立仍需要转矩。使用：

```math
C_{\mathrm{hold},t}
=
\Delta t\sum_{j=1}^{8}\tau_{j,t}^{2}
```

该项区分低转矩站立、高转矩支撑和斜坡艰难持姿，但不是经过电机电阻标定的
真实铜耗。

### 5.3 归一化相对任务能耗

```math
E_{\mathrm{rel}}
=
w_m\frac{\sum_t C_{\mathrm{move},t}}{S_m}
+
w_h\frac{\sum_t C_{\mathrm{hold},t}}{S_h}
```

- `Sm`、`Sh` 来自同一 Ant 的平地稳定直行、原地旋转和站立标定；
- `wm`、`wh` 在验证集上确定并在测试前冻结；
- 所有原始分量必须单独保存，以便离线重新计分；
- 当前系数尚未标定，不允许虚构物理意义；
- 输出名称固定为 `relative mission energy score`，方向为越低越好；
- 不能标成电池能量或真实焦耳。

各刚体外力、足端接触力和冲击力单独记录，用于稳定和滑动诊断，不能简单与
关节能耗相加，以免重复计量。

## 6. 地形梯度与候选路径预测

实际执行能耗由关节数据计算。地形梯度只用于执行前预测候选路径成本，避免
把上坡高度与已经增加的关节做功重复计入实际测量分数。

设地形为 `z=h(x,y)`，实际水平移动方向为：

```math
\hat{\boldsymbol v}_t
=
\frac{(\dot x_t,\dot y_t)}{\sqrt{\dot x_t^2+\dot y_t^2}}
```

沿行进方向的局部坡度为：

```math
s_t=\nabla h(x_t,y_t)\cdot\hat{\boldsymbol v}_t
```

- `s_t>0`：上坡；
- `s_t<0`：下坡；
- `s_t≈0`：平地或近似沿等高线移动；
- 速度接近零时不定义移动坡向，坡度移动成本为零，但持力继续累计。

使用实际 `(dx/dt,dy/dt)`，不是只使用身体朝向。如果出现斜行或滑动，实际
速度仍能反映真实路径。

```math
H^+=\sum_t\max(\Delta h_t,0),
\qquad
H^-=\sum_t\max(-\Delta h_t,0)
```

候选路径预测为：

```math
\widehat E_{\mathrm{path}}
=
k_LL+k_R|\Delta\psi|_{\mathrm{total}}+k_UH^++k_DH^-
```

其中 `L` 为路径长度，累计旋转量包括原地旋转和曲线转向。系数由已执行的
合格轨迹拟合或标定。当前不考虑能量回收，第一版只冻结：

```math
0<k_D<k_U
```

具体比例仍待标定。

## 7. 终点两秒站立

```math
E_{\mathrm{mission}}
=
E_{\mathrm{travel}}+E_{\mathrm{dwell}}
```

首次进入终点区域后：

1. 记录到达时间；
2. 平滑降低前进和旋转指令；
3. 继续仿真并保持站立 2.0 s；
4. 计算这两秒内的关节运动和持力；
5. 摔倒、严重滑动或离开终点区域则记为保持失败；
6. 完成两秒站立后结束 episode。

主导航实验的终点优先生成在局部平坦区域。斜坡终点站立先作为独立能力测试。

## 8. 安全诊断与有效性

### 8.1 逐足滑动

对每只脚独立判断：

```math
c_i=1
\quad\text{and}\quad
\|\boldsymbol v_{i,\mathrm{tan}}\|>v_{\mathrm{slip}}
```

连续满足冻结步数后记为滑动。记录脚编号、首次时间、持续时间、累计滑动
距离、接触脚数、法向力和切向力。滑动第一版是安全诊断和失败条件，不把
未经标定的滑动耗散直接加入主能耗。

可额外记录：

```math
C_{\mathrm{slip,diag}}
=
\int|\boldsymbol F_{\mathrm{tan}}\cdot
\boldsymbol v_{\mathrm{tan}}|\,dt
```

### 8.2 四足腾空

四只脚同时没有接触地面时：

- 记录首次时间、全部时间戳、持续时间和对应动作；
- 标记为违反“不跳跃”约束；
- 调试模式可以继续运行以便诊断；
- 正式评价中该 episode 为无效，不得参与节能排序；
- 不能让跳跃、摔倒或提前结束通过缩短任务时间取得低能耗优势。

### 8.3 有效 episode

```math
V=
\mathbb I(
\text{goal reached}
\land\text{2 s dwell completed}
\land\text{within horizon}
\land\text{no fall}
\land\text{no airborne violation}
\land\text{no disqualifying slip}
)
```

只有 `V=1` 的轨迹可以进入时间—能耗优化排序。失败轨迹继续保存失败前能耗，
但低失败能耗不表示节能性能更好。

## 9. 能源供应不设上限时的优化顺序

V2 使用分层/约束优先规则，不直接把所有量无条件压成一个分数：

1. 首先比较有效任务完成率和安全违规率；
2. 只保留达到冻结完成率门槛且满足安全规则的候选策略；
3. 只在合格策略的有效轨迹之间比较时间和相对任务能耗。

对有效轨迹定义：

```math
J_{\mathrm{valid}}
=
w_E\frac{E_{\mathrm{mission}}}{E_{\mathrm{ref}}}
+
w_T\frac{T_{\mathrm{mission}}}{T_{\mathrm{ref}}}
```

其中：

- `Jvalid` 越低越好；
- `wE>=0`、`wT>=0`，为便于解释可约束 `wE+wT=1`；
- `Eref`、`Tref` 只能使用训练/验证数据确定；
- 所有尺度和权重在最终测试前冻结；
- 能耗和时间原始量仍必须分别报告，不能只报告 `Jvalid`。

主要效率量使用整轮总能耗，而不是平均功率。只优化平均功率可能鼓励机器人
无限放慢；总能耗包含行走时间与静态持力，可以使无意义拖延继续付出代价。

## 10. 能耗训练信号及退化防护

如上层训练器需要逐步能耗信号，可使用：

```math
r_{E,t}=-\lambda_E\widetilde C_{E,t}
```

其中 `C~E,t` 是用训练集基准尺度归一化的每步相对能耗。能耗模块只输出该
信号，不决定完整 PPO 奖励。禁止使用在零能耗附近奇异的正奖励，例如
`1/(E+epsilon)`。

必须检查两类相反退化：

### 10.1 能耗被忽略

当 `lambdaE` 太小或能耗尺度远小于其他奖励时，策略可能与无能耗基线没有
实质差别。诊断包括：

- 总能耗和单位成功任务能耗没有下降；
- 关节动作、转矩和持力分布没有变化；
- 能耗项对总回报的占比接近零；
- 改变 `lambdaE` 后策略行为与能耗不敏感。

### 10.2 过度追求低能耗

当 `lambdaE` 太大时，策略可能：

- 原地不动或极慢移动；
- 拒绝转弯、爬坡或接近目标；
- 故意摔倒或主动触发提前结束；
- 以腾空、拖腿或减少有效支撑换取表面低能耗；
- 牺牲任务成功率、稳定性、时间或路径质量。

这些行为由有效性硬门槛拦截。若训练算法必须使用单一标量回报，完整奖励仍
必须保证失败不可能仅因少消耗能量而优于有效完成；具体终止代价和尺度要在
固定 horizon 与动作范围下标定，不能凭感觉设定。

## 11. 权重选择与实验设计

在训练前预先规定候选 `lambdaE` 或 `wE` 集合，不在 test 地图上调参。至少
保留 `lambdaE=0` 的无能耗基线，并保证各条件：

- 使用相同算法、网络、观察和动作接口；
- 使用相同训练步数、训练地形支持与调参预算；
- 使用配对训练 seed 和相同验证地图；
- 分别报告有效完成率、安全违规率、任务时间和能耗原始量；
- 不以“最佳单 seed”代替跨独立训练 run 的结果。

选择顺序：

1. 排除没有达到完成/安全门槛的候选；
2. 在验证集绘制时间—能耗 Pareto 前沿；
3. 按预先说明的偏好选择一个折中点；
4. 冻结 checkpoint、权重、尺度、失败规则和测试 seed；
5. 在未见 test 地图上只运行一次正式比较流程。

候选权重数值、有效完成率门槛和正式训练 seed 数目前仍待 pilot 确定，不在
缺少数据时虚构。

## 12. 不可行坡度与控制失败

先通过独立定角度坡台测试确定最大可行范围：

- 超出范围：路线不可通行，不判为节能策略失败；
- 范围内仍无法前进：运动控制、机械能力或稳定性失败；
- 底层只报告停滞、滑退或失败；
- 是否停止或重新规划由上层安全模块与路径规划器决定。

能源供应不设上限不会改变上述分类。

## 13. 软件接口

### 每步输入

```text
dt
joint_position[8]
joint_velocity[8]
actuator_joint_torque[8]
root_position_xy
terrain_height
terrain_gradient_xy
foot_contact_mask[4]
foot_normal_force[4]
foot_tangential_force[4]
foot_contact_tangential_velocity[4]
goal_reached
stable
```

### 每步输出

```text
joint_angle_change[8]
positive_mechanical_work[8]
negative_mechanical_work_abs[8]
holding_effort[8]
terrain_height_change
slope_class
slip_mask[4]
airborne_violation
relative_energy_step
```

### Episode 输出

```text
travel_energy_score
dwell_energy_score
mission_energy_score
normalised_mission_energy_score
per_joint_energy_contribution[8]
positive_mechanical_work_total
negative_mechanical_work_abs_total
holding_effort_total
cumulative_ascent
cumulative_descent
cumulative_turning_angle
arrival_time
dwell_completed
airborne_timestamps
per_foot_slip_time
per_foot_slip_distance
valid_episode
failure_reason
valid_time_energy_cost
```

V2 不输出 `battery_remaining`、`battery_fraction` 或 `energy_depleted`。

## 14. 验证检查

在能耗分数用于训练或路径规划前，至少完成：

1. `tau=0` 且关节不动时，运动项和持力项为零；
2. `tau!=0` 且关节不动时，运动项接近零、持力项大于零；
3. 站立时间加倍时，持力分数近似加倍；
4. 原地旋转具有正能耗，即使位移和高度变化为零；
5. 相同路线和 seed 的结果可复现；
6. 减小仿真步长后积分结果趋于稳定；
7. 上坡预测成本通常高于同长度平地，下坡具有非零制动成本；
8. 原始分量始终单独保存，允许离线重新计分；
9. 能耗模块关闭时，策略 observation、reward、termination 和动作不变；
10. 原地不动不能获得有效低能耗成绩；
11. 摔倒、腾空或提前结束不能优于成功完成；
12. 较小能耗权重不会被其他回报完全淹没；
13. 较大能耗权重不会破坏完成率、稳定和安全；
14. 0.7/0.8 m/s 等速度变化不能被误当作能耗机制改进；
15. test 地图不参与尺度、权重、checkpoint 或失败规则选择。

## 15. 当前证据、限制与下一道门槛

当前封存复杂地形的正式五 seed 配对显示：122D 基线为 `1/5` 空间到达、`0/5`
安全合格，平均四足端同时无接触比例约 `69.85%`；135D 局部预瞄 final 为
`0/5` 空间到达、`0/5` 安全合格，平均比例约 `66.88%`。后续 W12 支撑权重和
terrain-frame grouped ablation 均未通过预声明的悬空/推进保留门槛。因此当前
策略仍不满足 `valid_episode`，不能进入正式节能排序。较低动作平方、转矩积分
或机械功可能由少接触、少推进或异常腾空造成，不能单独解释为能耗优化成功。
证据见 `docs/FIXED_TERRAIN_PROGRESS_REPORT_20260819_0730_CN.md` 与
`docs/FIXED_MAP_FAILURE_MECHANISM_AUDIT_20260819_CN.md`。

```text
artifact: Relative Mission Energy Proxy V2
stage: design / pilot preparation
engineering: raw torque, velocity, mechanical-work and contact diagnostics available; V2 combined objective not integrated
scientific: energy scales, weights, success threshold and training coefficient set unresolved
evidence: formulas and interfaces traceable; physical battery calibration intentionally out of scope
release: internal design record; current preferred entry
next_gate: repair severe airborne behaviour, then use valid flat travel, in-place turn and 2 s dwell rollouts to calibrate Sm, Sh, Eref and Tref
```

自然步态外观仍不是下一道门槛。下一道门槛是先获得满足最低安全条件的稳定
平动策略，再标定能耗代理并进入斜面、凸/凹曲面和混合地形的时间—能耗
Pareto pilot。
