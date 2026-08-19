# ProxyGap 最终 Report 与 Presentation 交付蓝图

日期：2026-08-19（Europe/London）<br>
证据截点：PAIR0 L2b V3、坡度边界与视频、平地转向诊断、turn-balance V2 smoke、V2/V4 pre-training failure records、V5 final paired result、post-seal direct-goal map failure，以及最终 seal。<br>
状态：**交付设计稿，不是新增实验结果，不创建 PPTX，也不改变模型、奖励、摩擦、能耗或训练状态。**

若证据截点之后产生新的模型或评估，不得直接把新数字加入正文。必须先完成 raw rows、配置、checkpoint、manifest、SHA-256、失败记录和独立复核，再更新本文的证据状态表。

## 1. 交付定位

### 1.1 Communication job

> 答辩结束时，本科项目导师和评委应当理解：本项目没有证明随机复杂地形上的可靠最优导航，但通过可复现的故障隔离，识别并修复了 MuJoCo heightfield–capsule 接触表示中的主要支撑异常，使单个 PPO–Ant 策略在冻结标准坡面上取得条件性、可审计的运动能力；剩余的左右转向不对称和全局规划缺口被明确量化，而不是由一次成功视频掩盖。

### 1.2 推荐标题

中文：

> **从接触语义修复到坡面运动：PPO 四足机器人连续地形任务的可审计仿真研究**

英文：

> **From Contact Semantics to Slope Locomotion: An Auditable PPO–Ant Study on Continuous Terrain**

副标题建议：

> **Stable support improved under frozen conditions; reliable navigation remains unresolved**

### 1.3 中央结论

最终 report 和 presentation 应共同维护以下三层结论：

1. **已经建立的事实**：显式 floor–distal-ankle `PAIR0` 接触对与短预算续训，在一个训练 seed、固定标准场景和冻结评估 seeds 下，大幅降低四足端同时无接触，并保留平均推进；最终策略在所测上坡 4°、8°、12°通过预声明门槛。
2. **条件性解释**：接触 margin 组合是此前 heightfield 支撑异常的主要机制贡献者；这不等于 MuJoCo 存在缺陷，也不证明 `margin=0` 是普适物理真值。
3. **未解决结论**：左右转向仍显著不对称，固定地图可靠到达、未见地图泛化、路径最优性、正式相对任务能耗和自然步态均未建立。

不得把项目总结为“机器人已经学会在随机地图上找到最优路径”。更准确的总结是：**项目完成了从不可解释的复杂地图失败，到可审计的接触修复、坡度能力边界和转向瓶颈定位。**

## 2. 最终证据状态表

| Workstream | 直接证据 | 当前状态 | 允许的结论 | 禁止的结论 |
|---|---|---|---|---|
| 历史固定地图 | 五个评价 seed 中 `1/5` 空间到达、`0/5` 稳定安全合格、`2/5` 摔倒；空间到达轮在终点保持阶段仍大量失去足端支撑 | development diagnostic | 至少存在一条空间可达轨迹；旧系统不可靠 | 已完成固定地图导航；该路线最优或安全 |
| heightfield 接触机制 | 8 s 开环中，129/257 hfield 的四足端同时无接触由 `73.625%/77.500%` 降至 `21.500%/21.875%`，接近 plane margin=0 的 `22.125%` | controlled mechanism diagnostic | 默认双方 margin 组合是本设置下主要机制贡献者 | MuJoCo bug；margin=0 在所有地形都物理正确 |
| PAIR0 L2b V3 | held-out 每条件 20 episodes；zero-foot `0.24042→0.03208`，mean support `0.35335→1.34362`，mean progress ratio `1.0512`；PAIR0 最终安全检查通过 | final verified diagnostic，**一个训练 seed** | 在冻结标准场景与 seeds 下，支撑改善且平均推进得到保留 | 多训练 seed 稳健性；随机地图泛化；节能 |
| 标准坡度边界 | 11 场景 × 5 seeds = 55 episodes；上坡 4°/8°/12°通过，16°/20°失败；下坡结果非单调 | formally evaluated for one frozen policy | 12°是连续被测通过下界；16°是首个被测上坡失败点 | 物理最大爬坡角为12°；连续稳定下坡至16° |
| 坡面视频 | 上坡12° seed 94153、下坡16° seed 94137；均 32 s、1280×720、20 fps、640 frames；逐步 trace 和全帧 decode 已核验 | release-ready qualitative evidence | 可展示对应固定 episode 的姿态、轨迹和地势 | 视频替代多 seed 定量评价；下坡连续能力已证明 |
| 平地转向 | 9 条件 × 5 seeds = 45 episodes；安全门全部通过，但左转明显弱于右转 | final diagnostic；fixed-map readiness HOLD | 闭环转向存在方向不对称 | 符号反了；偏差必然只来自神经网络 |
| turn-balance V2 smoke | C0/C1 各 8,192 steps；8 workers、种子和对称命令暴露核验通过；无 checkpoint | engineering smoke only | wrapper、续训加载和命令曝光链可执行 | 转向已经改善；smoke 是科学结果 |
| turn-balance formal V2 | 在 `prepare_standard_slope_scenes` 因 Windows 260-character path 失败；训练未开始、无 checkpoint | failed before training; non-evaluable | 保存了可复核的工程失败证据 | 策略训练失败或转向方法无效 |
| turn-balance formal V4 | 短路径越过路径故障后，runner 因 audit 字典缺少 `passed` 字段而把4个实际合格的场景接触审计汇总为失败；训练未开始、无 checkpoint | failed before training; non-evaluable | 这是训练前接口字段故障，不是策略结果 | PAIR0 场景真实接触合同失败；C1 科学门失败 |
| turn-balance formal V5 | C0/C1 各续训65,536步；每分支45个转向+20个坡面episode；两者坡面PASS、转向FAIL；C1左0.10 ratio −0.288、右0.10 ratio 1.206 | final evaluable paired case study；一个训练 seed；hard stop | 对称命令曝光未在该预算和seed下消除左右偏置；Stage B按预声明规则HOLD | 多训练seed稳健性；方法在所有设置下必然无效；固定地图ready |
| 时间—能耗最优路径 | 三种 mission profiles、词典序安全门和 Pareto 规则已设计；合并相对能耗尚未标定或接入 | protocol design only | 可说明未来如何在合格路线间比较时间和能耗 | 当前已有最优路径；现有机械功等于电池能耗 |

## 3. Report 结构建议

建议正文约 7,000–9,000 中文字或相当英文篇幅；若课程 brief 给出其他字数，以 brief 为准并按比例压缩。摘要、参考文献和附录是否计入字数须在提交前确认。

### 摘要（约 250–300 字）

采用四句结构：

1. 问题：复杂地形上的视觉运动不等于稳定、安全和可复现的任务完成。
2. 方法：对 heightfield 接触语义、足端支撑、坡面推进和转向进行分层诊断，并在冻结摩擦与 8-DOF 结构下进行 PAIR0 续训。
3. 结果：报告 zero-foot、support、progress、坡度和转向的核心数值；明确只有一个训练 seed。
4. 结论：标准坡面能力得到条件性改善，但可靠导航、泛化和时间—能耗最优性仍未建立。

摘要不得出现“maximum climb angle”“generalises to random maps”“energy-efficient”或“solved navigation”。

### 第1章 Introduction：研究的是“可审计运动”，不是好看的视频（约 700 字）

- 解释四足机器人连续地形任务及工程意义。
- 区分 optimizer-facing reward、外部稳定/安全指标和人类任务意图。
- 提出三个可回答的研究问题：
  - RQ1：heightfield 接触表示是否解释此前异常失支撑？
  - RQ2：PAIR0 适配能否在冻结条件下改善支撑，同时保留坡面推进？
  - RQ3：修复支撑后，什么仍阻碍固定地图可靠导航？
- 将时间—能耗路径优化明确列为设计延伸，不冒充已完成的 RQ4。

### 第2章 Background and Related Work（约 800 字）

- Ant-v5 的状态、8维动作和奖励组成。
- PPO 的 clipped surrogate optimisation，只介绍支持方法理解的内容。
- reward misspecification：训练奖励与项目评价指标必须分离。
- 四足运动中的 command tracking、身体姿态、支撑和动作平滑；避免引用文献替代本项目阈值标定。
- MuJoCo contact margin、pair 和 heightfield 仅用于解释建模机制；所有软件语义须引用官方文档或源码版本。

主要外部文献入口使用 `docs/LITERATURE_EVIDENCE_REGISTER_V1_20260816.md`，最终提交前逐条复核题名、作者、年份、DOI/URL 和访问日期，采用 Leeds Harvard author–date。

### 第3章 System and Task Definition（约 700 字）

- Gymnasium MuJoCo Ant-v5、PPO、8个驱动关节、135D 最终局部观察（122D + 13D local terrain preview）。
- 机器人只接收前进速度与偏航/曲率命令，不接收横移或后退命令。
- 固定摩擦 `[1.0,0.5,0.5]`、`condim=3`；没有摩擦随机化或吸附力。
- 明确区分：
  - 低层运动：如何执行速度和转向；
  - 高层规划：走哪条路线；
  - 安全模块：摔倒、足端支撑、非足端接触、持续滑动；
  - 能耗模块：measurement-only diagnostics。
- 用一幅系统图标出哪些模块已实现，哪些只是未来设计；不得把 planned global planner 画成已运行模块。

### 第4章 Methodology and Evaluation Contract（约 1,000 字）

- 说明层级：training run → checkpoint → evaluation episode → control step → physics substep。
- 训练 seed 是策略层独立单位；同一策略的多个 evaluation episodes 是嵌套观测。
- 给出核心指标的操作定义：
  - mean best progress；
  - full-interval zero-foot fraction；
  - mean support count；
  - fall、torso-ground、sustained non-foot；
  - force-qualified corrected sustained slip；
  - target/actual cumulative yaw change 和 yaw ratio；
  - squared action、`∫|τ|dt`、positive/absolute mechanical work。
- 解释为什么旧“单步接触速度超限”不能直接叫持续滑动。
- 说明 checkpoint、seeds、终止、摩擦、episode horizon、门槛和失败处理如何预声明。

### 第5章 Contact-Mechanism Diagnosis and PAIR0 Intervention（约 900 字）

按因果链写作：

`异常失支撑 → plane/hfield matched probes → margin 单因素矩阵 → 显式四对 PAIR0 → paired continuation`

- 先报告受控开环与静态诊断，避免直接从视频猜测摩擦不足。
- 解释保留 geom margin `0.01 m`，只为 floor–四个 distal ankle 设置显式 pair `margin=0, gap=0`。
- 强调这是接触表示干预，不是提高摩擦、增加关节或加入吸附力。
- 说明 L2b V3 两条件从各自原 L2 endpoint 继续 65,536 timesteps，最终只评价固定 checkpoint `2,727,936`。

### 第6章 Results（约 1,600 字）

#### 6.1 PAIR0 支撑与推进

- 同时展示绝对量和差值，不只写“提高了”。
- zero-foot：`24.042% → 3.208%`；mean support：`0.353 → 1.344`；mean progress ratio：`1.051`。
- 显示 16,384/32,768/49,152/65,536 timesteps 的进展与 zero-foot，说明学习曲线非单调、最终 checkpoint 是预声明选择。
- 能耗四代理分开报告；结论是 mixed evidence，而不是“更节能”。

#### 6.2 坡度能力边界

- 上坡 4°、8°、12° PASS；16°因推进不足 FAIL；20°因推进与 zero-foot FAIL。
- 下坡 4°、12°、20°仅因固定推进门槛 FAIL，8°和16° PASS；明确非单调。
- 55/55 episodes 完整，均无 fall、torso-ground、sustained non-foot 或 corrected sustained slip event。
- 将 12°表述为连续被测通过下界，不写成最大角度。

#### 6.3 定性视频证据

- 使用预先固定 seed 的上坡12°与下坡16°双视角视频。
- 视频用于展示动作、遮挡处理、3D地势和轨迹；判定仍来自 CSV/JSON。

### 第7章 What Remains Unsolved: Turning and Map Navigation（约 1,000 字）

- 报告平地45轮安全通过后仍出现的方向不对称：
  - `|κ|=0.10`：left ratio `−0.203`、`0/5`同号；right ratio `1.120`、`5/5`同号；
  - `|κ|=0.20`：left `0.234`、`4/5`；right `0.874`、`5/5`；
  - `|κ|=0.35`：left `−0.016`、`3/5`；right `0.329`、`4/5`。
- 直行自身平均 yaw change 为 `−0.511 rad`，支持存在负向闭环漂移的观察。
- 不把偏差单独归因于 policy；可能机制包括策略权重、动作/关节排列、接触闭环和初态。
- 历史固定地图结果只作问题背景：`1/5`空间到达、`0/5`合格。
- 说明 turn-balance smoke 只证明工程链；V2/V4 formal 都在训练前失败。随后 V5 只修复缺失 `passed` 字段误判，正式生成 C0/C1 final checkpoints；两者坡面门通过但转向门失败，因此都不晋级，原 PAIR0 source 被保留。
- 报告 C1 的方向性证据：straight yaw −0.658 rad；left/right 0.10 yaw ratio −0.288/1.206；left/right 0.20 为0.171/0.658。所有转向安全子门通过不能替代转向跟踪门。

### 第8章 Time–Energy Path Objective: Designed but Not Yet Evaluated（约 550 字）

- 先给词典序门：成功与安全 → 时间/能耗 Pareto → mission profile。
- 展示 `time_prioritised 0.70/0.30`、`balanced_demo 0.50/0.50`、`energy_prioritised 0.30/0.70`。
- 明确能量供应“不设上限”只表示不模拟电池耗尽；低能耗不能奖励不动、早失败或少推进。
- 现有 energy quantities 是代理/机械量，不是电池焦耳；合并 V2 尚未标定。

### 第9章 Discussion and Limitations（约 900 字）

- 讨论为何 contact semantics 比直接提高摩擦或加关节更符合本轮证据。
- 解释支撑改善为何没有自动解决转向和全局导航：低层支撑、方向控制和全局选路是不同子问题。
- 对照至少两个替代解释或反证：
  - 若摩擦不足是主因，提高接触质量后应仍出现 force-qualified持续滑动；本轮没有观察到，但指标覆盖有限。
  - 若8-DOF在机械上完全不可行，则不应出现任何空间到达或标准坡面推进；已有反例否定“完全不可能”，但未证明充分。
- 逐条披露第7节列出的所有限制。

### 第10章 Conclusion and Future Work（约 350 字）

只回答三个研究问题：

- RQ1：在受控诊断中，margin 组合解释了大部分 heightfield 支撑差异。
- RQ2：PAIR0 在一个训练 seed 的冻结标准场景中改善支撑并保留推进，上坡连续被测通过至12°。
- RQ3：转向不对称仍阻塞可靠地图导航；最后的平衡续训产生了可评价负结果，但没有通过转向门。A*／waypoint Stage B 因预声明前置门失败而未启动。

未来工作只保留三项、按依赖顺序：

1. 当前优化 hard stop 后不再追加本项目模型训练；V2/V4 失败 root 和 V5 正式负结果均原样保留。
2. 若后续另立新研究，应先在独立多训练 seed 下重新设计并验证转向干预；只有通过冻结转向门后，才能接入可审查的 A*/Hybrid A* 高层规划与未见地图划分。
3. 只有产生合格路线后，才标定相对任务能耗并比较 Pareto 前沿；自然步态和结构升级保持后置。

### Appendices

- A：完整配置、软件版本、硬件与命令。
- B：全部 seed-level rows 与失败/排除记录。
- C：指标公式、阈值和敏感性分析。
- D：manifest、checkpoint、视频与核心文件 SHA-256。
- E：负结果时间线，包括 W12、terrain-frame、turn-balance V2/V4 工程阻断与 V5 正式转向失败。

## 4. Report 图表与数据表清单

### 4.1 主文图

| 编号 | 推荐 takeaway caption | 视觉形式 | 数据来源 | 重要语义 |
|---|---|---|---|---|
| Figure 1 | The system separates local locomotion, safety measurement and planned global routing | 单一水平流程图；implemented 实线，planned global planner 灰色虚线 | `src/proxygap/fixed_goal_terrain.py`、`src/proxygap/paired_turn_balance.py`、`docs/TIME_ENERGY_PATH_OBJECTIVE_V1_CN.md` | 不得把全局规划画成已实现 |
| Figure 2 | Contact margins, rather than friction changes, explained most of the flat-heightfield support gap | 129/257 hfield 与 plane 的 before/after zero-foot grouped bars；右侧小型 pair 示意 | `docs/HEIGHTFIELD_CAPSULE_MARGIN_ADDENDUM_20260819_CN.md` 和对应 CSV | 误差线不适用确定性开环；标为 mechanism diagnostic |
| Figure 3 | PAIR0 reduced zero-foot exposure while retaining mean progress | 双轴或上下两幅：zero-foot/support；progress ratio 单独点图 | `pair0_l2b_v3.../final_heldout/episode_metrics.csv` 与 `prospective_final_gate.json` | episode 嵌套于一个训练策略；不画伪置信区间 |
| Figure 4 | The selected final checkpoint passed the predeclared gate despite non-monotonic learning | 4个 checkpoint 的 progress 与 zero-foot 两条小图 | `pair0_l2b_v3.../evaluations/additional_*` | 标出 fixed final，而非“best checkpoint” |
| Figure 5 | Uphill progress degraded beyond 12°, while downhill performance was non-monotonic | x=有符号角度，y=mean best progress；上/下坡阈值虚线；PASS/FAIL形状冗余编码 | slope boundary `summary.json`/`episode_metrics.csv` | 不连接成物理极限曲线；显示每个 seed 点 |
| Figure 6 | Directional turn asymmetry, not immediate falling, blocks map readiness | 左右曲率的 target/actual yaw ratio 配对条或点图 | flat turn `summary.json`/`episode_metrics.csv` | 低速探针单独标为非原地旋转 |
| Figure 7 | One historical fixed-map trajectory reached the goal region but failed the safety contract | 封存地图3D帧、实际地面轨迹、1/5–0/5–2/5三项结果 | `docs/FIXED_TERRAIN_PROGRESS_REPORT_20260819_0730_CN.md` 与 relief video | 红色标注 historical unsafe diagnostic |
| Figure 8 | The evidence pipeline retains engineering stops and scientific negative results | 简洁时间线：diagnose → PAIR0 → slope → V2/V4 pre-train stop → V5 slope PASS/turn FAIL → Stage B HOLD | manifests、failure records、V5 final gate | 不把工程失败混成策略失败，也不把坡面PASS写成转向PASS |

所有结果图必须由 raw CSV/JSON 生成；禁止手工从报告表格抄数后作图。小样本时显示每个 seed 点，不用只有均值的柱状图掩盖离散度。颜色之外再使用形状或线型表示 PASS/FAIL，确保色觉可访问。

### 4.2 主文表

| 编号 | 内容 | 最低字段 |
|---|---|---|
| Table 1 | 冻结系统配置 | simulator/version、Ant XML、DOF/action、observation、dt/frame skip、friction、condim、PAIR0、training/evaluation seeds |
| Table 2 | 指标与门槛 | unit、aggregation grain、direction、threshold、failure semantics、是否用于 training reward |
| Table 3 | L2b final paired results | absolute values、difference/ratio、episode count、training-run count、gate result |
| Table 4 | 全部坡度点 | angle/direction、progress、zero-foot、slip events、safety events、PASS/FAIL、failed checks |
| Table 5 | 转向诊断 | target yaw、actual yaw、ratio、same-sign count、zero-foot、safety |
| Table 6 | Negative and failed attempts | intervention、stage reached、what changed、result、why not retained、artifact/failure hash |
| Table 7 | Claim boundary | established、supported inference、unresolved、required next evidence |

## 5. Presentation 设计：13页主线

建议按 10–12 分钟答辩设计；若实际时间更短，优先保留第1、2、5、6、7、9、10、13页。每页只保留一个主要 claim，标题直接说结论，不用“Methodology”“Results”等目录式标题。

### Slide 1 — Contact repair enabled slope locomotion, but navigation remains unresolved

- **Claim**：项目的真实贡献是把不可解释失败转化为可审计的接触修复与能力边界。
- **Visual**：极简标题；背景使用上坡12°视频的一张清晰全景帧，覆低透明度深色渐层。
- **Visible copy**：标题、副标题、项目名、作者/小组、日期；不放摘要段落。
- **Evidence/source**：project-generated uphill video；speaker notes 加 `[Sources]` 和视频路径、SHA-256。

### Slide 2 — A route counts only after arrival, stable dwell and safety all pass

- **Claim**：单次进入终点圈不能被称为任务完成，更不能进入最优路径排序。
- **Visual**：从左到右三个门：spatial arrival → stable 2 s dwell → safety-qualified completion；失败轨迹在对应门处停止。
- **Visible evidence**：历史固定地图 `1/5` spatial arrival、`0/5` qualified completion、`2/5` falls。
- **Evidence/source**：`docs/FIXED_TERRAIN_PROGRESS_REPORT_20260819_0730_CN.md`；`docs/TIME_ENERGY_PATH_OBJECTIVE_V1_CN.md`。

### Slide 3 — The learned controller is locally terrain-aware, not a global route planner

- **Claim**：135D 低层策略能观察局部地形并输出8个关节动作，但当前没有学习完整地图选路。
- **Visual**：一条主流程：local terrain preview + command → PPO low-level controller → 8 joint actions → MuJoCo → safety/energy logs；planned global planner 用灰色虚线置于上方。
- **Visible evidence**：122D + 13D、8 actions、fixed friction、forward+yaw commands。
- **Evidence/source**：`src/proxygap/fixed_goal_terrain.py`、`docs/FIXED_TERRAIN_PROGRESS_REPORT_20260819_0730_CN.md`；外部背景用 Farama Foundation Ant-v5 与 Schulman et al. (2017)。

### Slide 4 — The first fixed-map arrival exposed a support failure, not a solved task

- **Claim**：旧视频证明空间可达，但安全失败和低重复率否定可靠成果。
- **Visual**：左侧 relief 双视角历史视频帧；右侧一条真实轨迹与三项判定；在帧上明确写 `Historical unsafe diagnostic`。
- **Visible evidence**：seed 74803、473.75 s、最终距离1.373 m；终点保持40步中24步 zero-foot。
- **Evidence/source**：`artifacts/dev/fixed_map_reach_a_corrected_replication_v2_20260819/videos/seed_74803_dual_view_relief_v2/`；历史结果报告。

### Slide 5 — In controlled tests, contact margins explained most of the heightfield support gap

- **Claim**：把问题归因于“摩擦不足”并不符合对照证据；主要可复现实验因素是接触 margin 语义。
- **Visual**：三组 before/after bars：hfield129 `73.625→21.500%`、hfield257 `77.500→21.875%`、plane margin0 `22.125%`；下方仅一行“friction unchanged”。
- **Visible evidence**：固定动作、固定状态、固定摩擦；只有 floor/ankle margin 条件改变。
- **Evidence/source**：`docs/HEIGHTFIELD_CAPSULE_MARGIN_ADDENDUM_20260819_CN.md`；MuJoCo official contact documentation 在 notes 中引用。

### Slide 6 — PAIR0 cut zero-foot exposure by 20.8 percentage points without losing mean progress

- **Claim**：PAIR0 是本轮最强的定量改善，但结论只覆盖一个训练 seed 与冻结标准场景。
- **Visual**：左侧 slopegraph：zero-foot `24.042%→3.208%`；中间 support `0.353→1.344`；右侧 progress ratio `1.051`。顶部小字 `20 held-out episodes per condition; one training seed`。
- **Visible evidence**：12/12 held-out checks、9/9 continuity checks；final checkpoint `2,727,936`。
- **Evidence/source**：`docs/FIXED_STANDARD_PAIR0_ADAPTATION_L2B_V3_RESULT_20260819_CN.md`；manifest SHA `d9d6088a…e77ac`。

### Slide 7 — Uphill capability is demonstrated through 12°, not at an inferred maximum

- **Claim**：上坡12°是连续被测通过下界；16°首先因推进不足失败。
- **Visual**：上坡每个 seed 点 + mean line；推进门槛水平线；12°绿色实心、16°/20°红色空心。
- **Visible evidence**：progress 4° `10.05 m`、8° `8.10 m`、12° `7.33 m`、16° `5.53 m`、20° `2.24 m`；20° zero-foot `6.83%`。
- **Evidence/source**：slope boundary `summary.json`/`episode_metrics.csv`；manifest SHA `fd1995b2…60e3`。

### Slide 8 — Downhill performance is non-monotonic, so one passing angle is not a capability range

- **Claim**：下坡8°和16°离散点通过，但4°、12°、20°未达固定推进门槛，不能报告0–16°连续能力。
- **Visual**：左侧下坡 seed plot；右侧上坡12°/下坡16°视频静帧，点击播放完整视频。
- **Visible evidence**：55/55 episodes 无 fall、torso-ground、sustained non-foot 或 corrected sustained slip event。
- **Evidence/source**：同一 slope boundary 工件；上坡视频 SHA `bfca611d…6e19`，下坡视频 SHA `47565385…51c`；delivery manifest SHA `14a8f114…8bdb`。

### Slide 9 — Safe episodes still reveal a strong left–right turning asymmetry

- **Claim**：当前瓶颈不是立即摔倒，而是命令跟踪的方向不对称。
- **Visual**：三组镜像点图，x为 `|κ|`，y为 yaw target ratio；left 与 right 使用颜色+形状双编码；目标带 `[0.7,1.3]` 淡色背景。
- **Visible evidence**：κ0.10 `−0.203 vs 1.120`；κ0.20 `0.234 vs 0.874`；κ0.35 `−0.016 vs 0.329`；所有9条件 safety pass。
- **Evidence/source**：`docs/FIXED_STANDARD_PAIR0_FLAT_TURN_DIAGNOSTIC_20260819_CN.md`；manifest SHA `d0beda1f…f8d6`。

### Slide 10 — The last turn-balancing iteration stopped before training, so it produced no policy claim

- **Claim**：严格 fail-closed 流程防止把工程 smoke 或部分 root 误报成科学结果。
- **Visual**：单线时间轴：V2 smoke PASS → V2 formal pre-train path failure → V4 short-root pre-train audit-interface failure → hard stop；在 `learn()` 之前放明确停止线。
- **Visible evidence**：smoke C0/C1 each 8,192 steps；formal checkpoint count 0；fixed-map evaluated=false。
- **Evidence/source**：smoke manifest SHA `a02b8dad…32be`；V2 failure SHA `21ccdebc…9d23`；V4 failure SHA `9695bd3b…4812`；`docs/PAIR0_FINAL_OPTIMISATION_SEAL_20260819_CN.md`。

### Slide 11 — No existing trajectory is eligible for time–energy optimisation

- **Claim**：安全成功是硬约束；时间与能耗只在合格轨迹中形成 Pareto 折中。
- **Visual**：一条词典序流程，不画已存在的 Pareto 点：validity gate → `(T, E_rel)` → mission profile selection。
- **Visible evidence**：time-prioritised `0.70/0.30`、balanced `0.50/0.50`、energy-prioritised `0.30/0.70`；加 `Protocol design — not evaluated`。
- **Evidence/source**：`docs/TIME_ENERGY_PATH_OBJECTIVE_V1_CN.md`；`RELATIVE_MISSION_ENERGY_PROXY_V1.md`。

### Slide 12 — Reproducibility includes preserving negative results and exact provenance

- **Claim**：checkpoint、raw rows、运行依赖、视频和失败记录共同构成可审计成果。
- **Visual**：一条证据链：config → runtime snapshot → checkpoint → raw metrics → manifest/SHA → full-decode video；下方列出 V2/V4 failure records 被保留。
- **Visible evidence**：PAIR0 manifest 144/144 inventory verified；slope 55完整episodes；视频640/640 frames decoded。
- **Evidence/source**：L2b、slope boundary、slope delivery manifests；`docs/TRAINING_VIDEO_ARTIFACT_POLICY.md`。

### Slide 13 — The project established slope-capable support, while reliable navigation stays on HOLD

- **Claim**：最终结论必须同时给出已建立、未建立和下一道科学门。
- **Visual**：由左至右的三段结论线，不做密集卡片：contact mechanism established → conditional slope capability established → turning/global navigation unresolved。
- **Visible copy**：
  - Established: contact mechanism, PAIR0 support gain, uphill tested lower bound 12°;
  - Not established: balanced turning, fixed-map reliability, unseen-map generalisation, real energy;
  - Next gate: a new versioned multi-training-seed turn study, then global planning and energy calibration.
- **Evidence/source**：本文件第2节的 evidence ledger；所有核心报告。

### Backup slides（不计入13页）

1. 完整 slope 11行表。
2. 完整 flat-turn 9条件表。
3. 指标公式与 corrected slip 定义。
4. Energy proxy 四个 measurement-only quantities 与 V2设计公式。
5. Checkpoint、manifest、video 和 failure-record SHA-256。
6. 负结果矩阵：local preview、W12、terrain-frame、V2/V4。

## 6. Presentation 视觉规范

- 画幅：16:9，1920×1080 或 PowerPoint widescreen。
- 风格：研究答辩、低密度、真实证据优先；不使用通用AI机器人插画。
- 背景：暖白 `#F4F1EA`；正文深灰 `#20262E`；PAIR0/通过用蓝绿 `#168C8C`；诊断/提醒用琥珀 `#D8902F`；失败/HOLD用砖红 `#B84A3A`。
- 字号下限：deck title 50 pt、slide title 35 pt、subheading 24 pt、body 18–22 pt；图轴和标签不低于16 pt。
- 每页一个 takeaway title；正文最多4个短点，优先让图承载信息。
- 相邻页改变构图轮廓：全幅视频帧、左右图文、单图+大结论、时间线交替使用；不要做UI式卡片墙。
- 图表显示单位、`n`、阈值含义和 sampling unit；小样本显示全部 seed 点。
- PASS/FAIL 同时用颜色和符号表示；不要只靠红绿。
- 视频页在答辩电脑上优先嵌入短片，并在交付文件夹保留原始 MP4；若嵌入不稳定，使用静帧+本地超链接。正式演示前必须离线测试。
- 每页 speaker notes 必须包含：

```text
[Sources]
Project-generated: <artifact/report path>; <manifest or file SHA-256 when relevant>
External: <verified Leeds Harvard source, DOI/URL>
```

## 7. 推荐媒体与证据文件

### 必须优先使用

1. 上坡12°双视角视频：
   `artifacts/dev/fixed_standard_pair0_slope_delivery_video_v1_20260819/attempt_0/uphill_12deg_seed_94153/pair0_uphill_12deg_seed_94153_standard_slope_diagnostic.mp4`

   SHA-256：`bfca611de0bf947b0f6a27962aee35af0a28ee0983d00f3d3a5adfa3b1946e19`

2. 下坡16°双视角视频：
   `artifacts/dev/fixed_standard_pair0_slope_delivery_video_v1_20260819/attempt_0/downhill_16deg_seed_94137/pair0_downhill_16deg_seed_94137_standard_slope_diagnostic.mp4`

   SHA-256：`47565385ebbe06c45bcb4182198b986efa4307be0f6c7afb0ec8426b10cbd51c`

3. 坡面视频根 manifest：
   `artifacts/dev/fixed_standard_pair0_slope_delivery_video_v1_20260819/attempt_0/manifest.json`

   SHA-256：`14a8f1145e4e9e50a3c57f3985cde8070adc01884237c6e9f73239f6470b8bdb`

4. 最终运动优化封存：
   `docs/PAIR0_FINAL_OPTIMISATION_SEAL_20260819_CN.md`，SHA-256 `4e818850b9d1fa19b0ef4254566f0ec5a75a51f09185484c4916f6b9b41f6d92`；
   `artifacts/dev/pair0_turn_balance_final_seal_20260819/SEAL_MANIFEST.json`，SHA-256 `7d484a6ad01a830d4dbb286f05b731f8edaf51842054008d631b9e4af05cc6d2`。

### 只可作失败案例

历史固定地图 relief 视频：

`artifacts/dev/fixed_map_reach_a_corrected_replication_v2_20260819/videos/seed_74803_dual_view_relief_v2/fixed_map_final_policy_seed_74803_dual_view_relief-v2.mp4`

必须在画面和口头说明中标记：`Historical spatial arrival; safety failed; not the PAIR0 final policy`。不得把它放在标题页或结论页作为“最终成功演示”。

### 不建议进入主讲页

- smoke 地形展示视频：只证明渲染链，不证明运动能力。
- W12、terrain-frame 和 local-preview 失败视频：可放 backup 或回答问题时使用。
- 椭圆测试旧视频：除非 report 专门讨论从圆形到椭圆曲线的早期开发史，否则会分散最终叙事。

## 8. 必须披露的限制

正文 Discussion、结论前一页和答辩口头说明至少覆盖：

1. **单训练 seed**：PAIR0 改善没有跨独立训练 seed 复验；20/12/55/45 evaluation episodes 不能代替独立策略重复。
2. **条件性仿真范围**：结果仅适用于当前 Gymnasium Ant-v5、MuJoCo版本、XML、8-DOF动作、135D观察、PPO实现和CPU软件栈。
3. **接触建模选择**：显式 PAIR0 是本配置下的工程修复候选，不是唯一物理真值，也不能称 MuJoCo bug。
4. **固定摩擦**：地面摩擦保持恒定，未研究湿滑、材料变化或摩擦随机化；静态 `tan θ≤μ` 不能代替动态接触实验。
5. **slip 指标覆盖**：corrected slip 关注 force-qualified 足端持续切向运动；仍可能漏掉慢速滑退、阈值以下滑动、小腿或躯干接触滑动。
6. **坡度边界**：12°是被测上坡下界，不是物理最大角；下坡非单调，16°只是一个通过点。
7. **转向不足**：平地安全通过不等于曲率命令有效；左转明显弱于右转，固定地图 readiness 明确 HOLD。
8. **最后续训无科学结果**：turn-balance smoke 可执行；V2/V4 formal 都在训练前停止，无最终 checkpoint，不得推断干预有效或无效。V4 的实际 PAIR0 接触合同复核通过，停止原因是 audit 接口字段不一致。
9. **固定地图污染**：80 m × 80 m 地图被反复调试，只能作为 development map，不能再称未见测试地图。
10. **导航与最优性未建立**：历史地图只有 `1/5`空间到达、`0/5`合格；没有轨迹可进入时间—能耗 Pareto 比较。
11. **能耗不是真实电量**：squared action、torque-time 和机械功是不同代理；没有电机效率、电气损耗或电池模型。
12. **自然步态后置**：本轮只要求不出现严重协调故障；没有建立生物学自然步态指标或参考数据。
13. **结构充分性未识别**：现有证据既不证明8-DOF足以完成随机地图任务，也不支持立即增加关节、腿或吸附力。
14. **视频的角色**：视频是定性诊断和验收工件，不是成功率、泛化或安全结论的替代品。

## 9. 交付验收清单

### 9.1 Report

- [ ] 标题、摘要、研究问题和结论使用同一证据边界。
- [ ] 每个主要数字均能追溯到 raw CSV/JSON，而不是二手 Markdown 表。
- [ ] 所有表格标明单位、`n`、sampling unit、seed集合和 aggregation。
- [ ] 报告一个训练 seed，而不把 evaluation episodes 当成独立训练重复。
- [ ] 12°只写“continuous tested lower bound”；下坡16°只写“discrete passing point”。
- [ ] turn-balance smoke、V2 failure、V4 failure 分开陈述。
- [ ] energy measurement、relative proxy design、battery energy 三者不混用。
- [ ] 图表由脚本生成并与 analysis table 逐值核验。
- [ ] Leeds Harvard 文内引用与参考文献一一对应；DOI/URL 在提交前重新核验。
- [ ] 披露AI/外部工具使用方式，符合课程规则和小组贡献说明。
- [ ] 导出的 PDF 逐页渲染检查：无溢出、断图、缺字体或不可读表格。

### 9.2 Presentation

- [ ] 主讲页为13页，逐页只有一个主要 claim 和 takeaway title。
- [ ] 字号达到视觉规范；没有长段落、卡片墙或标题换行。
- [ ] 每个图的轴、单位、`n`、门槛和图例在投影尺寸可读。
- [ ] 每页 notes 含 `[Sources]`；project-generated 证据写路径与相关 SHA。
- [ ] 上坡/下坡视频在离线答辩电脑完整播放；保留静帧 fallback。
- [ ] 历史固定地图视频明确标为 unsafe diagnostic。
- [ ] 所有主结论均可在30秒内指出对应原始工件。
- [ ] PPTX 每页渲染并全尺寸检查；所有 unintended overlap、clipping 和 wrapping 清零。
- [ ] 练习完整版和缩短版；结尾回答开场研究问题，而不是以“Thank you”替代结论。

### 9.3 封存包

- [ ] report 源文件与 PDF。
- [ ] presentation 源 PPTX 与 PDF fallback。
- [ ] 两个正式坡面 MP4、静帧、contact sheet 和完整视频 manifest。
- [ ] PAIR0 final checkpoint、frozen configs、raw evaluation rows、manifests 与 SHA-256 清单。
- [ ] flat-turn summary/raw rows、turn-balance smoke manifest、V2/V4 failure records。
- [ ] 最终 seal report 与 `SEAL_MANIFEST.json`，并逐项复核其引用对象大小和 SHA-256。
- [ ] 软件环境、Git commit/dirty-state说明、复现命令和已知Windows路径限制。
- [ ] 不覆盖历史失败 root；不删除 negative results；不把 untracked/dirty 文件静默排除在 provenance 之外。

## 10. 最终讲述口径

推荐答辩结尾：

> The study did not establish reliable autonomous navigation. It established something narrower but defensible: a contact-representation mechanism was isolated, a fixed-policy slope capability was measured under frozen conditions, and the remaining turning bottleneck was quantified. The preserved failures show where the current evidence stops and define the next valid experiment.

中文口头解释：

> 本项目没有把一次到达视频包装成“导航成功”。真正完成的是：先找到 heightfield 支撑异常的主要接触机制，再得到一个在冻结标准坡面上明显更可靠的候选策略，并用坡度与转向实验界定它能做什么、不能做什么。可靠固定地图导航、未见地图泛化和时间—能耗最优路径仍然是下一阶段问题。
