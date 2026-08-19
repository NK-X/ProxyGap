# ProxyGap 最终 Presentation 内容设计 V2（15分钟）

日期：2026-08-20（Europe/London）
状态：已纳入最终完整地图成功结果与三段正式视频
建议可见语言：British Academic English；本文件用中文说明叙事、证据和口播

## 1. 汇报中心结论

建议全场只围绕一句话展开：

> **A matched PPO–Ant baseline was progressively extended from flat-ground locomotion to a hierarchical terrain-navigation system; on one frozen known map, two preselected route contracts completed 6/6 formal episodes without falls or sustained slip events, while important gait and generalisation limits remain.**

它同时回答四个问题：优化了谁、做了哪些改变、最后是否到达、证据到哪里为止。

## 2. 课程硬性要求

- 15分钟主讲，之后3–5分钟 Q&A；建议主讲控制在14:10，留50秒现场缓冲；
- 所有组员都发言，时间尽量平均；
- 每页右上角固定显示 speaker 拼音姓名；
- 每页右下角固定显示页码；
- 每个外部数据、观点、图片同页显示短来源，并在 references 页给完整来源；
- 不用 Cambridge logo；不用动画；
- 视频本地离线嵌入，同时准备 final-frame 静态 fallback。

四人建议分配：A 讲1–3，B讲4–6，C讲7–9，D讲10–13；分别约3:30、3:25、3:35、3:40。若人数不同，只需把连续叙事块重新分配，不应让成员只读参考文献。

## 3. 版本命名

避免把仓库内部 checkpoint V4/V5 与用户所说项目 V1–V3 混在一起：

| Presentation label | 内容 | 汇报角色 |
|---|---|---|
| Legacy V1 | 最早 control-cost / reward 尝试 | 仅用于说明问题起点 |
| Stage 1 / Project V2 | 平地直行、方向、姿态、动作平滑与接触 shaping | 主要 baseline 对照阶段 |
| Stage 2 / Project V3 | 连续起伏地图、局部地形、PAIR0、坡度、转向、规划与多目标路线 | 主要创新与系统集成阶段 |

仓库的 `V4 canonical-frame checkpoint` 只在 Stage 2 页面称为“archived bidirectional low-level expert”，不要新增第四个项目版本。

## 4. 13页结构与口播设计

### Slide 1 — From reward shaping to successful terrain navigation

**Speaker / time:** `[Speaker A Pinyin]`, 0:40
**Takeaway title:** `A two-stage PPO–Ant study progressed from flat walking to verified map completion.`

**可见内容：**

- `6/6 formal map episodes completed`
- `0 falls | 0 sustained slip events`
- `Two route contracts | three human preference profiles`
- 小字：`Known frozen map; candidate-bank near-optimal, not globally optimal`

**Visual：** 使用时间优先视频终帧作全幅图，右下放 6/6 和安全门；不要放抽象机器人插画。

**同页来源：** `Project formal result, 20 Aug 2026; video manifest 60498def…c024.`

**口播：** 开场先回答“最终是否走到终点”：是；随后立即给出边界——这是一张已知冻结地图上的分层系统结果，不是未见地图泛化。

---

### Slide 2 — Two stages increased task complexity while retaining one baseline

**Speaker / time:** `[Speaker A Pinyin]`, 1:10
**Takeaway title:** `V2 solved low-complexity locomotion diagnostics; V3 added terrain, safety and planning.`

**Visual：** 单线时间轴：Legacy V1 → Stage 1/V2 → Stage 2/V3。每段只放一个代表帧和一句 research question。

**可见内容：**

- Stage 1: `Can reward and constraints produce directed, stable flat locomotion?`
- Stage 2: `Can the locomotor complete a continuous terrain mission under safety and time–energy preferences?`
- 页脚：`Presentation stages are not repository filename suffixes.`

**同页来源：** `Project reward history; terrain protocols; final V3 system report.`

---

### Slide 3 — The defensible baseline is locally reproduced Ant-v5 + PPO

**Speaker / time:** `[Speaker A Pinyin]`, 1:40
**Takeaway title:** `Farama defines the environment; PPO is the algorithmic paper; matched local runs provide the numerical baseline.`

**Visual：** 三段箭头：

`Ant-v5 environment documentation` → `PPO method` → `matched project baseline and interventions`

**可见内容：**

- Gymnasium Ant-v5：8 torque actions；标准目标是向前移动；
- PPO：clipped surrogate policy-gradient family；
- Stage 1 数值 comparator：相同环境、预算、seed规则和评价协议下的 local PPO baseline；
- Stage 2 comparator：Stage 1/controller lineage 和 direct-goal baseline，而不是让默认 Ant 在复杂地图上“陪跑”。

**关键口播：** Farama Ant 页面不是 research paper。PPO paper 是算法 provenance，Ant 的研究历史可追溯到 GAE locomotion work。Fu et al. 是步态/机械能方法锚点，但机器人、DoF、控制和预算不同，不能直接减分数。

**同页来源：** `Farama Foundation (n.d.); Schulman et al. (2016, 2017); Fu et al. (2022).`

---

### Slide 4 — Human intent was separated from the reward optimised by PPO

**Speaker / time:** `[Speaker B Pinyin]`, 1:05
**Takeaway title:** `Reward was a training signal; task validity and safety were judged independently.`

**Visual：** 三层门：`optimised reward → independent diagnostics → hard mission gate`。

**可见内容：**

- Reward terms：速度、方向、姿态、动作率、落脚、支撑；
- Independent diagnostics：路径效率、heading error、tilt、contact sequence、mechanical proxies；
- Stage 2 hard gate：`arrival within 1.5 m → remain within 2 m for 2 s → no fall/torso/sustained non-foot/sustained slip`；
- 只有通过 hard gate 的轨迹才能进入 time–energy 排序。

**同页来源：** `Ng, Harada and Russell (1999); Pan, Bhatia and Steinhardt (2022); Skalse et al. (2022).`

**边界：** 本项目的多数 shaping 不是 potential-based shaping，不能假定最优策略不变；必须用独立指标检查副作用。

---

### Slide 5 — Stage 1 reshaped flat-ground locomotion without changing the 8-D robot action space

**Speaker / time:** `[Speaker B Pinyin]`, 1:05
**Takeaway title:** `Bounded reward and constraint terms targeted directed, smoother and safer locomotion.`

**Visual：** 从 standard Ant reward 向右依次加入 target speed、pitch/balance、action-rate、landing/contact diagnostics；外部 action limiter 单独标注为 controller intervention。

**可见内容：**

- target-speed and forward-direction tracking；
- torso vertical/angular and signed-pitch terms；
- action-rate / saturation diagnostics；
- foot-landing/contact terms；
- Ant body and 8 torque actions retained。

**同页来源：** `Project reward-iteration history; Raffin, Kober and Stulp (2022); Aractingi et al. (2023).`

---

### Slide 6 — Stage 1 improved selected diagnostics, not a biologically verified gait

**Speaker / time:** `[Speaker B Pinyin]`, 1:15
**Takeaway title:** `The intervention improved motion quality proxies, but “natural gait” remains a visual description.`

**Visual：** 两个 paired slope plots，不用柱状图：

- action roughness `0.0139 → 0.00985`；mean speed `0.844 → 0.918 m/s`；path efficiency `0.809 → 0.857`；
- mean take-offs `21.1 → 3.77`；no-floor fraction `0.526 → 0.465`；mean speed `0.961 → 0.931 m/s`。

**限制框：** `Development evidence; gait phase, duty factor and contact ordering were not frozen as biological validation metrics.`

**同页来源：** `Project Stage-1 development reports; Fu et al. (2022).`

**口播：** 可以说“the walk looked more coordinated and selected diagnostics improved”，不能说“natural gait was scientifically proven”。

---

### Slide 7 — Stage 2 exposed three different failure layers

**Speaker / time:** `[Speaker C Pinyin]`, 1:10
**Takeaway title:** `Support, turning and global route selection were separate problems.`

**Visual：** 三级结构：low-level contact/locomotion；mid-level turning/waypoint following；high-level map planning。

**可见内容：**

- 135-D observation = 122-D locomotion + 13-D local terrain preview；
- PAIR0 reduced heightfield contact-margin artefacts and improved distal support；
- direct-goal final PAIR0: best progress 14.51 m in 600 s, no arrival；
- V5 turn-balance: slope PASS, turn FAIL；
- low-level policy never receives the full 1025×1025 map。

**同页来源：** `Miki et al. (2022); PAIR0 V3 evaluation; V5 turn evaluation; post-seal full-map result.`

---

### Slide 8 — PAIR0 improved support and tested slope safety, but airborne gait remained

**Speaker / time:** `[Speaker C Pinyin]`, 1:10
**Takeaway title:** `Contact support improved strongly; that did not automatically yield navigation.`

**Visual：** 左侧 paired points：zero-foot `24.04% → 3.21%`、mean support `0.353 → 1.344`、best-progress ratio `1.051`；右侧坡度曲线。

**可见内容：**

- fixed final checkpoint、20 evaluation episodes/condition（nested within one training run）；
- uphill contiguous tested pass through12°，16° first tested fail；
- downhill non-monotonic；
- 55/55 slope episodes 无 fall/torso/sustained non-foot/corrected sustained-slip event。

**同页来源：** `Project PAIR0 L2b V3 and slope-boundary manifests; Lee et al. (2020).`

**边界：** 不称12°为物理最大爬坡角；不称PAIR0“修复MuJoCo”。

---

### Slide 9 — The successful solution was hierarchical, not another reward term

**Speaker / time:** `[Speaker C Pinyin]`, 1:15
**Takeaway title:** `A known-map route planner supplied global structure to a bidirectional low-level expert.`

**Visual：**

`Frozen 1025×1025 map → slope/turn-aware route candidates → 3 m lookahead waypoint → V4 low-level expert → 8 joint torques`。

**可见内容：**

- archived V4 expert was the only screened checkpoint with both left/right response；
- V4 + PAIR0 passed flat, ±8° and +12° safety screens；
- naive action blending fell after15.3 s and was rejected；
- known-map route length ≈109 m; maximum discrete corridor slope proxy15.9°；
- planner uses the full map; PPO receives only local commands and local terrain preview。

**同页来源：** `Project checkpoint screen, standard-slope screen and waypoint-route evidence.`

**边界：** 这是 system-level integration，不是证明 final PAIR0 policy 学会了全局规划。

---

### Slide 10 — Three preferences selected two routes from 15 feasible candidates

**Speaker / time:** `[Speaker D Pinyin]`, 1:10
**Takeaway title:** `Success and safety were hard constraints; time and mechanical work ranked only valid candidates.`

**Visual：** 15个候选的 time–positive-work scatter，所有可行点显示；两条选中路线高亮。横纵轴保留单位；不要把未评测连续空间画成平滑 Pareto front。

**可见公式：**

`J = wT(T/Tmin) + wE(W+/W+min)`

**可见结果：**

- time (0.8/0.2) → `time_and_balanced`；
- balanced (0.5/0.5) → same contract；
- energy (0.2/0.8) → `energy_priority`；
- 开发候选：238.85 s / 50.21 kJ proxy 与242.75 s / 49.89 kJ proxy。

**同页来源：** `Project candidate-selection JSON 212b7886…f20b; Fu et al. (2022) for mechanical-energy context.`

**边界：** `near-optimal within the 15-candidate bank`，不是 globally optimal。

---

### Slide 11 — All six formal episodes reached the goal without sustained slip events

**Speaker / time:** `[Speaker D Pinyin]`, 1:50（含约20–25秒视频）
**Takeaway title:** `The hierarchical system completed the known map repeatedly, with measurable trade-offs.`

**Visual：** 左侧播放三视频各约6–8 s；右侧显示逐 seed 点和均值表。

**正式表：**

| Contract | Success | Mean time | Mean positive work proxy | Mean path | Sustained slips | Falls |
|---|---:|---:|---:|---:|---:|---:|
| time/balanced | 3/3 | 264.55 s | 55.65 kJ | 153.23 m | 0 | 0 |
| energy | 3/3 | 259.37 s | 55.13 kJ | 152.14 m | 0 | 0 |

**视频提示：** 时间与平衡是同一路线合同、不同正式种子。三段复放的控制状态和五子步接触记录均 mismatch = 0；全部全帧解码通过。

**限制提示：** 代表回合仍有9.25–10.02%的完整控制区间四足无接触；no sustained slip 不等于 no instantaneous slip or natural gait。

**同页来源：** `Project final manifest 0bf2817c…8d83; video manifest 60498def…c024.`

---

### Slide 12 — References

**Speaker / time:** `[Speaker D Pinyin]`, 0:20
**Takeaway title:** `Method provenance and comparator literature`

两栏排版，16–17 pt，不口头逐条读。建议保留：

- Aractingi, M. et al. (2023) ‘Controlling the Solo12 quadruped robot with deep reinforcement learning’, *Scientific Reports*, 13, 11945. https://doi.org/10.1038/s41598-023-38259-7.
- Farama Foundation (n.d.) *Ant*. Gymnasium Documentation. Available at: https://gymnasium.farama.org/environments/mujoco/ant/ (Accessed: 20 August 2026).
- Fu, Z., Kumar, A., Malik, J. and Pathak, D. (2022) ‘Minimizing energy consumption leads to the emergence of gaits in legged robots’, *PMLR*, 164, pp. 928–937.
- Lee, J. et al. (2020) ‘Learning quadrupedal locomotion over challenging terrain’, *Science Robotics*, 5(47), eabc5986.
- Miki, T. et al. (2022) ‘Learning robust perceptive locomotion for quadrupedal robots in the wild’, *Science Robotics*, 7(62), eabk2822.
- Ng, A.Y., Harada, D. and Russell, S.J. (1999) ‘Policy invariance under reward transformations’, *ICML*, pp. 278–287.
- Pan, A., Bhatia, K. and Steinhardt, J. (2022) ‘The effects of reward misspecification’, *ICLR 2022*.
- Raffin, A., Kober, J. and Stulp, F. (2022) ‘Smooth exploration for robotic reinforcement learning’, *PMLR*, 164, pp. 1634–1644.
- Schulman, J. et al. (2016) ‘High-dimensional continuous control using generalized advantage estimation’, *ICLR*.
- Schulman, J. et al. (2017) ‘Proximal policy optimization algorithms’, arXiv:1707.06347.
- Skalse, J. et al. (2022) ‘Defining and characterizing reward hacking’, *NeurIPS*, 35.

同页底部：`Project configurations, raw traces, videos and SHA-256 manifests are listed in the evidence appendix.`

---

### Slide 13 — What is established, and where does the evidence stop?

**Speaker / time:** `[Speaker D Pinyin]`, 0:30；随后 Q&A
**Takeaway title:** `The project established a reproducible known-map system, not a universally optimal gait or planner.`

**可见内容：**

- Established: selected Stage-1 diagnostics; PAIR0 support; tested slope boundaries; 6/6 known-map completion；
- Contribution: failure-driven decomposition of contact, turning and planning；
- Not established: biological natural gait, unseen maps, multi-training-seed robustness, electrical energy, global mathematical optimality；
- Next valid study: train/test map splits + independent training seeds + calibrated mission energy。

大字：`Questions`。

**同页来源：** `Project final report and evidence manifests, 20 Aug 2026.`

**结束句：** `The final result is not that one reward solved everything; it is that a sequence of auditable failures revealed the architecture required for reliable completion.`

## 5. 每个 Stage 的正式评价指标

### Stage 1 / V2

1. Task validity：fixed horizon、mean velocity、target-speed error；
2. Direction：heading error、net displacement、path efficiency；
3. Safety/posture：fall、torso tilt、clearance、unhealthy termination；
4. Action quality：normalised roughness、saturation、action limiter intervention；
5. Contact/gait：per-foot duty factor、phase、contact-order、take-offs、zero-foot fraction；
6. Efficiency：action²、torque-time、positive/absolute mechanical work分开报告。

“自然步态”只有在第5组指标正式冻结并跨速度验证后才可以作为结论。

### Stage 2 / V3

采用词典序门控：

1. Arrival：进入1.5 m并在2.0 m圈保持2 s；
2. Safety：finite、0 fall、0 torso-ground、0 sustained non-foot、0 sustained corrected-slip event；
3. Locomotion：support count、zero-foot、terrain-relative attitude、slope grid；
4. Navigation：success rate、time、path length、cross-track、route progress；
5. Preference：仅对完成轨迹比较归一化 time 与 mechanical-work proxy；
6. Generalisation：独立 training seeds、未见 map seeds、未见 start-goal pairs。

## 6. Q&A 必答问题

1. **What is the baseline paper?** PPO paper is the algorithmic source; Ant documentation defines the environment; numerical comparison uses a matched local Ant-v5+PPO baseline.
2. **Did the robot learn a natural gait?** No. Selected motion diagnostics improved, but phase/duty/contact-order validation and airborne exposure prevent a biological claim.
3. **Why did the previous video fail?** The final PAIR0 policy had local support but no reliable bidirectional waypoint response or global map planner.
4. **What changed in the successful run?** A screened bidirectional V4 low-level expert was combined with PAIR0 and a known-map waypoint planner; no new training was hidden.
5. **Is the path optimal?** Only candidate-bank near-optimal among 15 evaluated feasible candidates.
6. **Did it never slip?** It had zero sustained corrected-slip events, not zero instantaneous candidate samples; airborne intervals remain.
7. **Why is the energy route faster in formal means?** Reset-seed variability changed the realised ordering; this is why all seed points and both metrics must be reported.
8. **Does it generalise?** Not yet. All formal episodes use one previously inspected frozen map.

## 7. 制作验收清单

- [ ] 每页右上 speaker 拼音、右下页码；
- [ ] 每页有可读同页短来源；references页metadata最终复核；
- [ ] Slide 6、8、11显示全部 seed 点，不把 episode 当 training replicate；
- [ ] Slide 10只画15个真实候选，不画虚构连续Pareto front；
- [ ] Slide 11本地离线嵌入三视频，并准备final-frame fallback；
- [ ] 不写 natural gait achieved、globally optimal、battery energy optimised、unseen-map generalisation；
- [ ] 不用Cambridge logo或动画；
- [ ] 完整彩排14:10；恢复版12:30；成员发言时间差尽量不超过30秒。
