# ProxyGap 最终 Presentation 内容设计（15分钟）

日期：2026-08-19（Europe/London）<br>
状态：内容蓝图；不生成 PPTX，不新增实验，不改变封存模型。<br>
预期受众：课程教授、助教和同学。<br>
建议可见语言：British Academic English；本文件以中文说明设计与证据边界。<br>

## 1. 控制要求与证据边界

### 1.1 Presentation 硬性要求

根据课程截图，最终演示必须满足：

- 总时长为15分钟，之后预留3–5分钟 Q&A；
- 每位小组成员都必须发言，并尽量平均分配时间；
- 每页右上角显示该页 speaker 的拼音姓名；
- 每页右下角显示页码；
- 外部数据、观点、文字或图片必须在同一页给出可读的短来源，并在 references 页列出完整来源；
- 不使用 University of Cambridge 标志；
- 不使用动画。

本设计安排14分05秒主讲内容，保留约55秒现场缓冲。若最终视频启动或成员交接较慢，优先删减口头解释，不删除证据边界和来源。

### 1.2 Report 模板与 Presentation 的关系

`D:\AI+ Project Report - Template 2026.docx` 是 report 模板，不是 presentation 模板。结构化只读检查确认其正文标题为：

1. Project Title；
2. Project Overview；
3. Research Context；
4. Division of Roles and Responsibilities；
5. Challenges and Solutions；
6. Contributions and Limitations；
7. Bibliography。

模板 FAQ 另规定：最终 report 为 PDF、2,000–2,500词（Bibliography 不计）、Times New Roman、最低11 pt、建议1.15倍行距、Harvard 引用；截止时间为2026年8月20日22:00（英国时间）。模板还要求保留 instructor evaluation sheet，并强调原创表达和完整引用。上述规则只约束 report，不应被复制成 PPT 页面结构。

当前运行环境没有 LibreOffice，标准 DOCX 渲染器无法生成页面 PNG；本轮只完成了全部正文、表格、标题和页面设置的结构化只读提取，没有修改原 DOCX。最终 report 制作时仍须进行逐页 PDF/PNG 视觉 QA。

### 1.3 汇报的唯一中心句

> **Reward and constraint design improved selected locomotion diagnostics as task complexity increased; however, neither natural gait nor reliable full-map navigation was established.**

答辩结束时，观众应理解：项目的贡献不是“机器人已经找到最优路线”，而是建立了一个从标准 Ant/PPO 基线、平地 reward shaping、接触语义修复，到坡度与完整地图失败边界的可审计优化过程。

## 2. 版本与 Stage 命名必须先澄清

用户口述中同时出现“抛开最初 V1”和“Stage 1 就是 V1”。仓库中还存在 `reward-v1-foot-landing`、`reward-v2-pitch-balance` 等局部标签。为避免答辩时把不同版本体系混用，建议采用下表并在 Slide 2 明示：

| Presentation label | 仓库对应 | 在汇报中的角色 |
|---|---|---|
| Legacy V1 exploration | `legacy/weight_sweep_v1/` | 早期 `ctrl_cost_weight` 探索；仅作研究动机，不并入主比较 |
| Stage 1: flat-ground locomotion | repository research direction V2；平地 reward/constraint experiments | 低复杂度阶段：定义人的意图，改善速度、方向、姿态、平滑和接触诊断 |
| Stage 2: continuous-terrain locomotion | terrain/PAIR0/final policy lineage，汇报中简称 V3 | 高复杂度阶段：局部地形预瞄、接触支撑、坡度能力、转向和完整地图测试 |

页脚可写：`Stage labels describe presentation complexity; they are not repository file-version suffixes.`

除非小组随后确认另一套命名，否则不要把 Legacy V1 包装为完成的 Stage 1，也不要把 `reward-v1` 当成 repository V1。

## 3. Baseline：有论文支撑，但不能把文献分数当成本项目对照结果

### 3.1 推荐采用三层 baseline

1. **可执行实验基线（主要 baseline）**：未加入项目干预的 Gymnasium `Ant-v5` + Stable-Baselines3 PPO，在相同 XML、seed 规则、训练预算、checkpoint 和评价协议下运行。
2. **算法来源（paper provenance）**：Schulman et al. (2017) 的 PPO 论文；Ant 环境的历史来源可追溯至 Schulman et al. (2016) 的高维连续控制/GAE 工作。
3. **最接近“自然步态—能耗”的 paper comparator（不是数值 baseline）**：Fu et al. (2022) 用足端接触序列、不同速度下的步态变化、energy per metre 和 Froude number 研究能耗与步态涌现；其12-DoF A1、PD控制、真实机器人和超大训练预算与本项目8-DoF Ant 不可直接比较。
4. **reward design 与复杂地形研究锚点（不是数值 baseline）**：Aractingi et al. (2023) 支持四足控制中同时评估命令跟踪、身体姿态、动作平滑和能量相关量；Lee et al. (2020) 与 Miki et al. (2022) 分别提供复杂地形指标和局部地形感知的研究背景；Pan, Bhatia and Steinhardt (2022) 支持把 optimiser-facing reward 与独立评价指标分开。

### 3.2 对 Farama Foundation Ant 页面引用的准确回答

Farama Foundation 的 `Ant-v5` 页面是**官方环境文档，不是 research article**。它适合引用环境定义、105维默认观察、8维动作、时间步和默认 reward components，但不能单独满足“有一篇论文作为实验 baseline”的要求。该网页没有明确页面出版年份，参考文献应写 `Farama Foundation (n.d.)` 并给出访问日期，不应仅因当前年份而写成2026。

若教授要求“我们优化了谁”：

> The executable baseline is the standard Gymnasium Ant-v5 task trained with PPO. PPO is attributed to Schulman et al. (2017), while Ant-v5 is defined by the Farama Foundation documentation. The published papers provide methodological provenance; all performance comparisons are made against locally reproduced, matched-budget baselines rather than incompatible published reward scores.

### 3.3 为什么不直接声称“优于 PPO 论文”

PPO/GAE 论文中的 MuJoCo Ant 与当前 `Ant-v5` 的版本、reward、termination、observation、软件栈和训练预算并不完全一致。没有在当前环境中复现实验前，跨版本 raw return 不可直接比较。因此：

- Stage 1 比较 `default/matched PPO baseline` 与 reward/constraint intervention；
- Stage 2 比较 `DEFAULT_CONTINUE` 与 `PAIR0_ADAPT`，并把 direct-goal controller 保留为地图导航 baseline；
- Fu et al. (2022) 是“如何以接触图和能耗讨论步态”的最接近论文比较对象，但不提供与本项目可直接相减的数值 baseline；
- Aractingi et al. (2023) 用于解释评价维度与工程意义，不用于声称本项目在数值上超过 Solo12；
- 论文 baseline 与项目 baseline 的角色必须在 Slide 3 分开标注。

## 4. 13页主线与时间分配

### 4.1 四人小组的建议分配

| Speaker placeholder | Slides | 主讲时间 |
|---|---:|---:|
| `[Speaker A Pinyin]` | 1–3 | 3:30 |
| `[Speaker B Pinyin]` | 4–6 | 3:30 |
| `[Speaker C Pinyin]` | 7–9 | 3:30 |
| `[Speaker D Pinyin]` | 10–13 | 3:35 |

总计14:05。若实际人数不是四人，应保留连续叙事块并按约14:05除以成员数重新分配；不要让某位成员只读 references 或只说“Questions”。每次交接只用一句因果过渡。

### 4.2 全局版式合同

- 16:9 widescreen；暖白背景 `#F4F1EA`，正文深灰 `#20262E`；改善/通过用蓝绿 `#168C8C`，警示用琥珀 `#D8902F`，失败/HOLD 用砖红 `#B84A3A`。
- deck title 不低于50 pt；slide title 不低于35 pt；正文18–22 pt；图轴与短来源不低于16 pt。
- 每页只表达一个 claim；最多四条短句；用图承载数据，不把 report 段落复制到幻灯片。
- 每页右上角固定 `[Speaker X Pinyin]`；右下角固定 `n / 13`。
- 每页底部保留单行或双行 `Source:` 区域；完整路径、SHA-256 和补充来源放 speaker notes 的 `[Sources]` block。
- project-generated 图表必须从 raw CSV/JSON 生成；小样本显示全部 seed 点；视频只能作定性证据。
- 不使用通用 AI 机器人插画、Cambridge 标志、动画、3D 柱图、装饰性仪表盘或卡片墙。

## 5. 逐页内容设计

### Slide 1 — Reward design improved locomotion, but did not solve navigation

**Speaker / time:** `[Speaker A Pinyin]`, 0:40<br>
**Narrative job:** 立即给出真实结论，而不是先讲软件清单。<br>
**Visible copy:**

- `From Reward Shaping to Terrain Locomotion`
- `A two-stage, auditable PPO–Ant study`
- `Support and tested slope capability improved; reliable navigation did not.`
- group name、成员姓名、日期。

**Visual:** 使用上坡12°正式双视角视频的清晰帧作为全幅背景，降低饱和度并加深色渐层；右下保留小型 `Project-generated evidence` 标识。不要使用历史固定地图“偶然到达”视频作封面。

**Short source on slide:** `Source: Project-generated uphill-12° episode; slope-video manifest (19 Aug 2026).`<br>
**Evidence:** 坡面视频 SHA-256 `bfca611d…6e19`；视频用于视觉引入，不承担成功率结论。<br>
**Transition:** `To understand what was actually improved, the three repository versions must first be separated from the two presentation stages.`

### Slide 2 — Two stages increased task complexity while preserving one audit trail

**Speaker / time:** `[Speaker A Pinyin]`, 1:20<br>
**Narrative job:** 解决 V1/V2/V3 与 Stage 1/2 的命名混乱。<br>
**Visible copy:**

- `Legacy V1 — control-cost exploration (context only)`
- `Stage 1 / project V2 — flat-ground reward and constraint design`
- `Stage 2 / project V3 — terrain-aware support, slopes and map traversal`
- 研究问题：`Can reward and constraint design preserve task performance as locomotion complexity increases?`

**Visual:** 一条从左到右的单线发展路径；Legacy V1 用浅灰，两个主 Stage 用实色。在线下方分别放一个扁平地面帧和一个连续地形帧，不做目录式卡片。

**Short source on slide:** `Sources: legacy/weight_sweep_v1/README; current/RESEARCH_DIRECTION_V2; final PAIR0 seal.`<br>
**Boundary:** Legacy V1 只提供动机；不得把三套标签的结果混合统计。

### Slide 3 — The defensible baseline is reproduced Ant-v5 + PPO, not a borrowed paper score

**Speaker / time:** `[Speaker A Pinyin]`, 1:30<br>
**Narrative job:** 直接回答教授“优化了谁、和谁比较”。<br>
**Visible copy:**

- `Environment baseline: Gymnasium Ant-v5`
- `Algorithm baseline: PPO`
- `Experimental comparator: matched local baseline, equal budget`
- `Published scores are context, not directly comparable outcomes`

**Visual:** 从左到右三段：`Published origin → Executable baseline → Project interventions`。在 executable baseline 下写 `105-D default observation | 8 torque actions | standard reward components`；Stage 2 旁注明观察扩展为135D，但动作仍为8D。

**Short source on slide:** `Farama Foundation (n.d.); Schulman et al. (2016, 2017); Fu et al. (2022); Aractingi et al. (2023).`<br>
**Speaker note:** 若被问“baseline paper是哪一篇”，先答 PPO paper 是算法来源，再强调真正的数值 comparator 是相同软件栈下的本地匹配基线。<br>
**Boundary:** 不声称超过 PPO paper 或 Solo12 的已发表性能。

### Slide 4 — Human intent was evaluated separately from the reward optimised by PPO

**Speaker / time:** `[Speaker B Pinyin]`, 1:10<br>
**Narrative job:** 把 reward misspecification 放在次要但清楚的位置，并建立全项目共同评价逻辑。<br>
**Visible copy:**

- `Optimised reward: what PPO receives`
- `Independent diagnostics: speed, direction, posture, support and smoothness`
- `Hard gates: full horizon / arrival, stable dwell and safety`
- `Efficiency is ranked only after task validity`

**Visual:** 单一三层漏斗或水平门：reward → independent diagnostics → validity gate。Stage 2 的成功门在下方显示为 `arrival → stable 2 s dwell → safety-qualified completion`。

**Short source on slide:** `Ng, Harada and Russell (1999); Pan, Bhatia and Steinhardt (2022); Skalse et al. (2022); project behaviour contract V2.`<br>
**Boundary:** external diagnostics 不是经验证的 universal `true reward`。本项目多数姿态、平滑和能耗 shaping 并非 potential-based shaping，因此不能宣称保留原任务的最优策略；必须检查失败是否转移到未优化指标。

### Slide 5 — Stage 1 reshaped reward signals for directed, stable flat-ground locomotion

**Speaker / time:** `[Speaker B Pinyin]`, 1:15<br>
**Narrative job:** 说明低复杂度阶段到底改了什么。<br>
**Visible copy:**

- `Target-speed tracking at 1.0 m/s`
- `Action-rate and body vertical/angular penalties`
- `Foot-landing velocity and signed-pitch balance terms`
- `Ant body, 8-D action space and PPO family retained`

**Visual:** 默认 Ant reward 在左，逐步加入的 bounded shaping terms 沿一条主线向右；外部 action-slew guardrail 放在主线下方并标 `controller intervention, not learned smoothness`。

**Evaluation footer:** `Primary: full horizon, command tracking, safety. Secondary: direction, path efficiency, tilt, roughness, saturation and contact diagnostics.`

**Short source on slide:** `Project reward iteration history (17 Aug 2026); Fu et al. (2022); Aractingi et al. (2023); Raffin, Kober and Stulp (2022).`<br>
**Boundary:** Slide 5 使用 `stable, coordinated locomotion diagnostics`，不使用 `natural gait achieved`。

### Slide 6 — Stage 1 improved selected diagnostics, but natural gait remained unverified

**Speaker / time:** `[Speaker B Pinyin]`, 1:05<br>
**Narrative job:** 同时报告改善和代价，防止只展示“最好看”的视频。<br>
**Visible evidence:**

- action-rate intervention：roughness `0.0139 → 0.00985`；mean speed `0.844 → 0.918 m/s`；path efficiency `0.809 → 0.857`；
- body shaping under ordinary exploration：mean take-offs `21.1 → 3.77`；no-floor fraction `0.526 → 0.465`；mean speed `0.961 → 0.931 m/s`；
- `Development evidence — no held-out formal confirmation`；
- `Biological gait phase, duty factor and contact-order metrics were not frozen.`

**Visual:** 两个简洁 paired slope plots；每个图保留原单位和方向，不用只有均值的柱状图。右下用一行琥珀色限制说明，而不是大段文字。

**Short source on slide:** `Project team progress update (16 Aug 2026); experiment status; intended-behaviour contract V2.`<br>
**Interpretation:** evidence supports partial mitigation of selected proxies. It does not establish a natural gait, biological fidelity or a held-out causal comparison. Fu et al. (2022) shows why a stronger natural-gait claim would require foot-contact patterns across speed, gait transitions and normalised energetic measures that are not present here.

### Slide 7 — Stage 2 added local terrain awareness and contact auditing, not global planning

**Speaker / time:** `[Speaker C Pinyin]`, 1:05<br>
**Narrative job:** 解释从平地到连续地形增加了哪些复杂度，以及什么仍未实现。<br>
**Visible copy:**

- `80 m × 80 m continuous heightfield`
- `135-D observation = 122-D locomotion state + 13-D local terrain preview`
- `8 joint actions; forward and yaw commands`
- `No full-map input and no learned global route planner`

**Visual:** `local preview + command → PPO low-level controller → 8 joints → MuJoCo → support/safety/energy logs`。planned global planner 用灰色虚线置于上方，且明确写 `not evaluated`。

**Short source on slide:** `Miki et al. (2022); project fixed-goal terrain implementation; full-map protocol (19 Aug 2026).`<br>
**Boundary:** 局部仿真真值高度预瞄不等于真实视觉感知，也不等于完整地图知识；低层 PPO 增加训练步数不会自动学会全局绕路。

### Slide 8 — PAIR0 cut zero-foot exposure by 20.8 percentage points while retaining mean progress

**Speaker / time:** `[Speaker C Pinyin]`, 1:20<br>
**Narrative job:** 展示 Stage 2 最强的内部对照结果。<br>
**Visible evidence:**

- pooled zero-foot fraction：`24.042% → 3.208%`；
- mean support count：`0.353 → 1.344 feet per physics substep`；
- mean best-progress ratio：`1.051`；
- `20 evaluation episodes per condition, nested within one training run`。

**Visual:** 两个 paired slope plots + 一个 progress ratio 点；标题下用小字写 `DEFAULT_CONTINUE vs PAIR0_ADAPT; fixed final checkpoint`。能耗结果用一个小型 `mixed` 注记：squared action/torque-time increased, positive work approximately unchanged, absolute work slightly lower。

**Short source on slide:** `Project PAIR0 L2b V3 final held-out evaluation; manifest d9d6088a…e77ac.`<br>
**Boundary:** 这是一个训练 seed 下的冻结诊断结果，不是跨训练 seed 稳健性、自然步态或节能证明。

### Slide 9 — The policy passed tested uphill gates through 12°, while downhill remained non-monotonic

**Speaker / time:** `[Speaker C Pinyin]`, 1:05<br>
**Narrative job:** 给出正式能力边界，同时阻止“最大坡度”误读。<br>
**Visible evidence:**

- uphill progress：4° `10.05 m`、8° `8.10 m`、12° `7.33 m`、16° `5.53 m`、20° `2.24 m`；
- 12°是 `continuous tested lower bound`，16°是首个被测失败点；
- downhill 8°和16°通过，4°、12°和20°失败，结果非单调；
- 55/55 episodes 无 fall、torso-ground、sustained non-foot 或 corrected sustained-slip event。

**Visual:** 有符号坡度的全 seed point plot，PASS 使用实心圆，FAIL 使用空心叉；显示推进门槛。上坡12°和下坡16°各放一个小型视频静帧，演示时只播放5–8秒片段。

**Short source on slide:** `Lee et al. (2020); project standard-slope evaluation, n=5 evaluation seeds per condition; manifest fd1995b2…60e3.`<br>
**Boundary:** 不写 `maximum climb angle = 12°`，也不写 `stable downhill capability to 16°`。

### Slide 10 — Balanced command exposure preserved slope safety but did not fix turning

**Speaker / time:** `[Speaker D Pinyin]`, 1:20<br>
**Narrative job:** 展示封存前最后一轮正式干预，并说明为何后续规划地图测试被预声明门槛阻止。<br>
**Visible evidence:**

- `2 matched branches × 65,536 training steps; 130 held-out evaluation episodes`；
- C0 与 C1 均 `slope PASS / turn FAIL`；
- C1：straight yaw `−0.658 rad`；left/right `0.10 m⁻¹` yaw ratio `−0.288 / 1.206`；
- C1 坡面：overall/uphill/downhill progress `8.751 / 8.039 / 9.993 m`，0 fall、0 sustained-slip event；
- 预声明规则 `C1 turn FAIL → Stage B HOLD`，所以没有用 A* 包装不合格的低层转向。

**Visual:** 左侧画 C0/C1 的 `target vs actual yaw ratio` 配对图；右侧放 seed 96131 的 C0/C1 左右 `0.20 m⁻¹` 对照视频或四格终帧。绿色小条标 `slope safety retained`，红色门标 `turn gate failed`。

**Short source on slide:** `Project V5 final turn-balance evaluation; 130 episodes; formal manifest f107543d…be7b.`<br>
**Interpretation:** 平衡训练曝光在这一训练 seed 和冻结预算下不足以消除闭环方向偏置；不能推断该方法对所有 seed 或结构必然无效。

### Slide 11 — The map run failed, so no trajectory entered the time–energy comparison

**Speaker / time:** `[Speaker D Pinyin]`, 1:35<br>
**Narrative job:** 用完整地图正式负结果回答“是否到达”，并保留用户认可的多目标框架。<br>
**Visible copy:**

1. `FAILED TO REACH: 12,000 steps / 600 s; best progress 14.508 m`
2. `Path 92.410 m; net-progress/path ratio 0.1356; full-control zero-foot 3.19%`
3. `Qualified completion and safety → Pareto(T, E_rel)`
4. `Human intent: 0.70/0.30 | 0.50/0.50 | 0.30/0.70; eligible trajectories: 0`

**Visual:** 左侧使用正式全地图 relief 与完整轨迹；右侧画词典序门控，不绘制虚构 Pareto 点：`validity → (T, E_rel) → mission profile`。在 energy 旁写 `mechanical proxies recorded; battery energy not modelled`。

**Short source on slide:** `Project post-seal full-map evaluation, seed 1763594348, manifest b3d37af1…6038; time–energy path objective V1.`<br>
**Boundary:** 当前没有最优能耗、最优时限、全局最优路线或真实电池能耗结论。W12 总机械功降低但推进下降49.1%的负例可在口头说明中用来解释为何“少动”不能叫节能。

### Slide 12 — References

**Speaker / time:** `[Speaker D Pinyin]`, 0:15<br>
**Narrative job:** 满足课程要求；不逐条朗读。<br>
**Visible references（两栏，Leeds Harvard）:**

- Aractingi, M., Léziart, P.-A., Flayols, T., Perez, J., Silander, T. and Souères, P. (2023) ‘Controlling the Solo12 quadruped robot with deep reinforcement learning’, *Scientific Reports*, 13, 11945. doi: 10.1038/s41598-023-38259-7.
- Farama Foundation (n.d.) *Ant*. Gymnasium Documentation. Available at: https://gymnasium.farama.org/environments/mujoco/ant/ (Accessed: 19 August 2026).
- Fu, Z., Kumar, A., Malik, J. and Pathak, D. (2022) ‘Minimizing energy consumption leads to the emergence of gaits in legged robots’, *Proceedings of the 5th Conference on Robot Learning*, Proceedings of Machine Learning Research, 164, pp. 928–937.
- Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V. and Hutter, M. (2020) ‘Learning quadrupedal locomotion over challenging terrain’, *Science Robotics*, 5(47), eabc5986. doi: 10.1126/scirobotics.abc5986.
- Miki, T., Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V. and Hutter, M. (2022) ‘Learning robust perceptive locomotion for quadrupedal robots in the wild’, *Science Robotics*, 7(62), eabk2822. doi: 10.1126/scirobotics.abk2822.
- Ng, A.Y., Harada, D. and Russell, S.J. (1999) ‘Policy invariance under reward transformations: theory and application to reward shaping’, *Proceedings of the Sixteenth International Conference on Machine Learning*, pp. 278–287.
- Pan, A., Bhatia, K. and Steinhardt, J. (2022) ‘The effects of reward misspecification: mapping and mitigating misaligned models’, *Proceedings of the 10th International Conference on Learning Representations*. Available at: https://openreview.net/forum?id=JYtwGwIL7ye (Accessed: 19 August 2026).
- Raffin, A., Kober, J. and Stulp, F. (2022) ‘Smooth exploration for robotic reinforcement learning’, *Proceedings of Machine Learning Research*, 164, pp. 1634–1644.
- Schulman, J., Moritz, P., Levine, S., Jordan, M.I. and Abbeel, P. (2016) ‘High-dimensional continuous control using generalized advantage estimation’, *Proceedings of the 4th International Conference on Learning Representations*. Available at: https://arxiv.org/abs/1506.02438 (Accessed: 19 August 2026).
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017) ‘Proximal policy optimization algorithms’, *arXiv:1707.06347*. Available at: https://arxiv.org/abs/1707.06347 (Accessed: 19 August 2026).
- Skalse, J., Howe, N.H.R., Krasheninnikov, D. and Krueger, D. (2022) ‘Defining and characterizing reward hacking’, *Advances in Neural Information Processing Systems*, 35, pp. 9460–9471.

**Project evidence line:** `Project-generated reports, raw rows, configurations and SHA-256 manifests are listed in the final evidence appendix.`<br>
**Note:** 提交前重新核对每条作者、年份、venue、页码、DOI/URL 和访问日期；不要只复制本设计稿。

### Slide 13 — What is established, and what is the next valid test?

**Speaker / time:** `[Speaker D Pinyin]`, 0:25；随后进入3–5分钟 Q&A。<br>
**Narrative job:** 用结论回答开场问题，并给 Q&A 一个有研究价值的入口。<br>
**Visible copy:**

- `Established: partial Stage-1 diagnostic gains; PAIR0 support improvement; tested uphill lower bound of 12°.`
- `Not established: natural gait, multi-training-seed robustness, balanced turning, global navigation or real energy efficiency.`
- `Final optimisation hard stop: the predeclared balanced-turn intervention failed; the planner stage was therefore withheld.`
- 大字：`Questions`。

**Visual:** 三段连续结论线：`evidence gained → boundary reached → next test`；不要只放“Thank you”。

**Short source on slide:** `Sources: project evidence ledger and final seal (19 Aug 2026).`<br>
**Closing sentence:** `The project improved what could be measured and preserved the failures that show where the evidence stops.`

## 6. 每个 Stage 的正式评价框架

### 6.1 Stage 1：平地 reward/constraint shaping

**主要 estimand（正式重做时）**：在相同训练预算下，干预条件相对于 default Ant-v5/PPO baseline，对新独立 training seed 的最终策略 intent-compliance 和连续诊断量的影响。

| Domain | Primary/secondary metrics | 当前状态 |
|---|---|---|
| Task | full 1,000-step horizon；fixed-horizon mean forward velocity；target range 0.8–1.2 m/s | 已定义；现有结果主要为 development evidence |
| Safety | unhealthy termination；sustained inversion；RMS torso tilt | 已定义，但门槛为项目选择而非外部安全标准 |
| Direction | displacement direction error；path efficiency | 已记录 |
| Action quality | normalised action roughness；action saturation；proposed/applied action difference | 已记录；外部 limiter 与 learned policy 必须分开 |
| Contact/gait | take-off count；no-floor/contact fraction；per-leg duty factor、phase、contact sequence | 前两项为诊断；自然步态的后几项尚未冻结 |
| Efficiency | squared action、torque-time、mechanical work 分开报告 | 不是电池能耗；不得合并成无标定的“energy efficiency” |

**正式比较要求：**同一训练 seed block、相同预算和调参预算；每个 training run 是独立单位；evaluation episodes 只在策略内聚合；保留全部失败；用 raw paired effects 和区间，不把10个 episode 写成 `n=10 independent policies`。

### 6.2 Stage 2：连续地形与任务完成

评价采用词典序，而不是把所有量先压成一个 reward：

1. **Validity:** 进入1.5 m终点圈，在2.0 m圈内稳定保持2秒；
2. **Safety:** fall、torso-ground、sustained non-foot、force-qualified corrected slip、full-interval zero-foot；
3. **Capability:** mean best progress、support count、terrain-relative attitude、上/下坡预声明网格；
4. **Navigation:** qualified success rate、time to completion、path length、net progress/path ratio、route deviation、replanning count；
5. **Efficiency:** 只在 qualified trajectories 中比较原始任务时间和标定后的 relative mission energy，并显示 Pareto front；
6. **Generalisation:** 地图 seed 以64/16/16训练/验证/测试划分；地图、起终点和训练 seed 分开记录。

当前只有接触/坡度能力的条件性证据、V5 转向门失败，以及一个 direct-goal 完整地图 fresh-seed 失败。A*／waypoint Stage B 因 C1 前置门失败而未执行；第4–6项不能写成已经完成。

## 7. Report 模板章节映射（2,000–2,500词）

建议目标约2,350词，不含 Bibliography：

| Template heading | 建议词数 | 内容映射 |
|---|---:|---|
| Project Title | 不计或极少 | `From Reward Shaping to Terrain Locomotion: An Auditable PPO–Ant Study` |
| Project Overview | 250 | 问题、两阶段设计、系统、最重要结果和明确失败边界；不写“optimal navigation achieved” |
| Research Context | 450 | Ant-v5、PPO、reward shaping/misspecification、quadruped evaluation；解释三层 baseline，Farama 是无年份官方文档而非论文；Fu et al. 是步态方法锚点而非数值对照 |
| Division of Roles and Responsibilities | 250 | 按真实成员填写 research design、implementation、training/evaluation、visualisation/reporting；不得由模型臆造姓名或贡献 |
| Challenges and Solutions | 950 | Stage 1 reward/constraint iterations（约350）；Stage 2 local terrain + contact diagnosis/PAIR0（约350）；slope/turn-balance/full-map negative results（约250） |
| Contributions and Limitations | 450 | partial Stage 1 gains、PAIR0 support、坡度边界、V5 转向正式负结果和可复现证据；单训练 seed、无自然步态验证、无全局规划、完整地图失败、能耗未标定 |
| Bibliography | 不计 | Leeds Harvard；正文所有来源一一对应；project artefacts 可另列 data/code evidence subsection |

### 7.1 Report 的三个研究问题

- **RQ1:** Can bounded reward and constraint interventions improve selected flat-ground locomotion diagnostics relative to a matched Ant-v5/PPO baseline?
- **RQ2:** Can local terrain observation and an explicit contact intervention improve support and preserve progress on frozen continuous-terrain tests?
- **RQ3:** Do those improvements suffice for reliable full-map navigation and time–energy optimisation?

当前允许的回答依次为：`partial development evidence`、`conditional yes under one training seed and frozen tests`、`no evidence of sufficiency; V5 turning failed and the formal direct-goal map episode failed`。

## 8. 必须避免的表述

| 不可写 | 建议替换 |
|---|---|
| `The robot learned a natural gait.` | `Selected stability and action-quality diagnostics improved; natural gait was not operationally verified.` |
| `The method outperformed PPO.` | `The intervention improved selected outcomes relative to a matched PPO-trained Ant baseline.` |
| `The robot can climb slopes up to 12°.` | `The fixed policy passed every tested uphill gate from 4° to 12°; 16° was the first tested failure.` |
| `The robot is stable downhill to 16°.` | `The discrete 16° downhill test passed, but the downhill pattern was non-monotonic.` |
| `Energy consumption was optimised.` | `Several mechanical/action proxies were measured; no calibrated mission-energy or battery model was evaluated.` |
| `The optimal path was found.` | `No trajectory qualified for time–energy ranking, and no global planner was evaluated in the final full-map episode.` |
| `PAIR0 fixed MuJoCo.` | `PAIR0 reduced the observed support gap under the frozen contact model; it is not a universal physical truth or a MuJoCo bug claim.` |
| `One fresh seed proves failure.` | `One predeclared fresh seed provides an interpretable negative case, not a reliability estimate.` |

## 9. 制作前证据与合规检查

- [ ] 用户确认实际成员数、拼音姓名和每人的连续 slide block。
- [ ] 用户确认 Stage 1/Stage 2 与 V2/V3 的最终命名；Slide 2 保留 crosswalk。
- [ ] 所有结果图从 raw CSV/JSON 生成，并逐值核对源表。
- [ ] 每个 plot 标明单位、seed、`n` 和 sampling unit；不把 evaluation episode 当 training replicate。
- [ ] Slide 6 的 Stage 1 数字标为 development evidence；不写 held-out formal claim。
- [ ] Slide 8 明示 one training run；能耗结果标 `mixed, measurement-only`。
- [ ] Slide 9 保留全部坡度点和 PASS/FAIL，不插值物理极限。
- [ ] Slide 10 明示 one formal seed、seen development map、no global planner。
- [ ] Slide 11 不绘制虚构 Pareto front；显示 eligible trajectory count = 0。
- [ ] 每页右上 speaker 拼音、右下页码、底部短来源均存在且投影可读。
- [ ] 每页 notes 含完整 `[Sources]` block；project-generated 项写文件路径和相关 SHA-256。
- [ ] References 页全部 metadata、DOI/URL 和访问日期重新核验。
- [ ] 不使用 Cambridge 标志和动画；视频离线播放并准备静帧 fallback。
- [ ] 练习14:05完整版与约12:30恢复版；所有成员发言时间差尽量不超过30秒。
- [ ] 小组每位成员能够解释本人页面的指标、比较条件、限制和来源，且按课程规则披露 AI assistance。

## 10. 推荐的 Q&A 准备

1. **Why is Farama not the paper baseline?** 它定义软件环境；PPO/GAE论文提供算法历史，本地匹配运行才是可比实验 baseline。
2. **Did the project achieve natural gait?** 没有；仅部分姿态、平滑、起跳和接触诊断改善，步态相位/占空比/接触顺序未冻结。
3. **Why did PAIR0 not solve the full map?** 支撑、转向和全局选路是不同层级；PAIR0改善接触不等于学会全局地图规划。
4. **Was the path time- or energy-optimal?** 没有合格完成轨迹，因此没有任何轨迹有资格进入 Pareto 排序。
5. **What is the strongest result?** 在一个训练 seed 的冻结标准场景中，PAIR0把 pooled zero-foot 从24.042%降至3.208%，并保持 mean best progress ratio 1.051；该结论不外推到未见地图。
6. **What should be tested next?** 先做多独立 training seed 的左右转向研究；通过后再引入全局规划和地图级未见测试；最后标定 relative mission energy。
