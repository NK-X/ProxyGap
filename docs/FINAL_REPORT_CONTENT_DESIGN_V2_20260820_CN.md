# AI+ Project Report 最终内容设计 V2

适用模板：`D:\AI+ Project Report - Template 2026.docx`
目标提交：PDF，2,000–2,500词（Bibliography不计）
建议正文：约2,350词；Times New Roman ≥11 pt；1.15倍行距；Harvard author–date

## 1. 建议标题与摘要句

**Project Title**

`From Reward Shaping to Hierarchical Terrain Navigation: An Auditable PPO–Ant Study`

**一句话结果**

`A matched Ant-v5/PPO baseline was progressively extended with bounded reward terms, contact auditing and hierarchical known-map planning; the final system completed 6/6 formal episodes on one frozen map without falls or sustained corrected-slip events, but natural-gait, unseen-map and electrical-energy claims remain unsupported.`

## 2. 三个 Research Questions

- **RQ1:** Can bounded reward and constraint interventions improve selected flat-ground locomotion diagnostics relative to a matched Ant-v5/PPO baseline?
- **RQ2:** Which contact, terrain and control interventions preserve support and slope capability as the task moves from flat ground to a continuous heightfield?
- **RQ3:** Can a hierarchical system complete a frozen complex-map mission under time-priority, balanced and energy-priority candidate-selection criteria while satisfying predeclared safety constraints?

建议回答：RQ1 = partial development evidence；RQ2 = conditional evidence with clear failed interventions；RQ3 = yes on one known frozen map and three reset seeds, but not unseen-map generalisation or global optimality。

## 3. 模板章节映射与词数

| Template section | 建议词数 | 核心任务 |
|---|---:|---|
| Project Title | — | 简洁标题；不写“optimal”或“natural gait achieved” |
| Project Overview | 260 | 问题、两阶段设计、系统、6/6最终结果、主要限制 |
| Research Context | 470 | baseline、PPO/Ant provenance、reward shaping/misspecification、quadruped terrain literature |
| Division of Roles and Responsibilities | 220 | 按真实成员填写；不得臆造姓名或贡献 |
| Challenges and Solutions | 980 | Stage 1约300；Stage 2接触/坡度约320；转向/规划/多目标约360 |
| Contributions and Limitations | 420 | 贡献、正式结果、失败边界、泛化/能耗/步态限制 |
| Bibliography | 不计 | Leeds Harvard，正文一一对应 |

目标总词数约2,350。

## 4. Project Overview（约260词）

建议段落顺序：

1. **Problem:** Standard reward return does not ensure directed, stable or mission-valid quadruped behaviour.
2. **Stage 1:** A matched Ant-v5/PPO baseline was extended with speed, orientation, smoothness and contact-related diagnostics on flat ground.
3. **Stage 2:** The task moved to an 80 m × 80 m continuous heightfield, adding 13-D local terrain preview, PAIR0 contact handling, slope tests, turn diagnostics and finally hierarchical known-map planning.
4. **Final result:** Two selected route contracts × three formal seeds = 6/6 completion; no falls, torso-ground, sustained non-foot contact or duration-corrected sustained-slip events.
5. **Boundary:** Same previously inspected map; two unique contracts for three preference profiles; mechanical work is a proxy; no biological gait or global optimum claim.

不要在 Overview 中写长实验流水账。最后一句应点出研究意义：the project shows why locomotion, contact, turning and global navigation must be evaluated as separate layers.

## 5. Research Context（约470词）

### 5.1 Baseline 的准确表述

主 baseline 是**本地可执行、匹配预算的 Gymnasium Ant-v5 + PPO**。Farama Foundation Ant 页面是环境文档，不是论文；PPO 的算法来源是 Schulman et al. (2017)，Ant locomotion 的历史来源可联系 Schulman et al. (2016) 的 GAE work。

报告应写：

> Published papers provide methodological provenance, whereas numerical claims use locally reproduced matched conditions because environment versions, observations, rewards and training budgets differ across publications.

Stage 1 可以直接对 matched Ant/PPO baseline；Stage 2 不应让 default Ant 在复杂地图上作为唯一 comparator，而应采用 sequential ablations and the direct-goal baseline。

### 5.2 Literature roles

- **PPO/GAE:** model-free continuous-control algorithm provenance；
- **Fu et al. (2022):** mechanical energy and gait emergence; provides gait/energy evaluation ideas, not comparable scores；
- **Lee et al. (2020):** robust locomotion on challenging terrain；
- **Miki et al. (2022):** exteroceptive/local terrain information for anticipatory locomotion；
- **Aractingi et al. (2023):** command tracking, body orientation, effort and smoothness in quadruped DRL；
- **Ng et al. (1999):** policy-invariant potential-based shaping boundary；
- **Pan et al. (2022), Skalse et al. (2022):** reward misspecification/hacking motivation and the need for independent evaluation；
- **Raffin et al. (2022):** jerky stepwise exploration and smooth exploration alternatives。

## 6. Challenges and Solutions（约980词）

### 6.1 Stage 1 / V2：从默认前进奖励到意图诊断（约300词）

建议只保留有研究价值的迭代：

1. target-speed/direction tracking；
2. torso vertical/angular and signed-pitch shaping；
3. action-rate/saturation diagnostics and external slew control boundary；
4. foot-landing/contact terms；
5. independent path efficiency, heading, tilt and contact measurements。

用两个 paired development results 说明部分改善，同时报告速度或推进的代价。不要把视觉更协调写成自然步态已证明。

### 6.2 Stage 2A：地形观察与接触问题（约180词）

保留：

- 80 m × 80 m冻结高度场；
- 135-D = 122-D locomotion + 13-D local terrain preview；
- flat-plane versus flat-heightfield contact diagnosis；
- MuJoCo hfield–capsule margin interaction；
- explicit four floor–distal PAIR0 contact contract；
- PAIR0 held-out diagnostic: zero-foot约24.04%→3.21%，support约0.353→1.344，mean best-progress ratio1.051。

报告要区分：PAIR0改善了冻结接触模型下的支撑诊断，但不是“MuJoCo bug fixed”。

### 6.3 Stage 2B：坡度、速度、相位与转向（约140词）

值得保留的负结果：

- speed reduction 0.20–0.40 m/s未达到完整区间支撑改善门；
- grouped terrain-frame reward switch降低推进并增加fall，不保留；
- phase-crawl smoke未达到正式训练门；
- slope capability: uphill tested pass through12°, first tested fail16°；downhill non-monotonic；
- V5 balanced-turn exposure kept slope safety but failed left/right tracking gate。

这些结果说明 reward scanning 不能代替接口、接触和规划分层。

### 6.4 Stage 2C：为什么 direct-goal 失败（约120词）

正式 direct-goal final PAIR0 在600 s内没有进终点圈；best progress 14.51 m，net progress12.53 m，路径92.41 m。Policy 只有局部地形与终点方向，没有全图，也缺可靠双向转向。该负结果必须保留，因为它构成引入高层 route planning 的因果动机。

### 6.5 Stage 2D：最终分层系统（约240词）

写清以下顺序：

1. read-only checkpoint screen identified archived V4 as the only bidirectional candidate；
2. V4+PAIR0 standard slopes showed no falls/sustained slips；
3. naive V4/PAIR0 action blending fell after15.3 s and was rejected；
4. known-map planner generated feasible waypoint routes under a16° discrete slope proxy；
5. 15 route/speed candidates all first passed completion and safety gates；
6. weighted normalised time and positive-work proxy selected two unique contracts for three preference profiles；
7. three new hash-derived reset seeds validated both contracts。

强调 no retraining in the final system integration。V4 checkpoint 是 archived expert；PAIR0 是 contact contract；planner provides global route；PPO only follows local waypoints。

## 7. Final Results 的表图设计

### 7.1 必须放入的主表

| Contract | Success | Mean time (s) | Mean positive work proxy (J) | Mean path (m) | Sustained slips | Falls |
|---|---:|---:|---:|---:|---:|---:|
| time/balanced | 3/3 | 264.550 | 55,651.43 | 153.226 | 0 | 0 |
| energy | 3/3 | 259.367 | 55,134.17 | 152.144 | 0 | 0 |

### 7.2 建议图

1. **Figure 1:** V1→V2→V3 research timeline，标出保留/拒绝；
2. **Figure 2:** matched baseline → reward → diagnostics → hard gates；
3. **Figure 3:** PAIR0 paired zero-foot/support plot；
4. **Figure 4:** signed slope grid with all seed points；
5. **Figure 5:** system architecture and information boundary；
6. **Figure 6:** 15 feasible candidates in time–positive-work space；
7. **Figure 7:** formal per-seed time/work/path results；
8. **Figure 8:** final map planned route versus actual trajectory。

所有图显示单位、seed、sampling unit。不能把evaluation episodes当作independent training runs。

## 8. Contributions and Limitations（约420词）

### Contributions

1. An auditable two-stage experimental framework separating optimised reward from independent task/safety metrics.
2. A contact-focused diagnosis showing how hfield/capsule margin semantics affected distal support under the frozen model.
3. A lexicographic mission evaluation: validity and safety before time/energy preference.
4. A hierarchical known-map system that achieved 6/6 formal completions using two route contracts and three new reset seeds.
5. Exact video provenance: action/state/contact replay mismatch=0 and full-frame decoding.
6. Preserved failures showing which interventions were rejected and why.

### Limitations

- one known, previously inspected map；
- only three reset seeds and no independent training-seed replication for final integration；
- archived V4 expert selection was development-driven；
- route optimisation covered15 candidates, not the continuous global optimum；
- mechanical work proxy, no actuator/battery efficiency model；
- no real robot or sensor noise；
- persistent airborne exposure: representative runs had9.25–10.02% full-control zero-foot intervals；
- zero sustained-slip events does not mean zero instantaneous slip；
- natural gait was not operationally validated。

## 9. Evidence Appendix（建议不计入主词数或作为补充材料）

必须列出：

- final config and runner SHA-256；
- final formal manifest `0bf2817c…8d83`；
- candidate selection `212b7886…f20b`；
- video root manifest `60498def…c024`；
- three MP4 SHA-256；
- frozen map heights/hfield/XML hashes；
- V4 checkpoint SHA-256；
- exact replay mismatch counts；
- test result；
- Git commit and branch after publication。

## 10. 必须避免的表述

| 不可写 | 建议替换 |
|---|---|
| `The robot learned a natural gait.` | `Selected coordination and stability diagnostics improved; biological gait was not validated.` |
| `We beat PPO.` | `Selected interventions improved outcomes relative to a matched PPO-trained Ant baseline.` |
| `The final model learned the whole map.` | `A high-level known-map planner supplied local waypoints to a frozen low-level policy.` |
| `The route is optimal.` | `The route was preferred within a 15-candidate feasible bank under a declared weighted objective.` |
| `No slipping occurred.` | `No duration-corrected sustained slip event occurred; transient candidates and airborne intervals remained.` |
| `Energy was minimised.` | `Positive mechanical-work proxy contributed to candidate ranking; battery energy was not modelled.` |
| `The method generalises.` | `The system completed one known frozen map over three reset seeds.` |

## 11. Word/PDF 验收

- [ ] 正文2,000–2,500词，Bibliography不计；
- [ ] Times New Roman≥11 pt，1.15倍行距；
- [ ] 标题层级与模板一致，保留 instructor evaluation sheet；
- [ ] 所有外部和项目来源在正文引用并列入Bibliography/evidence appendix；
- [ ] 逐页检查图表、跨页表格、孤行和空白页；
- [ ] 输出PDF后逐页渲染为PNG视觉QA；
- [ ] 文件名、团队成员、拼音、角色与最终Git commit全部核对；
- [ ] 按课程规则披露AI assistance，不提交模型生成但未经成员理解的内容。
