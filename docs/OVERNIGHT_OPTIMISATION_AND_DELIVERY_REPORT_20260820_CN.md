# ProxyGap 通宵优化与最终交付报告

日期：2026-08-20（Europe/London）

## 1. 最终结论

本轮已经把此前“支撑改善但完整地图不到达”的负结果，推进为一个能够在同一冻结已知地图上稳定到达终点的分层系统。最终系统采用冻结 V4 双向低层专家、PAIR0 接触合同、已知地图高层路径规划与局部 waypoint 跟随；并未把完整地图输入 PPO，也未隐蔽重训最终专家。

两条预选路线合同在三个新 hash-derived reset seeds 上均完成任务：共 6/6 正式成功，0 fall、0 torso-ground、0 sustained non-foot contact、0 duration-corrected sustained-slip event。成功判据为进入终点 1.5 m 范围，并在 2 m 范围内连续保持 2 s。

这不是 globally optimal、unseen-map generalisation 或 natural-gait 结论。路线只是在 15 个实际评估、且先通过安全门的候选中按三种偏好排序；机械正功是 simulation proxy，不是电池能耗。代表成功回合仍有 9.25–10.02% 的完整控制区间四足无接触。

## 2. 本轮值得保留的优化链

1. **Checkpoint 双向转向筛选。** 冻结 V4 canonical-frame checkpoint 是已审计候选中唯一具有左/右响应的低层专家；final PAIR0 checkpoint 支撑较好，但转向合同未通过。
2. **V4 直接终点试验。** 直接目标控制取得约 30.75 m 最佳推进，但在 168.3 s 因 terrain-relative tilt fall，证明单纯换 checkpoint 不足以完成任务。
3. **动作混合反例。** V4 与 PAIR0 的 naïve action blending 在 15.3 s 摔倒，因此拒绝；该结果排除了“把两个模型简单平均”这一不可靠捷径。
4. **标准坡面安全筛选。** V4 + PAIR0 在 flat、+8°、−8°、+12° 场景中通过 safety screen，无 fall、torso-ground 或 sustained slip。
5. **已知地图 waypoint 路线。** 高层搜索使用冻结 1025×1025 地图及 16° discrete corridor-slope proxy；低层只接收局部 waypoint、局部地形和自身状态。原始 3 m lookahead 比缩短 lookahead 更稳健。
6. **完整五子步 evaluator。** 每个 0.05 s 控制区间审计五个 0.01 s physics substeps，独立计算支撑、force-qualified slip、持续事件、fall 与能耗代理。
7. **多目标候选筛选。** 15 个候选全部先通过 arrival 与 safety gates，再用 `J = wT(T/Tmin) + wE(W+/W+min)` 排序。
8. **三种用户偏好。** 时间优先 `(0.8, 0.2)` 与平衡 `(0.5, 0.5)` 选择同一合同；节能优先 `(0.2, 0.8)` 选择第二条合同。三种偏好对应两条真实路线，而不是伪造三条不同轨迹。
9. **正式多 seed 复核。** 两合同各用三个新 reset seeds；不根据结果换 seed、换 checkpoint 或挑中间 checkpoint。
10. **视频 exact replay。** 三段视频均与正式控制 trace 和五子步接触记录逐值一致，mismatch=0，并通过逐帧完整解码。

## 3. 正式结果

| Route contract | Success | Mean time | Mean positive-work proxy | Mean path | Sustained slip events | Falls |
|---|---:|---:|---:|---:|---:|---:|
| time / balanced | 3/3 | 264.55 s | 55.65 kJ | 153.23 m | 0 | 0 |
| energy | 3/3 | 259.37 s | 55.13 kJ | 152.14 m | 0 | 0 |

正式 seed variability 使 energy-selected route 在平均时间上也略快，因此不能把开发阶段的排序解释为稳定的物理 Pareto 前沿。应展示全部 seed 点，并称为 candidate-bank preference selection。

## 4. 三段代表视频

- 时间优先：`artifacts/dev/v4_pair0_multiobjective_full_map_video_v1_20260820/time_priority/v4_pair0_time_priority_seed_690223864_full_map_relief_v1.mp4`
- 平衡：`artifacts/dev/v4_pair0_multiobjective_full_map_video_v1_20260820/balanced/v4_pair0_balanced_seed_1864999454_full_map_relief_v1.mp4`
- 节能优先：`artifacts/dev/v4_pair0_multiobjective_full_map_video_v1_20260820/energy_priority/v4_pair0_energy_priority_seed_952993985_full_map_relief_v1.mp4`

视频根 manifest：`60498def5e209959e0ea9fd09629b71bc2e6df292dfb00d2140cf838e1ddc024`。

## 5. Baseline 与研究叙事

Baseline 应定义为本地匹配的 **Gymnasium Ant-v5 + PPO**：Farama Ant 页面定义环境但不是论文；PPO 论文提供算法 provenance；本地相同环境、预算、seed 规则和评估协议提供数值比较。Fu et al. (2022)、Lee et al. (2020)、Miki et al. (2022) 等提供 gait、mechanical-energy 和 terrain-locomotion 方法锚点，但机器人、DoF、控制接口和训练预算不同，不能直接移植其分数。

最终 presentation/report 分为两阶段：

- **Stage 1 / Project V2：** 相对 matched Ant-v5/PPO baseline，讨论 target speed、方向、姿态、动作平滑和 contact diagnostics 的改变；只称 selected diagnostics improved，不称 biologically natural gait proven。
- **Stage 2 / Project V3：** 重点不是让 default Ant 在复杂地图中陪跑，而是在 Stage 1 lineage 上呈现地形观察、PAIR0、坡度、转向、direct-goal 失败、高层规划和多目标 preference 的 sequential ablations。

## 6. Presentation 设计

已生成 13 页、建议 14:10 主讲时长的可编辑 PPTX。每页右上角有 speaker 拼音占位符，右下角有页码；每页同页来源，另有 References 页；不使用 Cambridge logo 或动画。

叙事顺序为：结果先行 → V1/V2/V3 时间线 → baseline → reward 与 hard gate → Stage 1 证据边界 → Stage 2 三层失败 → PAIR0/坡度 → 分层架构 → 三偏好选择 → 6/6 正式结果 → references → 结论边界与 Q&A。

交付：`deliverables/ProxyGap_Final_Presentation_Draft_20260820.pptx`。对应逐页口播和证据设计：`docs/FINAL_PRESENTATION_CONTENT_DESIGN_V2_20260820_CN.md`。

## 7. Report 设计

报告草稿从课程 `AI+ Project Report - Template 2026.docx` 生成，正文约 2,087 词，不计 Bibliography；Times New Roman 11 pt、1.15 行距；保留 instructor evaluation sheet。内容映射到 Project Overview、Research Context、Division of Roles and Responsibilities、Challenges and Solutions、Contributions and Limitations、Bibliography 和 Evidence Appendix。

交付：

- `deliverables/ProxyGap_Final_Report_Draft_20260820.docx`
- `deliverables/ProxyGap_Final_Report_Draft_20260820.pdf`
- 内容设计：`docs/FINAL_REPORT_CONTENT_DESIGN_V2_20260820_CN.md`

提交前必须由小组填写成员姓名、student IDs、真实分工、speaker 拼音和 course group，并共同复核 AI assistance disclosure。

## 8. 证据与验证

- candidate selection SHA-256：`212b78865714332cf28c9c1dc5dc1b8f7fb74883f03ef1ebee3789161c31f20b`；
- formal manifest SHA-256：`0bf2817cbdaadc02929da91bae7acb04371ff9cf1ec43e1ab4efa3a8a4a08d83`；
- video manifest SHA-256：`60498def5e209959e0ea9fd09629b71bc2e6df292dfb00d2140cf838e1ddc024`；
- V4 checkpoint SHA-256：`6a0f6081e6aff4c85201242e53c44b0d057e96167336002da3a6e862fe134b6a`；
- full test suite：408 passed、3 skipped；
- PPTX：13 pages，rendered visual QA passed，overflow test passed；
- report：7 PDF pages，visual QA passed，evaluation sheet retained；
- results local commit：`fc7c87a252c17aa53d9662fd7178dafd1b905632`。

## 9. GitHub 状态

本地结果 commit 已生成；远端目标预定为 `https://github.com/NK-X/ProxyGap.git` 的 `agent/slope-support-relief-v1` 分支。由于发布包含约 44 MiB 的正式 traces、三段 MP4 和一个 model checkpoint，最终 push 需要用户在看到目标与内容后明确确认。未确认前不得声称 GitHub 上传完成。
