# ProxyGap 最终目标完成审计（2026-08-20）

## 1. 审计结论

本轮预声明目标已经完成，可以封存模型与实验结果：冻结四足机器人系统在同一已知完整地图上，分别按照时间优先、时间—能耗平衡、能耗优先三种评价体系选择可行方案，并完成了正式到达、两秒空间保持和安全门。两个唯一正式路线合同、每个合同三个全新哈希派生 reset seed，共 6/6 轮到达终点。

这里的“完成”严格限定为已知地图系统集成结果。它不等于低层策略独立学会全局规划，也不等于未知地图泛化、连续空间全局最优、自然步态或真实电池能耗最优。

## 2. 用户目标逐项验收

| 目标 | 证据 | 状态 |
|---|---|---|
| 三种时间—能耗权重 | 时间优先 `[0.8, 0.2]`、平衡 `[0.5, 0.5]`、能耗优先 `[0.2, 0.8]`；15 个通过有效性和安全门的候选进入排序 | 通过 |
| 完整地图到达终点 | 2 个唯一路线合同 × 3 个新 reset seed；6/6 轮进入 1.5 m 到达圈，并在 2 m 圈内连续保持 2 s | 通过 |
| 不滑倒或少滑倒 | 6/6 轮均无摔倒、躯干触地、持续非足端接触或时长修正的持续滑动事件 | 通过 |
| 可核验结果视频 | 三个代表性正式回放，均与正式 trace 状态及五个物理子步接触字段逐值一致，并完成全帧解码 | 通过 |
| 回顾 V1–V3 迭代 | Report、13 页 Presentation、15 分钟讲稿均包含 legacy V1、Project V2、Project V3、保留/拒绝的关键迭代 | 通过 |
| Baseline 研究设计 | 区分 Farama Ant-v5 环境文档、PPO 算法来源和本地 matched Ant-v5+PPO 数值 baseline；相关论文作为方法锚点而非伪造的同条件分数 | 通过 |
| 最终 Report | 基于课程模板生成可编辑 DOCX；正文 2,087 词；保留 evaluation sheet；另有 PDF 预览版 | 通过 |
| 15 分钟 Presentation | 13 页、预估 14:10；每页含 speaker 拼音占位、页码和同页来源提示；另附 1,682 词英文讲稿 | 通过 |

## 3. 三种评价体系与正式结果

只有先通过到达、安全和持续滑动门的候选才进入偏好排序。排序目标为：

`J = w_T (T / T_min) + w_E (W+ / W+_min)`

其中 `T` 是任务完成时间，`W+` 是正机械功代理。该能耗量不是电池焦耳。

| 偏好 | 权重 `[时间, 能耗]` | 选中候选/合同 | 正式结论 |
|---|---:|---|---|
| 时间优先 | `[0.8, 0.2]` | `s1p50_t1p00` / `time_and_balanced` | 使用该合同的 3/3 正式轮次到达 |
| 平衡 | `[0.5, 0.5]` | `s1p50_t1p00` / `time_and_balanced` | 使用该合同的 3/3 正式轮次到达 |
| 能耗优先 | `[0.2, 0.8]` | `balanced_speed_0p50` / `energy_priority` | 3/3 正式轮次到达 |

正式均值：

- 时间/平衡合同：264.55 s，正机械功代理 55,651.43 J，路径 153.226 m。
- 能耗优先合同：259.3667 s，正机械功代理 55,134.17 J，路径 152.144 m。
- 6/6 轮的结束原因均为 `arrival_and_two_second_spatial_hold`。
- 6/6 轮的摔倒、躯干触地、持续非足端接触和时长修正持续滑动事件均为 0。

reset 变化使能耗优先合同的正式平均时间反而略短。因此不能把开发候选上的排序差异描述为确定性的 Pareto 规律；应展示全部 seed 点，并将结论限定为已评估候选库中的近优选择。

## 4. 正式视频验收

视频根目录：

`artifacts/dev/v4_pair0_multiobjective_full_map_video_v1_20260820/`

视频 manifest SHA-256：

`60498def5e209959e0ea9fd09629b71bc2e6df292dfb00d2140cf838e1ddc024`

| 偏好 | 正式 seed | MP4 相对路径 | SHA-256 | 解码 |
|---|---:|---|---|---|
| 时间优先 | 690223864 | `time_priority/v4_pair0_time_priority_seed_690223864_full_map_relief_v1.mp4` | `a558886d4671525172f38ff54d84effd8bcffbf0db6fa94a50e0516f10b0808f` | 302/302，1280×720，20 fps |
| 平衡 | 1864999454 | `balanced/v4_pair0_balanced_seed_1864999454_full_map_relief_v1.mp4` | `073a419dd2cfd8c159c9bdc27b81912e7839cf2250bb896b10e90cf4f6b37ce8` | 315/315，1280×720，20 fps |
| 能耗优先 | 952993985 | `energy_priority/v4_pair0_energy_priority_seed_952993985_full_map_relief_v1.mp4` | `c47c8a3a54a676808802766896778cecee4e1d8e88432df8c8ffe9342be14c23` | 327/327，1280×720，20 fps |

三个回放与各自正式 trace 的状态字段和五个物理子步接触字段均为零差异。视频是压缩时间的正式回放，不是另选 seed 的演示轨迹。

## 5. 关键证据与交付文件

| 文件 | SHA-256 |
|---|---|
| `configs/v4_pair0_multiobjective_full_map_final_v1_20260820.json` | `18a377013dc45b31619a256ae172ec6d0066c4d0a02cce1d0a691c1ee8a9fe6f` |
| `artifacts/dev/v4_multiobjective_candidate_selection_v1_20260820/selection.json` | `212b78865714332cf28c9c1dc5dc1b8f7fb74883f03ef1ebee3789161c31f20b` |
| `artifacts/dev/v4_pair0_multiobjective_full_map_final_v1_20260820/attempt_0/manifest.json` | `0bf2817cbdaadc02929da91bae7acb04371ff9cf1ec43e1ab4efa3a8a4a08d83` |
| `deliverables/ProxyGap_Final_Presentation_Draft_20260820.pptx` | `d6bd4036d64c40457e9ce14681feb74e4b176439771e5cb4fd80982e497c9a94` |
| `deliverables/ProxyGap_Final_Report_Draft_20260820.docx` | `979f074fc9f9ba4815e74372efe8bf04a874326f5eb4613451535016c1eb1463` |
| `deliverables/ProxyGap_Final_Report_Draft_20260820.pdf` | `39e3791cd86a22cfc6c51d82ffa411004918e569bf422aa2bd30828d7a33987e` |
| `docs/FINAL_PRESENTATION_SPEAKER_SCRIPT_15MIN_20260820_EN.md` | `088d98a1460ce8b9943a9eed176c0872f0c6142056643300636f8ff3d6baebf4` |
| `docs/OVERNIGHT_OPTIMISATION_AND_DELIVERY_REPORT_20260820_CN.md` | `598392b009c504a3e1da7e0058bc1bbafae0e94560077593d8c7fc3797054952` |
| `docs/V4_PAIR0_MULTIOBJECTIVE_FULL_MAP_FINAL_REPORT_20260820_CN.md` | `98e9f6aeaf4bca51ddde95efb658f7e5a8510f5a70a092b8f6925797ba87b226` |

正式结果 artifact manifest 的 43/43 项、视频 manifest 的 16/16 项均已重新枚举并复算文件大小与 SHA-256：无缺失、无额外文件、无哈希或大小不匹配。

## 6. 验证状态

- 全库测试：`408 passed, 3 skipped`，用时 73.36 s。
- Git whitespace gate：`git -c core.whitespace=cr-at-eol diff --check origin/agent/slope-support-relief-v1..HEAD` 通过。
- 正式 runner 未训练、未写新 checkpoint，源 checkpoint timestep 与 SHA-256 保持不变。
- Report PDF 共 7 页并完成逐页渲染检查；PPTX 共 13 页并完成整页渲染及溢出检查。

## 7. 必须保留的限制

1. 最终证据来自一个已知、曾检查过的冻结地图，而不是未知地图集合。
2. 最终正式评估只有两个唯一路线合同、每个三个 reset seed；没有独立训练 seed 复验。
3. 15 个可行候选不能证明连续路线和控制器空间的全局最优。
4. 正机械功是仿真代理，不是经过标定的电池能耗。
5. 成功代表视频仍有约 9.25%–10.02% 的完整控制区间没有足端接触；因此不能声称自然步态或持续地面支撑。
6. “零持续滑动事件”不等于没有任何瞬时滑动候选。
7. 高层使用冻结已知地图；低层策略没有学会全图规划。
8. GitHub 上传尚未执行。提交和推送必须在用户确认远端、分支和约 53 MiB 的待推送内容后单独进行。

## 8. 封存决定

模型与实验层面停止继续无约束优化。后续工作只包括：将团队成员姓名和真实分工填入 Report/PPT、根据课程要求微调英文措辞、确认 GitHub 发布范围并推送。任何未知地图泛化、真实能耗、自然步态或训练 seed 稳健性研究，应作为新协议，不应回写本轮结论。
