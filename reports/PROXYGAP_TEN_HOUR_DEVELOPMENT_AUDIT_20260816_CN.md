# ProxyGap 十小时开发探索与证据审计

**日期：** 2026-08-16<br>
**状态：** Development evidence only；尚未进入正式 held-out 验证<br>
**适用范围：** 默认平地 `Ant-v5` 仿真、PPO、CPU。本文不讨论真实机器人部署，也不对地形、摩擦变化或外力扰动鲁棒性作出结论。

## 1. 结论先行

本轮完成了实现修正后的 6 条开发条件、3 个 training seeds、3 个 checkpoints 和每个 checkpoint 10 个固定 evaluation episodes，共训练 18 个策略并评价 540 个 episode。数据质量、奖励分解、逐步日志和视频均通过工程 QA。

然而，6 条候选条件在主要判定下的意图合规率均为 0%，没有任何条件通过预先声明的 development gate。因此，本轮没有合理证据支持“当前姿态 shaping、横向速度 shaping 与动作变化外部约束已经缓解 reward misspecification”，也没有启动 1M 延长训练或正式 held-out seeds。

当前最重要的机制发现是：动作变化约束显著改变了施加到环境的动作，并改善了部分前进、终止和路径指标，但策略自身仍提出非常粗糙的动作。约 98% 的步骤需要外部修正，说明“执行轨迹较平滑”主要是外部约束机械保证的，不等于 PPO 学会了平滑控制。横向 shaping 在三个 training seeds 上也没有形成一致改善。

## 2. 研究问题与边界

Gymnasium 对 Ant 的官方任务描述是协调四条腿向正 x 方向运动；默认奖励由前进、健康、控制代价和接触代价组成（Farama Foundation, 2026）。本项目额外关心的是：代理奖励是否能代表研究者预先声明的“有效、直立、方向可控且动作不过度剧烈的平地运动”。这属于对项目构念的审计，而不是断言 Ant-v5 或 PPO 普遍错误。

本文只检验默认平地 Ant-v5。没有改变地形、摩擦或施加外力，也没有真实硬件。因此即使候选方案成功，结论也只能是“在默认平地 Ant-v5 仿真中有效”。

## 3. 意图、代理奖励和研究目标的分离

三层概念必须分开：

1. **平台任务：** Ant 向正 x 方向移动，episode 最长 1,000 steps。
2. **PPO 可见代理：** 环境逐步返回的 scalar reward。
3. **研究者意图：** 在固定 1,000-step 评价期内有效前进、避免不健康终止和持续倒置、控制方向，并使用不过度剧烈的动作。

主要合规判定是预先声明的多域门槛，不把所有行为强行压成一个未经验证的“true reward”。净前进、路径效率、横向速度误差、躯干倾斜、终止、动作粗糙度和动作饱和均分别保留。`scalar_true_performance` 明确记为 `not_defined`。

## 4. 候选干预

基础代理为：

$$
r_t^{base}=r_t^{forward}+r_t^{healthy}-0.5\lVert a_t\rVert_2^2-r_t^{contact}.
$$

姿态 shaping 固定为：

$$
\phi_\theta(\theta_t)=\frac{1-\cos\theta_t}{2},\qquad
r_t^\theta=-0.1\phi_\theta(\theta_t).
$$

横向速度 shaping 使用 Ant 默认观测中可见的躯干 y 速度：

$$
\phi_y(v_{y,t})=\tanh\left[\left(\frac{v_{y,t}}{1\,\mathrm{m\,s^{-1}}}\right)^2\right],
\qquad r_t^y=-\lambda_y\phi_y(v_{y,t}),
$$

其中 $\lambda_y\in\{0,0.05,0.10\}$。动作外部约束把每步动作变化的 L2 范数限制为 1.1。它是执行层 projection，不是 reward 项。

Round 2 曾错误使用策略不可见的绝对横向位置作为 shaping 信号。该问题在查看结果前被识别并登记为设计偏差；Round 3 改用默认观测可见的横向速度，并为所有条件统一加入前一时刻施加动作，使比较保持可观测性一致。

## 5. 实验矩阵和随机性

- 6 个条件：3 个横向 shaping 权重 × 有/无动作变化约束；姿态 shaping 权重固定为 0.1。
- Training seeds：41301、41302、41303。每个 seed 从不同的网络初始化、环境随机序列和 PPO minibatch 顺序开始，形成三个独立训练策略。
- Evaluation seeds：51301–51310。它们为每个已训练策略生成相同的一组初始扰动，以便配对比较；它们不会重新训练网络，也不能代替独立 training seeds。
- 训练预算：每策略 300,000 steps；checkpoints 为 100k、200k、300k。
- 每个 checkpoint：10 个 evaluation episodes，每个最多 1,000 steps。

这些 seeds 已参与开发，因此只能用于 candidate selection 和工程诊断。预留的 42001–42008 尚未使用，只有冻结候选后才能用于正式验证。

## 6. 工程和数据质量

- 18/18 策略完成；54/54 checkpoints；540/540 episode rows。
- 540/540 step logs，共 334,966 个实际轨迹 steps。
- episode 主键重复数 0；step 主键重复数 0；缺失 episode keys 0。
- 基础奖励重算最大绝对误差 $3.18\times10^{-12}$。
- 控制代价重算最大绝对误差 0。
- 逐步奖励恒等式最大误差 $1.95\times10^{-15}$。
- 横向速度信号 QA：540 rows、0 重复，横向 reward 最大重算误差 $1.11\times10^{-16}$。

这些结果证明日志和公式实现一致，但不能代替 construct validity，也不能证明干预有效。

## 7. 主要结果

300k endpoint 下，所有条件的主要意图合规率均为 0%。无动作约束条件的平均前进速度为 0.064–0.141 m/s，不健康终止率为 0.60–0.83。加入动作变化约束后，平均前进速度提高至 0.488–0.686 m/s，不健康终止率降低至 0.13–0.37，但仍存在姿态、方向或路径域失败。

横向速度 shaping 没有跨三个 training seeds 形成一致效应：

- 无动作约束时，$\lambda_y=0.05$ 的平均绝对横向速度误差在三个 seeds 中全部恶化；$\lambda_y=0.10$ 只在一个 seed 改善。
- 有动作约束时，$\lambda_y=0.05$ 只在一个 seed 改善；$\lambda_y=0.10$ 在两个 seeds 改善、一个 seed 恶化。

阈值敏感性分析覆盖 729 个组合。$\lambda_y=0.05$ 加约束只在 14.8% 的阈值网格中呈现复制方向，平均配对合规率变化仅 +3.13 个百分点；$\lambda_y=0.10$ 加约束对应 6.58% 和 +0.71 个百分点。主要预声明阈值结论不变。

## 8. 外部约束的真实含义

动作变化约束后，施加动作的归一化 roughness 约为 0.0377，但策略提出动作的 roughness 仍为 0.249–0.271。约 98.4–98.7% 的 steps 发生约束干预，平均动作修正 L2 范数约 1.08–1.11。

因此不能说“PPO 学会了平滑动作”。更准确的结论是：外部 action projection 把粗糙提议转换为受限动作，并由此改变了运动结果。若未来要研究学习到的平滑控制，应把 proposed action 与 applied action 分开报告，并考虑在开发阶段引入动作变化 reward、合适的控制接口或平滑探索，而不能只看施加动作。

## 9. 视频证据

共生成并 QA 通过 8 个 MP4。每个视频为 20 fps、50.0 s 时间线、1,000 frames；提前终止后保留终止画面并明确填充，避免把短视频误解为完整成功轨迹。渲染地面仅扩大可视范围，没有改变物理世界或策略轨迹。

视频支持数值诊断：姿态-only baseline 在 300k checkpoint 的示例因 high-z excursion 提前终止；$\lambda_y=0.05$ 加约束的示例完成 horizon，但产生明显横向位移；$\lambda_y=0.10$ 加约束的示例后期躯干接近倒置。视频只作定性解释，不代替多 seed 数值证据。

## 10. 可支持与不可支持的结论

**当前可支持：**

- Round 3 的实现、日志、奖励分解和视频管线通过工程 QA。
- 在当前 300k、3 个 development seeds 中，动作外部约束显著改变了执行行为，但现有候选未达到完整意图门槛。
- 横向速度 shaping 的效应高度依赖 training seed，没有形成一致缓解证据。
- 外部约束改善的 applied action 不等于策略学会生成平滑 proposed action。

**当前不可支持：**

- 不能宣称已正式证明 reward hacking。
- 不能宣称当前 shaped reward 或约束已经缓解 reward misspecification。
- 不能把开发 seeds 当作 held-out confirmation。
- 不能把结果外推到其他地形、摩擦、外力、算法或真实机器人。
- 不能把 0% 合规简单解释为“奖励一定错误”；训练预算、动作接口和 PPO 优化也仍是替代解释。

## 11. 停止决定与三日安排

因为没有候选通过 development gate，本轮停止继续调权重，未启动 1M 扩展和正式训练。这避免了在看见结果后不断增加 reward 项直到得到阳性结果。

未来三天最有效率的路径是：

1. **第 1 天：** 冻结本轮负面开发结论；在“继续一次受限的控制设计开发”与“把项目定稿为默认奖励构念审计加失败缓解尝试”之间作出选择。若继续，最多批准一个新的机制问题，不进行开放式权重搜索。
2. **第 2 天：** 只有新候选通过独立开发门槛才动用 held-out training seeds；否则不伪造正式阳性结果，转入负面结果报告。
3. **第 3 天：** 完成证据矩阵、seed-level 图、视频索引、重现命令、限制和报告。
4. **随后 1.5 天：** 只做表达、图表、引用、PPT 和交付 QA，不再改变科学问题。

如批准最后一次受限开发，最合理的单一问题不是继续微调 $\lambda_y$，而是检验“策略为什么持续提出粗糙动作”。可比较一种预先固定的 action-difference reward 或平滑探索/控制接口，同时保持 flat-ground Ant-v5、PPO、训练预算和评价合同不变。该步骤属于新版本开发，不能与本轮数据合并为同一次确认实验。

## 12. 量化评价矩阵

助教要求的“accuracy matrix”在本项目中不应伪装成分类混淆矩阵。最合适的是 domain-compliance matrix：每个 condition × training seed 的单元格报告符合意图的 evaluation episodes 百分比，并同时呈现前进、姿态、方向、路径、动作和健康各域通过率。阶段二若存在正式候选，可再报告配对 mitigation success rate：

$$
\text{Mitigation success rate}=\frac{\text{同时改善意图且保持前进能力的 held-out seed pairs}}{\text{全部 held-out seed pairs}}\times100\%.
$$

该百分数必须在正式阶段前冻结“改善”和“保持”的阈值；当前尚未冻结，也没有候选可用于计算正式缓解成功率。

## 13. 参考文献

Achiam, J., Held, D., Tamar, A. and Abbeel, P. (2017) ‘Constrained policy optimization’, *Proceedings of the 34th International Conference on Machine Learning*, 70, pp. 22–31. Available at: https://proceedings.mlr.press/v70/achiam17a.html.

Aractingi, M., Cumin, J., Stasse, O. and Righetti, L. (2023) ‘Learning agile locomotion skills with a mentor’, *Scientific Reports*, 13, 11045. Available at: https://www.nature.com/articles/s41598-023-38259-7.

Farama Foundation (2026) ‘Ant’. *Gymnasium Documentation*. Available at: https://gymnasium.farama.org/environments/mujoco/ant/ (Accessed: 16 August 2026).

Pan, A., Bhatia, K. and Steinhardt, J. (2022) ‘The effects of reward misspecification: mapping and mitigating misaligned models’, *arXiv*. Available at: https://arxiv.org/abs/2201.03544.

Raffin, A., Kober, J. and Stulp, F. (2022) ‘Smooth exploration for robotic reinforcement learning’, *Proceedings of the 5th Conference on Robot Learning*, 164, pp. 1634–1644. Available at: https://proceedings.mlr.press/v164/raffin22a.html.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017) ‘Proximal policy optimization algorithms’, *arXiv*. Available at: https://arxiv.org/abs/1707.06347.

Skalse, J., Howe, N., Krasheninnikov, D. and Krueger, D. (2022) ‘Defining and characterizing reward gaming’, *Advances in Neural Information Processing Systems*, 35. Available at: https://arxiv.org/abs/2209.13085.
