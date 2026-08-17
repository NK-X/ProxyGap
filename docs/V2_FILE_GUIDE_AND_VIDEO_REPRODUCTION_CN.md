# ProxyGap V2 重要文件导览与视频复现说明

**适用版本：** V2 current

**核对日期：** 17 August 2026

**用途：** 帮助新组员先找到正确文件，再区分“运行代码”“重新训练”和“复现已有视频”。

## 1. 先说结论

当前公开仓库的训练和渲染代码可以运行，但 GitHub 副本**不能单独重建已有的对应视频**。原因如下：

1. 公开仓库有渲染脚本、配置和视频索引，但按发布边界排除了训练模型 `*.zip` 和完整视频 `*.mp4`；
2. 完整视频渲染必须读取一个明确的 PPO checkpoint；
3. `environment.yml` 尚未声明 MP4 编码需要的 `imageio` 和 `imageio-ffmpeg`；
4. 精确复现还需要相同的配置、渲染 XML、training seed、evaluation seed、软件版本和模型哈希。

因此，这不是 PPO 或 MuJoCo 渲染逻辑已经损坏。更准确的诊断是：

- **核心渲染路径：已实测可用；**
- **公开交接包：有意不包含大文件，因此不足以独立重建已有视频；**
- **依赖声明：存在一个需要后续修复的打包缺口。**

17 August 2026 的本地验证使用本仓库的 `scripts/render_stage1_full_video.py` 和外部保存的 V2 checkpoint，成功生成了 1,000 帧、20 fps、50.0 s、播放倍率 1.0 的完整 MP4。该验证没有修改训练、奖励、评价或渲染源代码。

## 2. 三分钟阅读顺序

| 顺序 | 文件 | 先从这里获得什么 |
|---|---|---|
| 1 | [`README.md`](../README.md) | 项目目标、V1/V2 边界和目录总图 |
| 2 | [`STATUS.md`](../STATUS.md) | 现在已经完成什么，什么还没有冻结 |
| 3 | [`current/README.md`](../current/README.md) | V2 的工作顺序和当前科学门槛 |
| 4 | [`current/RESEARCH_DIRECTION_V2.md`](../current/RESEARCH_DIRECTION_V2.md) | 当前研究问题、人的意图、约束与 shaping 的角色 |
| 5 | [`docs/INTENDED_BEHAVIOUR_CONTRACT_V2_20260816.md`](INTENDED_BEHAVIOUR_CONTRACT_V2_20260816.md) | 已有行为指标和阈值；它仍需为 gait/contact 评价继续修订 |
| 6 | [`handoff/RUN_REGISTRY.csv`](../handoff/RUN_REGISTRY.csv) | 每一批实验属于 smoke、development、legacy 还是未来 formal |
| 7 | [`handoff/KNOWN_ISSUES.md`](../handoff/KNOWN_ISSUES.md) | 已知缺陷、解释边界和不能声称的结论 |

不要从 `configs/formal_v1_*.yaml` 或 `legacy/weight_sweep_v1/` 开始新的 V2 实验。它们用于保存旧设计的历史证据。

## 3. V2 研究逻辑文件

### 3.1 当前方向与科学边界

| 文件 | 作用 | 是否直接运行 |
|---|---|---|
| [`current/RESEARCH_DIRECTION_V2.md`](../current/RESEARCH_DIRECTION_V2.md) | V2 的 canonical research direction；区分核心任务、安全约束和步态质量偏好 | 否 |
| [`STATUS.md`](../STATUS.md) | 各工作流当前状态和下一道 gate | 否 |
| [`docs/INTENDED_BEHAVIOUR_CONTRACT_V2_20260816.md`](INTENDED_BEHAVIOUR_CONTRACT_V2_20260816.md) | 将“希望机器人怎样运动”转换为可计算诊断；不能当作 Gymnasium 官方保证 | 否 |
| [`protocols/BODY_DYNAMICS_REPLICATION_GATE_V1_20260817.md`](../protocols/BODY_DYNAMICS_REPLICATION_GATE_V1_20260817.md) | 最近一次 body-dynamics development replication 的冻结规则 | 否 |
| [`configs/body_dynamics_replication_v1_20260817.json`](../configs/body_dynamics_replication_v1_20260817.json) | 上述 replication 的可执行配置、seeds、PPO 参数、预算和输出路径 | 是，作为脚本输入 |

配置文件公开并不表示实验已经公开完成。V2 的新模型、原始日志和视频目前仍保留在本地证据包中。

### 3.2 核心 Python 模块

| 文件 | 负责什么 | 初学者应关注什么 |
|---|---|---|
| [`src/proxygap/ant_wrapper.py`](../src/proxygap/ant_wrapper.py) | 创建 Ant-v5，分解 reward，加入 shaping/constraint 候选，记录逐步信息 | PPO 实际看到的 reward 和研究者记录的 diagnostics 在这里分开 |
| [`src/proxygap/metrics.py`](../src/proxygap/metrics.py) | 计算前进、方向、路径效率、姿态、终止、动作粗糙度、饱和率等 episode 指标 | 没有把所有表现偷偷合并成一个 `true_performance` 总分 |
| [`src/proxygap/experiment.py`](../src/proxygap/experiment.py) | PPO 训练、25/50/75/100% checkpoints、重复评价和 CSV 输出 | training seed 生成独立策略；evaluation episodes 是嵌套重复测量 |
| [`src/proxygap/protocol.py`](../src/proxygap/protocol.py) | 检查配置是否满足协议冻结条件 | 工程测试通过不等于科学设计已经冻结 |
| [`src/proxygap/__init__.py`](../src/proxygap/__init__.py) | 对脚本暴露常用环境和指标接口 | 通常不需要直接修改 |

`reference_baseline.py`、`selection.py`、`stage1.py` 和部分旧分析模块仍保留以支持历史可追溯性。新组员不应仅凭文件名把它们当作当前 V2 主入口。

## 4. 最近 V2 development replication 的数据流

```text
configs/body_dynamics_replication_v1_20260817.json
                      |
                      v
scripts/run_body_smoothness_gsde_matrix.py
                      |
                      +--> artifacts/.../models/.../checkpoint_*.zip
                      +--> artifacts/.../logs/*.csv
                      |
                      v
scripts/analyse_body_dynamics_replication.py
                      |
                      +--> development summary tables and figures
                      |
                      v
scripts/render_body_smoothness_gsde_videos.py
                      |
                      v
scripts/render_stage1_full_video.py
                      |
                      +--> complete MP4
                      +--> JSON manifest with model/video hashes and episode metrics
```

对应文件的角色如下：

| 文件 | 作用 |
|---|---|
| [`scripts/run_body_smoothness_gsde_matrix.py`](../scripts/run_body_smoothness_gsde_matrix.py) | 读取 matrix 或 replication 配置，训练每个 condition × training-seed 任务，并拒绝覆盖非空输出目录 |
| [`scripts/analyse_body_dynamics_replication.py`](../scripts/analyse_body_dynamics_replication.py) | 对 `B0__G0_REP` 与 `B1__G0_REP` 做配对 development 分析 |
| [`scripts/render_body_smoothness_gsde_videos.py`](../scripts/render_body_smoothness_gsde_videos.py) | 按配置寻找 endpoint checkpoints，并批量调用完整视频渲染器 |
| [`scripts/render_stage1_full_video.py`](../scripts/render_stage1_full_video.py) | 从一个明确 checkpoint 重放一个完整 episode，写 MP4 和 JSON manifest |
| [`assets/ant_render_large_floor.xml`](../assets/ant_render_large_floor.xml) | 扩大视觉地面；只改变渲染场景，不改变已训练策略的物理轨迹 |
| [`scripts/_portable_runtime.py`](../scripts/_portable_runtime.py) | 查找字体和可选 `imageio-ffmpeg` 安装位置 |
| [`tests/test_body_dynamics_replication_analysis.py`](../tests/test_body_dynamics_replication_analysis.py) | 检查 replication 分析中的关键计算和 schema |

## 5. “生成对应视频”到底有三种不同要求

### A. 只检查本机能否显示 MuJoCo

这不需要训练模型：

```powershell
python scripts/smoke_render_video.py `
  --frames 24 `
  --backend glfw `
  --output artifacts/videos/ant_v5_render_smoke.gif
```

它使用随机动作，只验证渲染通路，不代表训练结果，也不能替代正式视频。

### B. 从已有 checkpoint 重建相应视频

这需要：

- `checkpoint_*.zip`；
- checkpoint 对应的 JSON 配置；
- 相同的渲染 XML；
- training seed 与 evaluation seed；
- 与训练模型兼容的软件版本；
- `imageio` 和 `imageio-ffmpeg`。

当完整 replication 的 `artifacts/` 证据包已经按配置中的相对路径放回仓库副本后，可运行：

```powershell
python scripts/render_body_smoothness_gsde_videos.py `
  --config configs/body_dynamics_replication_v1_20260817.json `
  --evaluation-seed 51601 `
  --output-dir-name full_horizon_videos
```

该命令会寻找以下结构：

```text
artifacts/dev/body_dynamics_replication_v1_20260817/
  runs/seed_41601/B0__G0_REP/models/B0__G0_REP/checkpoint_1000000.zip
  runs/seed_41601/B1__G0_REP/models/B1__G0_REP/checkpoint_1000000.zip
  ...
```

如果这些文件不存在，脚本失败是正确行为：它不能仅凭一个配置文件恢复已经学习到的神经网络参数。

### C. 从代码重新训练，再生成一批新视频

```powershell
python scripts/run_body_smoothness_gsde_matrix.py `
  --config configs/body_dynamics_replication_v1_20260817.json
```

训练完成后再运行上一节的批量视频命令。该配置需要完成 2 conditions × 3 training seeds × 1,000,000 timesteps，不能把它当作快速视频命令。即使 seeds 相同，不同 CPU、库版本或底层数值实现也可能使重新训练结果无法与旧模型逐位一致，因此应称为**重复运行设计**，而不是自动等同于**重建同一模型**。

## 6. Windows 安装与视频依赖

仓库现有基础安装命令为：

```powershell
conda env create -f environment.yml
conda activate proxygap-ant
python -m pip install -e .
```

当前 `environment.yml` 没有声明视频编码依赖。生成 MP4 前还需运行：

```powershell
python -m pip install imageio imageio-ffmpeg==0.6.0
```

快速核对：

```powershell
python -c "import gymnasium, mujoco, imageio, imageio_ffmpeg; print('video dependencies OK')"
```

本说明记录缺口，但遵守本轮范围，没有修改 `environment.yml` 或任何源代码。后续应通过单独的依赖修复提交把它加入环境声明，并增加渲染依赖测试。

## 7. 常见报错如何判断

| 现象或报错 | 最可能原因 | 处理 |
|---|---|---|
| `FileNotFoundError: ...checkpoint_*.zip` | GitHub 没有发布模型，或证据包放错路径 | 获取正确 checkpoint；核对 config、seed、timesteps 和 SHA-256 |
| `No module named imageio_ffmpeg` | MP4 编码依赖未安装 | 安装 `imageio` 与 `imageio-ffmpeg==0.6.0` |
| GLFW/OpenGL/MuJoCo context error | 机器没有可用渲染后端或驱动 | 先运行 GIF smoke；核对显卡驱动和 `--backend glfw` |
| 只能看到 JSON/CSV 中的视频名称 | public results 只保存 sanitised index | 从独立证据包取得 MP4；索引中的 `not_committed/` 是有意标记 |
| 新生成视频与旧视频动作不同 | checkpoint、seed、config、XML 或软件版本不一致 | 比较 model hash、config hash、commit SHA 和环境版本 |
| 50 秒视频中机器人提前停止，后面画面静止 | episode 已提前终止且使用 `--pad_to_horizon` | 查看 JSON 中的 `trajectory_frames`、`padded_frames` 和 `termination_category` |

## 8. 给新组员的最小交接包

若目标是让组员**查看和核验已有视频**，最小包应包含：

1. 对应的 MP4；
2. 同名 JSON manifest；
3. 配置文件；
4. checkpoint，或明确说明 checkpoint 不分发；
5. 源代码 commit SHA；
6. package-version freeze；
7. 文件 SHA-256 manifest。

若目标是让组员**重新生成已有视频**，checkpoint 不能省略。视频与模型不宜直接混入 Git 历史；应放入单独的、带 SHA-256 清单的证据包或经过审核的 GitHub Release，并继续遵守仓库的数据发布边界。

## 9. 结论边界

- 一条视频是定性审计证据，不是独立 replication；
- evaluation seed 产生同一类可重复初始扰动，但不能替代独立 training seeds；
- 视频成功生成只证明软件通路可用，不证明步态自然、shaping 有效或 reward misspecification 已被正式确认；
- V2 当前仍需先冻结 gait/contact 的可测定义，再授权 held-out formal comparison。
