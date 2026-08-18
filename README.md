# Ant-v5 可复现三维随机平滑地形生成器

## 状态与边界

本目录是独立环境模块。当前状态为 **engineering-validated（工程验证通过）**：已验证确定性地形生成、查询接口、MuJoCo heightfield 加载、Gymnasium `Ant-v5` reset，以及 10 个零动作仿真步。它没有训练机器人，也没有修改 PPO checkpoint、Ant 的奖励、动作、平动、转向、步态或能耗模型。

因此，本模块不能支持“机器人已经能在随机地形上稳定行走”“不会打滑”或“能耗得到改善”等结论。`sample_safe_goal()` 只返回一个候选坐标，不会添加目标奖励或终止条件。

`configs/terrain_development.json` 中的 16 m × 16 m 仅为 development smoke test 配置。它不是正式实验尺寸，也没有替用户选择正式坡度范围或 test seeds。

## 实际环境

验证解释器为：

```text
D:\ProxyGap\envs\proxygap-ant\python.exe
Python 3.11.15
Gymnasium 1.3.0
MuJoCo 3.11.0
NumPy 2.4.6
Matplotlib 3.11.1
pytest 9.1.1
SciPy: 未安装
```

SciPy 缺失不构成运行缺项。本实现使用 NumPy 的反射边界可分离高斯卷积完成平滑，没有下载新依赖。当前 shell 中 `python` 和 `py` 不在 `PATH`，所以以下命令使用完整解释器路径。

## 支持的地形

通过同一个配置模型可生成：

- 平地；
- 正或负 `global_slope_x` 纵坡；
- 正或负 `global_slope_y` 侧坡；
- 一个或多个 Gaussian 山丘；
- 一个或多个 Gaussian 凹地；
- 低频 Fourier 起伏；
- 上述成分的受控组合。

实现没有尖刺、台阶、墙面、石块、断崖、洞穴、悬垂或不连续高度的生成入口，也不接受既有地图的平移、翻转或旋转作为“新地图”。地面摩擦只从配置读取一次，并固定写入 XML；不会随 seed 随机化。

## 关键定义

| 配置项 | 工程定义 |
|---|---|
| `terrain_length_m`, `terrain_width_m` | heightfield 在 x、y 方向的完整物理尺寸，单位 m |
| `nrow`, `ncol` | 分别对应 y 行、x 列；支持 257、513、1025 |
| `maximum_height_m` | 以 start 中心高度为 0 m 后的 `max(abs(h))` |
| `maximum_absolute_slope` | `max(||gradient(h)||_2)`，同时检查所有 cell 的保守三角面坡度组合；rise/run，无量纲 |
| `maximum_curvature` | 量化后物理节点上有限差分 Hessian 的最大绝对特征值，单位 m⁻¹ |
| `minimum_feature_width_m` | 保守构造尺度：Gaussian sigma、Fourier 四分之一波长或安全区过渡宽度的最小值 |
| `global_slope_x`, `global_slope_y` | 加入复合地形的全局平面梯度分量；随机残差可缩放，但该基底不会被静默缩放 |
| `smoothing_strength` | 高斯平滑 sigma，单位 m，不是像素数 |
| `friction` | MuJoCo 的 sliding、torsional、rolling 三个固定系数；默认保持参考 Ant XML 的 `[1.0, 0.5, 0.5]` |
| `terrain_seed`, `split` | 地形 RNG seed 与其命名空间 |
| `start_safe_region`, `goal_safe_region` | 圆形可用核心、C2 过渡宽度及核心坡度上限 |

`minimum_feature_width_m` 必须至少等于已检查 Ant 足端 capsule 的 0.16 m 直径，并至少跨 8 个网格间距。该规则是生成基函数的构造保证，不是对任意复合场局部峰宽的逆向数学证明。

`maximum_curvature` 是平滑配方离散采样后的有限差分诊断。MuJoCo 的实际碰撞面由三角形组成，三角形内部曲率为 0、边上经典曲率不连续，因此不能把该数值表述为碰撞网格的连续主曲率定理。

## 生成与约束顺序

1. 用独立的 PCG64 子流抽取山丘、凹地和 Fourier 参数。
2. 在物理坐标中构造全局平面和随机残差。
3. 用反射边界高斯核平滑随机残差。
4. 对 start/goal 核心使用 `q(t)=6t^5-15t^4+10t^3` 的径向 C2 smootherstep 融合；没有突然截平。
5. 先验证“全局坡度＋安全区融合”基底。如果它自身越界，配置被拒绝。
6. 只对随机残差应用保守幅值缩放，使高度、梯度、三角面坡度与 Hessian 谱范数均不超过上限。
7. 归一化到 `[0,1]`，转换为 MuJoCo 实际保存的 `float32`，再反解为物理节点并重新认证全部边界。

物理恢复公式保存在每个 manifest：

```text
height_m = physical_offset_m + normalised_height * physical_scale_m
```

平地的 `physical_scale_m=0`，而 XML 使用正的 `1e-6 m` MuJoCo elevation scale 和全零数据，以满足 MuJoCo 对 `hfield size` 的要求。

## 查询接口

`TerrainQueries` 提供：

- `height(x, y)`：m；
- `gradient(x, y)`：`(dh/dx, dh/dy)`；
- `normal(x, y)`：向上的单位法向量；
- `slope_along(x, y, vx, vy)`：沿水平运动方向的有符号坡度；
- `is_safe_spawn(x, y)`：检查中心及默认 0.65 m 足迹 stencil；
- `sample_safe_goal(seed)`：使用独立 RNG 流采样，不会改变地图 RNG。

查询采用 MuJoCo 3.11 的同一左下至右上分片三角约定，节点则来自实际写入 MuJoCo 的 `float32` 数值。`slope_along` 实现为：

```text
s = gradient(h) dot normalise([vx, vy])
```

`s>0` 为上坡，`s<0` 为下坡；零水平方向会抛出 `ValueError`。接口没有使用身体的 `dz/dt`。

默认 0.65 m spawn footprint 是保守开发假设，不是正式机器人几何证明。默认配置把安全区放在原点并覆盖 Ant reset noise；`make_ant_terrain_env()` 不平移机器人状态。如果正式配置把 start 中心移离原点，后续集成方必须显式处理初始放置，并单独验证该更改。

## Seed 隔离与可复现记录

命名空间仅定义范围，不选择正式实验 seed：

```text
development:       0 ..   999999
train:       1000000 ..  1999999
validation:  2000000 ..  2999999
test:        3000000 ..  3999999
```

配置若把 seed 标为错误 split 会被拒绝。测试还检查 train/validation/test 物理数组不相同，且不会通过 90° 旋转或翻转互相复制。

每个 bundle 保存：完整配置、split、seed、生成器版本、UTC 时间、物理/归一化数组、规范数组 SHA-256、`.npy` 文件 SHA-256、归一化恢复元数据、生成成分和实测边界。

当前生成器版本为 `1.1.0`。分辨率收敛脚本先分别计算 257、513、1025 网格的原生约束缩放，再冻结三者共同的最保守随机残差缩放；该专用 cap 不改变默认生成路径，也不缩放全局坡度基底。

## 运行命令

PowerShell：

```powershell
$PY = 'D:\ProxyGap\envs\proxygap-ant\python.exe'
Set-Location 'D:\nn_lecture\ant_random_terrain'

& $PY -m pytest
& $PY scripts\generate_terrain.py
& $PY scripts\render_terrain_preview.py
& $PY scripts\run_resolution_check.py
```

使用自定义配置：

```powershell
& $PY scripts\generate_terrain.py `
  --config 'D:\path\to\terrain.json' `
  --output-dir 'D:\nn_lecture\ant_random_terrain\outputs\manifests' `
  --stem 'my_terrain'
```

## 最小对接接口

```python
from pathlib import Path
import sys

root = Path(r"D:\nn_lecture\ant_random_terrain")
sys.path.insert(0, str(root / "src"))

from terrain_generator import generate_terrain, load_config
from terrain_queries import TerrainQueries
from mujoco_heightfield import make_ant_terrain_env

config = load_config(root / "configs" / "terrain_development.json")
terrain = generate_terrain(config)
queries = TerrainQueries(terrain)

z_m = queries.height(0.0, 0.0)
uphill = queries.slope_along(0.0, 0.0, vx=1.0, vy=0.0)
goal_xy = queries.sample_safe_goal(seed=12345)

env = make_ant_terrain_env(
    terrain,
    root / "outputs" / "mujoco" / "ant_runtime.xml",
)
observation, info = env.reset(seed=202608018)
env.close()
```

后续平动模型只需要接收 `TerrainData`/`TerrainQueries`，或使用生成的 XML 与已安装的数据。无需改写地形生成器，也不应让目标坐标自动进入奖励或终止逻辑。

## 实际验证结果

最终完整测试：

```text
23 passed, 0 failed, 0 skipped
```

513×513 development mixed terrain：

| 指标 | 实测 | 配置上限 |
|---|---:|---:|
| `max(abs(height))` | 0.212861 m | 0.45 m |
| 最大三角面坡度 | 0.081396 | 0.22 |
| Hessian 谱范数 | 0.190497 m⁻¹ | 0.30 m⁻¹ |
| start 核心最大坡度 | 0 | 0.035 |
| goal 核心最大坡度 | 0 | 0.040 |
| 最小构造尺度 | 1.04645 m | 1.0 m |
| 归一化反解最大误差 | 0 m | — |

同一组 Gaussian/Fourier 参数、共同随机残差缩放 `0.2289141562`、384 个确定性 off-grid probes、1025 作为数值参考：

| 网格 | 间距 (m) | height RMSE vs 1025 (m) | gradient RMSE vs 1025 | MuJoCo load 中位数 (s), n=3 | `mj_step` 中位数 (s), 5×50 steps | 最大相邻法向角 | 最大同时接触数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 257 | 0.062500 | 1.063e-5 | 4.896e-4 | 0.00534 | 5.902e-5 | 0.651° | 11 |
| 513 | 0.031250 | 2.113e-6 | 2.483e-4 | 0.00809 | 1.298e-4 | 0.332° | 33 |
| 1025 | 0.015625 | 0（参考网格） | 0（参考网格） | 0.01593 | 2.688e-4 | 0.167° | 50 |

未冻结时三档原生随机残差缩放分别为 `0.2395019392`、`0.2309885411`、`0.2289141562`；脚本采用最小值后，三档实际应用值逐值相同。height 与 gradient 的 probe RMSE 随分辨率单调下降。最大相邻法向角也下降，预览自审未发现明显局部折面。1025 的零误差是因为它被用作参考，不是连续真值误差为零。计时只代表本机和本次进程；完整校准证据、原始计时、随机化 fresh-load 顺序及 IQR 位于 `outputs/manifests/resolution_check.json`。

所有三种分辨率均成功加载，Ant 初始接触数为 0，10 步内没有终止、截断、NaN、Inf 或 MuJoCo warning。与此同时，heightfield 的最大同时接触数为 11/33/50，而匹配的默认 plane smoke 为 2。该接触多重性随分辨率上升，是已观察到的接触几何差异；它没有造成这次短时崩溃，但在正式训练前必须做专门的接触动力学、滑动与稳定性检查。因此本模块不声称接触行为已经等价于默认 plane。

## 已知限制与下一道验证门

- 只支持连续、单值的 `z=h(x,y)`；不支持洞穴、悬垂、石块、台阶或断崖。
- 数值平滑使用相同物理 sigma，但在每个网格上离散求值；分辨率比较是收敛 smoke test，不是连续域误差证明。
- 法向角是局部折面启发式诊断；没有替代完整接触分析。
- 0.65 m footprint stencil 是开发假设，不能自动推广到其他机器人或姿态。
- 没有运行 PPO、没有评估训练鲁棒性、稳定行走、滑动、跌倒率或能耗。
- 正式实验仍需由用户冻结物理尺寸、坡度/曲率范围、分辨率、seed 列表和接触接受标准。
- 在第一次渲染之后修改 hfield 时，MuJoCo 需要 `mjr_uploadHField`；本模块把静态数据放在首次 reset/render 之前，避免了该路径。

建议下一道门是：在不改变奖励或 PPO checkpoint 的前提下，先做默认 plane 与各分辨率 flat hfield 的配对接触/滑动基准；确定可接受的接触多重性与 signed distance 后，再把 `TerrainQueries` 和 XML 工厂接到平动环境。只有该门通过后，才适合讨论训练协议。
