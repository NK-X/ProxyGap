# 固定地图16°候选路线探索性评估（smoke）

## 结论边界

Exploratory deterministic-policy evaluation on one repeatedly inspected fixed map. All failures are retained. Existing mechanical-work quantities are diagnostics, not V2 relative mission energy, and no route is eligible for time-energy ranking unless it passes the separately computed arrival, stable-dwell and whole-episode safety gates.

路线是根据封存高度图重新构造的候选，不是对丢失原始 waypoints 的精确复现。
已有机械功只作诊断，不是V2相对任务能耗，也没有进入奖励或路线成本。

## 路线重建

- 长度：106.150 m
- 累计爬升/下降：5.711/3.481 m
- 最大走廊坡度：15.968°
- 首段航向：90.000°
- 累计转向：129.073°

## 条件汇总

| 条件 | n | 进入终点 | 位置保持 | 稳定dwell | 安全合格 | 摔倒 | 平均净推进(m) | 平均腾空率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R45_route_slope_speed_v050_l050_k020 | 1 | 0 | 0 | 0 | 0 | 0 | 6.314 | 67.33% |

## 解释规则

- `进入终点`只表示曾进入1.5 m范围。
- `位置保持`表示进入后在2.0 m范围内连续40步，不代表站立稳定。
- `稳定dwell`还要求支撑、低速度、低角速度、健康姿态和无接触速度超限。
- `安全合格`进一步要求全轮无摔倒、无四足同时腾空且无接触速度超限。
- 所有失败和超时均保留在 `episode_rows.csv`。
