# 平面直行—急停—横移交付说明（2026-08-18）

## 这次使用的起点

本次从“未加入俯仰平衡奖励”的 `F1__FOOT_LANDING` 模型继续训练，未使用
俯仰版本模型。源模型种子为 `41703`，检查点为 100 万步。

## 新模型的输入与输出

- 策略输出仍是 8 个电机动作；
- 旧模型 113 维观测完整保留；
- 新增 `(vx_command, vy_command)` 两维，合计 115 维；
- 命令顺序为 `(1,0) -> (0,0) -> (0,1)`；
- 90° 电机索引映射为 `[6,7,0,1,2,3,4,5]`；
- 躯干没有 z 轴旋转命令，偏航角相对每回合初始值受惩罚。

## 选中的版本

- 配置：`planar_translation_transition_v3_20260818`；
- 训练种子：`42001`；
- 检查点：`1,000,000`；
- 模型：交付包 `deliverables/models/selected_planar_transition_1m.zip`；
- 视频：交付包 `deliverables/video/FINAL_planar_transition.mp4`。

三种子训练共生成 12 个候选检查点并完成 120 个固定评估回合。选中模型的
十回合均值为：直行 `(vx,vy)≈(0.57,0.24) m/s`，横移
`(vx,vy)≈(0.12,0.76) m/s`，偏航 RMS `12.8°`，无摔倒。严格连续停车率为
`5/10`；`8/10` 回合在制动期间至少达到 `0.15 m/s` 以下。

最终视频使用 GPU 推理和固定评估种子 `52011`，在 `1.5 s` 内满足连续停车
条件，随后进入正 y 平移，25 秒视频没有摔倒。

## 复现

在仓库根目录和 `proxygap-ant` Conda 环境中运行：

```powershell
python scripts/run_planar_translation_transition.py `
  --config configs/planar_translation_transition_v3_20260818.json `
  --max-workers 3
```

渲染命令见 `docs/PLANAR_TRANSLATION_TRANSITION_20260818.md`。Windows 下 MuJoCo
不能从含中文字符的路径直接打开自定义 XML；新渲染脚本会自动复制到纯 ASCII
临时路径再加载。

## 注意

这是固定 90° 平动切换的开发模型，还不是任意方向平面导航。当前也没有训练
躯干绕 z 轴转向、复杂地形、扰动恢复或真实机器人迁移。
