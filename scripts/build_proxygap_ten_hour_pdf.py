from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    from _portable_runtime import CJK_FONT_PAIRS, FONT_DIR_ENV, iter_font_pairs
except ModuleNotFoundError:  # Support module-style execution from the repository root.
    from scripts._portable_runtime import CJK_FONT_PAIRS, FONT_DIR_ENV, iter_font_pairs


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "dev" / "hg_r3_obsfix_v1"
ANALYSIS = RESULTS / "analysis"
ASSETS = ROOT / "output" / "report_assets_20260816"
OUTPUT = ROOT / "output" / "pdf" / "PROXYGAP_TEN_HOUR_DEVELOPMENT_AUDIT_20260816_CN.pdf"

NAVY = colors.HexColor("#173753")
TEAL = colors.HexColor("#0F766E")
GOLD = colors.HexColor("#C99528")
RED = colors.HexColor("#A63A3A")
PALE_BLUE = colors.HexColor("#EAF1F6")
PALE_GOLD = colors.HexColor("#FFF6DF")
PALE_RED = colors.HexColor("#FBECEC")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#566573")
GRID = colors.HexColor("#CAD5DE")


def register_fonts() -> tuple[str, str]:
    for regular, bold in iter_font_pairs(CJK_FONT_PAIRS):
        try:
            pdfmetrics.registerFont(TTFont("CJKRegular", str(regular), subfontIndex=0))
            pdfmetrics.registerFont(TTFont("CJKBold", str(bold), subfontIndex=0))
            return "CJKRegular", "CJKBold"
        except Exception:
            continue
    raise RuntimeError(
        f"No usable Chinese font pair was found; set {FONT_DIR_ENV} to a font directory."
    )


REGULAR_FONT, BOLD_FONT = register_fonts()


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=BOLD_FONT,
            fontSize=25,
            leading=33,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=8 * mm,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName=REGULAR_FONT,
            fontSize=11.5,
            leading=18,
            textColor=MUTED,
            spaceAfter=4 * mm,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=BOLD_FONT,
            fontSize=17,
            leading=23,
            textColor=NAVY,
            spaceBefore=3 * mm,
            spaceAfter=3 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=BOLD_FONT,
            fontSize=12.5,
            leading=18,
            textColor=TEAL,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=9.4,
            leading=15.5,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=2.2 * mm,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=7.7,
            leading=11.3,
            textColor=INK,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=7.7,
            leading=11.5,
            textColor=MUTED,
            spaceBefore=1.5 * mm,
            spaceAfter=3 * mm,
            wordWrap="CJK",
        ),
        "callout": ParagraphStyle(
            "callout",
            parent=base["BodyText"],
            fontName=BOLD_FONT,
            fontSize=10.2,
            leading=16,
            textColor=NAVY,
            wordWrap="CJK",
        ),
        "reference": ParagraphStyle(
            "reference",
            parent=base["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=7.3,
            leading=10.8,
            textColor=INK,
            spaceAfter=2 * mm,
            wordWrap="CJK",
        ),
    }


S = styles()


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def callout(text: str, colour: colors.Color = PALE_BLUE) -> Table:
    table = Table([[p(text, "callout")]], colWidths=[174 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colour),
                ("BOX", (0, 0), (-1, -1), 0.7, NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )
    return table


def bullet(text: str) -> Paragraph:
    return Paragraph(f"• {text}", S["body"])


def figure(name: str, caption: str, width_mm: float = 174) -> list:
    image_path = ASSETS / name
    image = Image(str(image_path))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width_mm * mm
    image.drawHeight = width_mm * ratio * mm
    return [image, p(caption, "caption")]


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict, dict]:
    endpoint = pd.read_csv(ANALYSIS / "endpoint_condition_summary.csv")
    lateral = pd.read_csv(
        ANALYSIS / "lateral_velocity" / "endpoint_condition_lateral_velocity_summary.csv"
    )
    qa = json.loads((ANALYSIS / "data_quality_qa.json").read_text(encoding="utf-8"))
    gate = json.loads(
        (ANALYSIS / "development_gate_adjudication.json").read_text(encoding="utf-8")
    )
    sensitivity = json.loads(
        (ANALYSIS / "intent_sensitivity" / "sensitivity_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    video_qa = json.loads((RESULTS / "videos" / "VIDEO_QA.json").read_text(encoding="utf-8"))
    return endpoint, lateral, qa, gate, sensitivity, video_qa


def result_table(endpoint: pd.DataFrame, lateral: pd.DataFrame) -> Table:
    merged = endpoint.merge(lateral, on="condition_id", how="left")
    aliases = {
        "Rt0p1_Rvy0__K0": "λy=0; K=off",
        "Rt0p1_Rvy0__K1p1": "λy=0; K=1.1",
        "Rt0p1_Rvy0p05__K0": "λy=.05; K=off",
        "Rt0p1_Rvy0p05__K1p1": "λy=.05; K=1.1",
        "Rt0p1_Rvy0p1__K0": "λy=.10; K=off",
        "Rt0p1_Rvy0p1__K1p1": "λy=.10; K=1.1",
    }
    rows = [[
        "条件",
        "合规率",
        "vx (m/s)",
        "|vy| (m/s)",
        "倾斜 RMS",
        "方向误差",
        "不健康终止",
        "base proxy",
    ]]
    for _, row in merged.iterrows():
        rows.append(
            [
                aliases[row["condition_id"]],
                f"{100 * row['intent_compliance_rate_mean']:.0f}%",
                f"{row['fixed_horizon_mean_forward_velocity_mean']:.3f}",
                f"{row['mean_abs_lateral_velocity_error_mean']:.3f}",
                f"{row['torso_tilt_rms_degrees_mean']:.1f}°",
                f"{row['net_displacement_direction_error_degrees_mean']:.1f}°",
                f"{100 * row['unhealthy_termination_mean']:.0f}%",
                f"{row['base_proxy_return_mean']:.1f}",
            ]
        )
    data = [[p(str(value), "small") for value in row] for row in rows]
    table = Table(
        data,
        colWidths=[28 * mm, 15 * mm, 20 * mm, 20 * mm, 20 * mm, 20 * mm, 23 * mm, 22 * mm],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), BOLD_FONT),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]),
                ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )
    return table


def footer(canvas, doc) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
    canvas.setFont(REGULAR_FONT, 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8.5 * mm, "ProxyGap | Development evidence only | 2026-08-16")
    canvas.drawRightString(width - 18 * mm, 8.5 * mm, f"{doc.page}")
    canvas.restoreState()


def build() -> Path:
    endpoint, lateral, qa, gate, sensitivity, video_qa = read_inputs()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="ProxyGap 十小时开发探索与证据审计",
        author="ProxyGap project",
        subject="Ant-v5 PPO reward misspecification and mitigation development audit",
    )
    story: list = []

    # Cover
    story.extend(
        [
            Spacer(1, 15 * mm),
            p("ProxyGap 十小时开发探索与证据审计", "title"),
            p("默认平地 Ant-v5 · PPO · Reward Misspecification · Reward Shaping / External Constraint", "subtitle"),
            Spacer(1, 12 * mm),
            callout(
                "核心判断：工程管线通过 QA，但 6 个开发候选在主要判定下均未达到意图合规门槛。"
                "因此没有启动 1M 延长或正式 held-out 训练，也不能宣称 mitigation 已成功。",
                PALE_GOLD,
            ),
            Spacer(1, 10 * mm),
            p("报告日期：2026-08-16", "subtitle"),
            p("研究状态：Scientifically unresolved；development evidence only", "subtitle"),
            p("计算环境：Windows · CPU-only · Gymnasium Ant-v5 · Stable-Baselines3 PPO", "subtitle"),
            Spacer(1, 22 * mm),
            p(
                "范围声明：本报告只讨论默认平地 Ant-v5 仿真。没有真实机器人、复杂地形、摩擦变化或外力扰动测试，"
                "也不作相应外推。",
                "callout",
            ),
            PageBreak(),
        ]
    )

    # Executive status
    story.extend(
        [
            p("1  结论先行", "h1"),
            p(
                "本轮完成 6 条开发条件、3 个 training seeds、3 个 checkpoints 和每个 checkpoint 10 个固定 "
                "evaluation episodes，共训练 18 个策略并评价 540 个 episode。奖励分解、逐步日志、横向速度信号和视频均通过工程 QA。"
            ),
            p(
                "但所有条件在 300k endpoint 的主要意图合规率均为 0%。动作变化约束改善了部分前进、终止和路径指标，"
                "却没有形成完整意图合规；横向 shaping 也没有跨三个 training seeds 一致改善。因此没有候选进入下一阶段。"
            ),
            callout(
                "停止并不是训练失败后的逃避，而是预先声明 gate 的正常执行：没有达标候选，就不动用正式 seeds，"
                "不在看完结果后继续调到出现阳性。",
                PALE_BLUE,
            ),
            p("证据状态", "h2"),
            bullet("Engineering-validated：代码、奖励重算、日志主键、逐步恒等式、视频长度与解码。"),
            bullet("Development evidence：3 个已重复使用的 training seeds；可用于诊断和候选筛选。"),
            bullet("Scientifically unresolved：reward misspecification 的机制归因与 mitigation 效果尚未 held-out 确认。"),
            bullet("Formal inference：未启动；预留 training seeds 42001–42008 未使用。"),
            p("2  研究问题与不可越界的范围", "h1"),
            p(
                "Gymnasium 官方把 Ant 任务描述为协调四条腿向正 x 方向运动，episode 最长 1,000 steps。"
                "本项目检验默认 scalar reward 是否能代表研究者额外声明的有效、直立、方向可控和动作质量要求。"
                "这是一项项目内构念审计，不等于断言 Ant-v5 或 PPO 普遍错误。"
            ),
            callout(
                "最终结论必须写成“在默认平地 Ant-v5 仿真和本项目 PPO 设置中观察到……”。"
                "不得写成真实四足机器人已经安全、自然或稳健。",
                PALE_RED,
            ),
            PageBreak(),
        ]
    )

    # Contract and design
    story.extend(
        [
            p("3  意图、代理奖励和研究目标", "h1"),
            p("三层对象必须分开，否则很容易把环境奖励误称为人的真实目标："),
            bullet("平台任务：向正 x 方向移动；最长 1,000 steps。"),
            bullet("PPO 可见代理：环境逐步返回的 scalar reward。"),
            bullet("研究者意图：有效前进、避免不健康终止和持续倒置、方向可控、动作不过度剧烈。"),
            p(
                "主要评价保留多个领域，不定义一个未经验证的 scalar true performance。配置中的 "
                "scalar_true_performance 明确为 not_defined。这样可以检查前进改善是否以姿态、方向或控制质量为代价。"
            ),
            p("4  候选干预与可观测性修正", "h1"),
            *figure(
                "figure_experiment_equations.png",
                "图 1  实验中实际使用的基础代理、余弦姿态 shaping、横向速度 shaping 和动作变化 projection。"
                "外部 projection 不是 reward 项，必须与 learned behaviour 分开解释。",
            ),
            p(
                "Round 2 曾使用策略不可见的绝对横向位置作为 shaping 信号。该偏差在查看结果前被识别并登记。"
                "Round 3 改用 Ant 默认观测中可见的躯干 y 速度，并为所有条件统一加入前一时刻施加动作，"
                "从而使动作变化约束具备 Markov 相关信息。"
            ),
            p("5  矩阵、seeds 与重复单位", "h1"),
            bullet("6 条条件：横向 shaping 权重 0 / 0.05 / 0.10 × 动作变化约束 off / L2≤1.1。"),
            bullet("姿态 shaping 固定为 λθ=0.1；ctrl_cost_weight 固定为默认 0.5。"),
            bullet("Training seeds 41301–41303：产生三个独立训练策略，是复制单位。"),
            bullet("Evaluation seeds 51301–51310：给每个策略相同初始扰动，形成配对考题；不重新训练。"),
            bullet("每策略 300k steps；100k / 200k / 300k checkpoints；每 checkpoint 10 episodes。"),
            PageBreak(),
        ]
    )

    # QA and table
    episode_qa = qa["episode_table"]
    step_qa = qa["step_logs"]
    story.extend(
        [
            p("6  工程与数据质量", "h1"),
            callout(
                f"QA 状态：{qa['status'].upper()}。18/18 策略、54/54 checkpoints、"
                f"{episode_qa['observed_rows']}/540 episode rows、{step_qa['step_log_files']}/540 step logs。"
            ),
            bullet(f"实际逐步记录：{step_qa['total_logged_steps']:,} steps；episode 和 step 主键重复均为 0。"),
            bullet(
                "基础奖励重算最大绝对误差 "
                f"{episode_qa['max_abs_base_reward_reconciliation_error']:.2e}；控制代价误差 "
                f"{episode_qa['max_abs_ctrl_cost_reconciliation_error']:.1f}。"
            ),
            bullet(
                "逐步奖励恒等式最大误差 "
                f"{step_qa['max_step_reward_identity_error']:.2e}；无缺失或额外 episode keys。"
            ),
            bullet("横向速度分析为 540 rows、0 重复，reward 重算最大误差 1.11e-16。"),
            p(
                "这些检查排除了常见的日志、奖励分解和关联错误，但不能证明人的意图定义完美，"
                "也不能用工程测试通过替代 construct validity。"
            ),
            p("7  300k endpoint 结果", "h1"),
            result_table(endpoint, lateral),
            p(
                "表 1  三个 training seeds 的 condition-level 均值。K=1.1 表示动作变化 L2 projection；"
                "vx 与 |vy| 分别为固定评价期内平均前进速度与平均绝对横向速度误差。",
                "caption",
            ),
            callout(
                f"Development gate 结果：{len(gate['advanced_conditions'])} 个候选进入下一阶段。"
                "所有候选均为 does_not_pass_prespecified_development_gate。",
                PALE_GOLD,
            ),
            PageBreak(),
        ]
    )

    # Compliance and seed chart
    story.extend(
        [
            p("8  多域合规：平均表现不能遮住失败领域", "h1"),
            *figure(
                "figure_domain_compliance.png",
                "图 2  主要阈值下的 domain-compliance matrix。所有条件的整体意图合规率均为 0%；"
                "外部约束机械保证 applied-action smoothness，但路径、姿态、方向或健康仍存在失败。",
            ),
            p(
                "这张矩阵就是本项目适合报告的量化“accuracy matrix”：单元格是 evaluation episodes 的通过百分比，"
                "而不是把连续控制任务硬改成没有真实类别标签的分类混淆矩阵。"
            ),
            p("9  横向 shaping 的 seed 依赖", "h1"),
            *figure(
                "figure_lateral_velocity_by_seed.png",
                "图 3  300k endpoint 下，各 training seed 的平均绝对横向速度误差。相同颜色代表相同 seed；"
                "方形与虚线表示动作约束。改善方向没有跨三个 seeds 一致复制。",
            ),
            p(
                "无动作约束时，λy=0.05 在三个 seeds 中全部恶化；λy=0.10 仅 seed 41302 改善。"
                "有约束时，λy=0.05 仅 seed 41303 改善；λy=0.10 在 41302 和 41303 改善、41301 恶化。"
                "因此不能把 aggregate mean 的变化解释为稳定 mitigation。"
            ),
            PageBreak(),
        ]
    )

    # Mechanism and sensitivity
    story.extend(
        [
            p("10  外部约束改善了执行，不等于策略学会了平滑", "h1"),
            *figure(
                "figure_guardrail_mechanism.png",
                "图 4  外部约束条件下 proposed action 与 applied action 的机制差异。"
                "约 98% steps 需要 correction；策略提议仍显著粗糙。",
            ),
            p(
                "动作约束条件的 applied roughness 约 0.0377，而 proposed roughness 为 0.249–0.271；"
                "干预率 98.4–98.7%，平均动作修正 L2 约 1.08–1.11。"
                "所以准确表述是“projection 产生平滑执行动作”，而不是“PPO 学会平滑动作”。"
            ),
            p("11  阈值敏感性", "h1"),
            p(
                f"敏感性分析覆盖 {sensitivity['threshold_grid_cells']} 个阈值组合和 "
                f"{sensitivity['candidate_cell_evaluations']} 个候选-网格比较。主要阈值结论保持不变。"
            ),
            bullet("λy=0.05 + K=1.1：复制方向覆盖 14.8% 网格；平均配对合规率变化 +3.13 个百分点。"),
            bullet("λy=0.10 + K=1.1：复制方向覆盖 6.58% 网格；平均配对合规率变化 +0.71 个百分点。"),
            p(
                "敏感性分析说明结论不是只由一个阈值点决定，但它不能把 development evidence 升格为 formal inference。"
                "所有主要结果仍来自已用于开发的 3 个 training seeds。"
            ),
            PageBreak(),
        ]
    )

    # Video
    story.extend(
        [
            p("12  完整视频证据", "h1"),
            callout(
                f"视频 QA：{video_qa['video_count']}/{video_qa['expected_video_count']} MP4 通过；"
                f"统一 20 fps、{video_qa['all_timeline_durations_seconds'][0]:.1f} s、"
                f"{video_qa['all_manifest_frames'][0]} frames。"
            ),
            *figure(
                "figure_video_evidence.png",
                "图 5  固定 training seed 41301 与 evaluation seed 51301 的完整轨迹截帧。"
                "提前终止视频用最终状态补足 50 s，并在清单中记录真实 trajectory frames；"
                "视频只解释数值现象，不用于选择最戏剧性的案例。",
            ),
            p(
                "渲染器已扩大棋盘可视范围，解决旧视频后段“地面消失”的显示问题；这只是视觉修复，"
                "没有改变物理环境、轨迹或 reward。姿态-only baseline 的 300k 示例发生 high-z excursion；"
                "λy=0.05 + K=1.1 示例完成 horizon 但有大横向位移；λy=0.10 + K=1.1 示例后期接近倒置。"
            ),
            PageBreak(),
        ]
    )

    # Claims and plan
    story.extend(
        [
            p("13  结论边界", "h1"),
            p("当前可支持", "h2"),
            bullet("Round 3 实现、日志、奖励分解和视频管线通过工程 QA。"),
            bullet("动作 projection 显著改变执行行为，但候选没有达到完整意图门槛。"),
            bullet("横向速度 shaping 的效果依赖 training seed，未形成一致缓解证据。"),
            bullet("applied-action 平滑主要由外部约束保证，不是已证明的 learned smoothness。"),
            p("当前不可支持", "h2"),
            bullet("不能宣称已正式证明 reward hacking。"),
            bullet("不能宣称 shaped reward 或约束已经缓解 reward misspecification。"),
            bullet("不能用更多 evaluation episodes 替代独立 held-out training seeds。"),
            bullet("不能外推到复杂地形、摩擦变化、外力扰动、其他算法或真实机器人。"),
            bullet("不能把 0% 合规单独归因于 reward；训练预算、动作接口与 PPO 优化仍是替代解释。"),
            p("14  三天完成路径", "h1"),
            bullet(
                "第 1 天：冻结本轮负面开发结论；决定是否只批准一个新的机制问题。"
                "不再开放式搜索 λy。"
            ),
            bullet(
                "第 2 天：只有新候选通过独立 development gate 才动用 held-out seeds；"
                "否则转为默认奖励构念审计加失败缓解尝试的负面结果报告。"
            ),
            bullet("第 3 天：完成证据矩阵、seed 图、视频索引、重现命令、限制和报告。"),
            bullet("随后 1.5 天：只优化图表、引用、PPT 和交付 QA，不改变科学问题。"),
            callout(
                "若批准最后一次受限开发，最值得检验的单一机制是策略持续提出粗糙动作的原因，"
                "而不是继续细调横向权重。可预先固定 action-difference reward 或平滑探索/控制接口中的一个；"
                "该实验必须作为新版本开发，不能与本轮数据合并成确认结果。",
                PALE_GOLD,
            ),
            PageBreak(),
        ]
    )

    # Evaluation metric and references
    story.extend(
        [
            p("15  量化评价矩阵", "h1"),
            p(
                "当前最合适的 accuracy-style 输出是 condition × training seed 的 domain-compliance matrix。"
                "每格报告通过 evaluation episodes 的百分比，并分解前进、姿态、方向、路径、动作和健康。"
                "它可审计，也不会虚构分类标签。"
            ),
            p(
                "若后续存在正式 shaping 候选，才计算配对 mitigation success rate：同时改善意图且保持前进能力的 "
                "held-out seed pairs / 全部 held-out seed pairs × 100%。阈值必须在正式训练前冻结；当前不提前伪定。"
            ),
            p("16  参考文献", "h1"),
            p(
                "Achiam, J., Held, D., Tamar, A. and Abbeel, P. (2017) ‘Constrained policy optimization’, "
                "<i>Proceedings of the 34th International Conference on Machine Learning</i>, 70, pp. 22–31. "
                "https://proceedings.mlr.press/v70/achiam17a.html",
                "reference",
            ),
            p(
                "Aractingi, M., Cumin, J., Stasse, O. and Righetti, L. (2023) ‘Learning agile locomotion skills "
                "with a mentor’, <i>Scientific Reports</i>, 13, 11045. "
                "https://www.nature.com/articles/s41598-023-38259-7",
                "reference",
            ),
            p(
                "Farama Foundation (2026) ‘Ant’. <i>Gymnasium Documentation</i>. "
                "https://gymnasium.farama.org/environments/mujoco/ant/ (Accessed: 16 August 2026).",
                "reference",
            ),
            p(
                "Pan, A., Bhatia, K. and Steinhardt, J. (2022) ‘The effects of reward misspecification: mapping "
                "and mitigating misaligned models’, <i>arXiv</i>. https://arxiv.org/abs/2201.03544",
                "reference",
            ),
            p(
                "Raffin, A., Kober, J. and Stulp, F. (2022) ‘Smooth exploration for robotic reinforcement learning’, "
                "<i>Proceedings of the 5th Conference on Robot Learning</i>, 164, pp. 1634–1644. "
                "https://proceedings.mlr.press/v164/raffin22a.html",
                "reference",
            ),
            p(
                "Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017) ‘Proximal policy "
                "optimization algorithms’, <i>arXiv</i>. https://arxiv.org/abs/1707.06347",
                "reference",
            ),
            p(
                "Skalse, J., Howe, N., Krasheninnikov, D. and Krueger, D. (2022) ‘Defining and characterizing "
                "reward gaming’, <i>Advances in Neural Information Processing Systems</i>, 35. "
                "https://arxiv.org/abs/2209.13085",
                "reference",
            ),
            Spacer(1, 4 * mm),
            p("关键证据路径", "h2"),
            p("配置：configs/hybrid_guardrail_observability_correction_v1_20260816.json", "reference"),
            p("结果：artifacts/dev/hg_r3_obsfix_v1/analysis/", "reference"),
            p("视频：artifacts/dev/hg_r3_obsfix_v1/videos/", "reference"),
            p("实验日志：experiment_logs/ProxyGap/hybrid_guardrail/PG-B-260816-003.md", "reference"),
        ]
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(build())
