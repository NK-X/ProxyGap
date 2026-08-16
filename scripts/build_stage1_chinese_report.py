"""Build the Chinese stage-one exploration report as a verified PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    from _portable_runtime import (
        DENG_FIRST_CJK_FONT_PAIRS,
        FONT_DIR_ENV,
        iter_font_pairs,
    )
except ModuleNotFoundError:  # Support module-style execution from the repository root.
    from scripts._portable_runtime import (
        DENG_FIRST_CJK_FONT_PAIRS,
        FONT_DIR_ENV,
        iter_font_pairs,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#007C83")
ORANGE = colors.HexColor("#D55E00")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_GREY = colors.HexColor("#F4F5F6")
MID_GREY = colors.HexColor("#66727D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_root", required=True)
    parser.add_argument("--dense_run_root", required=True)
    parser.add_argument("--reevaluation_root", required=True)
    parser.add_argument("--forensic_json", required=True)
    parser.add_argument("--bootstrap_csv", required=True)
    parser.add_argument("--video_manifest", action="append", default=[])
    parser.add_argument("--video_contact_sheet")
    parser.add_argument("--validation_json", required=True)
    parser.add_argument("--equation_dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def register_fonts() -> None:
    for regular, bold in iter_font_pairs(DENG_FIRST_CJK_FONT_PAIRS):
        try:
            pdfmetrics.registerFont(TTFont("Deng", str(regular), subfontIndex=0))
            pdfmetrics.registerFont(TTFont("DengBold", str(bold), subfontIndex=0))
            return
        except Exception:
            continue
    raise RuntimeError(
        f"No suitable Chinese font pair was found; set {FONT_DIR_ENV} to a font directory."
    )


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=base["Title"],
            fontName="DengBold",
            fontSize=24,
            leading=31,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=8 * mm,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN",
            parent=base["Normal"],
            fontName="Deng",
            fontSize=11,
            leading=17,
            textColor=MID_GREY,
            alignment=TA_LEFT,
        ),
        "h1": ParagraphStyle(
            "H1CN",
            parent=base["Heading1"],
            fontName="DengBold",
            fontSize=16,
            leading=21,
            textColor=NAVY,
            spaceBefore=6 * mm,
            spaceAfter=3 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2CN",
            parent=base["Heading2"],
            fontName="DengBold",
            fontSize=12,
            leading=17,
            textColor=TEAL,
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=9.4,
            leading=15.2,
            textColor=colors.HexColor("#20262C"),
            # Left alignment avoids large inter-word gaps in mixed Chinese/English text.
            alignment=TA_LEFT,
            firstLineIndent=2 * 9.4,
            spaceAfter=2.2 * mm,
            splitLongWords=True,
        ),
        "body_noindent": ParagraphStyle(
            "BodyNoIndentCN",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=9.4,
            leading=15.2,
            textColor=colors.HexColor("#20262C"),
            alignment=TA_LEFT,
            spaceAfter=2.2 * mm,
            splitLongWords=True,
        ),
        "bullet": ParagraphStyle(
            "BulletCN",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=9.2,
            leading=14.5,
            leftIndent=5 * mm,
            firstLineIndent=-3.5 * mm,
            bulletIndent=0,
            spaceAfter=1.4 * mm,
        ),
        "caption": ParagraphStyle(
            "CaptionCN",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=8.2,
            leading=12,
            textColor=MID_GREY,
            alignment=TA_LEFT,
            spaceBefore=1.5 * mm,
            spaceAfter=4 * mm,
        ),
        "small": ParagraphStyle(
            "SmallCN",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=7.7,
            leading=11,
            textColor=colors.HexColor("#343A40"),
        ),
        "file_path": ParagraphStyle(
            "FilePathCN",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=7.6,
            leading=10.4,
            textColor=colors.HexColor("#20262C"),
            alignment=TA_LEFT,
            splitLongWords=True,
        ),
        "table_head": ParagraphStyle(
            "TableHeadCN",
            parent=base["BodyText"],
            fontName="DengBold",
            fontSize=7.8,
            leading=10.2,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "table": ParagraphStyle(
            "TableCN",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=7.6,
            leading=10.2,
            textColor=colors.HexColor("#20262C"),
        ),
        "callout": ParagraphStyle(
            "CalloutCN",
            parent=base["BodyText"],
            fontName="DengBold",
            fontSize=9.7,
            leading=15,
            textColor=NAVY,
            leftIndent=4 * mm,
            rightIndent=4 * mm,
            spaceBefore=2 * mm,
            spaceAfter=3 * mm,
        ),
    }


def page_header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#C8D2DC"))
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, height - 13 * mm, width - 18 * mm, height - 13 * mm)
    canvas.setFont("Deng", 7.5)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(18 * mm, height - 10 * mm, "ProxyGap 阶段一开发性探索")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def para(text: str, styles: dict[str, ParagraphStyle], style: str = "body") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(f"• {text}", styles["bullet"])


def table(
    headers: list[str],
    rows: list[list[Any]],
    styles: dict[str, ParagraphStyle],
    widths: list[float] | None = None,
) -> Table:
    values = [
        [Paragraph(str(value), styles["table_head"]) for value in headers]
    ] + [
        [Paragraph(str(value), styles["table"]) for value in row]
        for row in rows
    ]
    item = Table(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
    item.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD3DA")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_GREY]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return item


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def domain_cn(name: str) -> str:
    return {
        "locomotion_effectiveness": "前进有效性",
        "environment_health": "环境健康终止",
        "lateral_control": "横向控制",
        "posture_stability": "躯干姿态",
        "command_quality": "动作指令质量",
    }.get(name, name)


def add_figure(
    story: list[Any],
    path: Path,
    caption: str,
    styles: dict[str, ParagraphStyle],
    *,
    max_height: float = 118 * mm,
) -> None:
    story.append(
        KeepTogether(
            [
                scaled_image(path, 174 * mm, max_height),
                para(caption, styles, "caption"),
            ]
        )
    )


def main() -> None:
    args = parse_args()
    register_fonts()
    styles = build_styles()
    analysis_root = Path(args.analysis_root).resolve()
    dense_run_root = Path(args.dense_run_root).resolve()
    reevaluation_root = Path(args.reevaluation_root).resolve()
    forensic_path = Path(args.forensic_json).resolve()
    bootstrap_path = Path(args.bootstrap_csv).resolve()
    validation_path = Path(args.validation_json).resolve()
    equation_dir = Path(args.equation_dir).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    result = json.loads(
        (analysis_root / "stage1_development_result.json").read_text(encoding="utf-8")
    )
    completion = json.loads(
        (dense_run_root / "parallel_completion.json").read_text(encoding="utf-8")
    )
    reevaluation = json.loads(
        (reevaluation_root / "reevaluation_manifest.json").read_text(encoding="utf-8")
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    forensic = json.loads(forensic_path.read_text(encoding="utf-8"))
    bootstrap = pd.read_csv(bootstrap_path)
    video_manifests = [
        json.loads(Path(path).resolve().read_text(encoding="utf-8"))
        for path in args.video_manifest
    ]
    video_contact_sheet = (
        Path(args.video_contact_sheet).resolve() if args.video_contact_sheet else None
    )
    sensitivity = pd.read_csv(analysis_root / "margin_sensitivity.csv")
    persistence = pd.read_csv(analysis_root / "late_checkpoint_persistence.csv")
    runtime = pd.read_csv(dense_run_root / "logs" / "training_runtime.csv")

    screens = result["endpoint_screens"]
    candidates = sorted(
        [item["candidate_weight"] for item in screens if item["strong_development_candidate"]],
        reverse=True,
    )
    transitions = result["discrete_onset_intervals"]
    first_onset = result.get("first_discrete_onset_interval")
    reentries = result.get("candidate_reentry_intervals", [])
    exits = result.get("candidate_exit_intervals", [])
    onset_text = (
        f"{first_onset['candidate_lower_weight']:g}–{first_onset['noncandidate_upper_weight']:g}"
        if first_onset
        else "未识别"
    )
    total_train_seconds = float(runtime["train_elapsed_sec"].sum())
    total_eval_seconds = float(runtime["eval_elapsed_sec"].sum())

    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=17 * mm,
        title="ProxyGap 阶段一严谨性重构与开发性探索报告",
        author="ProxyGap research workflow",
        subject="Ant-v5 PPO reward misspecification stage-one development study",
    )
    story: list[Any] = []

    story.append(Spacer(1, 18 * mm))
    story.append(para("ProxyGap 阶段一严谨性重构与开发性探索报告", styles, "title"))
    story.append(
        para(
            "Ant-v5 + PPO 奖励错位检测：旧方案回顾、证据修复、密集权重探索与正式实验闸门",
            styles,
            "subtitle",
        )
    )
    story.append(Spacer(1, 12 * mm))
    cover_table = Table(
        [
            [para("报告日期", styles, "small"), para("2026-08-14", styles, "body_noindent")],
            [para("研究阶段", styles, "small"), para("阶段一开发性探索；非正式确认", styles, "body_noindent")],
            [para("核心变量", styles, "small"), para("ctrl_cost_weight", styles, "body_noindent")],
            [para("当前状态", styles, "small"), para("Scientifically unresolved; formal run blocked", styles, "body_noindent")],
        ],
        colWidths=[36 * mm, 126 * mm],
    )
    cover_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.7, NAVY),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C8D2DC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(cover_table)
    story.append(Spacer(1, 13 * mm))
    story.append(
        para(
            "结论边界：本报告只能判断开发性候选及其离散起始区间，不能把两枚 training seeds 当作正式复现，不能声称找到全局最优奖励权重，也不能证明连续意义上的相变。",
            styles,
            "callout",
        )
    )
    story.append(PageBreak())

    story.append(para("执行摘要", styles, "h1"))
    if candidates:
        story.append(
            para(
                f"在名义实际意义阈值下，开发性候选权重为 {', '.join(f'{x:g}' for x in candidates)}；首个离散起始区间为 {onset_text}。这表示现有两枚开发 seed 中，候选策略在相同 R_w 尺度下获得更高代理回报，同时至少一个预声明行为领域越过阈值。它仍不是正式奖励错位结论。",
                styles,
            )
        )
        if reentries or exits:
            reentry_text = ", ".join(
                f"{item['candidate_lower_weight']:g}–{item['noncandidate_upper_weight']:g}"
                for item in reentries
            ) or "无"
            exit_text = ", ".join(
                f"{item['noncandidate_lower_weight']:g}–{item['candidate_upper_weight']:g}"
                for item in exits
            ) or "无"
            story.append(
                para(
                    f"候选状态随权重降低并非单调：首次进入区间为 {onset_text}，退出区间为 {exit_text}，再次进入区间为 {reentry_text}。因此结果不能解释为一个权重以下持续发生错位的单一阈值规律。",
                    styles,
                    "callout",
                )
            )
    else:
        story.append(
            para(
                "名义阈值下未找到强开发性候选。按冻结规则，应保留负结果并停止正式确认，而不是继续降低权重制造阳性现象。",
                styles,
            )
        )
    story.append(
        para(
            f"新密集网格完成 {completion['completed_policies']} 个策略、{completion['evaluation_rows']} 个评价 episode 行；旧模型统一重评价覆盖 {reevaluation['source_model_count']} 个 checkpoint 和 {reevaluation['evaluation_row_count']} 个 episode 行。最终合并数据共有 {result['data_quality']['row_count']} 行，无重复 episode 键。",
            styles,
        )
    )
    story.append(bullet("保留：Ant-v5、PPO、单一主变量、300k 训练预算、六个 checkpoint、配对 training seed 与统一 R_w 重评分。", styles))
    story.append(bullet("修改：阶段二 shaping 完全延期；增加实际意义阈值、局部等距权重网格、速度与归一化动作粗糙度。", styles))
    story.append(bullet("废止：把任意正差值叫作恶化、把相关动作指标计为两个独立伤害、用 0.5 统一量尺挑所谓最优权重。", styles))
    story.append(bullet("正式实验仍被阻断：阈值、六条件矩阵及五枚描述性或八枚较强方向性 training-seed 方案尚待批准。", styles))

    story.append(para("1. 已确认、已决定与仍未知", styles, "h1"))
    rows = [
        ["已确认事实", "Ant-v5 的动作维度为 8、默认 dt=0.05 s、默认健康区间为 [0.2,1.0]；当前环境与依赖均可运行。"],
        ["已确认事实", f"统一重评价：{reevaluation['source_model_count']} 个模型、{reevaluation['evaluation_row_count']} 行；最大基础奖励重构误差 {result['data_quality']['base_reward_reconciliation_max_abs']:.3e}。"],
        ["已确认事实", "PPO checkpoint 名称是目标步数；实际模型步数因 2,048-step rollout 向上越过，300k 终点实际为 " + str(result["data_quality"]["actual_timesteps_by_target"]["300000"]) + "。所有条件的对应实际步数一致。"],
        ["研究决定", "阶段一只检测 proxy–diagnostic divergence；阶段二 shaping 本轮不运行、不调参、不讨论效果。"],
        ["研究决定", "开发网格在 [0.125,0.25] 内加入 0.15625、0.1875、0.21875，形成 0.03125 的等距细分。"],
        ["未验证假设", "开发候选能否在一组全新的 held-out training seeds 上复现，并在晚期 checkpoint 中持续。"],
        ["未决事项", "实际意义阈值、六条件或缩减矩阵，以及五枚描述性或八枚较强方向性 seed 方案。"],
        ["证据不支持", "全局最优权重、唯一阈值、连续相变、真实机器人安全、跨算法或跨环境普遍性。"],
    ]
    story.append(table(["类别", "内容"], rows, styles, [31 * mm, 139 * mm]))
    story.append(para("1.1 Proposal_G6 规定方案与实际方案", styles, "h2"))
    rows = [
        ["权重范围", "以 0.5 为中心的小范围", "0.125–0.5，仅向下", "改为针对控制代价低估的单向机制检验；不能称双向或 centred sweep。"],
        ["条件", "参考 + 三个减弱 + 一个 shaping", "开发阶段参考 + 六个减弱；无 shaping", "提高局部定位分辨率，但 proposal 的完整缓解目标尚未完成。"],
        ["checkpoint", "25%/50%/75%/100%", "50k 至 300k 共六个", "提高时间分辨率和优化压力检查；不增加独立复制数。"],
        ["seed", "单主 seed；额外 seed 可选", "两枚开发 seed；五枚或八枚正式 seed 待选", "降低单次偶然性风险，但增加正式计算成本。"],
        ["阈值主张", "识别 divergence threshold", "只报告离散起始区间", "避免由七个离散点推断连续临界点或相变。"],
    ]
    story.append(table(["项目", "Proposal_G6", "本轮实际", "偏差影响"], rows, styles, [25*mm, 42*mm, 45*mm, 58*mm]))
    story.append(
        para(
            "上述偏差均不改变 Ant-v5、PPO 与 ctrl_cost_weight 单变量设计，但会改变可支持的结论与时间成本。它们必须在正式 protocol freeze 前由小组和导师知情接受，不能用本轮开发结果反向改写 proposal 已规定的历史事实。",
            styles,
        )
    )

    story.append(para("2. 两小时前方案与本轮修订", styles, "h1"))
    rows = [
        ["保留", "同一候选公式 R_w 对候选与参考轨迹重评分；避免比较不同公式的原始 return。"],
        ["保留", "training seed 是独立复制单位；evaluation episode 嵌套在策略内并先聚合。"],
        ["保留", "300k 与 50k 间隔的六个 checkpoint；终点为主，时间趋势为次。"],
        ["修改", "从 0.5/0.375/0.25/0.125 的粗网格，增加 0.21875/0.1875/0.15625 定位离散起始区间。"],
        ["修改", "原先“至少两个伤害指标”改为“至少一个预声明领域越过实际阈值”；相关指标只属于同一领域。"],
        ["修改", "增加 mean_forward_velocity，但不把步频误称为速度；增加 normalised_action_roughness。"],
        ["废止", "prospective-v3 的“代理挑最佳权重”逻辑；它回答调参可靠性，不直接回答本阶段要制造并检测合理错位的研究问题。"],
        ["废止", "所有 delta>0 即恶化；这会把浮点噪声和极小差异伪装成构念伤害。"],
        ["延期", "所有 shaping 条件、参数和因果解释；只有阶段一正式确认后才进入第二实验。"],
    ]
    story.append(table(["处理", "方案要点"], rows, styles, [23 * mm, 147 * mm]))

    story.append(para("3. “消失成果”与未完成任务取证", styles, "h1"))
    story.append(
        para(
            "文件系统与聊天记录只支持一个被中断的四策略扩展任务，而不支持“两份已经完成后又消失的正式成果”。该任务包含 0.125 与 0.375 两个系数、每个系数两枚 seed；停止时四个策略都只到达 50k 和 100k，因此留下 8 个局部 checkpoint。它们没有生成 300k endpoint，故不存在可恢复的完整 endpoint 结论。",
            styles,
        )
    )
    rows = [
        ["首次目录", "expanded_core_300k_20260813", "17", "8", "2,588,618", "中断；仅 50k/100k"],
        ["重启目录", "expanded_core_300k_restarted_20260813", "49", "24", "8,010,671", "完整；4/4 策略，240 行"],
    ]
    story.append(table(["证据", "目录", "文件", "模型", "字节", "判定"], rows, styles, [18*mm, 61*mm, 14*mm, 14*mm, 25*mm, 38*mm]))
    story.append(
        para(
            f"两套 ZIP 的 SHA-256 均不同，但张量级审计显示 {forensic['shared_checkpoint_count']} 对共有 checkpoint 的策略参数逐项完全相同，最大绝对差为 0。因此 ZIP 差异只能说明归档字节不同，不能证明策略不同。两套文件均保留；分析只使用完整重启目录。此前未完成的统一重评价、实际意义阈值、密集局部网格和完整视频修复已在本轮完成；全新 seed 正式确认及阶段二 shaping 仍未运行。",
            styles,
        )
    )

    story.append(para("4. 阶段一研究逻辑", styles, "h1"))
    story.append(para("4.1 两个不同层级的目标", styles, "h2"))
    story.append(
        para(
            "机器人任务目标是在固定时域内持续向 +x 方向移动并保持 Gymnasium 所定义的健康状态，同时接受动作幅度与接触代价。研究者目标则是在合理的控制代价区间内，检验 PPO 是否能提高代理目标却损害未直接优化的行为诊断。前者定义 MDP 中的任务，后者定义研究问题；二者不能混写成同一个“目标”。",
            styles,
        )
    )
    story.append(
        KeepTogether(
            [
                scaled_image(equation_dir / "reward_equation.png", 165 * mm, 23 * mm),
                para("式 1　不同条件只改变控制代价系数 w；本轮所有 shaping 权重均为 0。", styles, "caption"),
            ]
        )
    )
    story.append(
        KeepTogether(
            [
                scaled_image(equation_dir / "matched_rescore_equation.png", 120 * mm, 18 * mm),
                para("式 2　候选与参考必须在同一候选公式下比较。ΔR_w>0 表示候选在该代理尺上更高。", styles, "caption"),
            ]
        )
    )
    story.append(para("4.2 假说", styles, "h2"))
    story.append(
        para(
            "阶段一研究问题保留“代理表现相近或较高”的原始表述。为避免事后任意定义“相近”，强候选的可执行主假说采用更保守的严格条件：在预先声明的合理 ctrl_cost_weight 区间内，至少一个降低权重的策略在配对 R_w 比较中获得更高代理回报，同时至少一个预声明行为领域恶化并越过实际意义阈值。没有预先锁定非劣效界值的“相近”结果只能描述，不能触发正式实验。次假说定位分歧首次出现或明显加剧的离散相邻区间，不证明连续函数的不连续点。",
            styles,
        )
    )
    story.append(
        para(
            "这一设计与文献中的核心逻辑有关但不等同：Pan, Bhatia and Steinhardt (2022, p. 1) 在人工构造且具有 true reward 的环境中沿能力轴观察更高代理与更低真实奖励；Skalse et al. (2022, p. 4, Definition 1) 用两种奖励对同一策略对产生相反排序来定义 hackability；Karwowski et al. (2024, pp. 1, 6) 讨论优化压力超过临界点后的 Goodhart 行为。但本项目没有标注的 true reward，因此只能报告分解诊断，不得照搬这些文献中的 true-reward、hackability 或 critical-point 结论。",
            styles,
        )
    )

    story.append(para("5. 指标、健康与实际意义", styles, "h1"))
    story.append(
        KeepTogether(
            [
                scaled_image(equation_dir / "locomotion_equations.png", 160 * mm, 20 * mm),
                para("式 3　速度是 x 方向平移速度，不是腿的步频。完整 1,000-step episode 对应 50 个仿真秒。", styles, "caption"),
            ]
        )
    )
    story.append(
        KeepTogether(
            [
                scaled_image(equation_dir / "diagnostic_equations.png", 174 * mm, 28 * mm),
                para("式 4　路径效率、动作边界占用与归一化动作粗糙度。S 不是物理电机饱和，Q 不是实测机械冲击。", styles, "caption"),
            ]
        )
    )
    rows = [
        ["前进有效性", "净前进降低 ≥1.0，或路径效率降低 ≥0.10", "至少同一个具体指标跨 seed 达标"],
        ["环境健康", "不健康终止率增加 ≥0.20", "10 episodes 下等于至少 2 局差异"],
        ["横向控制", "平均绝对横向漂移增加 ≥0.50", "仿真位置单位；非硬件安全线"],
        ["姿态稳定", "躯干倾角 RMS 增加 ≥5°", "由 0.087266 rad 换算"],
        ["指令质量", "S 与 Q 均增加 ≥0.02", "两指标共同构成一个领域"],
    ]
    story.append(table(["领域", "名义阈值", "一致性规则"], rows, styles, [31*mm, 77*mm, 62*mm]))
    story.append(
        para(
            "Gymnasium 的健康定义较窄：状态必须有限，躯干 z 必须位于 [0.2,1.0]。低 z、高 z 和非有限状态分别记录。横向漂移、倾斜和动作粗糙度属于研究者的扩展诊断，不等同于环境健康，也不自动等同于真实机器人安全。",
            styles,
        )
    )

    story.append(para("6. 实验矩阵与随机性控制", styles, "h1"))
    rows = [
        ["环境 / 算法", "Ant-v5 / PPO (MlpPolicy, 64–64 Tanh actor and critic)"],
        ["本机硬件", "Intel Core i7-13700H；20 logical CPUs；16.67 GB RAM；CPU-only"],
        ["关键软件", f"Python {reevaluation['software']['python']}；PyTorch {reevaluation['software']['torch']}；Gymnasium {reevaluation['software']['gymnasium']}；SB3 {reevaluation['software']['stable_baselines3']}"],
        ["已有开发权重", "0.5, 0.375, 0.25, 0.125"],
        ["新增密集权重", "0.21875, 0.1875, 0.15625"],
        ["开发 training seeds", "41101, 41102"],
        ["evaluation seeds", "51101–51110；每个策略/checkpoint 采用相同起始扰动"],
        ["训练预算", "300,000 timesteps；50k/100k/150k/200k/250k/300k"],
        ["独立单位", "trained policy / training seed；episode 与 checkpoint 均不是独立复制"],
        ["正式预留", "training seeds 42001–42005；evaluation block 从 52001 开始"],
    ]
    story.append(table(["项目", "锁定内容"], rows, styles, [39*mm, 131*mm]))
    story.append(
        para(
            "training seed 同时影响网络初值、策略采样、环境初始扰动序列和 minibatch 顺序；evaluation seed 只在模型冻结后决定初始扰动。相同 evaluation seed 让条件间起点可比，但不能替代新的训练复制。Agarwal et al. (2021) 强调少量 RL 运行下的不确定性，因此本开发结果展示全部 seed 原始点，不依赖单一均值。",
            styles,
        )
    )
    story.append(
        para(
            "本阶段把“合理区间”操作化为官方默认值 0.5 到其四分之一 0.125：不包含零控制代价，也不沿旧探索继续降到 0.0625。该范围服务于一个有方向的机制假说，即控制努力被低估时是否出现分歧；它不检验过高控制代价导致的停滞，因此不能代表完整的双向权重敏感性研究。",
            styles,
        )
    )

    story.append(para("7. 工程修订与验证", styles, "h1"))
    rows = [
        ["测量", "新增 environment_dt、episode_duration_seconds、mean_forward_velocity、normalised_action_roughness。"],
        ["判定", "新增按领域的实际阈值；修复不同 seed 伤害不同子指标却被误称一致的问题。"],
        ["来源", "旧模型统一重评价；每个模型写入 SHA-256；分析输入和代码均写哈希。"],
        ["步数", "checkpoint 标签与 actual_model_timesteps 分开记录；同一目标下实际步数必须跨条件一致。"],
        ["纠错", "首版统一重评价曾把目标步数误写为实际步数；该目录保留但排除，v2 从 PPO 模型元数据读取实际值后重新生成。"],
        ["QA 自纠", "验收脚本首次错误依赖 pytest 的摘要字符串，后改为退出码与 collect-only 数量；第二次错误假定了不存在的合并字段，后改按实际 schema 检查分项不变量。两次均未修改训练或评价数据。"],
        ["报告重建自纠", "最终排版重建首次误选了名称相近但不存在运行清单的目录；读取在生成前即失败。随后以 parallel_completion.json 定位真实训练目录，未改动任何实验输入或结果。"],
        ["调度", "新权重任务按固定 task_order_seed 打乱；每个完整策略结束即写增量 CSV。"],
        ["隔离", "所有输出位于 C 盘 revision 目录；D 盘 canonical、formal-v1、旧模型和 Git 历史未修改。"],
    ]
    story.append(table(["层面", "修改"], rows, styles, [28*mm, 142*mm]))
    validation_rows = [
        [item["name"], item["status"], item.get("detail", "")]
        for item in validation["checks"]
    ]
    story.append(Spacer(1, 2 * mm))
    story.append(table(["验证", "状态", "证据"], validation_rows, styles, [50*mm, 20*mm, 100*mm]))
    if video_manifests:
        video_rows = [
            [
                item["condition_id"],
                str(item["training_seed"]),
                f"{item['frames']} frames / {item['video_duration_seconds']:.2f} s",
                "不健康终止" if item["episode_summary"]["terminated"] else "时间上限",
                "1.0×",
            ]
            for item in video_manifests
        ]
        story.append(Spacer(1, 2 * mm))
        story.append(table(["完整轨迹", "training seed", "长度", "结束原因", "播放速度"], video_rows, styles, [38*mm, 29*mm, 41*mm, 38*mm, 24*mm]))
        if video_contact_sheet is not None:
            add_figure(
                story,
                video_contact_sheet,
                "图 A　统一 evaluation seed 51101 下六段完整轨迹的固定中点帧。截图只用于行为解释；候选筛选在视频生成前由数值规则完成。参考 seed 41102 与低权重 seed 41102 的翻转姿态说明 torso-height health 与广义姿态质量不是同一构念。",
                styles,
                max_height=150 * mm,
            )

    story.append(PageBreak())
    story.append(para("8. 开发性结果", styles, "h1"))
    quality = result["data_quality"]
    story.append(
        para(
            f"合并数据包含 {quality['row_count']} 行、{quality['cell_count']} 个策略/checkpoint cell，每个 cell 为 {quality['episodes_per_cell']} 个 episode；重复键为 {quality['duplicate_episode_keys']}。基础奖励与控制代价重构误差最大值分别为 {quality['base_reward_reconciliation_max_abs']:.3e} 和 {quality['ctrl_cost_reconciliation_max_abs']:.3e}。",
            styles,
        )
    )
    endpoint_rows = []
    for screen in sorted(screens, key=lambda item: item["candidate_weight"], reverse=True):
        advantages = [
            fmt(item["candidate_proxy_advantage_under_R_w"], 1)
            for item in screen["contrasts"]
        ]
        endpoint_rows.append(
            [
                f"{screen['candidate_weight']:g}",
                " / ".join(advantages),
                "；".join(domain_cn(name) for name in screen["consistently_harmed_domains"]) or "无",
                "通过" if screen["strong_development_candidate"] else "不通过",
            ]
        )
    story.append(table(["w", "ΔR_w：seed 41101 / 41102", "跨 seed 越阈值领域", "强候选"], endpoint_rows, styles, [18*mm, 51*mm, 72*mm, 29*mm]))
    if candidates:
        story.append(
            para(
                f"名义阈值下的候选集合为 {', '.join(f'{x:g}' for x in candidates)}，首个离散起始区间为 {onset_text}。这不是“从头到尾稳定错位”：晚期持续性另按 200k/250k/300k 三个依赖 checkpoint 检查，且正式结论仍须新 training seeds。",
                styles,
            )
        )
    add_figure(
        story,
        analysis_root / "endpoint_seed_contrasts.png",
        "图 1　终点处每枚 training seed 的配对代理优势与全部七项诊断伤害。代理面板的虚线 0 表示优势边界；诊断面板的虚线 1 表示实际意义阈值。蓝色圆点与橙色方点分别表示两枚 training seed。",
        styles,
        max_height=112 * mm,
    )
    add_figure(
        story,
        analysis_root / "domain_replication_matrix.png",
        "图 2　每个权重下越过各领域名义阈值的 development training seed 数。数字 2 表示两枚 seed 均达到该领域的单-seed规则；跨 seed 一致性仍要求相同具体子指标。",
        styles,
        max_height=80 * mm,
    )
    add_figure(
        story,
        analysis_root / "cross_rescore_matrix.png",
        "图 3　终点轨迹的交叉重评分矩阵。行是策略训练权重，列是用于评分的 R_w。该矩阵展示策略排名对评价尺的敏感性；任何一列都不是 true performance。",
        styles,
        max_height=115 * mm,
    )
    add_figure(
        story,
        analysis_root / "checkpoint_replication_matrix.png",
        "图 4　各 checkpoint 中同时出现正代理优势和至少一个越阈值领域的 training seed 数。checkpoint 是同一策略训练轨迹的重复测量，不能把六列当作六次独立实验。",
        styles,
        max_height=95 * mm,
    )
    add_figure(
        story,
        analysis_root / "progress_effort_map.png",
        "图 5　终点净前进与累计动作努力的关系图。每个点是一枚 training seed 下的策略均值；连线仅表示权重顺序，不表示连续函数或统计拟合。该图辅助解释多目标权衡，不参与候选判定。",
        styles,
        max_height=105 * mm,
    )

    story.append(para("9. 阈值与时间敏感性", styles, "h1"))
    sensitivity_rows = []
    for scale in sorted(sensitivity["margin_scale"].unique()):
        subset = sensitivity[
            (sensitivity["margin_scale"] == scale)
            & (sensitivity["strong_development_candidate"] == True)  # noqa: E712
        ]
        weights = ", ".join(f"{value:g}" for value in sorted(subset["candidate_weight"], reverse=True)) or "无"
        sensitivity_rows.append([f"{scale:g}×", weights])
    story.append(table(["阈值尺度", "强开发性候选权重"], sensitivity_rows, styles, [38*mm, 132*mm]))
    persistence_rows = [
        [f"{row.candidate_weight:g}", f"{int(row.late_checkpoints_strong_count)}/3", "是" if row.late_window_persistent else "否"]
        for row in persistence.itertuples(index=False)
    ]
    story.append(Spacer(1, 2 * mm))
    story.append(table(["w", "晚期强候选 checkpoint", "≥2/3"], persistence_rows, styles, [35*mm, 85*mm, 50*mm]))
    story.append(
        para(
            "阈值敏感性回答的是“结论是否完全依赖一组任意门槛”。若候选集合在 0.5×、1×、2× 下大幅变化，正式协议应降低措辞强度并把结果描述为阈值依赖；即使不变化，也不能证明阈值具备外部安全效度。",
            styles,
        )
    )
    story.append(para("9.1 评价初始扰动的不确定性", styles, "h2"))
    metric_labels = {
        "proxy_advantage_under_R_w": "代理优势",
        "forward_path_efficiency": "路径效率伤害",
        "lateral_drift_mean_abs": "横向漂移伤害",
        "action_saturation_rate": "动作饱和伤害",
        "normalised_action_roughness": "动作粗糙度伤害",
    }
    selected_quantities = {
        0.21875: {"proxy_advantage_under_R_w", "lateral_drift_mean_abs"},
        0.125: {
            "proxy_advantage_under_R_w",
            "forward_path_efficiency",
            "action_saturation_rate",
            "normalised_action_roughness",
        },
    }
    bootstrap_rows = []
    for row in bootstrap.itertuples(index=False):
        allowed = selected_quantities.get(float(row.candidate_weight), set())
        if row.quantity not in allowed:
            continue
        bootstrap_rows.append(
            [
                f"{row.candidate_weight:g}",
                str(int(row.training_seed)),
                metric_labels[row.quantity],
                f"{row.mean_delta_or_directed_harm:.3f} "
                f"[{row.bootstrap_95pct_lower:.3f}, {row.bootstrap_95pct_upper:.3f}]",
                f"{row.bootstrap_fraction_above_boundary:.3f}",
            ]
        )
    story.append(
        table(
            ["w", "training seed", "量", "均值 [2.5%, 97.5%]", "越界比例"],
            bootstrap_rows,
            styles,
            [15 * mm, 27 * mm, 42 * mm, 57 * mm, 29 * mm],
        )
    )
    story.append(
        para(
            "这里的 20,000 次配对 bootstrap 只重采样同一已训练策略下的 10 个 evaluation seeds，描述初始状态扰动敏感性；它不是跨 training seed 的置信区间。0.125 的两项动作质量伤害在两枚 training seeds 中均较稳定，而 0.21875 的横向漂移区间均跨过名义 0.50 边界，0.125/41102 的代理优势区间也跨过 0，因此这些候选仍须新的独立训练复制。",
            styles,
        )
    )

    story.append(para("10. 替代解释与混杂因素", styles, "h1"))
    story.append(bullet("PPO 随机性：两枚 seed 可能恰好产生相似行为；正式阶段必须使用与开发阶段不重叠的新 seed，并在运行前冻结五枚或八枚方案。", styles))
    story.append(bullet("权重与代理尺耦合：每个候选用自身 R_w 比较是公平的目标内比较，但不同 R_w 的排名可能变化；交叉重评分矩阵用于公开这种敏感性。", styles))
    story.append(bullet("存活奖励混杂：健康策略每步 +1，提前终止同时改变 episode length、总回报和速度分母；必须联合报告。", styles))
    story.append(bullet("指标构念局限：路径效率与动作粗糙度是项目定义，不是权威四足机器人评分标准；视频只能支持解释，不能替代量化门槛。", styles))
    story.append(bullet("开发者自由度：局部网格和指标是在旧结果之后提出；通过保留开发标签并用全新正式 seed 控制，而不是把探索包装成预注册。", styles))
    story.append(bullet("局部而非连续：七个离散权重只能夹逼起始区间，不能估计唯一临界点或证明骤变。", styles))
    story.append(bullet("单向范围：本阶段只检验降低控制代价的预期机制；没有证据说明增大到 1.0 或 2.0 时的行为，也不能宣称扫描了所有合理权重。", styles))

    story.append(para("11. 正式阶段建议与成本", styles, "h1"))
    if first_onset:
        minimal_conditions = [0.5, 0.25, 0.21875, 0.125]
        five_conditions = [0.5, 0.25, 0.21875, 0.1875, 0.125]
        transition_conditions = [0.5, 0.25, 0.21875, 0.1875, 0.15625, 0.125]
        recommended = transition_conditions
        story.append(
            para(
                "最小四条件矩阵为 "
                + ", ".join(f"w={value:g}" for value in minimal_conditions)
                + "，只能验证两个候选点的存在。五条件矩阵再加入 w=0.1875，可检验首个候选后的立即退出，但不能夹住低权重再次进入。推荐的六条件矩阵为 "
                + ", ".join(f"w={value:g}" for value in transition_conditions)
                + "；它保留 0.15625，才能完整复核本轮的进入、退出和再次进入图样。该选择必须在 held-out 结果不可见时冻结。",
                styles,
            )
        )
        policy_count = len(recommended) * 5
    else:
        story.append(para("未识别起始区间，因此不建议启动正式长训练。", styles))
        policy_count = 0
    observed_train_per_policy = max(
        float(runtime.groupby(["training_seed", "ctrl_cost_weight"])["train_elapsed_sec"].sum().max()),
        1.0,
    )
    observed_eval_per_policy = max(
        float(runtime.groupby(["training_seed", "ctrl_cost_weight"])["eval_elapsed_sec"].sum().max()),
        1.0,
    )
    estimated_batches = (policy_count + 3) // 4 if policy_count else 0
    estimate_minutes = estimated_batches * (
        observed_train_per_policy + 2.0 * observed_eval_per_policy
    ) / 60.0
    option_rows = []
    for label, conditions in (
        ("四条件：候选存在", minimal_conditions if first_onset else []),
        ("五条件：加立即退出", five_conditions if first_onset else []),
        ("六条件：完整非单调图样", recommended if first_onset else []),
    ):
        option_policies = len(conditions) * 5
        option_batches = (option_policies + 3) // 4 if option_policies else 0
        option_minutes = option_batches * (
            observed_train_per_policy + 2.0 * observed_eval_per_policy
        ) / 60.0
        option_rows.append(
            [label, str(len(conditions)), str(option_policies), f"约 {option_minutes:.0f} min"]
        )
    story.append(para("下表统一按资源受限的五枚 training seeds 与四个并行 worker 估算。", styles, "small"))
    story.append(table(["方案", "条件", "策略", "训练+评价墙钟粗估"], option_rows, styles, [66*mm, 25*mm, 25*mm, 54*mm]))
    rows = [
        ["本轮新增训练 CPU 累计", f"{total_train_seconds/60:.1f} min"],
        ["本轮新增评价 CPU 累计", f"{total_eval_seconds/60:.1f} min"],
        ["六条件 × 五 seeds", f"{policy_count} 个策略；约 {estimate_minutes:.0f} min"],
        ["六条件 × 八 seeds", "48 个策略；约 348 min"],
        ["存储粗估", "30/48 个策略的六个模型 checkpoint 约 56/90 MB；连同 CSV、图和完整 MP4，建议预留 1 GB"],
    ]
    story.append(table(["项目", "估计"], rows, styles, [70*mm, 100*mm]))
    story.append(
        para(
            "停止标准：开发候选为空则停止；五-seed 方案要求同一批至少 4/5 training seeds 联合达到代理与同一诊断规则，八-seed 方案要求至少 7/8；未达到则报告未确认。任何失败 seed 必须按全条件一致的预声明规则处理，不能替换“不好看”的 seed。",
            styles,
        )
    )
    add_figure(
        story,
        equation_dir / "seed_consistency_equation.png",
        "式 5　若把每枚 training seed 的方向视为 p=0.5 的符号结果，4/5 的单侧尾概率仍为 0.1875，因此五枚 seed 的 4/5 规则只能作为描述性复现门槛，不能包装成常规显著性检验。八枚 seed 中至少 7 枚同向的对应尾概率为 0.0352，但仍不解决构念效度与多指标选择问题。",
        styles,
        max_height=24 * mm,
    )

    story.append(para("12. 当前状态判断", styles, "h1"))
    story.append(
        para(
            "工程状态：通过。开发性数据状态：完成并可审计。科学状态：仍未解决。Protocol-freeze 状态：阶段一正式协议尚未冻结，因为实际意义阈值、条件矩阵、one-sided proposal 偏差和 seed 方案仍需明确批准。Release-ready 状态：否。",
            styles,
            "callout",
        )
    )
    story.append(
        para(
            "最重要的边界是：现有结果可以支持“在两枚开发 seed 上观察到候选 proxy–diagnostic divergence”，不能支持“已经复现奖励错位”或“找到了最优权重下的奖励错位”。正式验证的价值正是检验该候选是否能跨全新训练随机性复现，而不是继续调权重追逐阳性结果。",
            styles,
        )
    )

    story.append(para("参考文献", styles, "h1"))
    references = [
        "Agarwal, R., Schwarzer, M., Castro, P.S., Courville, A.C. and Bellemare, M.G. (2021) ‘Deep reinforcement learning at the edge of the statistical precipice’, Advances in Neural Information Processing Systems, 34.",
        "Farama Foundation (2026) ‘Ant - Gymnasium documentation’. Available at: https://gymnasium.farama.org/environments/mujoco/ant/ (Accessed: 14 August 2026).",
        "Group 6 (2026) Proposal_G6: Reward Misspecification and Reward Shaping in Deep Reinforcement Learning for Robotic Automation. Internal project proposal.",
        "Karwowski, J., Hayman, O., Bai, X., Kiendlhofer, K., Griffin, C. and Skalse, J. (2024) ‘Goodhart’s law in reinforcement learning’, International Conference on Learning Representations.",
        "Pan, A., Bhatia, K. and Steinhardt, J. (2022) ‘The effects of reward misspecification: Mapping and mitigating misaligned models’, International Conference on Learning Representations.",
        "Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017) ‘Proximal policy optimization algorithms’, arXiv:1707.06347.",
        "Skalse, J., Howe, N.H.R., Krasheninnikov, D. and Krueger, D. (2022) ‘Defining and characterizing reward gaming’, Advances in Neural Information Processing Systems, 35.",
    ]
    for reference in references:
        story.append(para(reference, styles, "body_noindent"))

    story.append(para("附录 A：关键文件", styles, "h1"))
    files = [
        "configs/stage1_dense_development_v1_20260814.json",
        "configs/stage1_formal_confirmation_proposal_v1_20260814.json",
        "protocols/STAGE1_PROXY_DIVERGENCE_PROTOCOL_DRAFT_20260814.md",
        "protocols/STAGE1_FORMAL_CONFIRMATION_PROPOSAL_20260814.md",
        "docs/STAGE1_METRIC_CONTRACT_20260814.md",
        "docs/STAGE1_DEVIATION_REGISTER_20260814.md",
        "docs/STAGE1_METHOD_CRITIC_20260814.md",
        "protocols/STAGE1_DEVELOPMENT_ANALYSIS_AMENDMENT_20260814.md",
        "src/proxygap/stage1.py",
        "scripts/reevaluate_stage1_models.py",
        "scripts/analyse_stage1_dense_development.py",
        "scripts/render_stage1_full_video.py",
        "scripts/audit_interrupted_model_equivalence.py",
        "scripts/bootstrap_stage1_evaluation_episodes.py",
        "scripts/build_stage1_video_contact_sheet.py",
        "scripts/build_stage1_validation_summary.py",
        str(forensic_path.relative_to(PROJECT_ROOT)),
        str(bootstrap_path.relative_to(PROJECT_ROOT)),
        str((analysis_root / "stage1_development_result.json").relative_to(PROJECT_ROOT)),
        str((analysis_root / "output_manifest.csv").relative_to(PROJECT_ROOT)),
        str((analysis_root / "trajectory_midpoint_contact_sheet.png").relative_to(PROJECT_ROOT)),
        "artifacts/exploration/stage1_harmonised_existing_models_20260814/SUPERSEDED_NOTICE.md",
        "Proposal_G6.pdf (external controlling source; not distributed in this repository)",
    ]
    file_rows = [
        [
            Paragraph(f"{index}.", styles["file_path"]),
            Paragraph(path, styles["file_path"]),
        ]
        for index, path in enumerate(files, start=1)
    ]
    file_table = Table(file_rows, colWidths=[8 * mm, 162 * mm], hAlign="LEFT")
    file_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
            ]
        )
    )
    story.append(file_table)

    doc.build(story, onFirstPage=page_header_footer, onLaterPages=page_header_footer)
    print(output)


if __name__ == "__main__":
    main()
