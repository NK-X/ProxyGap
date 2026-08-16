"""Build the audited Chinese stage-one ProxyGap review PDF.

Run this script with the bundled document Python after equations have been
rendered by ``render_stage1_report_equations.py`` in the ProxyGap environment.
The report reads frozen JSON and image evidence; it does not train or evaluate
an agent and does not alter experiment artefacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

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
    LongTable,
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


NAVY = colors.HexColor("#16324F")
TEAL = colors.HexColor("#008C7A")
ORANGE = colors.HexColor("#E07A1F")
RED = colors.HexColor("#B33A3A")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_TEAL = colors.HexColor("#E7F5F2")
LIGHT_ORANGE = colors.HexColor("#FFF2E5")
LIGHT_RED = colors.HexColor("#FBECEC")
LIGHT_GREY = colors.HexColor("#F2F4F5")
MID_GREY = colors.HexColor("#69757F")
GRID = colors.HexColor("#CBD3D8")
WHITE = colors.white


def register_fonts() -> None:
    for regular, bold in iter_font_pairs(DENG_FIRST_CJK_FONT_PAIRS):
        try:
            pdfmetrics.registerFont(TTFont("Deng", str(regular), subfontIndex=0))
            pdfmetrics.registerFont(TTFont("DengBold", str(bold), subfontIndex=0))
            pdfmetrics.registerFontFamily(
                "Deng",
                normal="Deng",
                bold="DengBold",
                italic="Deng",
                boldItalic="DengBold",
            )
            return
        except Exception:
            continue
    raise RuntimeError(
        f"No suitable Chinese font pair was found; set {FONT_DIR_ENV} to a font directory."
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="DengBold",
            fontSize=25,
            leading=34,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=7 * mm,
            wordWrap="CJK",
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName="Deng",
            fontSize=12,
            leading=18,
            textColor=MID_GREY,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="DengBold",
            fontSize=17,
            leading=23,
            textColor=NAVY,
            spaceBefore=3 * mm,
            spaceAfter=3 * mm,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="DengBold",
            fontSize=12.5,
            leading=18,
            textColor=TEAL,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=9.5,
            leading=15,
            textColor=colors.HexColor("#20282E"),
            alignment=TA_JUSTIFY,
            spaceAfter=2.2 * mm,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=7.7,
            leading=11.2,
            textColor=colors.HexColor("#2E3941"),
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "tiny": ParagraphStyle(
            "tiny",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=6.5,
            leading=9.0,
            textColor=colors.HexColor("#2E3941"),
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=7.5,
            leading=10.5,
            textColor=MID_GREY,
            alignment=TA_LEFT,
            spaceBefore=1.2 * mm,
            spaceAfter=2.5 * mm,
            wordWrap="CJK",
        ),
        "quote": ParagraphStyle(
            "quote",
            parent=base["BodyText"],
            fontName="DengBold",
            fontSize=11,
            leading=17,
            textColor=NAVY,
            alignment=TA_LEFT,
            leftIndent=5 * mm,
            rightIndent=5 * mm,
            spaceBefore=2 * mm,
            spaceAfter=2 * mm,
            wordWrap="CJK",
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta",
            parent=base["BodyText"],
            fontName="Deng",
            fontSize=9,
            leading=14,
            textColor=colors.HexColor("#46545F"),
            wordWrap="CJK",
        ),
    }
    return styles


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullet(text: str, styles: dict[str, ParagraphStyle], colour=TEAL) -> Table:
    return Table(
        [[para("●", styles["small"]), para(text, styles["body"])]],
        colWidths=[4.5 * mm, 166 * mm],
        style=TableStyle(
            [
                ("TEXTCOLOR", (0, 0), (0, 0), colour),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.5 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        ),
    )


def box(
    title: str,
    body: str,
    styles: dict[str, ParagraphStyle],
    background=LIGHT_BLUE,
    accent=NAVY,
) -> Table:
    data = [[para(title, styles["h2"])], [para(body, styles["body"])]]
    table = Table(data, colWidths=[174 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.7, accent),
                ("LINEBEFORE", (0, 0), (0, -1), 3.0, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2.3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.3 * mm),
            ]
        )
    )
    return table


def data_table(
    rows: list[list[object]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
    header=True,
    font_size="small",
) -> LongTable:
    formatted = []
    for row_index, row in enumerate(rows):
        current = []
        for value in row:
            value_text = escape(str(value)).replace("\n", "<br/>")
            if header and row_index == 0:
                current.append(
                    para(
                        f'<font color="#FFFFFF"><b>{value_text}</b></font>',
                        styles[font_size],
                    )
                )
            else:
                current.append(para(value_text, styles[font_size]))
        formatted.append(current)
    table = LongTable(formatted, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY if header else WHITE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE if header else colors.black),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.0 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.0 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5 * mm),
    ]
    for row_index in range(1 if header else 0, len(formatted)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_GREY))
    table.setStyle(TableStyle(commands))
    return table


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    image = Image(str(path))
    ratio = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * ratio
    image.drawHeight = image.imageHeight * ratio
    return image


def figure(
    path: Path,
    caption: str,
    styles: dict[str, ParagraphStyle],
    max_width=174 * mm,
    max_height=113 * mm,
) -> list[object]:
    return [
        scaled_image(path, max_width, max_height),
        para(caption, styles["caption"]),
    ]


def page_header_footer(canvas, document) -> None:
    canvas.saveState()
    page_width, page_height = A4
    canvas.setStrokeColor(colors.HexColor("#C8D1D7"))
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, page_height - 14 * mm, page_width - 18 * mm, page_height - 14 * mm)
    canvas.setFont("Deng", 7.5)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(18 * mm, page_height - 10.5 * mm, "ProxyGap | 阶段一双向权重开发审查")
    canvas.drawRightString(page_width - 18 * mm, 10 * mm, f"第 {document.page} 页")
    canvas.restoreState()


def first_page(canvas, document) -> None:
    canvas.saveState()
    page_width, page_height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, page_height - 24 * mm, page_width, 24 * mm, stroke=0, fill=1)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, page_width, 7 * mm, stroke=0, fill=1)
    canvas.setFont("Deng", 7.5)
    canvas.setFillColor(MID_GREY)
    canvas.drawRightString(page_width - 18 * mm, 11 * mm, "2026-08-14 | Development evidence only")
    canvas.restoreState()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def format_weight(value: float) -> str:
    return f"{value:g}"


def build_story(root: Path, equations: Path, styles: dict[str, ParagraphStyle]) -> list[object]:
    analysis = root / "artifacts" / "analysis" / "stage1_bidirectional_development_v2_20260814"
    result_path = analysis / "stage1_development_result.json"
    audit_path = analysis / "stage1_bidirectional_audit.json"
    result = load_json(result_path)
    audit = load_json(audit_path)

    story: list[object] = []

    # Cover.
    story.extend(
        [
            Spacer(1, 25 * mm),
            para("ProxyGap 阶段一", styles["subtitle"]),
            para("双向权重开发审查报告", styles["title"]),
            para(
                "Ant-v5 + PPO 奖励错位研究：证据裁决、构念风险与正式实验冻结条件",
                styles["subtitle"],
            ),
            Spacer(1, 18 * mm),
            box(
                "当前裁决",
                "保留 <b>w=0.21875</b> 与 <b>w=0.125</b> 作为开发候选；正式 held-out training 与 reward shaping 继续阻止。当前状态是 <b>scientifically unresolved</b>，不是 protocol-freeze-ready。",
                styles,
                background=LIGHT_ORANGE,
                accent=ORANGE,
            ),
            Spacer(1, 9 * mm),
            data_table(
                [
                    ["项目", "实际状态"],
                    ["算法与环境", "Stable-Baselines3 PPO；Gymnasium Ant-v5；CPU-only"],
                    ["开发网格", "9 weights × 2 training seeds × 6 checkpoints × 10 evaluation episodes"],
                    ["证据量", "1,080 episode rows；59 automated tests passed"],
                    ["本轮新增", "上侧 0.625 与 0.75；完整视频；替代解释审计；revision gate V3"],
                    ["禁止越界", "没有启动 formal seeds；没有启动 shaping；没有改写 D 盘 canonical 历史项目"],
                ],
                [42 * mm, 132 * mm],
                styles,
            ),
            Spacer(1, 13 * mm),
            para(
                "报告日期：2026 年 8 月 14 日<br/>"
                "报告语言：中文；参考文献按 Leeds Harvard author-date 结构整理<br/>"
                "结论层级：development nomination only",
                styles["cover_meta"],
            ),
            PageBreak(),
        ]
    )

    # Executive verdict.
    story.append(para("执行摘要", styles["h1"]))
    story.append(
        para(
            "本轮解决了原设计只向较小控制代价搜索的问题。以默认 <b>w=0.5</b> 为中心，"
            "开发网格扩展至 <b>0.125-0.75</b>，并保持奖励函数除控制代价权重外完全不变。"
            "这使研究问题成为可证伪的双向检验，而不是预设“减小权重必然出错”。",
            styles["body"],
        )
    )
    story.append(
        box(
            "最重要的阳性信号",
            "在 300k endpoint，<b>w=0.21875</b> 的两个策略都在匹配的 R<sub>0.21875</sub> 量尺上超过同 seed 的 w=0.5 参考；平均绝对横向漂移分别增加 <b>1.026</b> 与 <b>0.669</b>。信号在 200k、250k、300k 均出现。",
            styles,
            background=LIGHT_TEAL,
            accent=TEAL,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        box(
            "为什么还不能称为奖励错位",
            "w=0.5 参考策略在两个 development seeds 下的不健康终止率为 <b>0.90</b> 和 <b>0.70</b>。候选运行满 1,000 steps，而参考平均只运行 331.4 与 485.9 steps。按位移归一化的横向指标没有跨两个 seeds 一致恶化，因此“走得更久导致绝对漂移更大”仍是合理替代解释。",
            styles,
            background=LIGHT_RED,
            accent=RED,
        )
    )
    story.append(Spacer(1, 4 * mm))
    for item in [
        "<b>保留：</b> 双向开发网格、匹配重评分、seed 分层、奖励分解、六 checkpoints、完整视频和敏感性审计。",
        "<b>必须冻结：</b> 是否把走廊遵循定义为隐藏行为要求；参考最低能力；formal 条件与 seeds；停止和排除规则。",
        "<b>推荐下一步：</b> 只把 w=0.5、0.21875、0.125 的现有 development 策略延长至 1M steps，不同时改变网络或 normalization。",
        "<b>阶段边界：</b> 阶段二 shaping 暂停，直到阶段一正式判定完成。",
    ]:
        story.append(bullet(item, styles))
    story.append(PageBreak())

    # Evidence classification.
    story.append(para("1. 证据状态分类", styles["h1"]))
    rows = [
        ["类别", "当前内容", "允许的表述"],
        ["已确认事实", "环境、算法、实际 PPO 配置、数据规模、测试结果、候选筛选输出", "可作为方法与工程事实报告"],
        ["研究决定", "双向权重、matched rescoring、development/formal seed 隔离、视频选择规则", "需在版本化协议中保持"],
        ["未验证假设", "候选是低总体表现；参考欠训练；候选关系能在新 seeds 复制", "只能写为待检验"],
        ["未决事项", "走廊意图、能力门、1M 扩展、formal matrix、5 或 8 seeds", "需要在 formal launch 前冻结"],
        ["不支持结论", "全局最优权重、true reward、普遍 reward hacking、现实硬件安全", "不得进入摘要、结果或结论"],
    ]
    story.append(data_table(rows, [28 * mm, 92 * mm, 54 * mm], styles))
    story.append(Spacer(1, 4 * mm))
    story.append(
        para(
            "这里的关键边界是：项目已定义研究者希望观察的行为 guardrails，但并没有一个独立、完整、经外部标定的 true reward。"
            "因此可检验的是 <b>proxy-diagnostic divergence</b>，而不能直接套用“代理奖励提高且 true reward 降低”的最强形式定义。",
            styles["body"],
        )
    )
    story.append(
        box(
            "反事实检查",
            "若 1M 后参考仍不合格，或候选的归一化行为诊断不恶化，当前“奖励错位”解释应被削弱或拒绝。不能通过临时改变指标、阈值或选择另一个最有故事的 seed 来挽救阳性结果。",
            styles,
            background=LIGHT_ORANGE,
            accent=ORANGE,
        )
    )
    story.append(PageBreak())

    # Methods and equations.
    story.append(para("2. 阶段一设计与数学定义", styles["h1"]))
    story.append(para("2.1 三层目标必须分开", styles["h2"]))
    target_rows = [
        ["层级", "含义", "本项目实现"],
        ["机器人任务", "一局中机器人被要求做什么", "最多 1,000 steps 内持续向 +x 前进并保持健康；是否必须守走廊尚未冻结"],
        ["代理奖励", "PPO 实际优化的标量", "forward + survive + contact - w × squared action"],
        ["研究目标", "研究者想检验什么", "合理权重网格中是否出现代理提高而预声明行为 guardrail 恶化"],
    ]
    story.append(data_table(target_rows, [30 * mm, 55 * mm, 89 * mm], styles))
    story.append(para("2.2 奖励分解", styles["h2"]))
    story.append(scaled_image(equations / "equation_reward.png", 160 * mm, 18 * mm))
    story.append(
        para(
            "F、S、C 分别为 forward、survive/healthy 与有符号 contact reward sums；A 为累计平方动作。"
            "控制代价权重 w 是本阶段唯一操纵变量，所有 shaping 项均为 0。",
            styles["caption"],
        )
    )
    story.append(para("2.3 匹配重评分与可证伪假说", styles["h2"]))
    story.append(scaled_image(equations / "equation_matched_contrast.png", 160 * mm, 18 * mm))
    story.append(scaled_image(equations / "equation_hypothesis.png", 168 * mm, 18 * mm))
    story.append(
        para(
            "每个候选与同 training seed 的参考策略都使用同一个候选 R<sub>w</sub> 重评分。"
            "这避免比较不同单位的 condition-specific returns。严格代理增益与 5% 相对非劣性同时保存，主开发筛选要求两个 seeds 均满足且至少一个行为领域跨过预设实用阈值。",
            styles["body"],
        )
    )
    story.append(para("2.4 行为指标", styles["h2"]))
    story.append(scaled_image(equations / "equation_locomotion.png", 170 * mm, 25 * mm))
    story.append(scaled_image(equations / "equation_diagnostics.png", 165 * mm, 20 * mm))
    story.append(
        para(
            "速度是质心 x 位移除以仿真时间，不是腿部步频。动作边界占用表示动作命令接近 [-1,1] 的比例；"
            "它不是电机物理饱和。动作粗糙度是连续动作向量变化的代理，也不是实测机械冲击。",
            styles["caption"],
        )
    )
    story.append(PageBreak())

    # Matrix and QA.
    story.append(para("3. 实验矩阵与数据质量", styles["h1"]))
    matrix_rows = [
        ["设计项", "实际执行"],
        ["权重", "0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.375, 0.5, 0.625, 0.75"],
        ["Training seeds", "41101, 41102；仅用于 development"],
        ["Checkpoints", "50k, 100k, 150k, 200k, 250k, 300k target；实际受 rollout 对齐"],
        ["Evaluation seeds", "51101-51110；每个已训练策略 10 episodes；跨条件配对"],
        ["Episode", "最多 1,000 steps；dt=0.05 s；最多 50 s 仿真时间"],
        ["数据量", "108 cells；1,080 rows；70 columns"],
        ["上侧运行", "240 rows；24 models；墙钟 869.1 s；无失败"],
    ]
    story.append(data_table(matrix_rows, [42 * mm, 132 * mm], styles))
    story.append(Spacer(1, 4 * mm))
    qa_rows = [["审计项", "结果", "裁决"]]
    dq = result["data_quality"]
    qa_rows.extend(
        [
            ["Cell 完整性", f"{dq['cell_count']} cells；每 cell {dq['episodes_per_cell']} episodes", "PASS"],
            ["重复 episode keys", dq["duplicate_episode_keys"], "PASS"],
            ["非有限 decision metrics", sum(dq["non_finite_decision_metric_counts"].values()), "PASS"],
            ["非法 episode 结束", dq["invalid_episode_end_state_count"], "PASS"],
            ["Duration 最大误差", f"{dq['duration_reconciliation_max_abs']:.2e}", "PASS"],
            ["Velocity 最大误差", f"{dq['velocity_reconciliation_max_abs']:.2e}", "PASS"],
            ["Reward 最大重构误差", f"{dq['base_reward_reconciliation_max_abs']:.2e} < 1e-3", "PASS"],
        ]
    )
    story.append(data_table(qa_rows, [74 * mm, 66 * mm, 34 * mm], styles))
    story.append(Spacer(1, 3 * mm))
    story.append(
        box(
            "工程验证不等于科学有效性",
            "这些 PASS 证明日志、公式、schema 与聚合在工程上自洽。它们不能证明横向漂移代表人的真实意图，也不能把两个 training seeds 变成充分的统计复制。",
            styles,
            background=LIGHT_BLUE,
            accent=NAVY,
        )
    )
    story.append(PageBreak())

    # Results summary.
    story.append(para("4. 开发结果", styles["h1"]))
    persistence = {
        float(item["candidate_weight"]): int(item["late_checkpoints_strong_count"])
        for item in result["late_checkpoint_persistence"]
    }
    screen_rows = [["w", "严格增益 seeds", "一致受损领域", "300k", "晚期/3"]]
    for item in result["endpoint_screens"]:
        weight = float(item["candidate_weight"])
        if weight == 0.5:
            continue
        domains = item["consistently_harmed_domains"]
        domain_text = "、".join(
            {
                "locomotion_effectiveness": "前进有效性",
                "environment_health": "环境健康",
                "lateral_control": "横向控制",
                "posture_stability": "姿态稳定",
                "command_quality": "命令质量",
            }.get(domain, domain)
            for domain in domains
        ) or "无"
        status = "通过" if item["strong_development_candidate"] else "不通过"
        if weight == 0.21875:
            status += "；主候选"
        elif weight == 0.125:
            status += "；次级候选"
        screen_rows.append(
            [
                format_weight(weight),
                f"{item['positive_proxy_seed_count']}/2",
                domain_text,
                status,
                f"{persistence.get(weight, 0)}/3",
            ]
        )
    story.append(data_table(screen_rows, [19 * mm, 31 * mm, 52 * mm, 49 * mm, 23 * mm], styles))
    story.append(Spacer(1, 4 * mm))
    story.extend(
        figure(
            analysis / "domain_replication_matrix.png",
            "图 1. 每个权重在 300k 时跨两个 development training seeds 越过实用阈值的行为领域数。绿色 2 表示两个 seeds 都越过；橙色 1 表示仅一个；灰色 0 表示没有。该图只显示 guardrail 复制，不显示代理门。",
            styles,
            max_height=91 * mm,
        )
    )
    story.append(PageBreak())

    # Candidate evidence and checkpoints.
    story.append(para("5. 主候选、次级候选与时间轨迹", styles["h1"]))
    selected = next(
        item for item in result["endpoint_screens"] if float(item["candidate_weight"]) == 0.21875
    )
    selected_rows = [["Training seed", "Δ matched R", "Δ net progress", "Δ path efficiency", "Δ mean |y|"]]
    for contrast in selected["contrasts"]:
        deltas = contrast["raw_metric_deltas_candidate_minus_reference"]
        selected_rows.append(
            [
                contrast["training_seed"],
                f"{contrast['candidate_proxy_advantage_under_R_w']:+.2f}",
                f"{deltas['net_forward_progress']:+.3f}",
                f"{deltas['forward_path_efficiency']:+.3f}",
                f"{deltas['lateral_drift_mean_abs']:+.3f}",
            ]
        )
    story.append(data_table(selected_rows, [30 * mm, 34 * mm, 36 * mm, 39 * mm, 35 * mm], styles))
    story.append(
        para(
            "两个 seeds 的 matched proxy 都提高；横向漂移都超过 +0.5。seed 41101 的净前进和路径效率明显改善，而 seed 41102 的净前进与路径效率下降。"
            "因此主候选不是“所有行为都更差”，而是一个多目标权衡中的局部 proxy-guardrail 分离。",
            styles["body"],
        )
    )
    story.extend(
        figure(
            analysis / "checkpoint_replication_matrix.png",
            "图 2. 各权重在每个 checkpoint 满足完整开发筛选的 training seed 数。深蓝 2 表示两个 seeds 同时满足。w=0.21875 与 w=0.125 在 200k-300k 三个晚期点均为 2；checkpoints 是同一策略训练轨迹中的重复测量，不是独立复制。",
            styles,
            max_height=92 * mm,
        )
    )
    story.append(
        box(
            "次级候选 w=0.125 的价值",
            "它同样在两个 seeds 下产生严格 matched proxy gain，但一致恶化集中在路径效率、动作边界占用和动作粗糙度。该机制不依赖绝对横向漂移，因此可作为构念检查；它比 0.21875 离默认值更远，不应替代最近边界候选。",
            styles,
            background=LIGHT_TEAL,
            accent=TEAL,
        )
    )
    story.append(PageBreak())

    # Rescore and sensitivity.
    story.append(para("6. 代理量尺与敏感性", styles["h1"]))
    story.extend(
        figure(
            analysis / "cross_rescore_matrix.png",
            "图 3. 300k 时各训练权重策略在九个评分权重 Rw 下的平均重评分 return。横轴改变评价公式中的控制代价，纵轴是策略训练时的权重。排序随评分量尺改变，说明不存在由当前数据自动给出的唯一 true weight。",
            styles,
            max_height=112 * mm,
        )
    )
    story.append(
        para(
            "w=0.21875 在代理相对非劣界 0%、2.5%、5% 下均通过；诊断阈值为原值的 0.5 倍与 1 倍时通过，2 倍时失败。"
            "因此代理结论不是仅靠宽松非劣界成立，但“横向恶化是否足够大”仍取决于实用阈值。正式报告必须同时给出原始 seed-level effect，而不是只给通过/失败标签。",
            styles["body"],
        )
    )
    story.append(
        box(
            "上侧结果",
            "w=0.625 只有 1/2 seeds 的代理不低于参考；w=0.75 为 0/2。该结果只排除当前预算与两个离散上侧点上的同类信号，不排除更大权重导致“不愿移动”等另一种失败模式。",
            styles,
            background=LIGHT_BLUE,
            accent=NAVY,
        )
    )
    story.append(PageBreak())

    # Competence and confounding.
    story.append(para("7. 必须关闭的科学阻碍", styles["h1"]))
    story.append(para("7.1 参考能力不足", styles["h2"]))
    competence_rows = [["Seed", "Matched return", "Net progress", "v_x", "Unhealthy", "Steps"]]
    for row in audit["reference_competence_audit"]["policy_rows"]:
        competence_rows.append(
            [
                row["training_seed"],
                f"{row['matched_proxy_return']:.2f}",
                f"{row['net_forward_progress']:.3f}",
                f"{row['mean_forward_velocity']:.3f}",
                f"{row['unhealthy_termination']:.2f}",
                f"{row['episode_length']:.1f}",
            ]
        )
    story.append(data_table(competence_rows, [25 * mm, 34 * mm, 32 * mm, 24 * mm, 30 * mm, 29 * mm], styles))
    story.append(
        para(
            "两个参考策略在多数 evaluation episodes 中提前不健康终止。Gymnasium 注册 threshold=6000 只作上下文，不直接充当项目能力门；"
            "但高终止率本身已经说明 300k 参考未被证明为合格比较器。",
            styles["body"],
        )
    )
    story.append(para("7.2 暴露时间与任务意图", styles["h2"]))
    story.append(
        para(
            "候选两个 seeds 均运行满 1,000 steps；参考平均只运行 331.4 与 485.9 steps。绝对横向漂移通过，但 final |y| / |forward|、lateral path fraction 和 path efficiency 没有同时在两个 seeds 证明同一方向的实用恶化。",
            styles["body"],
        )
    )
    alt_rows = [
        ["可能解释", "何时成立", "对结论的影响"],
        ["走廊违规", "预先定义机器人应沿初始 x 轴附近前进", "绝对 y 偏移可作为独立 guardrail"],
        ["暴露时间", "任务只要求前进，不限制横向位置", "更长运行自然累积偏移；当前错位解释不充分"],
        ["参考欠训练", "1M 后 w=0.5 能显著改善稳定性", "300k 候选优势可能被夸大"],
        ["候选机制真实", "1M 与新 formal seeds 仍出现 matched proxy gain + guardrail harm", "才可升级为 held-out confirmation"],
    ]
    story.append(data_table(alt_rows, [32 * mm, 78 * mm, 64 * mm], styles))
    story.append(Spacer(1, 3 * mm))
    story.append(
        box(
            "当前 formal launch 状态",
            "BLOCKED。阻碍包括：参考能力规则未冻结、参考在两个 development seeds 均多数不健康终止、绝对漂移构念取决于走廊意图、300k 低于 1M benchmark scale，以及 formal 条件与 seed gate 未冻结。",
            styles,
            background=LIGHT_RED,
            accent=RED,
        )
    )
    story.append(PageBreak())

    # PPO configuration.
    story.append(para("8. PPO 网络与超参数裁决", styles["h1"]))
    ppo_rows = [
        ["元素", "实际值", "裁决"],
        ["Observation / action", "105 / 8", "由 Ant-v5 接口决定"],
        ["Actor", "105 → 64 Tanh → 64 Tanh → 8 means；另有 8 log_std", "标准 PPO 连续动作策略"],
        ["Critic", "105 → 64 Tanh → 64 Tanh → 1 value", "与 Actor 共享输入但隐藏层独立"],
        ["参数量", "22,481", "已从保存模型实测"],
        ["Optimizer", "Adam；lr=3e-4；betas=(0.9,0.999)；eps=1e-5", "标准起点，不是已证明最优"],
        ["Rollout / batch", "n_steps=2048；batch=64；epochs=10", "固定以隔离奖励权重"],
        ["PPO", "gamma=.99；GAE=.95；clip=.2；entropy=0", "与常见基线相容"],
        ["Normalization", "未使用", "与当前 RL Zoo Ant 基线不同；暂不与预算同时改"],
        ["Budget", "约 301k actual timesteps", "适合 development；基线能力尚不足"],
    ]
    story.append(data_table(ppo_rows, [32 * mm, 76 * mm, 66 * mm], styles))
    story.append(Spacer(1, 4 * mm))
    story.append(
        para(
            "Actor 不把动作参数“交给”Critic。两者接收同一 105 维状态；Actor 产生动作分布，Critic 估计该状态的长期价值。"
            "训练时，环境奖励和下一状态形成 advantage/return 目标，分别更新 Actor 与 Critic。8 个 log_std 控制八个关节动作分布的探索宽度。",
            styles["body"],
        )
    )
    story.append(
        box(
            "为什么现在不全面调参",
            "同时改变网络层数、激活函数、learning rate、batch、regularization、normalization 和总步数，会让奖励权重效应与训练设置效应不可区分。下一道门只改变训练预算；若仍失败，再独立进行 baseline-configuration pilot。",
            styles,
            background=LIGHT_BLUE,
            accent=NAVY,
        )
    )
    story.append(PageBreak())

    # Video evidence.
    story.append(para("9. 完整视频与视觉审计", styles["h1"]))
    videos = analysis / "videos"
    frame_table = Table(
        [
            [
                scaled_image(videos / "qa_reference_mid.png", 83 * mm, 58 * mm),
                scaled_image(videos / "qa_candidate_mid.png", 83 * mm, 58 * mm),
            ],
            [
                para("参考：seed 41101 / eval 51103，中段", styles["caption"]),
                para("w=0.21875：相同 seed / eval，中段", styles["caption"]),
            ],
        ],
        colWidths=[87 * mm, 87 * mm],
        style=TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1 * mm),
            ]
        ),
    )
    story.append(frame_table)
    story.append(scaled_image(videos / "qa_candidate_end.png", 116 * mm, 76 * mm))
    story.append(
        para(
            "图 4. 完整 episode 视频的 QA 帧。参考视频 328 frames / 16.4 s 后 high-z unhealthy termination；候选视频 1,000 frames / 50.0 s 到 time limit。候选末段可见显著横向位置和躯干倾斜，但视频不能替代数值门。",
            styles["caption"],
        )
    )
    video_rows = [
        ["视频", "帧 / 时长", "SHA-256"],
        [
            "reference_seed41101_eval51103_300k.mp4",
            "328 / 16.4 s",
            "0cd33640de7740f04b0b27d52418b3355fc919974a518967eebd5a0f8d423120",
        ],
        [
            "ctrl_0p21875_seed41101_eval51103_300k.mp4",
            "1,000 / 50.0 s",
            "30bc9ca7b0a4f9eb38f246b370fd6599856bb6ed851bd6f9dabd2ff7f5dc0e19",
        ],
    ]
    story.append(data_table(video_rows, [61 * mm, 30 * mm, 83 * mm], styles, font_size="tiny"))
    story.append(PageBreak())

    # Freeze list and next gate.
    story.append(para("10. 正式实验前必须冻结的内容", styles["h1"]))
    freeze_rows = [
        ["冻结项", "最低要求", "当前状态"],
        ["任务意图", "是否 corridor-constrained；绝对 y 偏移为何有害", "未冻结"],
        ["权重域", "[0.125,0.75] 与九点离散网格；停止事后加点", "开发域已记录"],
        ["参考能力", "指标、阈值、按 seed 判定及失败处理", "阈值待批准"],
        ["PPO 配置", "网络、optimizer、normalization、预算", "300k 配置已记录；1M 待批"],
        ["数据分区", "development/formal training 与 evaluation seeds 完全隔离", "原则已定；formal 列表待定"],
        ["主比较", "matched Rw；严格 gain 与非劣性分报", "已实现并测试"],
        ["诊断指标", "定义、方向、聚合、实用阈值、领域规则", "开发版已实现；走廊构念待定"],
        ["Formal matrix", "0.5 + 候选；主 endpoint；checkpoints 角色", "待定"],
        ["复制数", "5 descriptive 或 8 stronger directional training seeds", "待定"],
        ["停止/排除", "NaN、crash、缺 checkpoint、能力门失败、重跑规则", "待冻结"],
        ["视频", "参考选 case；跨条件匹配；20 fps；完整 episode", "已实现并测试"],
        ["结论语言", "development、held-out、negative、unresolved 分级", "已写入 revision gate"],
    ]
    story.append(data_table(freeze_rows, [33 * mm, 96 * mm, 45 * mm], styles))
    story.append(Spacer(1, 4 * mm))
    story.append(
        box(
            "推荐的一步修改",
            "只把现有 development 策略 w=0.5、0.21875、0.125 从约 300k 延长到 1M，并在 500k、750k、1M 评价。网络、optimizer、reward、seeds、无 normalization 均保持不变。",
            styles,
            background=LIGHT_TEAL,
            accent=TEAL,
        )
    )
    story.append(Spacer(1, 3 * mm))
    next_rows = [
        ["项目", "估计"],
        ["新增训练", "3 weights × 2 development seeds × 约 700k steps"],
        ["墙钟时间", "约 65-75 min，基于本机四并发实测；受休眠和评价开销影响"],
        ["新增 checkpoints", "500k、750k、1M；共 18 个模型文件"],
        ["存储", "约 6-10 MB"],
        ["建议能力门", "每个参考 training seed：unhealthy termination ≤ 0.20 且 mean forward velocity ≥ 0.10"],
        ["失败处理", "formal 保持阻止；另开 normalization / baseline-configuration pilot"],
    ]
    story.append(data_table(next_rows, [43 * mm, 131 * mm], styles))
    story.append(PageBreak())

    # Final adjudication and references.
    story.append(para("11. 最终裁决与下一步选择", styles["h1"]))
    story.append(
        box(
            "可以接受",
            "双向权重设计、matched rescoring、training/evaluation seed 分层、九点开发网格、奖励分解、完整 CSV schema、数据 QA、checkpoint 轨迹、敏感性分析和完整视频，已经构成可复现的开发证据链。",
            styles,
            background=LIGHT_TEAL,
            accent=TEAL,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        box(
            "必须修改或冻结",
            "参考最低能力、走廊构念、formal 条件与新 seeds、停止和排除规则，以及是否批准 1M budget-only extension。未关闭前，不进入 protocol-freeze-ready，不启动 shaping。",
            styles,
            background=LIGHT_RED,
            accent=RED,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(para("需要用户冻结的五项选择", styles["h2"]))
    for text in [
        "是否把隐藏任务定义为 corridor-constrained forward locomotion？",
        "是否接受参考能力门：每个 seed unhealthy ≤ 0.20 且 mean forward velocity ≥ 0.10？",
        "是否批准三组现有 development 策略进行 1M budget-only extension？",
        "formal matrix 是否同时保留 0.21875 与 0.125？",
        "正式训练采用 5 个还是 8 个全新 training seeds？",
    ]:
        story.append(bullet(text, styles, colour=ORANGE))
    story.append(Spacer(1, 4 * mm))
    story.append(
        para(
            "<b>本轮最严格结论：</b> 在当前 Ant-v5、PPO、约 300k 预算、两个 development seeds 与离散权重网格下，"
            "w=0.21875 和 w=0.125 产生开发级 proxy-diagnostic divergence 信号；w=0.625 和 0.75 未产生同类信号。"
            "参考能力不足与横向漂移替代解释尚未关闭，因此奖励错位未获正式确认。",
            styles["quote"],
        )
    )
    story.append(PageBreak())

    story.append(para("参考文献", styles["h1"]))
    references = [
        "Agarwal, R. et al. (2021) 'Deep reinforcement learning at the edge of the statistical precipice', Advances in Neural Information Processing Systems, 34.",
        "Farama Foundation (2026) Ant - Gymnasium documentation. Available at: https://gymnasium.farama.org/environments/mujoco/ant/ (Accessed: 14 August 2026).",
        "Henderson, P. et al. (2018) 'Deep reinforcement learning that matters', Proceedings of the AAAI Conference on Artificial Intelligence, 32(1).",
        "Pan, A., Bhatia, K. and Steinhardt, J. (2022) 'The effects of reward misspecification: mapping and mitigating misaligned models', International Conference on Learning Representations.",
        "Raffin, A. (2026) RL Baselines3 Zoo PPO hyperparameters. Available at: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml (Accessed: 14 August 2026).",
        "Schulman, J. et al. (2017) 'Proximal policy optimization algorithms', arXiv:1707.06347.",
        "Skalse, J. et al. (2022) 'Defining and characterizing reward hacking', Advances in Neural Information Processing Systems, 35.",
    ]
    for reference in references:
        story.append(para(reference, styles["small"]))
        story.append(Spacer(1, 1.5 * mm))

    story.append(para("证据与可追溯性", styles["h2"]))
    provenance_rows = [["证据", "SHA-256"]]
    provenance_paths = [
        result_path,
        audit_path,
        root / "configs" / "stage1_bidirectional_development_v2_20260814.json",
        root / "protocols" / "STAGE1_PREFORMAL_REVISION_GATE_V3_20260814.md",
        root / "src" / "proxygap" / "stage1.py",
        root / "scripts" / "analyse_stage1_dense_development.py",
    ]
    for path in provenance_paths:
        provenance_rows.append([str(path.relative_to(root)), sha256(path)])
    story.append(data_table(provenance_rows, [84 * mm, 90 * mm], styles, font_size="tiny"))
    story.append(Spacer(1, 3 * mm))
    story.append(
        para(
            "完整逐文件 manifest 与 PDF 自身哈希在构建后另行生成。报告正文来源为版本化 JSON/CSV；视频与图表只作辅助呈现。",
            styles["caption"],
        )
    )
    return story


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--equations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    register_fonts()
    styles = make_styles()

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=17 * mm,
        title="ProxyGap 阶段一双向权重开发审查报告",
        author="ProxyGap research workflow",
        subject="Ant-v5 PPO reward misspecification development adjudication",
    )
    story = build_story(root, args.equations.resolve(), styles)
    document.build(story, onFirstPage=first_page, onLaterPages=page_header_footer)
    print(output)
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()
