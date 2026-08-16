"""Build the Chinese PDF addendum for the stage-one 1M adjudication."""

from __future__ import annotations

import argparse
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
ORANGE = colors.HexColor("#D97706")
RED = colors.HexColor("#B42318")
INK = colors.HexColor("#20282E")
MID = colors.HexColor("#65717A")
GRID = colors.HexColor("#C8D1D7")
PALE_BLUE = colors.HexColor("#EDF4F8")
PALE_TEAL = colors.HexColor("#E8F5F1")
PALE_ORANGE = colors.HexColor("#FFF3E5")
PALE_RED = colors.HexColor("#FCECEB")
WHITE = colors.white


def register_fonts() -> None:
    for regular, bold in iter_font_pairs(DENG_FIRST_CJK_FONT_PAIRS):
        try:
            pdfmetrics.registerFont(TTFont("Deng", str(regular), subfontIndex=0))
            pdfmetrics.registerFont(TTFont("DengBold", str(bold), subfontIndex=0))
            pdfmetrics.registerFontFamily(
                "Deng", normal="Deng", bold="DengBold", italic="Deng", boldItalic="DengBold"
            )
            return
        except Exception:
            continue
    raise RuntimeError(
        f"No suitable Chinese font pair was found; set {FONT_DIR_ENV} to a font directory."
    )


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="DengBold", fontSize=24,
            leading=33, textColor=NAVY, alignment=TA_LEFT, wordWrap="CJK"
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName="Deng", fontSize=11,
            leading=17, textColor=MID, wordWrap="CJK"
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="DengBold", fontSize=17,
            leading=23, textColor=NAVY, spaceAfter=3 * mm, keepWithNext=True,
            wordWrap="CJK"
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="DengBold", fontSize=12,
            leading=17, textColor=TEAL, spaceBefore=2 * mm, spaceAfter=1.5 * mm,
            keepWithNext=True, wordWrap="CJK"
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontName="Deng", fontSize=9.4,
            leading=15, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=2.1 * mm,
            wordWrap="CJK"
        ),
        "small": ParagraphStyle(
            "small", parent=base["BodyText"], fontName="Deng", fontSize=7.7,
            leading=11.5, textColor=INK, alignment=TA_LEFT, wordWrap="CJK"
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["BodyText"], fontName="Deng", fontSize=7.4,
            leading=10.5, textColor=MID, alignment=TA_LEFT, spaceAfter=2 * mm,
            wordWrap="CJK"
        ),
        "verdict": ParagraphStyle(
            "verdict", parent=base["BodyText"], fontName="DengBold", fontSize=12,
            leading=18, textColor=NAVY, alignment=TA_LEFT, wordWrap="CJK"
        ),
        "footer": ParagraphStyle(
            "footer", parent=base["BodyText"], fontName="Deng", fontSize=7,
            leading=9, textColor=MID, alignment=TA_CENTER, wordWrap="CJK"
        ),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def info_box(title: str, body: str, s: dict[str, ParagraphStyle], colour, fill) -> Table:
    table = Table(
        [[p(title, s["h2"])], [p(body, s["body"])]],
        colWidths=[174 * mm],
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("BOX", (0, 0), (-1, -1), 0.7, colour),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colour),
        ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2 * mm),
    ]))
    return table


def table(rows: list[list[str]], widths: list[float], s: dict[str, ParagraphStyle]) -> Table:
    formatted = []
    for row_index, row in enumerate(rows):
        formatted.append([
            p(
                (f'<font color="#FFFFFF"><b>{escape(str(value))}</b></font>'
                 if row_index == 0 else escape(str(value))),
                s["small"],
            )
            for value in row
        ])
    result = Table(formatted, colWidths=widths, repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#F5F7F8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8 * mm),
    ]))
    return result


def bullet(text: str, s: dict[str, ParagraphStyle]) -> Table:
    result = Table(
        [[p('<font color="#008C7A"><b>•</b></font>', s["body"]), p(text, s["body"])]],
        colWidths=[5 * mm, 169 * mm],
    )
    result.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return result


def on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5DCE0"))
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("Deng", 7)
    canvas.setFillColor(MID)
    canvas.drawString(18 * mm, 9 * mm, "ProxyGap | Stage-one development evidence | 14 August 2026")
    canvas.drawRightString(192 * mm, 9 * mm, str(doc.page))
    canvas.restoreState()


def build(root: Path, output: Path) -> None:
    register_fonts()
    s = styles()
    analysis = root / "artifacts" / "analysis" / "stage1_budget_extension_1m_v4_20260814_attempt2"
    adjudication = json.loads((analysis / "stage1_budget_extension_adjudication.json").read_text(encoding="utf-8"))
    verification = json.loads((analysis / "independent_verification.json").read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title="ProxyGap 阶段一 1M 预算延长：证据裁决",
        author="ProxyGap research audit",
    )
    story = []

    story += [Spacer(1, 17 * mm), p("ProxyGap 阶段一 1M 预算延长", s["title"]),
              p("证据裁决与下一道研究门槛", s["title"]), Spacer(1, 6 * mm),
              p("Ant-v5 + PPO | ctrl_cost_weight | development evidence only", s["subtitle"]),
              Spacer(1, 18 * mm),
              info_box("一句话结论", "<b>w=0.21875</b> 在两个 development training seeds 上均取得更高的同尺度 proxy score，同时前进有效性和姿态稳定性变差；但 <b>w=0.5</b> 参考组自身未通过健康能力门槛，所以当前是<strong>强候选</strong>，还不是正式确认的 reward misspecification。", s, TEAL, PALE_TEAL),
              Spacer(1, 6 * mm),
              info_box("当前状态", "Engineering-validated → Scientifically unresolved。正式 held-out training 与 stage-two shaping 均保持禁止。", s, ORANGE, PALE_ORANGE),
              Spacer(1, 18 * mm),
              p("报告日期：14 August 2026<br/>证据范围：V4 预算延长与 V5 裁决<br/>语言：中文；术语与公式保留英文/数学记号", s["subtitle"]),
              PageBreak()]

    story += [p("1. 这次到底检验了什么", s["h1"]),
              p("本轮只检验阶段一：在预先限定的合理 control-cost 范围内，是否能够找到“同尺度代理分数更高，但预声明行为指标更差”的可重复候选。它没有修改神经网络、优化器、奖励结构或 normalization，也没有运行 shaping。", s["body"]),
              info_box("为什么延长到 1M", "300k 时参考组较弱，无法判断候选现象来自奖励设置还是训练预算不足。把已有策略继续训练到 1M，可以先隔离 budget sufficiency，而不同时改变多个因素。", s, NAVY, PALE_BLUE),
              p("冻结实验矩阵", s["h2"]),
              table([
                  ["维度", "冻结值"],
                  ["Environment / algorithm", "Gymnasium Ant-v5 / SB3 PPO / CPU"],
                  ["ctrl_cost_weight", "0.5, 0.21875, 0.125"],
                  ["Training seeds", "41101, 41102"],
                  ["Evaluation seeds", "51101-51110，跨策略配对"],
                  ["Checkpoint targets", "500k, 750k, 1M"],
                  ["Shaping / normalization", "全部为 0 / 保持关闭"],
              ], [49 * mm, 125 * mm], s),
              Spacer(1, 3 * mm),
              p("实际完成 6 个策略、18 个模型和 180 个 evaluation episodes；墙钟时间为 4,469.7 s（74.50 min）。由于 PPO 按 2,048-step rollout 更新，实际 checkpoint 为 501,760、751,616 和 1,001,472。", s["body"]),
              info_box("独立单位", "Training seed 产生一条独立训练策略，是 development replication unit。10 个 evaluation episodes 只是同一策略下的嵌套测量，checkpoint 也是同一训练过程的重复观察。", s, TEAL, PALE_TEAL),
              PageBreak()]

    eq = root / "artifacts" / "reports" / "stage1_exploration_20260814" / "equations" / "matched_rescore_equation.png"
    story += [p("2. 为什么要使用同尺度 proxy", s["h1"]),
              p("不同 ctrl_cost_weight 产生不同奖励公式，因此不能把候选组自己的 raw return 与 0.5 组自己的 raw return 直接比较。对候选权重 w，必须把候选与参考轨迹都放进同一个 R_w 重新评分。", s["body"])]
    if eq.is_file():
        story += [Image(str(eq), width=163 * mm, height=17 * mm),
                  p("图 1. 同一候选权重下的 matched rescore。公式由 LaTeX/Matplotlib mathtext 渲染。", s["caption"])]
    story += [p("使用的代理公式为 R_w = Σ_t(r_forward + r_survive + r_contact - w||a_t||²)。这一处理解决了数值量尺不一致，却没有创造 true reward；它仍只是公开、可复算的 proxy comparator。", s["body"]),
              p("3. 参考组能力门槛", s["h1"]),
              table([
                  ["Training seed", "Unhealthy rate", "Mean v_x", "Joint gate"],
                  ["41101", "0.90", "1.166", "Fail"],
                  ["41102", "0.60", "1.075", "Fail"],
              ], [35 * mm, 45 * mm, 45 * mm, 49 * mm], s),
              Spacer(1, 3 * mm),
              p("冻结门槛要求每个参考策略同时满足 unhealthy termination rate ≤ 0.20 与 mean forward velocity ≥ 0.10 position units s<super>-1</super>。两条 seed 都能向前移动，但都在健康门槛失败。不能因为结果不方便而事后降低门槛。", s["body"]),
              Image(str(analysis / "reference_competence_trajectory.png"), width=174 * mm, height=91 * mm),
              p("图 2. 参考策略随训练预算的能力轨迹。虚线为预先冻结的门槛；蓝/橙分别代表两条 training seed。", s["caption"]),
              PageBreak()]

    candidate = adjudication["candidate_audits"]["0.21875"]["seed_level_rows"]
    story += [p("4. 主要候选：w = 0.21875", s["h1"]),
              p("在 1M endpoint，两条 development seed 均取得正的 matched proxy advantage，同时在 locomotion effectiveness 与 posture stability 两个预声明领域达到受损判据。", s["body"]),
              table([
                  ["Seed", "Δ matched proxy", "Δ net progress", "Δ path efficiency", "Δ tilt RMS", "Δ unhealthy"],
                  ["41101", f"{candidate[0]['delta_matched_proxy_return']:+.2f}", f"{candidate[0]['delta_net_forward_progress']:+.2f}", f"{candidate[0]['delta_forward_path_efficiency']:+.3f}", f"{candidate[0]['delta_torso_tilt_rms']:+.3f} rad", f"{candidate[0]['delta_unhealthy_termination']:+.2f}"],
                  ["41102", f"{candidate[1]['delta_matched_proxy_return']:+.2f}", f"{candidate[1]['delta_net_forward_progress']:+.2f}", f"{candidate[1]['delta_forward_path_efficiency']:+.3f}", f"{candidate[1]['delta_torso_tilt_rms']:+.3f} rad", f"{candidate[1]['delta_unhealthy_termination']:+.2f}"],
              ], [20 * mm, 32 * mm, 29 * mm, 32 * mm, 31 * mm, 30 * mm], s),
              Spacer(1, 3 * mm),
              Image(str(analysis / "candidate_contrast_trajectory.png"), width=174 * mm, height=118 * mm),
              p("图 3. 候选与参考的配对差值随 checkpoint 变化。圆/方表示两个候选权重；实线/虚线表示两条 training seed；灰线为零差值。", s["caption"]),
              info_box("正确解释", "0.21875 是 strong development candidate。它不能被描述为“机器人所有方面都更差”，因为 unhealthy termination 在两条 seed 上都改善。现有证据是 proxy-diagnostic divergence 与 multi-objective trade-off。", s, ORANGE, PALE_ORANGE),
              PageBreak()]

    story += [p("5. 时间轨迹、负结果与替代解释", s["h1"]),
              p("0.21875 的 checkpoint 屏幕结果不是单调序列：300k 通过；500k 不通过；750k 有 proxy gain，但没有两条 seed 一致受损的领域；1M 再次通过。因此当前不能声称 divergence 从某一点开始持续存在，更不能声称越训练越严重。", s["body"]),
              table([
                  ["Checkpoint", "0.21875 proxy-positive seeds", "Common harmed domain", "Screen"],
                  ["300k", "2/2", "Lateral control", "Pass"],
                  ["500k", "0/2", "Locomotion + posture", "Fail"],
                  ["750k", "2/2", "None", "Fail"],
                  ["1M", "2/2", "Locomotion + posture", "Pass"],
              ], [30 * mm, 49 * mm, 55 * mm, 40 * mm], s),
              Spacer(1, 4 * mm),
              p("300k 的候选解释主要依赖 absolute lateral drift，可能受 episode 更长、移动更多影响。1M 时，两条 seed 共同受损的是 net progress/path efficiency 与 torso tilt RMS，因此主要机制已不再只依赖 absolute drift；但 lateral diagnostics 仍须完整保留。", s["body"]),
              p("w = 0.125 是重要负结果", s["h2"]),
              p("1M 时它的 matched proxy differences 为 -282.12 与 -354.65。即便多项行为指标更差，它没有满足“代理相近或更高”这一必要条件，所以不能用作阶段一 primary candidate。保留该负结果可以防止只展示最有故事的条件。", s["body"]),
              info_box("Continuation 限制", "模型、价值网络、优化器和 timestep 被恢复；300k 保存瞬间的 MuJoCo 状态与完整随机流未被保存。V4 是可复现的 policy continuation，不是 bitwise-equivalent uninterrupted training。", s, NAVY, PALE_BLUE),
              PageBreak()]

    story += [p("6. 可追溯性与失败记录", s["h1"]),
              p("主分析与独立 verifier 都通过。独立 verifier 不导入主分析函数，从 raw episode CSV 重新聚合并核对关键差值。", s["body"]),
              table([
                  ["检查", "结果"],
                  ["Extension rows / duplicate keys", "180 / 0"],
                  ["Shaping terms", "All exactly zero"],
                  ["Reward reconstruction max error", "≈ 1.44 × 10^-6; contract 1 × 10^-3"],
                  ["Source model hashes", "6/6 unchanged"],
                  ["Continued models", "18/18 hashes and timesteps reloaded"],
                  ["Independent recomputation", verification["status"].upper()],
                  ["Full automated test suite", "77/77 passed"],
              ], [72 * mm, 102 * mm], s),
              p("两类失败证据被保留而非删除：第一次 smoke 使用了缺少 MuJoCo 的错误 Python 环境；第一次分析采用了比既有 CSV contract 更严格但未预声明的 1e-6 tolerance；第一次 independent verifier 未解析 CSV boolean。修复分别是切换到既有 D 盘 ProxyGap 环境、恢复 1e-3 contract 并加回归测试、增加显式 boolean parser 并加回归测试。", s["body"]),
              p("7. 当前允许与禁止的结论", s["h1"]),
              table([
                  ["允许", "禁止"],
                  ["1M endpoint 存在 strong development candidate", "已证明 reward hacking 或 true reward 下降"],
                  ["两条 development seeds 方向一致", "把 20 个 evaluation episodes 当作 n=20"],
                  ["proxy 更高而 locomotion/posture 更差", "uniformly low overall performance"],
                  ["checkpoint pattern 非单调", "持续、稳定或单调放大"],
                  ["仅限 Ant-v5/PPO/当前实现", "推广到所有机器人、RL 系统或真实硬件"],
              ], [87 * mm, 87 * mm], s),
              PageBreak()]

    story += [p("8. 下一道门槛仍属于阶段一", s["h1"]),
              info_box("首选下一步", "冻结一个 fresh uninterrupted 1M reference-only development replication。它只检验：当前参考失败是否由 baseline configuration 本身造成，还是 continuation discontinuity 或两条 seed 的偶然性。", s, TEAL, PALE_TEAL),
              p("若 fresh reference 仍失败，再建立独立 baseline-configuration pilot，比较有实现依据的 normalization 或其他基线设置。任何 normalization/architecture 变化都不能与 V4 extension 合并分析，也不能在看到候选结果后偷偷改变正式 comparator。", s["body"]),
              p("仍然阻止 protocol freeze 的事项", s["h2"]),
              bullet("参考组未通过已冻结的 competence gate。", s),
              bullet("现有两条 training seeds 属于 development discovery，不是 held-out confirmation。", s),
              bullet("正式 condition matrix 与五或八条 held-out training seeds 尚未冻结。", s),
              bullet("助教所说 accuracy matrix 的两条轴与适用性尚未澄清。", s),
              Spacer(1, 3 * mm),
              info_box("Accuracy matrix", "标准 confusion matrix 需要 ground-truth classes 与 predicted classes。当前连续控制实验没有这些标签。请向助教确认：Do you mean a normalised confusion matrix? If yes, what are the ground-truth classes and predicted classes in this reinforcement-learning project?", s, RED, PALE_RED),
              p("最终裁决", s["h2"]),
              p("Candidate identified, formal confirmation not yet authorised. Stage two remains out of scope until stage-one reference competence and held-out confirmation logic are resolved.", s["verdict"]),
              Spacer(1, 7 * mm),
              p("核心证据：V4 frozen config；raw 180-row evaluation CSV；primary adjudication JSON；independent verification JSON；V5 machine-readable gate；SHA-256 delivery manifest。", s["caption"])]

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.root.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
