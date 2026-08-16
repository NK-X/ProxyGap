"""Build the audited Chinese PDF for the body-smoothness mechanism exploration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
OLD = ROOT / "artifacts" / "dev" / "hg_r3_obsfix_v1"
TARGET = ROOT / "artifacts" / "dev" / "smoothness_target_extension_1m_v1"
MATRIX = ROOT / "artifacts" / "dev" / "body_smoothness_gsde_matrix_v1"
ANALYSIS = MATRIX / "analysis"
ASSETS = ROOT / "output" / "smooth_locomotion_report_assets_20260816"
OUTPUT = ROOT / "output" / "pdf" / "SMOOTH_LOCOMOTION_MECHANISM_EXPLORATION_20260816_CN.pdf"
QA_OUTPUT = ROOT / "output" / "pdf" / "SMOOTH_LOCOMOTION_MECHANISM_EXPLORATION_20260816_CN_QA.json"

NAVY = colors.HexColor("#173753")
TEAL = colors.HexColor("#0F766E")
GOLD = colors.HexColor("#C99528")
RED = colors.HexColor("#A63A3A")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#566573")
GRID = colors.HexColor("#CAD5DE")
PALE_BLUE = colors.HexColor("#EAF1F6")
PALE_TEAL = colors.HexColor("#E8F5F2")
PALE_GOLD = colors.HexColor("#FFF6DF")
PALE_RED = colors.HexColor("#FBECEC")


def register_fonts() -> tuple[str, str]:
    for regular, bold in iter_font_pairs(CJK_FONT_PAIRS):
        try:
            pdfmetrics.registerFont(TTFont("CJKRegular", str(regular), subfontIndex=0))
            pdfmetrics.registerFont(TTFont("CJKBold", str(bold), subfontIndex=0))
            return "CJKRegular", "CJKBold"
        except Exception:
            continue
    raise RuntimeError(
        f"No suitable Chinese font pair was found; set {FONT_DIR_ENV} to a font directory."
    )


REGULAR, BOLD = register_fonts()


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName=BOLD, fontSize=24,
            leading=32, textColor=NAVY, alignment=TA_LEFT, spaceAfter=6 * mm,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName=REGULAR, fontSize=10.5,
            leading=17, textColor=MUTED, spaceAfter=4 * mm, wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName=BOLD, fontSize=16,
            leading=22, textColor=NAVY, spaceBefore=3 * mm, spaceAfter=3 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName=BOLD, fontSize=12,
            leading=17, textColor=TEAL, spaceBefore=2.5 * mm, spaceAfter=2 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontName=REGULAR, fontSize=9.3,
            leading=15.2, textColor=INK, alignment=TA_LEFT, spaceAfter=2.2 * mm,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "small", parent=base["BodyText"], fontName=REGULAR, fontSize=7.2,
            leading=10.5, textColor=INK, wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["BodyText"], fontName=REGULAR, fontSize=7.4,
            leading=10.8, textColor=MUTED, spaceBefore=1.2 * mm,
            spaceAfter=3 * mm, wordWrap="CJK",
        ),
        "callout": ParagraphStyle(
            "callout", parent=base["BodyText"], fontName=BOLD, fontSize=10,
            leading=15.5, textColor=NAVY, wordWrap="CJK",
        ),
        "reference": ParagraphStyle(
            "reference", parent=base["BodyText"], fontName=REGULAR, fontSize=7.1,
            leading=10.3, textColor=INK, spaceAfter=1.8 * mm, wordWrap="CJK",
        ),
    }


S = make_styles()


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def callout(text: str, background=PALE_BLUE, border=NAVY) -> Table:
    table = Table([[p(text, "callout")]], colWidths=[174 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.7, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
    ]))
    return table


def bullet(text: str) -> Paragraph:
    return Paragraph(f"&#8226;&nbsp;&nbsp;{text}", S["body"])


def styled_table(rows: list[list], widths: list[float], header_rows: int = 1) -> Table:
    wrapped = []
    for row_index, row in enumerate(rows):
        style = "small"
        wrapped.append([cell if hasattr(cell, "wrap") else p(str(cell), style) for cell in row])
    table = Table(wrapped, colWidths=[value * mm for value in widths], repeatRows=header_rows)
    commands = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
        ("FONTNAME", (0, 0), (-1, header_rows - 1), BOLD),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.6 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.6 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5 * mm),
    ]
    for row_index in range(header_rows, len(rows)):
        if (row_index - header_rows) % 2:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F5F8FA")))
    table.setStyle(TableStyle(commands))
    return table


def add_image(story: list, path: Path, caption: str, width_mm: float = 174) -> None:
    image = Image(str(path))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width_mm * mm
    image.drawHeight = width_mm * ratio * mm
    story.extend([image, p(caption, "caption")])


def equation_asset(name: str, lines: list[str]) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    output = ASSETS / name
    fig, axis = plt.subplots(figsize=(10.8, 0.75 + 0.58 * len(lines)))
    axis.axis("off")
    for index, line in enumerate(lines):
        axis.text(0.5, 1 - (index + 0.62) / len(lines), line, ha="center", va="center", fontsize=17)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont(REGULAR, 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "ProxyGap development mechanism report | 16 August 2026")
    canvas.drawRightString(192 * mm, 9 * mm, f"{doc.page}")
    canvas.restoreState()


def read_inputs() -> dict:
    required = [
        ANALYSIS / "body_smoothness_gsde_summary.json",
        ANALYSIS / "checkpoint_policy_means.csv",
        ANALYSIS / "endpoint_body_contact_policy_means.csv",
        ANALYSIS / "body_contact_paired_factorial_contrasts.csv",
        OLD / "analysis" / "jump_contact_gait" / "jump_contact_gait_summary.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing analysis inputs: {missing}")
    return {
        "matrix_summary": json.loads(required[0].read_text(encoding="utf-8")),
        "checkpoint": pd.read_csv(required[1]),
        "body": pd.read_csv(required[2]),
        "contrasts": pd.read_csv(required[3]),
        "jump": json.loads(required[4].read_text(encoding="utf-8")),
        "compliance": pd.read_csv(
            OLD / "analysis" / "intent_sensitivity" / "condition_domain_compliance.csv"
        ),
        "target": json.loads(
            (TARGET / "analysis" / "target_budget_extension_summary.json").read_text(encoding="utf-8")
        ),
    }


def compliance_table(frame: pd.DataFrame) -> Table:
    columns = [
        ("condition_id", "条件"),
        ("horizon_and_health", "健康/时长"),
        ("forward_tracking", "前进"),
        ("torso_stability", "躯干"),
        ("directional_control", "方向"),
        ("path_directness", "路径"),
        ("action_smoothness", "动作"),
    ]
    rows = [[label for _, label in columns]]
    for _, record in frame.iterrows():
        rows.append([
            record[column] if column == "condition_id" else f"{100 * float(record[column]):.0f}%"
            for column, _ in columns
        ])
    return styled_table(rows, [47, 22, 18, 18, 18, 18, 18])


def condition_results(checkpoint: pd.DataFrame, body: pd.DataFrame) -> tuple[Table, pd.DataFrame]:
    endpoint = checkpoint.loc[checkpoint["target_timesteps"] == 1_000_000]
    endpoint = endpoint.groupby("condition_id", as_index=False).mean(numeric_only=True)
    merged = endpoint.merge(body.groupby("condition_id", as_index=False).mean(numeric_only=True), on="condition_id", suffixes=("", "_body"))
    rows = [["条件", "vx m/s", "路径效率", "方向误差", "倾斜 RMS", "动作粗糙度", "意图合规"]]
    for _, record in merged.iterrows():
        rows.append([
            record["condition_id"],
            f"{record['fixed_horizon_forward_velocity_m_per_s']:.3f}",
            f"{record['forward_path_efficiency_body']:.3f}",
            f"{record['direction_error_degrees']:.1f}°",
            f"{np.degrees(record['torso_tilt_rms_rad']):.1f}°",
            f"{record['normalised_action_roughness_body']:.4f}",
            f"{100 * record['intent_compliant_body']:.1f}%",
        ])
    return styled_table(rows, [30, 21, 24, 24, 24, 27, 24]), merged


def body_results(body: pd.DataFrame) -> Table:
    grouped = body.groupby("condition_id", as_index=False).mean(numeric_only=True)
    rows = [["条件", "RMS vz", "RMS roll/pitch", "无地面接触", "起跳次数", "P95 接触诊断"]]
    for _, record in grouped.iterrows():
        rows.append([
            record["condition_id"],
            f"{record['rms_root_vertical_velocity_m_per_s']:.3f}",
            f"{record['rms_root_roll_pitch_angular_speed_rad_per_s']:.3f}",
            f"{100 * record['no_floor_contact_step_fraction']:.1f}%",
            f"{record['prominent_takeoff_count_vz_ge_1p25']:.1f}",
            f"{record['p95_raw_floor_force_norm']:.1f}",
        ])
    return styled_table(rows, [32, 26, 34, 32, 25, 32])


def contrast_table(frame: pd.DataFrame) -> Table:
    labels = {
        "rms_root_vertical_velocity_m_per_s": "RMS vz",
        "rms_root_roll_pitch_angular_speed_rad_per_s": "RMS roll/pitch",
        "no_floor_contact_step_fraction": "无地面接触比例",
        "prominent_takeoff_count_vz_ge_1p25": "突出起跳次数",
        "fixed_horizon_forward_velocity_m_per_s": "前进速度",
        "forward_path_efficiency": "路径效率",
    }
    rows = [["指标", "身体惩罚主效应", "gSDE 主效应", "交互效应"]]
    for metric, label in labels.items():
        cells = [label]
        for effect in ("body_main", "gsde_main", "interaction"):
            values = frame.loc[(frame["metric"] == metric) & (frame["effect"] == effect), "contrast"]
            cells.append(f"{values.mean():+.3f} [{values.min():+.3f}, {values.max():+.3f}]")
        rows.append(cells)
    return styled_table(rows, [42, 44, 44, 44])


def video_summary() -> tuple[int, int, int, str]:
    index = MATRIX / "8_16_trials_4" / "VIDEO_INDEX.csv"
    frame = pd.read_csv(index)
    videos = list((MATRIX / "8_16_trials_4").glob("*.mp4"))
    hashes_ok = len(videos) == len(frame)
    for _, record in frame.iterrows():
        path = Path(record["video_path"])
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != record["video_sha256"]:
            hashes_ok = False
            break
    return (
        len(frame),
        int(frame["trajectory_frames"].min()),
        int(frame["padded_frames"].sum()),
        "PASS" if hashes_ok else "FAIL",
    )


def report_narrative(merged: pd.DataFrame, contrasts: pd.DataFrame) -> str:
    def effect(metric: str, name: str) -> tuple[float, int]:
        values = contrasts.loc[(contrasts["metric"] == metric) & (contrasts["effect"] == name), "contrast"]
        return float(values.mean()), int((values < 0).sum())

    b_vz, b_vz_n = effect("rms_root_vertical_velocity_m_per_s", "body_main")
    b_angular, b_angular_n = effect("rms_root_roll_pitch_angular_speed_rad_per_s", "body_main")
    b_air, b_air_n = effect("no_floor_contact_step_fraction", "body_main")
    b_takeoff, b_takeoff_n = effect("prominent_takeoff_count_vz_ge_1p25", "body_main")
    g_vz, g_vz_n = effect("rms_root_vertical_velocity_m_per_s", "gsde_main")
    g_angular, g_angular_n = effect("rms_root_roll_pitch_angular_speed_rad_per_s", "gsde_main")
    return (
        f"在三个配对 training seeds 中，身体动态惩罚对 RMS 垂直速度的平均对比为 {b_vz:+.3f} "
        f"m/s（{b_vz_n}/3 seeds 向较低方向），对 roll/pitch 角速度为 {b_angular:+.3f} rad/s "
        f"（{b_angular_n}/3），对无地面接触比例为 {100*b_air:+.1f} 个百分点（{b_air_n}/3），"
        f"对突出起跳次数为 {b_takeoff:+.2f}（{b_takeoff_n}/3）。gSDE 对前两项的平均对比分别为 "
        f"{g_vz:+.3f} m/s（{g_vz_n}/3）和 {g_angular:+.3f} rad/s（{g_angular_n}/3）。"
        "这些是 development 中的配对描述性效应；只有方向一致且没有明显损害任务跟踪时，才构成机制支持。"
    )


def build() -> None:
    data = read_inputs()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    equation_action = equation_asset(
        "equation_action_roughness.png",
        [r"$Q_a=\frac{1}{32(T-1)}\sum_{t=2}^{T}\lVert a_t-a_{t-1}\rVert_2^2$"],
    )
    equation_body = equation_asset(
        "equation_body_penalty.png",
        [
            r"$p_z(t)=\tanh[(v_z/1.0141)^2],\quad p_\omega(t)=\tanh[(\sqrt{\omega_x^2+\omega_y^2}/1.9893)^2]$",
            r"$r_{body}(t)=-0.05p_z(t)-0.05p_\omega(t),\qquad -0.1\leq r_{body}\leq0$",
        ],
    )
    condition_table, merged = condition_results(data["checkpoint"], data["body"])
    videos, min_frames, padded, video_qa = video_summary()
    jump = data["jump"]
    contact = jump["contact_and_gait"]
    events = jump["prominent_takeoff_events_vz_at_least_1p25_m_per_s"]
    max_force = max(item["preceding_four_step_max_floor_force_norm"] for item in events)

    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=18 * mm, title="ProxyGap Ant-v5 平滑运动机制探索报告",
        author="ProxyGap project",
    )
    story: list = []
    story.extend([
        Spacer(1, 8 * mm),
        p("ProxyGap Ant-v5 平滑运动机制探索报告", "title"),
        p("从“动作变化小”到“身体运动平滑”：跳跃、接触、gSDE 与身体动态 shaping 的受控 development 研究", "subtitle"),
        callout(
            "结论边界：本报告记录 development mechanism evidence。它不等于 held-out 正式实验，"
            "不定义生物学自然步态，也不把 MuJoCo 接触诊断量解释为真实机器人安全载荷。",
            PALE_GOLD, GOLD,
        ),
        Spacer(1, 5 * mm),
        p("核心回答", "h1"),
        p(
            "视频中的不平滑不是一个单独的“动作抖动”问题。直接力矩控制、非线性接触、奖励未直接评价身体上下/角运动，"
            "以及训练探索噪声共同形成了跳跃与侧向偏移的可行策略。旧 action-rate 指标只测策略输出，不足以保证躯干轨迹平滑。"
        ),
        p(
            "六个候选合规率为 0 的直接原因是严格 AND 合规规则下，六组均未通过路径直接性；它不表示所有指标都是 0。"
            "现有完整视频也不是只选择异常样本，而是按全部训练端点、固定评价起点保存。"
        ),
        p("证据状态", "h2"),
        styled_table([
            ["层级", "当前状态", "允许的说法"],
            ["工程", "145 项自动测试通过；日志、模型、重放和视频均有版本记录", "实现与数据链可复核"],
            ["科学", "3 个 development training seeds；未使用 held-out formal seeds", "可报告机制线索和配对描述性结果"],
            ["外推", "默认平地 Ant-v5、PPO、CPU 仿真", "不得外推至真实四足机器人或所有 DRL 系统"],
        ], [27, 69, 78]),
        PageBreak(),
        p("1. 三个容易混淆的“平滑”概念", "h1"),
        styled_table([
            ["构念", "问题", "当前指标/状态"],
            ["策略输出变化率", "相邻动作指令是否突变？", "Q_a；已有实现"],
            ["身体层面平滑", "躯干是否跳跃、翻滚、长时间腾空或强接触？", "本轮新增 vz、roll/pitch rate、腾空和接触诊断"],
            ["指定步态", "是否符合 crawl、trot、pace 或 bound 的相位结构？", "未预声明；不得事后从视频定义"],
        ], [38, 67, 69]),
    ])
    add_image(story, equation_action, "式 1　旧动作粗糙度只比较相邻策略输出；它不测身体垂直运动、接触冲击或足端相位。", 150)
    story.extend([
        p(
            "即使每一步动作变化很小，连续多步力矩仍可积累，并在足端接触时转化为起跳。"
            "因此“动作输出平滑”与“身体运动平滑”不是同一个构念。"
        ),
        p("理论依据", "h2"),
        bullet("Ant-v5 使用 8 个直接力矩动作；官方奖励不包含 action-rate、躯干垂直速度或 gait-phase 项（Farama Foundation, 2026）。"),
        bullet("逐步独立高斯探索可在机器人控制中产生不连续轨迹；gSDE 提供时间相关、状态相关的探索机制（Raffin, Kober and Stulp, 2022）。"),
        bullet("真实四足研究通常同时处理速度跟踪、动作差分、姿态、足端滑移和控制接口；但这些设计不能直接证明本 Ant 参数有效（Aractingi et al., 2023）。"),
        PageBreak(),
        p("2. 精确重放：为什么会突然跳", "h1"),
        callout(
            f"固定重放包含 {len(events)} 次突出起跳；{100*contact['no_floor_contact_step_fraction']:.1f}% steps 无地面接触；"
            f"起跳前四步原始地面合力峰值达到 {max_force:.1f}。重放与原日志的最大奖励误差为 "
            f"{jump['replay_errors']['max_abs_reward_error']:.2e}。",
            PALE_RED, RED,
        ),
        p(
            "落地时的非线性接触会产生短时峰值，随后关节力矩把身体推离地面。当前线性前进收益和存活收益仍可奖励这种策略；"
            "接触代价对每个接触力分量先截断到 [-1,1]，再使用很小的权重，因此不会与未截断原始冲击同比增加。"
        ),
    ])
    add_image(
        story,
        OLD / "analysis" / "jump_contact_gait" / "jump_event_diagnostic.png",
        "图 1　单一精确重放 episode 的跳跃、接触与奖励诊断。该图用于定位机制，不用于估计总体发生率。",
    )
    story.extend([
        p("接触力解释限制", "h2"),
        p(
            "图中的力是 MuJoCo 约束求解器诊断量，不是硬件标定的牛顿安全阈值。当前能说的是“存在较大的仿真接触诊断峰值”，"
            "不能说“真实机器人承受了同样大小的冲击”。"
        ),
        PageBreak(),
        p("3. 6 个候选为什么都是 0% 合规", "h1"),
        p(
            "旧总合规要求所有领域同时通过。六个条件都通过了 action smoothness，但都没有通过 path directness，"
            "所以总合规全部为 0。这是合取规则的信息压缩结果，不是六组所有能力为零。"
        ),
        compliance_table(data["compliance"]),
        p(
            "这些阈值是在 development 阶段形成，尚未获得外部效度。后续不能只报告一个 accuracy/compliance 百分数，"
            "必须并列报告各领域通过率、连续指标和阈值敏感性。"
        ),
        p("视频抽样规则", "h2"),
        p(
            "旧六条件共有 18 段完整端点视频（6 条件 × 3 training seeds）；动作机制试验有 12 段；1M 扩展有 6 段。"
            "本轮矩阵生成 12 段：每个训练端点固定同一个 evaluation seed，完整 1,000 frames/50 s。视频不是独立统计重复，"
            "也不是只挑最异常策略。"
        ),
        PageBreak(),
        p("4. 1M 扩展揭示：动作变化率惩罚仍不够", "h1"),
        p(
            "把目标速度策略延长至 1M steps 后，action-rate weight 0.2 相比 0 改善了平均前进速度、路径效率、躯干倾斜和动作粗糙度，"
            "但方向误差更大，意图合规率更低；六个端点在固定重放中仍呈现大量腾空。"
        ),
    ])
    add_image(
        story,
        TARGET / "analysis" / "target_budget_extension_summary.png",
        "图 2　100k 至 1M 的目标速度扩展。每个点为独立 training seed 策略均值；结果显示多个目标之间存在权衡。",
    )
    story.extend([
        callout(
            "这解释了为什么“修正一处，另一处可能变差”：四足运动是多目标控制问题。"
            "一个奖励项可以改变优化方向，却不能自动满足未写入奖励或约束的其他构念。",
            PALE_TEAL, TEAL,
        ),
        PageBreak(),
        p("5. 本轮冻结的受控机制实验", "h1"),
        p(
            "本轮不继续盲目细调横向权重，而是用 2x2 因子设计分离两条有理论依据但作用层次不同的机制："
            "身体动态 shaping 直接评价身体结果；gSDE 改变训练探索的时间结构。"
        ),
        styled_table([
            ["条件", "身体动态惩罚", "训练探索"],
            ["B0__G0", "无", "普通逐步高斯"],
            ["B1__G0", "有", "普通逐步高斯"],
            ["B0__G8", "无", "gSDE，每 8 steps 重采样"],
            ["B1__G8", "有", "gSDE，每 8 steps 重采样"],
        ], [35, 64, 75]),
    ])
    add_image(story, equation_body, "式 2　有界身体动态惩罚。两个尺度来自既有 development 轨迹的合并第 90 百分位。", 164)
    story.extend([
        p(
            "全部条件共享 PPO 2x64 Tanh、1M steps、目标速度 1 m/s、默认控制代价、相同姿态/横向/action-rate shaping、"
            "三个配对 training seeds、十个配对 evaluation seeds 和 250k/500k/750k/1M checkpoints。"
        ),
        p("解释门槛", "h2"),
        p(
            "只有 matched-seed 对比在身体垂直/角运动或腾空指标上改善，并且没有明显破坏目标速度和路径跟踪，"
            "才支持相应机制。一个更好看的视频或一个异常 seed 都不够。"
        ),
        PageBreak(),
        p("6. 2x2 矩阵结果", "h1"),
        condition_table,
        Spacer(1, 3 * mm),
        body_results(data["body"]),
        p("表 1–2　条件均值先对每个 training seed 的 10 个 evaluation episodes 聚合，再在三个独立策略之间取均值。", "caption"),
    ])
    add_image(
        story,
        ANALYSIS / "body_smoothness_gsde_endpoint_summary.png",
        "图 3　四条件配对端点。颜色和形状区分 training seeds；浅线连接同一 seed；黑色菱形及数字为条件均值。",
    )
    story.extend([
        PageBreak(),
        p("7. 配对因子效应与裁决", "h1"),
        contrast_table(data["contrasts"]),
        p("表 3　效应写作平均值 [最小值, 最大值]。对身体动态、腾空和方向误差，负值通常表示改善；对速度和路径效率，解释必须结合目标值。", "caption"),
        p(report_narrative(merged, data["contrasts"])),
        p("不能把 development 结果升级为正式证明", "h2"),
        p(
            "本轮三个 training seeds 用于机制开发，且身体惩罚尺度来自既有 development 轨迹。它能帮助决定下一轮保留或拒绝哪些机制，"
            "但不能替代从未参与调参的 held-out training seeds。若三个 seed 的效应方向不一致，应报告不稳定性，而不是只展示最好视频。"
        ),
        PageBreak(),
        p("8. 视频与可复现性 QA", "h1"),
        styled_table([
            ["检查", "结果"],
            ["本轮完整端点视频", f"{videos} 段"],
            ["每段最少轨迹 frames", str(min_frames)],
            ["总填充 frames", str(padded)],
            ["视频索引与文件数量", video_qa],
            ["自动测试", "145 passed"],
            ["训练重复单位", "12 个独立策略：4 条件 × 3 training seeds"],
            ["数值评价", "每个策略/checkpoint 10 个配对 evaluation episodes"],
        ], [66, 108]),
        p(
            "training seed 决定网络初始权重、训练时采样和 minibatch 等随机过程，因此每个 seed 会得到一个新的策略；"
            "evaluation seed 只决定固定模型面对的初始扰动。十个 evaluation episodes 不能代替三个独立训练策略。"
        ),
        p("步态边界", "h2"),
        p(
            "Ant-v5 没有规定必须逐足爬行。两只前足同时接触可能类似 bound，也可能是高冲击异常；在未定义足端相位前不能自动判错。"
            "若以后需要 crawl/trot，必须单独预声明相位、占空比与容许误差，或使用参考控制器/模仿数据。"
        ),
        p(
            f"在被精确重放的旧 episode 中，双前足同时接触只占 {100*contact['front_pair_simultaneous_contact_fraction']:.1f}% steps，"
            f"同侧双足接触占 {100*contact['same_side_pair_contact_fraction']:.1f}%，对角双足接触占 "
            f"{100*contact['diagonal_pair_contact_fraction']:.1f}%。视频中“两条腿同时摆动”不等于它们同时承重；"
            "这些比例目前仅为诊断，尚未构成经验证的 gait classifier。"
        ),
        PageBreak(),
        p("9. 下一步决策", "h1"),
        styled_table([
            ["矩阵可能结果", "下一步"],
            ["身体惩罚一致改善，任务保持", "保留为 shaping candidate；用有限新系数确认尺度后冻结"],
            ["gSDE 一致改善", "保留训练探索机制；正式比较时固定同一实现"],
            ["组合最好但有交互", "保留组合；报告两因素不可简单相加"],
            ["改善不一致或任务崩溃", "拒绝该机制；不通过继续叠加奖励项掩盖负结果"],
            ["仍有极端跳跃", "考虑独立外部姿态/动作速率约束或 PD 动作接口，作为新的实验因素"],
        ], [55, 119]),
        p(
            "最终路线仍应是：冻结人的任务、最低安全底线与质量偏好；完成有限 development；选择一个候选；冻结 reward、约束、PPO、"
            "seeds、排除规则和视频规则；再用全新 held-out training seeds 正式比较。奖励负责平均行为质量，外部约束负责不可接受的底线，"
            "两者不能混成一个不断加项的大公式。"
        ),
        callout(
            "本轮最重要的认识：平滑动作、平滑身体和指定步态是三个不同问题。"
            "严谨优化不是“看哪里不好就无限加哪里”，而是先定位构念，再用受控消融确认机制。",
            PALE_GOLD, GOLD,
        ),
        p("References", "h1"),
        p("Aractingi, M., Desbiez, A., Ferrari, R., Le Moal, C., Ivaldi, S. and Mouret, J.-B. (2023) ‘Controlling the Solo12 quadruped robot with deep reinforcement learning’, <i>Scientific Reports</i>, 13, 11945. https://doi.org/10.1038/s41598-023-38259-7.", "reference"),
        p("Farama Foundation (2026) ‘Ant - Gymnasium documentation’. Available at: https://gymnasium.farama.org/environments/mujoco/ant/ (Accessed: 16 August 2026).", "reference"),
        p("Raffin, A., Kober, J. and Stulp, F. (2022) ‘Smooth exploration for robotic reinforcement learning’, <i>Proceedings of Machine Learning Research</i>, 164, pp. 1634–1644. Available at: https://proceedings.mlr.press/v164/raffin22a.html (Accessed: 16 August 2026).", "reference"),
    ])
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    sha = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    qa = {
        "status": "created_pending_visual_qa",
        "pdf": str(OUTPUT),
        "sha256": sha,
        "matrix_status": data["matrix_summary"]["status"],
        "videos": videos,
        "minimum_video_frames": min_frames,
        "padded_frames": padded,
    }
    QA_OUTPUT.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    build()
