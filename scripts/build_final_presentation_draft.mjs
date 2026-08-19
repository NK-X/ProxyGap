import fs from "fs";
import path from "path";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, m => m.slice(1))), "..");
const OUT = path.join(ROOT, "deliverables");
fs.mkdirSync(OUT, { recursive: true });

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "ProxyGap project team";
pptx.subject = "Final 15-minute group presentation draft";
pptx.title = "From reward shaping to hierarchical terrain navigation";
pptx.company = "Southwest Jiaotong University–Leeds Joint School";
pptx.lang = "en-GB";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-GB",
};
pptx.defineLayout({ name: "CUSTOM_WIDE", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_WIDE";

const C = {
  paper: "F4F1EA", white: "FFFFFF", ink: "20262E", muted: "65717D",
  teal: "168C8C", blue: "2E6E9E", amber: "D8902F", red: "B84A3A",
  cyan: "44D7E4", line: "CCD2D6", pale: "E7ECEB", dark: "15232B",
};
const W = 13.333, H = 7.5;
const finalFrame = path.join(ROOT, "artifacts/dev/v4_pair0_multiobjective_full_map_video_v1_20260820/time_priority/v4_pair0_time_priority_seed_690223864_full_map_relief_v1_final_frame.png");
const balancedFrame = path.join(ROOT, "artifacts/dev/v4_pair0_multiobjective_full_map_video_v1_20260820/balanced/v4_pair0_balanced_seed_1864999454_full_map_relief_v1_final_frame.png");
const candidatePlot = path.join(ROOT, "docs/figures/v4_multiobjective_final/candidate_time_work_scatter.png");
const formalPlot = path.join(ROOT, "docs/figures/v4_multiobjective_final/formal_per_seed_outcomes.png");
const routePlot = path.join(ROOT, "docs/figures/v4_multiobjective_final/representative_planned_vs_actual_routes.png");

function addText(slide, text, x, y, w, h, options = {}) {
  slide.addText(text, {
    x, y, w, h, margin: 0, fontFace: "Aptos", fontSize: 20,
    color: C.ink, breakLine: false, valign: "mid", ...options,
  });
}

function baseSlide(title, speaker, n, source) {
  const slide = pptx.addSlide();
  slide.background = { color: C.paper };
  slide.addShape(pptx.ShapeType.line, { x: 0.65, y: 1.12, w: 12.0, h: 0, line: { color: C.teal, width: 1.8 } });
  addText(slide, title, 0.65, 0.34, 10.7, 0.62, { fontFace: "Aptos Display", fontSize: 27, bold: true });
  addText(slide, speaker, 10.95, 0.16, 1.72, 0.25, { fontSize: 10, color: C.muted, align: "right", italic: true });
  addText(slide, source, 0.68, 7.13, 11.5, 0.18, { fontSize: 7.2, color: C.muted, valign: "bottom" });
  addText(slide, String(n).padStart(2, "0"), 12.38, 7.06, 0.30, 0.22, { fontSize: 9, color: C.muted, align: "right" });
  return slide;
}

function card(slide, x, y, w, h, title, body, colour = C.teal) {
  slide.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.06, fill: { color: C.white }, line: { color: C.line, width: 1 } });
  slide.addShape(pptx.ShapeType.rect, { x, y, w: 0.08, h, fill: { color: colour }, line: { color: colour } });
  addText(slide, title, x + 0.22, y + 0.12, w - 0.34, 0.36, { fontSize: 18, bold: true, color: colour });
  addText(slide, body, x + 0.22, y + 0.55, w - 0.34, h - 0.68, { fontSize: 13.5, color: C.ink, valign: "top", breakLine: true });
}

function pill(slide, text, x, y, w, colour) {
  slide.addShape(pptx.ShapeType.roundRect, { x, y, w, h: 0.42, rectRadius: 0.18, fill: { color: colour }, line: { color: colour } });
  addText(slide, text, x, y + 0.01, w, 0.38, { fontSize: 13, bold: true, color: C.white, align: "center" });
}

function arrow(slide, x1, y1, x2, y2, colour = C.muted) {
  slide.addShape(pptx.ShapeType.line, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color: colour, width: 1.8, beginArrowType: "none", endArrowType: "triangle" } });
}

// 1 — outcome first
{
  const slide = pptx.addSlide();
  slide.background = { color: C.dark };
  slide.addImage({ path: finalFrame, x: 0, y: 0, w: W, h: H });
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: W, h: H, fill: { color: "071117", transparency: 33 }, line: { color: "071117", transparency: 100 } });
  addText(slide, "From reward shaping to verified terrain navigation", 0.72, 0.72, 8.9, 1.12, { fontFace: "Aptos Display", fontSize: 34, bold: true, color: C.white, valign: "top" });
  addText(slide, "A two-stage PPO–Ant study", 0.76, 1.83, 5.3, 0.44, { fontSize: 20, color: "DCEBEC" });
  pill(slide, "6 / 6 formal completions", 0.76, 5.35, 2.68, C.teal);
  pill(slide, "0 falls", 3.58, 5.35, 1.35, C.blue);
  pill(slide, "0 sustained slip events", 5.07, 5.35, 2.55, C.amber);
  addText(slide, "Known frozen map · candidate-bank near-optimal · not unseen-map generalisation", 0.76, 5.92, 8.0, 0.36, { fontSize: 13.5, color: C.white });
  addText(slide, "[Speaker A Pinyin]", 10.76, 0.18, 1.9, 0.25, { fontSize: 10, color: C.white, align: "right", italic: true });
  addText(slide, "Project formal result, 20 Aug 2026; video manifest 60498def…c024.", 0.76, 7.08, 10.9, 0.20, { fontSize: 7.2, color: "D8E0E3" });
  addText(slide, "01", 12.38, 7.05, 0.3, 0.22, { fontSize: 9, color: C.white, align: "right" });
}

// 2 — stage timeline
{
  const slide = baseSlide("Two stages increased task complexity while retaining one baseline", "[Speaker A Pinyin]", 2, "Project reward history; terrain protocols; final V3 system report.");
  const xs = [1.0, 4.9, 8.8];
  const titles = ["Legacy V1", "Stage 1 / Project V2", "Stage 2 / Project V3"];
  const bodies = ["Early control-cost and reward attempts\nProblem discovery", "Directed flat locomotion\nReward, posture and contact diagnostics", "Continuous terrain mission\nPAIR0, slopes, turning, planning and preferences"];
  const colours = [C.muted, C.blue, C.teal];
  for (let i = 0; i < 3; i++) {
    slide.addShape(pptx.ShapeType.ellipse, { x: xs[i], y: 2.05, w: 0.54, h: 0.54, fill: { color: colours[i] }, line: { color: colours[i] } });
    addText(slide, String(i + 1), xs[i], 2.04, 0.54, 0.54, { fontSize: 16, bold: true, color: C.white, align: "center" });
    card(slide, xs[i] - 0.32, 3.02, 3.35, 2.25, titles[i], bodies[i], colours[i]);
    if (i < 2) arrow(slide, xs[i] + 0.64, 2.32, xs[i + 1] - 0.18, 2.32, colours[i + 1]);
  }
  addText(slide, "Research question", 0.68, 5.75, 1.55, 0.35, { fontSize: 15, bold: true, color: C.red });
  addText(slide, "Can a locomotor progress from stable direction control to safe, preference-aware completion on continuous terrain?", 2.25, 5.72, 9.85, 0.50, { fontSize: 19, bold: true });
}

// 3 — baseline
{
  const slide = baseSlide("The defensible baseline is locally reproduced Ant-v5 + PPO", "[Speaker A Pinyin]", 3, "Farama Foundation (n.d.); Schulman et al. (2016, 2017); Fu et al. (2022).");
  card(slide, 0.8, 1.55, 3.25, 2.45, "Ant-v5", "Environment definition\n8 torque actions\nStandard forward-locomotion objective", C.blue);
  card(slide, 5.02, 1.55, 3.25, 2.45, "PPO", "Algorithmic provenance\nClipped surrogate policy-gradient family\nMatched budgets and seeds", C.teal);
  card(slide, 9.18, 1.55, 3.25, 2.45, "Project baseline", "Local Ant-v5 + PPO reproduction\nNumerical comparator for Stage 1\nController lineage for Stage 2", C.amber);
  arrow(slide, 4.12, 2.78, 4.87, 2.78, C.muted); arrow(slide, 8.34, 2.78, 9.02, 2.78, C.muted);
  slide.addShape(pptx.ShapeType.roundRect, { x: 1.25, y: 4.55, w: 10.85, h: 1.25, rectRadius: 0.08, fill: { color: "E8F1F1" }, line: { color: C.teal, width: 1 } });
  addText(slide, "Key distinction", 1.55, 4.75, 1.55, 0.38, { fontSize: 16, bold: true, color: C.teal });
  addText(slide, "Farama is documentation, not the baseline paper. Published studies anchor method choices; matched local runs support numerical claims.", 3.12, 4.62, 8.52, 0.76, { fontSize: 17, bold: true });
}

// 4 — reward vs gates
{
  const slide = baseSlide("Human intent was separated from the reward optimised by PPO", "[Speaker B Pinyin]", 4, "Ng, Harada and Russell (1999); Pan, Bhatia and Steinhardt (2022); Skalse et al. (2022).");
  const layers = [
    [1.10, 1.55, 11.0, 1.12, "1  Optimised reward", "speed · direction · posture · action rate · landing · support", C.blue],
    [1.55, 3.02, 10.1, 1.12, "2  Independent diagnostics", "path efficiency · heading error · tilt · contact sequence · mechanical proxies", C.teal],
    [2.05, 4.49, 9.1, 1.30, "3  Hard mission gate", "arrival 1.5 m → 2 m dwell for 2 s → finite + no fall/torso/sustained non-foot/slip", C.red],
  ];
  for (const [x,y,w,h,t,b,c] of layers) {
    slide.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.06, fill: { color: C.white }, line: { color: c, width: 2 } });
    addText(slide, t, x+0.25, y+0.12, 2.5, 0.4, { fontSize: 17, bold: true, color: c });
    addText(slide, b, x+2.7, y+0.12, w-2.95, h-0.24, { fontSize: 15.3, bold: true });
  }
  addText(slide, "Only trajectories that passed validity and safety were ranked by time and work proxy.", 2.0, 6.15, 9.2, 0.45, { fontSize: 19, bold: true, color: C.teal, align: "center" });
}

// 5 — stage 1 interventions
{
  const slide = baseSlide("Stage 1 reshaped locomotion without changing the 8-D action space", "[Speaker B Pinyin]", 5, "Project Stage-1 reward history; Raffin, Kober and Stulp (2022); Aractingi et al. (2023).");
  const items = [
    ["Target motion", "speed and forward-direction tracking", C.blue],
    ["Posture", "vertical, angular and signed-pitch terms", C.teal],
    ["Action quality", "action-rate and saturation diagnostics", C.amber],
    ["Contact", "foot-landing and support diagnostics", C.red],
  ];
  let x = 0.78;
  for (const [t,b,c] of items) { card(slide, x, 1.58, 2.86, 2.2, t, b, c); x += 3.05; }
  slide.addShape(pptx.ShapeType.roundRect, { x: 1.0, y: 4.45, w: 11.25, h: 1.18, rectRadius: 0.08, fill: { color: C.dark }, line: { color: C.dark } });
  addText(slide, "Retained", 1.32, 4.78, 1.15, 0.32, { fontSize: 15, bold: true, color: C.cyan });
  addText(slide, "Ant body · 8 torque actions · matched PPO environment and budget", 2.52, 4.62, 8.95, 0.62, { fontSize: 20, bold: true, color: C.white });
  addText(slide, "The external action limiter is a controller intervention, not a learned biological gait mechanism.", 1.02, 6.02, 11.2, 0.42, { fontSize: 16, italic: true, color: C.muted, align: "center" });
}

// 6 — stage 1 evidence
{
  const slide = baseSlide("Stage 1 improved selected diagnostics—not a biologically verified gait", "[Speaker B Pinyin]", 6, "Project Stage-1 development reports; Fu et al. (2022).");
  card(slide, 0.82, 1.52, 5.74, 3.28, "Direction and smoothness", "Action roughness   0.0139 → 0.00985\nMean speed             0.844 → 0.918 m/s\nPath efficiency         0.809 → 0.857", C.teal);
  card(slide, 6.78, 1.52, 5.74, 3.28, "Contact coordination", "Mean take-offs         21.1 → 3.77\nNo-floor fraction       0.526 → 0.465\nMean speed              0.961 → 0.931 m/s", C.blue);
  slide.addShape(pptx.ShapeType.roundRect, { x: 1.23, y: 5.30, w: 10.87, h: 1.08, rectRadius: 0.08, fill: { color: "F5E9E4" }, line: { color: C.red, width: 1.2 } });
  addText(slide, "Boundary", 1.52, 5.55, 1.15, 0.34, { fontSize: 15, bold: true, color: C.red });
  addText(slide, "The walk looked more coordinated, but duty factor, phase and contact ordering were not frozen as biological validation metrics.", 2.75, 5.38, 8.85, 0.66, { fontSize: 16.2, bold: true });
}

// 7 — failure layers
{
  const slide = baseSlide("Stage 2 exposed three different failure layers", "[Speaker C Pinyin]", 7, "Miki et al. (2022); PAIR0 V3, V5 turn and post-seal map evaluations.");
  card(slide, 0.83, 1.55, 3.7, 3.98, "Low level", "Contact and locomotion\n\n135-D observation = 122-D locomotion + 13-D local terrain preview\n\nPAIR0 improved distal support", C.blue);
  card(slide, 4.82, 1.55, 3.7, 3.98, "Mid level", "Turning and waypoint following\n\nV5: slope PASS, turn FAIL\n\nReliable left/right response remained checkpoint-dependent", C.amber);
  card(slide, 8.82, 1.55, 3.7, 3.98, "High level", "Global route selection\n\nDirect-goal PAIR0 best progress: 14.51 m in 600 s\n\nLow-level policy never saw the full map", C.teal);
  addText(slide, "Improving support did not automatically solve steering or route choice.", 1.0, 6.02, 11.25, 0.44, { fontSize: 21, bold: true, color: C.red, align: "center" });
}

// 8 — PAIR0 and slope
{
  const slide = baseSlide("PAIR0 improved support and tested slope safety—but airborne gait remained", "[Speaker C Pinyin]", 8, "Project PAIR0 L2b V3 and slope-boundary manifests; Lee et al. (2020).");
  addText(slide, "24.04%", 0.95, 1.78, 2.2, 0.68, { fontSize: 31, bold: true, color: C.muted, align: "center" });
  addText(slide, "→ 3.21%", 2.77, 1.78, 2.2, 0.68, { fontSize: 31, bold: true, color: C.teal, align: "center" });
  addText(slide, "full-control zero-foot", 1.25, 2.50, 3.45, 0.35, { fontSize: 15, color: C.muted, align: "center" });
  addText(slide, "0.353", 0.95, 3.40, 2.2, 0.68, { fontSize: 31, bold: true, color: C.muted, align: "center" });
  addText(slide, "→ 1.344", 2.77, 3.40, 2.2, 0.68, { fontSize: 31, bold: true, color: C.blue, align: "center" });
  addText(slide, "mean supporting feet", 1.25, 4.12, 3.45, 0.35, { fontSize: 15, color: C.muted, align: "center" });
  card(slide, 5.62, 1.52, 6.73, 3.78, "Slope boundary (tested grid)", "Uphill: contiguous pass through 12°\n16°: first tested failure\nDownhill: non-monotonic\n\n55 / 55 episodes: 0 fall, torso-ground, sustained non-foot or sustained corrected-slip events", C.amber);
  addText(slide, "12° is not a physical maximum, and PAIR0 is not a claim that MuJoCo was “fixed”.", 1.25, 5.92, 10.9, 0.46, { fontSize: 17, bold: true, color: C.red, align: "center" });
}

// 9 — hierarchy
{
  const slide = baseSlide("The successful solution was hierarchical—not another reward term", "[Speaker C Pinyin]", 9, "Project checkpoint screen, slope screen and waypoint-route evidence.");
  const labels = [
    ["Frozen 1025×1025 map", C.blue], ["15 route / speed candidates", C.teal],
    ["3 m waypoint lookahead", C.amber], ["Archived V4 expert", C.red], ["8 joint torques", C.dark],
  ];
  let x = 0.48;
  for (let i=0;i<labels.length;i++) {
    const w = i === 0 ? 2.25 : (i === 1 ? 2.35 : (i === 2 ? 2.15 : 2.03));
    slide.addShape(pptx.ShapeType.roundRect, { x, y: 2.18, w, h: 1.28, rectRadius: 0.07, fill: { color: labels[i][1] }, line: { color: labels[i][1] } });
    addText(slide, labels[i][0], x+0.1, 2.33, w-0.2, 0.88, { fontSize: 15.3, bold: true, color: C.white, align: "center" });
    if (i<labels.length-1) arrow(slide, x+w+0.06, 2.82, x+w+0.42, 2.82, C.muted);
    x += w + 0.52;
  }
  addText(slide, "Full map", 0.77, 4.20, 2.1, 0.35, { fontSize: 14, bold: true, color: C.blue, align: "center" });
  addText(slide, "Global structure", 3.25, 4.20, 2.3, 0.35, { fontSize: 14, bold: true, color: C.teal, align: "center" });
  addText(slide, "Local command", 5.98, 4.20, 2.1, 0.35, { fontSize: 14, bold: true, color: C.amber, align: "center" });
  addText(slide, "135-D local sensing", 8.60, 4.20, 2.2, 0.35, { fontSize: 14, bold: true, color: C.red, align: "center" });
  slide.addShape(pptx.ShapeType.roundRect, { x: 0.85, y: 5.18, w: 11.65, h: 1.02, rectRadius: 0.08, fill: { color: C.white }, line: { color: C.line } });
  addText(slide, "Rejected en route", 1.12, 5.47, 1.7, 0.32, { fontSize: 15, bold: true, color: C.red });
  addText(slide, "Naïve V4 / PAIR0 action blending fell after 15.3 s; the final architecture kept contact and planning roles explicit.", 2.82, 5.32, 9.1, 0.62, { fontSize: 16.2, bold: true });
}

// 10 — multi-objective selection
{
  const slide = baseSlide("Three preferences selected two routes from 15 feasible candidates", "[Speaker D Pinyin]", 10, "Project candidate-selection JSON 212b7886…f20b; Fu et al. (2022) for mechanical-energy context.");
  slide.addImage({ path: candidatePlot, x: 0.72, y: 1.38, w: 8.15, h: 4.72 });
  card(slide, 9.15, 1.48, 3.45, 1.10, "Time 0.8 / 0.2", "s1p50_t1p00", C.teal);
  card(slide, 9.15, 2.79, 3.45, 1.10, "Balanced 0.5 / 0.5", "same route contract", C.blue);
  card(slide, 9.15, 4.10, 3.45, 1.10, "Energy 0.2 / 0.8", "balanced_speed_0p50", C.amber);
  addText(slide, "J = wT(T/Tmin) + wE(W+/W+min)", 8.98, 5.62, 3.82, 0.45, { fontSize: 15, bold: true, color: C.ink, align: "center" });
  addText(slide, "Near-optimal within the candidate bank; not a global optimum.", 8.92, 6.20, 3.95, 0.38, { fontSize: 11.5, italic: true, color: C.red, align: "center" });
}

// 11 — final outcome
{
  const slide = baseSlide("All six formal episodes reached the goal without sustained slip events", "[Speaker D Pinyin]", 11, "Project formal manifest 0bf2817c…8d83; video manifest 60498def…c024.");
  slide.addImage({ path: formalPlot, x: 0.55, y: 1.38, w: 7.72, h: 3.63 });
  slide.addImage({ path: balancedFrame, x: 8.58, y: 1.46, w: 4.18, h: 2.35 });
  addText(slide, "Video 2: balanced route · exact formal replay", 8.58, 3.86, 4.18, 0.26, { fontSize: 10, italic: true, color: C.muted, align: "center" });
  slide.addShape(pptx.ShapeType.roundRect, { x: 0.70, y: 5.25, w: 12.0, h: 1.10, rectRadius: 0.06, fill: { color: C.white }, line: { color: C.line } });
  addText(slide, "time / balanced", 0.98, 5.42, 1.58, 0.26, { fontSize: 13, bold: true, color: C.teal });
  addText(slide, "3/3 · 264.55 s · 55.65 kJ proxy · 153.23 m", 2.62, 5.35, 3.74, 0.42, { fontSize: 14.4, bold: true });
  addText(slide, "energy", 6.53, 5.42, 0.75, 0.26, { fontSize: 13, bold: true, color: C.amber });
  addText(slide, "3/3 · 259.37 s · 55.13 kJ proxy · 152.14 m", 7.32, 5.35, 3.72, 0.42, { fontSize: 14.4, bold: true });
  addText(slide, "0 falls · 0 sustained slip events", 10.92, 5.32, 1.55, 0.50, { fontSize: 12, bold: true, color: C.red, align: "center" });
  addText(slide, "Airborne exposure remained: representative runs had 9.25–10.02% full-control zero-foot intervals.", 1.18, 6.53, 10.95, 0.34, { fontSize: 12.5, italic: true, color: C.red, align: "center" });
}

// 12 — references
{
  const slide = baseSlide("Method provenance and comparator literature", "[Speaker D Pinyin]", 12, "Full URLs/DOIs are retained in the report bibliography; accessed 20 Aug 2026 where applicable.");
  const left = [
    "Aractingi et al. (2023) Scientific Reports 13, 11945. doi:10.1038/s41598-023-38259-7.",
    "Farama Foundation (n.d.) Ant. Gymnasium Documentation.",
    "Fu et al. (2022) PMLR 164, pp. 928–937.",
    "Lee et al. (2020) Science Robotics 5(47), eabc5986.",
    "Miki et al. (2022) Science Robotics 7(62), eabk2822.",
    "Ng, Harada and Russell (1999) ICML, pp. 278–287.",
  ].join("\n\n");
  const right = [
    "Pan, Bhatia and Steinhardt (2022) ICLR 2022.",
    "Raffin, Kober and Stulp (2022) PMLR 164, pp. 1634–1644.",
    "Schulman et al. (2016) ICLR.",
    "Schulman et al. (2017) arXiv:1707.06347.",
    "Skalse et al. (2022) NeurIPS 35.",
    "Project configurations, raw traces, videos and SHA-256 manifests are listed in the evidence appendix.",
  ].join("\n\n");
  addText(slide, left, 0.82, 1.46, 5.85, 5.30, { fontSize: 14.1, valign: "top", breakLine: true });
  slide.addShape(pptx.ShapeType.line, { x: 6.66, y: 1.42, w: 0, h: 5.30, line: { color: C.line, width: 1 } });
  addText(slide, right, 6.98, 1.46, 5.55, 5.30, { fontSize: 14.1, valign: "top", breakLine: true });
}

// 13 — evidence boundary
{
  const slide = baseSlide("What is established—and where does the evidence stop?", "[Speaker D Pinyin]", 13, "Project final report and evidence manifests, 20 Aug 2026.");
  card(slide, 0.78, 1.45, 5.92, 4.12, "Established", "• selected Stage-1 motion diagnostics\n• PAIR0 support under the frozen model\n• tested slope boundaries\n• 6/6 known-map completion\n• exact replay and full video decode", C.teal);
  card(slide, 6.92, 1.45, 5.62, 4.12, "Not established", "• biologically natural gait\n• unseen-map generalisation\n• independent training-seed robustness\n• electrical energy\n• a continuous global optimum", C.red);
  addText(slide, "The result is architectural: auditable failures revealed the layers required for reliable completion.", 1.05, 5.93, 11.15, 0.55, { fontSize: 20, bold: true, color: C.dark, align: "center" });
  addText(slide, "Questions", 4.73, 6.53, 3.86, 0.55, { fontFace: "Aptos Display", fontSize: 27, bold: true, color: C.teal, align: "center" });
}

for (const slide of pptx._slides) {
  if (typeof slide.addNotes === "function") {
    slide.addNotes("Speaker notes: see docs/FINAL_PRESENTATION_CONTENT_DESIGN_V2_20260820_CN.md for the timed script and evidence boundaries.");
  }
}

const output = path.join(OUT, "ProxyGap_Final_Presentation_Draft_20260820.pptx");
await pptx.writeFile({ fileName: output });
console.log(output);
