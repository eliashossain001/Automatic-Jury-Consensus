#!/usr/bin/env python3
"""Build the lab-meeting PowerPoint from the deck outline + real repo figures.

Progress-report talk for: "Structured Co-Failure in LLM Judge Banks:
Measurement, Correlated Failure Injection, and Adaptive CorrFilter".

Mirrors reports/lab_meeting_deck.md. Embeds the actual generated figures where
they exist; for the 3 slides whose figure does not exist yet (pipeline schematic,
position-bias bar chart, timeline) it synthesizes a clean matplotlib figure so
the deck is self-contained.

Usage:
    python scripts/make_lab_meeting_deck.py --project-root .
Output:
    reports/lab_meeting_deck.pptx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

# ---------------------------------------------------------------- palette
NAVY = RGBColor(0x1A, 0x23, 0x47)
CRIMSON = RGBColor(0xA4, 0x1E, 0x34)
SLATE = RGBColor(0x3C, 0x47, 0x5A)
LIGHT = RGBColor(0xF4, 0xF5, 0xF8)
GREY = RGBColor(0x6B, 0x72, 0x80)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT = RGBColor(0x2E, 0x5A, 0x88)
GREEN = RGBColor(0x1F, 0x7A, 0x3D)
AMBER = RGBColor(0xB0, 0x6A, 0x00)

# 16:9
SW, SH = Inches(13.333), Inches(7.5)


def _root() -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    a = ap.parse_args()
    return Path(a.project_root).resolve()


# ---------------------------------------------------------------- synthesized figures
def fig_pipeline(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 3.4), dpi=160)
    ax.axis("off")
    boxes = [
        ("Preference\npairs", "#3C475A"),
        ("Judge bank\n(k-of-n)", "#A41E34"),
        ("Consensus\nfilter", "#2E5A88"),
        ("Filtered\ntraining data", "#3C475A"),
        ("DPO / IPO\npolicy", "#1F7A3D"),
    ]
    n = len(boxes)
    x = 0.03
    w, h, y = 0.155, 0.42, 0.30
    for i, (label, color) in enumerate(boxes):
        ax.add_patch(plt.Rectangle((x, y), w, h, color=color, ec="white", lw=2,
                                   transform=ax.transAxes, zorder=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                color="white", fontsize=13, fontweight="bold",
                transform=ax.transAxes, zorder=3)
        if i < n - 1:
            ax.annotate("", xy=(x + w + 0.038, y + h / 2), xytext=(x + w, y + h / 2),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color="#1A2347", lw=2.5))
        x += w + 0.048
    ax.text(0.5, 0.86, "The judge bank is the quality gate upstream of every preference-tuned model",
            ha="center", va="center", fontsize=12.5, style="italic", color="#1A2347",
            transform=ax.transAxes)
    ax.text(0.5, 0.04, "Hidden assumption: judges fail INDEPENDENTLY  →  more judges = more reliable",
            ha="center", va="center", fontsize=12, color="#A41E34", fontweight="bold",
            transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_position_bias(out: Path) -> Path:
    # From reports/h1_measurement.md section 2 (no-swap vs swap accuracy).
    judges = [
        ("mistral-7b::pair", 96.7, 26.5), ("llama-3.1-8b::pair", 95.3, 31.6),
        ("qwen-2.5-7b::pair", 93.8, 47.1), ("qwen-2.5-7b::likert", 88.9, 37.2),
        ("gemma-2-9b::likert", 85.7, 45.0), ("llama-3.1-8b::likert", 71.4, 53.9),
        ("gemma-2-9b::pair", 69.0, 79.4), ("mistral-7b::likert", 64.9, 49.0),
        ("phi-3.5-mini::likert", 50.5, 62.4), ("phi-3.5-mini::pair", 43.9, 74.0),
    ]
    labels = [j[0] for j in judges]
    noswap = [j[1] for j in judges]
    swap = [j[2] for j in judges]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=160)
    ax.barh(y - 0.2, noswap, height=0.4, color="#2E5A88", label="chosen shown as A (no-swap)")
    ax.barh(y + 0.2, swap, height=0.4, color="#A41E34", label="chosen shown as B (swap)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(50, color="#6B7280", ls="--", lw=1, label="chance (50%)")
    ax.set_xlabel("accuracy (%)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_title("Position bias is universal: accuracy flips with answer slot\n"
                 "(Mistral-pairwise 96.7% → 26.5%, a 70-pt swing)", fontsize=12.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_timeline(out: Path) -> Path:
    # label ABOVE each bar in dark text (always on white, never clipped);
    # short status tag centered inside the bar.
    rows = [
        ("Measurement, CFI, Adaptive CorrFilter", 0.0, 4.4, "#1F7A3D", "DONE"),
        ("H2b: Synthetic-Poisoned UltraFeedback", 4.4, 1.4, "#1F7A3D", "DONE (neg.)"),
        ("Position-Aligned Poisoning", 5.8, 1.4, "#B06A00", "NEXT"),
        ("DPO / IPO downstream training", 7.2, 2.4, "#3C475A", "PENDING"),
        ("AlpacaEval + PKU-SafeRLHF", 9.6, 1.8, "#3C475A", "PENDING"),
        ("Paper draft", 11.4, 1.8, "#A41E34", "DRAFTING"),
    ]
    n = len(rows)
    rowh, barh = 1.0, 0.5
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=160)
    ax.axis("off")
    here = 5.8
    for i, (label, start, dur, color, tag) in enumerate(rows):
        y = (n - 1 - i) * rowh
        ax.add_patch(plt.Rectangle((start, y), dur, barh, facecolor=color,
                                   edgecolor="white", lw=1.5, alpha=0.95, zorder=2))
        ax.text(start, y + barh + 0.13, label, ha="left", va="bottom",
                fontsize=11.5, fontweight="bold", color="#1A2347", zorder=3)
        ax.text(start + dur / 2, y + barh / 2, tag, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white", zorder=4)
    ax.axvline(here, color="#A41E34", ls="--", lw=2, zorder=1)
    ax.text(here, n * rowh + 0.18, "we are here", color="#A41E34", fontsize=11,
            fontweight="bold", ha="center")
    ax.set_xlim(-0.3, 13.8)
    ax.set_ylim(-0.3, n * rowh + 0.7)
    ax.set_title("Roadmap to completion (relative effort, not calendar)",
                 fontsize=13, color="#1A2347")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_alignment_path(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=160)
    ax.axis("off")

    def flow(y, steps, color, title, title_color):
        ax.text(0.5, y + 0.17, title, ha="center", va="center", fontsize=13.5,
                fontweight="bold", color=title_color, transform=ax.transAxes)
        n = len(steps)
        x = 0.015
        w, h = 0.165, 0.135
        for i, label in enumerate(steps):
            ax.add_patch(plt.Rectangle((x, y - h / 2), w, h, color=color, ec="white",
                                       lw=2, transform=ax.transAxes, zorder=2))
            ax.text(x + w / 2, y, label, ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold", transform=ax.transAxes, zorder=3)
            if i < n - 1:
                ax.annotate("", xy=(x + w + 0.028, y), xytext=(x + w, y),
                            xycoords=ax.transAxes, textcoords=ax.transAxes,
                            arrowprops=dict(arrowstyle="-|>", color="#1A2347", lw=2.2))
            x += w + 0.037

    flow(0.80,
         ["Raw preference\ndata", "LLM judge\nfiltering", "bad labels survive\nconsensus",
          "DPO / RLHF\ntraining", "biased model\nbehavior"],
         "#A41E34", "The problem today", "#A41E34")
    flow(0.22,
         ["Adaptive\nCorrFilter", "cleaner\npreference data", "safer, more reliable\npreference training"],
         "#1F7A3D", "The proposed path", "#1F7A3D")
    # short downward cue in the clear band, well above the lower title
    ax.annotate("", xy=(0.5, 0.52), xytext=(0.5, 0.64), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color="#2E5A88", lw=3))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_corrfilter_intuition(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=160)
    ax.axis("off")
    # Left: naive majority counts 3 correlated yes as 3
    ax.text(0.24, 0.93, "Naive majority", ha="center", fontsize=14, fontweight="bold",
            color="#A41E34", transform=ax.transAxes)
    ax.text(0.76, 0.93, "CorrFilter (decorrelated)", ha="center", fontsize=14,
            fontweight="bold", color="#1F7A3D", transform=ax.transAxes)
    for cx, color in [(0.10, "#A41E34"), (0.20, "#A41E34"), (0.30, "#A41E34")]:
        ax.add_patch(plt.Circle((cx, 0.60), 0.035, color=color, transform=ax.transAxes))
        ax.text(cx, 0.60, "Y", ha="center", va="center", color="white",
                fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.annotate("", xy=(0.18, 0.74), xytext=(0.22, 0.74), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-", color="#6B7280", lw=1))
    ax.text(0.20, 0.78, "share a bias\n(co-fail together)", ha="center", fontsize=9.5,
            color="#6B7280", transform=ax.transAxes)
    ax.text(0.24, 0.40, "= 3 votes\n\"strong agreement\"", ha="center", fontsize=12,
            color="#A41E34", transform=ax.transAxes)
    ax.text(0.24, 0.16, "bad label PASSES", ha="center", fontsize=12.5,
            fontweight="bold", color="#A41E34", transform=ax.transAxes)

    for cx in [0.66, 0.76, 0.86]:
        ax.add_patch(plt.Circle((cx, 0.60), 0.035, color="#1F7A3D", transform=ax.transAxes))
        ax.text(cx, 0.60, "Y", ha="center", va="center", color="white",
                fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(0.76, 0.40, "discount judges that\nhistorically fail together", ha="center",
            fontsize=12, color="#1F7A3D", fontweight="bold", transform=ax.transAxes)
    ax.text(0.76, 0.26, "3 co-failing Y  ~=  1 effective vote",
            ha="center", fontsize=10.5, color="#3C475A", transform=ax.transAxes)
    ax.text(0.76, 0.04, r"$\alpha_{subset}=|S|/\sqrt{1^\top_S R_S 1_S}$", ha="center",
            fontsize=8, color="#9AA0AA", transform=ax.transAxes)
    ax.text(0.76, 0.16, "bad label FLAGGED", ha="center", fontsize=12.5,
            fontweight="bold", color="#1F7A3D", transform=ax.transAxes)
    ax.axvline(0.5, color="#CCCCCC", lw=1)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_h2b_bars(out: Path) -> Path:
    # Completed H2b precision at matched retention (outputs/.../summary.md).
    rates = ["5% poison", "10% poison", "20% poison"]
    majority = [0.980, 0.956, 0.901]
    adaptive = [0.978, 0.949, 0.895]
    oracle = [0.979, 0.953, 0.892]   # oracle_variant_R (oracle correlation matrix)
    x = np.arange(len(rates))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=160)
    b1 = ax.bar(x - w, majority, w, label="Naive consensus", color="#3C475A")
    b2 = ax.bar(x, adaptive, w, label="Adaptive CorrFilter", color="#A41E34")
    b3 = ax.bar(x + w, oracle, w, label="Oracle-R CorrFilter", color="#6B7280")
    for bars in (b1, b2, b3):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.006,
                    f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=9,
                    color="#1A2347")
    ax.set_xticks(x)
    ax.set_xticklabels(rates, fontsize=12)
    ax.set_ylabel("precision at matched retention", fontsize=11)
    ax.set_ylim(0, 1.12)          # full scale: bars are honestly near-identical
    ax.set_title("Adaptive CorrFilter ~= Consensus  (no meaningful gain)", fontsize=13,
                 color="#1A2347")
    ax.legend(loc="lower left", fontsize=10, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_final_picture(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 5.0), dpi=160)
    ax.axis("off")
    cols = [
        ("Weak Dependence", "Consensus Works", "#3C475A"),
        ("Globally Correlated\nCo-Failure", "CorrFilter Helps", "#1F7A3D"),
        ("Dominant Vulnerable\nSubgroup", "Bias-Cluster Helps", "#B06A00"),
    ]
    w = 0.30
    xs = [0.015, 0.35, 0.685]
    for (regime, mitig, color), x in zip(cols, xs):
        ax.add_patch(plt.Rectangle((x, 0.66), w, 0.22, facecolor=color, edgecolor="white",
                                   lw=2, transform=ax.transAxes, zorder=2))
        ax.text(x + w / 2, 0.77, regime, ha="center", va="center", color="white",
                fontsize=13, fontweight="bold", transform=ax.transAxes, zorder=3)
        ax.annotate("", xy=(x + w / 2, 0.40), xytext=(x + w / 2, 0.63),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", color="#1A2347", lw=3))
        ax.add_patch(plt.Rectangle((x, 0.16), w, 0.22, facecolor="white", edgecolor=color,
                                   lw=3, transform=ax.transAxes, zorder=2))
        ax.text(x + w / 2, 0.27, mitig, ha="center", va="center", color=color,
                fontsize=13.5, fontweight="bold", transform=ax.transAxes, zorder=3)
    ax.text(0.5, 0.03, "Different dependence regimes require different mitigations.",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#A41E34",
            transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_surprising(out: Path) -> Path:
    # position attack, 10% poisoning, gain over naive majority (pts).
    labels = ["Label-free\n(0 labels)", "Learned\n(100 labels)", "H1 imported\ncluster"]
    gains = [4.8, 4.9, 4.8]
    colors = ["#B06A00", "#2E5A88", "#1F7A3D"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    bars = ax.bar(labels, gains, color=colors, width=0.6)
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.12,
                f"+{r.get_height():.1f}", ha="center", va="bottom", fontsize=14,
                fontweight="bold", color="#1A2347")
    ax.set_ylabel("precision gain over majority (pts)", fontsize=11)
    ax.set_ylim(0, 6.2)
    ax.set_title("All three recover roughly the same gain\n(position attack, 10% poisoning)",
                 fontsize=13, color="#1A2347")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_flow(ax, y, steps, color, title, title_color, fs=9.5):
    ax.text(0.5, y + 0.20, title, ha="center", va="center", fontsize=13.5,
            fontweight="bold", color=title_color, transform=ax.transAxes)
    n = len(steps)
    gap = 0.022
    w = (0.98 - gap * (n - 1)) / n
    h = 0.135
    x = 0.01
    for i, label in enumerate(steps):
        ax.add_patch(plt.Rectangle((x, y - h / 2), w, h, color=color, ec="white",
                                   lw=2, transform=ax.transAxes, zorder=2))
        ax.text(x + w / 2, y, label, ha="center", va="center", color="white",
                fontsize=fs, fontweight="bold", transform=ax.transAxes, zorder=3)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap * 0.92, y), xytext=(x + w, y),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color="#1A2347", lw=2))
        x += w + gap


def fig_h2b_failed(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=160)
    ax.axis("off")
    _draw_flow(ax, 0.82,
               ["Structured\npoisoning", "shared judge\nfailures", "consensus\nerror",
                "CorrFilter\nadvantage"],
               "#A41E34", "Expected", "#A41E34")
    _draw_flow(ax, 0.30,
               ["Structured\npoisoning", "little correlation\ndrift", "judges remain\nindependent",
                "consensus already\nworks well", "CorrFilter has\nnothing to correct"],
               "#3C475A", "Observed", "#3C475A", fs=9.0)
    ax.annotate("", xy=(0.5, 0.54), xytext=(0.5, 0.66), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color="#2E5A88", lw=3))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- pptx helpers
def add_bg(slide, color=WHITE):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def textbox(slide, l, t, w, h):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    return tb, tf


def set_run(run, text, size, color, bold=False, italic=False, font="Calibri"):
    run.text = text
    f = run.font
    f.size = Pt(size)
    f.color.rgb = color
    f.bold = bold
    f.italic = italic
    f.name = font


def accent_bar(slide, color=CRIMSON):
    bar = slide.shapes.add_shape(1, 0, 0, Inches(0.18), SH)
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()


def section_tag(slide, text, color=CRIMSON):
    tb, tf = textbox(slide, Inches(0.45), Inches(0.30), Inches(9.5), Inches(0.4))
    set_run(tf.paragraphs[0].add_run(), text.upper(), 12, color, bold=True)


def title_block(slide, title, sub=None):
    tb, tf = textbox(slide, Inches(0.45), Inches(0.62), Inches(12.4), Inches(1.1))
    set_run(tf.paragraphs[0].add_run(), title, 30, NAVY, bold=True)
    if sub:
        p = tf.add_paragraph()
        set_run(p.add_run(), sub, 15, SLATE, italic=True)


def bullets(slide, items, l=Inches(0.55), t=Inches(1.85), w=Inches(6.2), h=Inches(5.0),
            size=16):
    tb, tf = textbox(slide, l, t, w, h)
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            text, level, color, bold = item
        else:
            text, level, color, bold = item, 0, SLATE, False
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        p.space_after = Pt(7)
        bullet = "•  " if level == 0 else "–  "
        if text.startswith("@"):  # no bullet marker
            bullet, text = "", text[1:]
        set_run(p.add_run(), bullet + text, size - level * 1, color, bold=bold)
    return tb


def add_image_fit(slide, img: Path, l, t, max_w, max_h):
    from PIL import Image

    with Image.open(img) as im:
        iw, ih = im.size
    ar = iw / ih
    box_ar = max_w / max_h
    if ar > box_ar:
        w = max_w
        h = Emu(int(max_w / ar))
    else:
        h = max_h
        w = Emu(int(max_h * ar))
    left = Emu(int(l + (max_w - w) / 2))
    top = Emu(int(t + (max_h - h) / 2))
    slide.shapes.add_picture(str(img), left, top, width=w, height=h)


def speaker_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def status_chip(slide, l, t, text, color):
    chip = slide.shapes.add_shape(5, l, t, Inches(2.05), Inches(0.55))
    chip.fill.solid()
    chip.fill.fore_color.rgb = color
    chip.line.fill.background()
    tf = chip.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), text, 12, WHITE, bold=True)


TAGLINE = "Consensus can fail in multiple ways, and no single filter fixes them all."


def footer_tagline(slide, color=GREY):
    tb, tf = textbox(slide, Inches(0.45), Inches(7.04), Inches(12.4), Inches(0.38))
    set_run(tf.paragraphs[0].add_run(), TAGLINE, 10.5, color, italic=True)


def big_stat(slide, l, t, number, label, color=CRIMSON, num_size=48, w=Inches(3.6)):
    tb, tf = textbox(slide, l, t, w, Inches(1.4))
    set_run(tf.paragraphs[0].add_run(), number, num_size, color, bold=True)
    p = tf.add_paragraph()
    set_run(p.add_run(), label, 13, SLATE)


# ---------------------------------------------------------------- slide builders
def content_slide(prs, section, title, blts, img: Path | None, notes,
                  sub=None, bullet_size=16):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s)
    accent_bar(s)
    section_tag(s, section)
    title_block(s, title, sub)
    if img and img.exists():
        bullets(s, blts, w=Inches(6.0), size=bullet_size)
        add_image_fit(s, img, Inches(6.85), Inches(1.85), Inches(6.1), Inches(5.0))
    else:
        bullets(s, blts, w=Inches(12.3), size=bullet_size + 1)
    footer_tagline(s)
    speaker_notes(s, notes)
    return s


def main():
    root = _root()
    figdir = root / "reports" / "deck_assets"
    figdir.mkdir(parents=True, exist_ok=True)

    H1 = root / "experiments" / "h1_measurement" / "figures"
    EIG = root / "outputs" / "eigen_failure_modes"
    CFI = root / "outputs" / "cfi" / "figures"
    ADA = root / "outputs" / "adaptive_r" / "figures"
    DIR = root / "outputs" / "direction_randomized" / "figures"
    UF = root / "outputs" / "synthetic_poisoned_ultrafeedback" / "figures"

    # synthesized
    f_pipe = fig_pipeline(figdir / "s_pipeline.png")
    f_pos = fig_position_bias(figdir / "s_position_bias.png")
    f_time = fig_timeline(figdir / "s_timeline.png")
    f_cfintu = fig_corrfilter_intuition(figdir / "s_corrfilter_intuition.png")
    f_align = fig_alignment_path(figdir / "s_alignment_path.png")
    f_h2b_bars = fig_h2b_bars(figdir / "s_h2b_bars.png")
    f_h2b_failed = fig_h2b_failed(figdir / "s_h2b_failed.png")
    f_final = fig_final_picture(figdir / "s_final_picture.png")
    f_surprising = fig_surprising(figdir / "s_surprising.png")

    # repo figures for the taxonomy + router experiments
    POS = root / "outputs" / "position_poisoning" / "figures"
    CLU = root / "outputs" / "cluster_filter" / "figures"
    LBC = root / "outputs" / "learned_bias_cluster" / "figures"
    RTR = root / "outputs" / "regime_router" / "figures"

    prs = Presentation()
    prs.slide_width = SW
    prs.slide_height = SH

    # ===================================================== Slide 1: title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    bar = s.shapes.add_shape(1, 0, Inches(2.95), SW, Inches(0.06))
    bar.fill.solid(); bar.fill.fore_color.rgb = CRIMSON; bar.line.fill.background()
    tb, tf = textbox(s, Inches(0.7), Inches(1.05), Inches(12.0), Inches(1.9))
    set_run(tf.paragraphs[0].add_run(),
            "Consensus Is Not Reliability:", 38, WHITE, bold=True)
    p = tf.add_paragraph()
    set_run(p.add_run(), "A Taxonomy of Dependence Failures in LLM Judge Banks", 24, WHITE, bold=True)
    tb, tf = textbox(s, Inches(0.7), Inches(3.2), Inches(11.8), Inches(0.6))
    set_run(tf.paragraphs[0].add_run(), "ICLR 2026 Submission – Progress Report",
            20, RGBColor(0xF4, 0xC4, 0x30), bold=True)
    tb, tf = textbox(s, Inches(0.7), Inches(4.4), Inches(11.0), Inches(1.0))
    set_run(tf.paragraphs[0].add_run(), "Elias Hossain", 20, WHITE, bold=True)
    p = tf.add_paragraph()
    set_run(p.add_run(), "University of Central Florida", 16, RGBColor(0xC9, 0xCE, 0xD6))
    tb, tf = textbox(s, Inches(0.7), Inches(6.25), Inches(12.0), Inches(0.8))
    set_run(tf.paragraphs[0].add_run(), TAGLINE, 14, RGBColor(0xD9, 0xB8, 0xC0), italic=True)
    speaker_notes(s, "Frame the talk around the new thesis: consensus is not reliability. We use "
        "panels of LLM judges to decide which preference data is good enough to train on, and we "
        "assume agreement implies correctness. The project shows that consensus fails in several "
        "DISTINCT ways, and that different dependence regimes need different mitigations. The "
        "strongest contribution now is a taxonomy of consensus failure in LLM judge banks, not a "
        "single filter. This is a progress report: measurement and the diagnostic experiments are "
        "done with real numbers; downstream DPO is still ahead.")

    # ===================================================== SECTION 1
    content_slide(prs, "Section 1 · Motivation",
        "Why LLM judges matter in RLHF / DPO",
        [
            "Alignment (RLHF, DPO, IPO) needs a preference signal: which response is better.",
            "Human labels are expensive → we use LLM judges to label / filter data at scale.",
            "Common safety move: a bank of judges; keep an item only if k-of-n agree.",
            ("@ ", 0, SLATE, False),
            ("Implicit promise: more judges → more reliable label → cleaner data.", 0, CRIMSON, True),
            "The judge bank is the quality gate upstream of every preference-tuned model.",
        ],
        f_pipe,
        "Walk the pipeline left to right. The judge bank is the quality gate that decides what "
        "the model learns from. If the gate is wrong in a CORRELATED way, bad labels pass through "
        "and get baked into the trained policy. Stress the stakes: this is upstream of every "
        "preference-tuned model. The k-of-n rule is everywhere because it feels like a vote, and "
        "votes feel trustworthy. The rest of the talk is why that intuition fails.",
        sub="The judge bank is the quality gate for alignment data")

    content_slide(prs, "Section 1 · Motivation",
        "The hidden assumption behind k-of-n consensus",
        [
            "k-of-n consensus is only safe if judges fail INDEPENDENTLY.",
            ("If errors are correlated, agreement is a herd, not a signal: judges co-fail on the same items.", 0, CRIMSON, True),
            "Two consequences of average pairwise correlation ρ̄ > 0:",
            ("variance of mean error inflated by 1 + (n−1)ρ̄", 1, SLATE, False),
            ("effective independent judges bounded by n_eff ≤ 1/ρ̄", 1, SLATE, False),
            "A 10-judge bank can be worth far fewer than 10, and adding judges barely helps.",
        ],
        H1 / "neff_collapse.png",
        "This is the conceptual crux. The standard defense ('we used many judges and diversified "
        "across families') silently assumes independence. Show the collapse curve: under the "
        "MEASURED correlation, a bank of 10 behaves like ~3.5 independent judges and asymptotes "
        "below 5 no matter how many you add. Punchline: correlated agreement is exactly what fools "
        "a consensus filter, because the wrong answer also gets a confident majority. This is "
        "'consensus is not reliability under dependence' made quantitative.",
        sub="Correlated agreement is what fools a consensus filter")

    # why this matters for alignment (new motivation slide)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s)
    section_tag(s, "Section 1 · Motivation")
    title_block(s, "Why This Matters for Alignment",
                "Correlated judge failures propagate straight into the trained model")
    add_image_fit(s, f_align, Inches(0.6), Inches(2.0), Inches(12.1), Inches(4.6))
    footer_tagline(s)
    speaker_notes(s, "Make the stakes concrete. Top row is the failure path as it stands today: "
        "raw preference data goes through an LLM-judge filter, correlated co-failure lets bad "
        "labels survive the consensus vote, those labels train the policy via DPO/RLHF, and the "
        "result is biased model behavior. The bad labels don't get caught downstream, they get "
        "amplified. Bottom row is what we're proposing: Adaptive CorrFilter produces cleaner "
        "preference data, which gives safer and more reliable preference training. The whole "
        "project is about converting the top path into the bottom path.")

    # research questions slide (custom, with chips)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s)
    section_tag(s, "Section 1 · Motivation")
    title_block(s, "Research questions (C1–C4)", "The spine of the talk; each maps to one section")
    rq = [
        ("C1 (Measurement)", "Are judge errors correlated, and does the correlation have identifiable structure?", "C1  DONE", GREEN),
        ("C2 (Mechanism)", "Does naive k-of-n over-retain wrong labels, concentrated on triggered items?", "C2  DONE (partial)", GREEN),
        ("C3 (Mitigation)", "What kinds of dependence cause consensus failure, and which mitigation works in each regime?", "C3  REFRAMED", AMBER),
        ("C4 (Downstream)", "Does DPO/IPO on correlation-aware-filtered data give a better help–safety trade-off?", "C4  NOT STARTED", GREY),
    ]
    y = Inches(1.95)
    for name, desc, chip, color in rq:
        tb, tf = textbox(s, Inches(0.55), y, Inches(8.2), Inches(1.1))
        set_run(tf.paragraphs[0].add_run(), name, 16, NAVY, bold=True)
        p = tf.add_paragraph(); set_run(p.add_run(), desc, 13.5, SLATE)
        status_chip(s, Inches(10.7), y + Inches(0.12), chip, color)
        y += Inches(1.22)
    footer_tagline(s)
    speaker_notes(s, "Lay out the spine of the talk. The key change is C3. We are no longer asking "
        "'can CorrFilter improve filtering quality?'. We are asking what KINDS of dependence cause "
        "consensus to fail, and which mitigation works in each regime. That shift, from a single "
        "method to a taxonomy, is the scientific contribution this update is built around. C1 and "
        "C2 are done; C4 (downstream DPO) is not started.")

    # ----- ANCHOR SLIDE: The Final Picture (taxonomy overview)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s)
    section_tag(s, "Section 1 · Motivation")
    title_block(s, "The Final Picture",
                "Consensus failure is heterogeneous: each regime has its own mitigation")
    add_image_fit(s, f_final, Inches(0.6), Inches(1.8), Inches(12.1), Inches(5.0))
    footer_tagline(s)
    speaker_notes(s, "This is the anchor slide for the whole talk. Across the experiments, consensus "
        "does not fail in one way, it fails in distinct regimes, and each regime has a different "
        "right answer. Weak dependence: consensus already works. Globally correlated co-failure: "
        "CorrFilter helps. Dominant vulnerable subgroup: bias-cluster filtering helps. The takeaway "
        "at the bottom is the thesis: different dependence regimes require different mitigations. "
        "Everything that follows is evidence for these three columns, and an attempt to detect the "
        "regime automatically.")

    # ----- Three Dependence Regimes (formal table)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s)
    section_tag(s, "Section 1 · Motivation")
    title_block(s, "Three Dependence Regimes")
    rows = [
        ("Regime", "Example", "Best Filter"),
        ("Weak Dependence", "H2b", "Supermajority"),
        ("Globally Correlated Co-Failure", "CFI", "CorrFilter"),
        ("Dominant Vulnerable Subgroup", "Position Attack", "Bias Cluster"),
    ]
    tbl = s.shapes.add_table(len(rows), 3, Inches(0.7), Inches(2.1),
                             Inches(11.9), Inches(2.7)).table
    for j, wd in enumerate((Inches(5.3), Inches(3.0), Inches(3.6))):
        tbl.columns[j].width = wd
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = NAVY; col = WHITE; bold = True
            elif j == 0:
                cell.fill.fore_color.rgb = LIGHT; col = NAVY; bold = True
            else:
                cell.fill.fore_color.rgb = WHITE if i % 2 else LIGHT; col = SLATE; bold = (j == 2)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            set_run(p.add_run(), val, 15, col, bold=bold)
    band = s.shapes.add_shape(1, Inches(0.7), Inches(5.2), Inches(11.9), Inches(0.95))
    band.fill.solid(); band.fill.fore_color.rgb = CRIMSON; band.line.fill.background()
    bt = band.text_frame; bt.vertical_anchor = MSO_ANCHOR.MIDDLE
    bp = bt.paragraphs[0]; bp.alignment = PP_ALIGN.CENTER
    set_run(bp.add_run(), "No single filter dominates all regimes.", 19, WHITE, bold=True)
    footer_tagline(s)
    speaker_notes(s, "The formal version of the anchor. Three regimes, each with the example "
        "experiment that exhibits it and the filter that wins there. Weak dependence (H2b): plain "
        "supermajority is enough. Globally correlated co-failure (CFI): CorrFilter. Dominant "
        "vulnerable subgroup (position attack): bias-cluster filtering. The banner is the one-line "
        "claim of the paper: no single filter dominates all regimes.")

    # ===================================================== SECTION 2
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s)
    section_tag(s, "Section 2 · Measurement study  (COMPLETED)")
    title_block(s, "Judge bank & RewardBench setup")
    bullets(s, [
        ("Judge bank", 0, CRIMSON, True),
        ("5 open-weight bases × 2 prompt styles = 10 logical judges", 1, SLATE, False),
        ("Llama-3.1-8B, Qwen-2.5-7B, Gemma-2-9B, Mistral-7B-v0.3, Phi-3.5-mini (4-bit nf4)", 1, SLATE, False),
        ("pairwise (A/B) and likert (1–5, ties forbidden)", 1, SLATE, False),
        ("per-item seeded position swap controls order bias", 1, SLATE, False),
        ("Calibration", 0, CRIMSON, True),
        ("reward-bench-2, stratified to 1195 items (≤250/subset)", 1, SLATE, False),
        ("gold = 1 by construction (chosen ≻ rejected): remember this for Section 5", 1, AMBER, True),
        ("R via Ledoit-Wolf shrinkage, 94.8% listwise retention; 1000-bootstrap CIs", 1, SLATE, False),
    ], w=Inches(12.3), size=15)
    footer_tagline(s)
    speaker_notes(s, "Logical judge = base model x prompt protocol, which lets us separate family "
        "lineage from prompt-template effects. Credibility choices: open weights (reproducible), "
        "quantized (fits one TITAN RTX), seeded position swap (don't confuse order bias for "
        "preference). The fixed-direction gold (chosen is always right) is deliberate and "
        "important: it makes analysis clean but returns as a load-bearing caveat in Section 5. "
        "Pre-registration: we locked the H1 threshold before looking, so we can report a rejection "
        "honestly. Full vote pass was ~6 GPU-hours, parquet-cached.")

    content_slide(prs, "Section 2 · Measurement study  (COMPLETED)",
        "Discovery 1: position bias is universal",
        [
            "EVERY judge shows a position-accuracy gap (chosen shown as A vs B).",
            ("Mistral-pairwise: 96.7% → 26.5% (a 70-pt swing)", 1, CRIMSON, True),
            ("Llama-pair +63.7, Qwen-pair +46.7", 1, SLATE, False),
            "Polarity varies by family; presence is universal.",
            "Without per-item swap, naive scoring over-credits A-biased judges.",
            ("A bias every judge shares survives a consensus vote.", 0, CRIMSON, True),
        ],
        f_pos,
        "First concrete shared failure mode and a great intuition pump: these models aren't fully "
        "reading the answer, they're partly reading the SLOT. A 70-point swing means "
        "Mistral-pairwise is basically answering 'A' regardless of content. Meta-point: a bias "
        "every judge shares is exactly the kind that survives a consensus vote, because they all "
        "lean the same way, so the majority is confidently wrong together. We control it per-item, "
        "so the downstream correlations are not just order artifacts. (Note: this figure is "
        "synthesized from the H1 report table; the repo doesn't yet have a standalone PNG.)")

    # Discovery 2, numerical summary (heatmap removed)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s)
    section_tag(s, "Section 2 · Measurement study  (COMPLETED)")
    title_block(s, "Discovery 2: Judge Errors Are Correlated")
    big_stat(s, Inches(0.7), Inches(1.95), "ρ̄ = 0.206", "mean error correlation", CRIMSON)
    big_stat(s, Inches(7.0), Inches(1.95), "3.51 / 10", "effective ensemble size (n_eff)", CRIMSON)
    bullets(s, [
        ("Strongest correlated pair is cross-family / cross-prompt:", 0, NAVY, True),
        ("Qwen-pairwise ↔ Gemma-likert  =  +0.565", 1, SLATE, False),
        ("Strongest intra-family pair is negative:", 0, NAVY, True),
        ("Qwen-pairwise ↔ Qwen-likert  =  −0.186", 1, SLATE, False),
        ("Takeaway: model-family diversity alone does not remove shared judge failures.", 0, CRIMSON, True),
    ], l=Inches(0.7), t=Inches(3.55), w=Inches(12.0), size=18)
    footer_tagline(s)
    speaker_notes(s, "Two numbers carry this slide. rho-bar = 0.206 means errors are substantially "
        "correlated, not independent. n_eff = 3.51 of 10 means the bank behaves like only ~3.5 "
        "independent judges. The twist: the two judges that co-fail most are from DIFFERENT "
        "families with DIFFERENT prompts (+0.565), while the same base model under two prompts can "
        "be anti-correlated (-0.186). So the standard mitigation, 'diversify across families,' "
        "targets the wrong axis. This sets up why H1 as originally framed gets rejected.")

    content_slide(prs, "Section 2 · Measurement study  (COMPLETED)",
        "Discovery 3: effective ensemble size collapses",
        [
            ("n_eff (random-effects) = 3.51 of 10", 0, CRIMSON, True),
            "n_eff^eig (participation ratio) = 6.74 of 10.",
            "Prop. 1 bound: n_eff ≤ 1/ρ̄ ≈ 4.86, regardless of bank size.",
            ("Leading eigen-direction carries 29.2% of error variance (vs 10% if independent).", 0, SLATE, False),
            "The 29% spike IS the dominant shared failure mode, decomposed next.",
        ],
        H1 / "R_eigenspectrum.png",
        "Two ways to count independent judges, both say 'far fewer than 10.' The eigenspectrum "
        "shows why: one big spike (29% of variance) plus a second above the independence line. "
        "That spike is the shared failure mode the rest of the project decomposes and then "
        "exploits. Tie back: 10 judges, ~3.5 independent votes, hard ceiling under 5.")

    content_slide(prs, "Section 2 · Measurement study  (COMPLETED)",
        "H1: pre-registered REJECT, but C1 strongly supported",
        [
            "Pre-registered H1: intra > cross by Δ ≥ 0.10 (p<0.01), family OR prompt.",
            ("Family: ρ̄_intra 0.136 vs ρ̄_cross 0.214 → Δ = −0.078 (wrong sign). REJECT.", 1, CRIMSON, True),
            ("Prompt style: Δ = +0.0003 (≈ zero). REJECT.", 1, CRIMSON, True),
            ("But the bigger claim wins: errors ARE heavily correlated,", 0, NAVY, True),
            ("just not along family / prompt lines.", 1, NAVY, True),
            "Reframed headline: shared failure modes are bank-wide and cross-family.",
            "Diversification alone is not a sufficient fix.",
        ],
        None,
        "The honesty slide. We pre-registered the wrong structure (family/prompt) and the data "
        "rejected it cleanly on the locked threshold. But the proposal's fallback ('maybe errors "
        "are independent') is also decisively false: rho-bar ~0.21, n_eff 3.5. So we pivot the "
        "headline from 'which lineage correlates' to 'correlation is real, structured, and crosses "
        "family boundaries.' That's stronger and more actionable than the original H1. Rejecting a "
        "pre-registered hypothesis and reporting it is what makes the rest credible.",
        sub="Rejecting our own hypothesis is what makes the rest credible")

    content_slide(prs, "Section 2 · Measurement study  (COMPLETED)",
        "Failure-mode decomposition: eigenvectors → semantics",
        [
            ("v1 (29.2%): near-uniform sign → common difficult-item factor", 0, NAVY, True),
            ("item-loading vs error-count r=+0.99; cross-family co-failure r=+0.93", 1, SLATE, False),
            ("v2 (14.5%): within-Qwen pairwise-vs-likert flip → prompt-divergence axis", 0, NAVY, True),
            ("v3 (9.9%): prompt contrast on Llama/Mistral; top items skew safety/refusal", 0, NAVY, True),
            "Determinism verified (identical CSV checksums across reruns).",
            "Dominant shared direction is semantic (hard items), not lineage.",
        ],
        EIG / "failure_tag_by_eigenvector.png",
        "We turn linear-algebra directions into human-readable failure families. v1 = 'items that "
        "fool everyone at once,' the dangerous ones for consensus. v2 = the prompt-style axis that "
        "explains the negative intra-Qwen correlation. v3 picks up safety/refusal items. Takeaway: "
        "the dominant shared-error direction is semantic (hard/ambiguous items), which is exactly "
        "why cross-family diversification doesn't decorrelate it.")

    # ===================================================== SECTION 3
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s)
    section_tag(s, "Section 3 · Correlated Failure Injection  (COMPLETED)")
    title_block(s, "CFI design: from observed to causal",
                "Construct correlation to test cause & effect")
    bullets(s, [
        "Measurement shows correlation exists; CFI CONSTRUCTS it to test causality.",
        "Inject a mild biasing system prompt + heuristic trigger into part of the bank.",
        "Bank-ratio conditions: all_clean → mixed_25/50/75 → all_biased.",
        "Real GPU run: 5 bases × 2 prompts × 7 mechanisms, triggered items on a 400-item subset.",
        ("Run completed: 9.87 h, 70/70 passes, 10,932 biased votes cached.", 0, GREEN, True),
        ("7 mechanisms: verbosity, sycophancy, jailbreak-shell, polite-hallucination,", 0, SLATE, False),
        ("length-target, style-template, position-bias-stress.", 1, SLATE, False),
    ], w=Inches(12.3), size=15.5)
    footer_tagline(s)
    speaker_notes(s, "CFI is the controlled complement to the observational study. Instead of "
        "waiting for natural co-failure, we plant a known shared bias in some judges and dial up "
        "how many are infected. This answers C2 causally: when judges share a mechanism, do bad "
        "labels survive the vote, and are survivors CONCENTRATED on triggered items? This was a "
        "real ~10-hour GPU run on one TITAN RTX, not a simulation. Each mechanism stands in for a "
        "documented LLM-judge pathology (length preference, sycophancy, polished-but-wrong text).")

    content_slide(prs, "Section 3 · Correlated Failure Injection  (COMPLETED)",
        "Result: false retention rises with the biased fraction",
        [
            "Naive-majority false retention climbs for 3 of 7 mechanisms (peaks ~75% biased):",
            ("polite_hallucination +0.059", 1, CRIMSON, True),
            ("position_bias +0.056", 1, CRIMSON, True),
            ("verbosity +0.027", 1, CRIMSON, True),
            "Other 4 ~null (sycophancy only 9 items trigger; jailbreak/length/style flat).",
            ("Confirms C2 (partial): shared mechanisms push wrong labels through the vote,", 0, NAVY, True),
            ("concentrated on triggered items.", 1, NAVY, True),
        ],
        CFI / "cfi_false_retention_vs_ratio.png",
        "Core causal result. As more of the bank shares the bias, the majority vote retains more "
        "wrong labels, exactly the predicted failure. Be precise and honest: it fires strongly for "
        "mechanisms with a real correlated signal (polite hallucination, position bias, verbosity) "
        "and is null for weak-trigger mechanisms. The non-monotone dip-then-peak near 75% is worth "
        "a sentence: at 100% biased some mechanisms partially cancel. Bottom line: when co-failure "
        "is real, consensus is provably contaminated, motivating a correlation-aware filter.")

    # ===================================================== SECTION 4
    content_slide(prs, "Section 4 · CorrFilter  (COMPLETED)",
        "Why majority fails, and the CorrFilter intuition",
        [
            ("CorrFilter discounts agreement among judges that historically fail together.", 0, GREEN, True),
            "Majority counts votes as if independent, correlated agreement is double-counting.",
            "Three judges that always agree because they share a bias ≈ one real vote.",
            "Rewards genuinely independent agreement.",
            "Keeps top-scored items at a matched retention (vs supermajority's top raw-consensus).",
            ("(formula kept in the notes, not on the slide)", 0, GREY, False),
        ],
        f_cfintu,
        "Plain language: if three judges always agree because they share a bias, their joint 'yes' "
        "should count as roughly one vote, not three. The score (kept off the slide) is "
        "alpha_subset = |S| / sqrt(1^T_S R_S 1_S): it uses the correlation matrix R to deflate "
        "redundant agreement and inflate independent agreement. We compare at MATCHED retention so "
        "it's apples-to-apples: same number of items kept, fewer of them wrong is the win condition.")

    content_slide(prs, "Section 4 · CorrFilter  (COMPLETED)",
        "Static CorrFilter: an honest negative result",
        [
            "On the clean bank, CorrFilter ≈ majority (sanity check passes).",
            ("On biased banks with clean R, CorrFilter ties or slightly loses to supermajority.", 0, CRIMSON, True),
            ("Δ(super−CF): +0.006 on 3 mechanisms; −0.03 to −0.005 on the 4 that matter.", 1, SLATE, False),
            "Diagnosis: clean R cannot SEE the injected correlation → α can't discount it.",
            ("This is correlation DRIFT: deployment bank is more correlated than calibration measured.", 0, NAVY, True),
        ],
        CFI / "cfi_majority_vs_corrfilter.png",
        "Don't hide this, lead with it. With R measured on clean calibration data, CorrFilter "
        "doesn't beat a plain supermajority. That LOOKS like the method failing. Real cause: the R "
        "we plugged in was estimated before the bias appeared, so it literally can't discount a "
        "correlation it never saw. Name it: correlation drift between calibration and deployment. "
        "This reframes the negative as a measurement problem (which R?), not a flaw in alpha_subset, "
        "and sets up the fix.",
        sub="The method looks broken, but it's the R that's stale")

    # variant-R slide (big numbers, no table)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s)
    section_tag(s, "Section 4 · CorrFilter  (COMPLETED)")
    title_block(s, "With the right R, CorrFilter wins",
                "Position-bias mechanism, matched retention, false-retention rate")
    big_stat(s, Inches(0.8), Inches(2.1), "0.304", "clean-R CorrFilter", GREY, 50)
    tb, tf = textbox(s, Inches(4.4), Inches(2.35), Inches(1.0), Inches(1.0))
    set_run(tf.paragraphs[0].add_run(), "→", 44, SLATE, bold=True)
    big_stat(s, Inches(5.5), Inches(2.1), "0.183", "with the RIGHT R", GREEN, 50)
    tb, tf = textbox(s, Inches(9.3), Inches(2.35), Inches(3.6), Inches(1.2))
    set_run(tf.paragraphs[0].add_run(), "vs 0.274", 26, SLATE, bold=True)
    p = tf.add_paragraph(); set_run(p.add_run(), "naive supermajority", 13, SLATE)
    bullets(s, [
        ("~40% fewer wrong labels retained, beats both supermajority and clean-R.", 0, GREEN, True),
        "Re-estimate R from the biased bank's own votes (it now SEES the injected correlation).",
        "The correlation it acts on rises ~0.20 → ~0.32, it is discounting the real co-failure.",
        ("So the method works. Open problem: get deployment-time R without gold labels →", 0, CRIMSON, True),
    ], l=Inches(0.7), t=Inches(4.1), w=Inches(11.9), size=17)
    footer_tagline(s)
    speaker_notes(s, "When CorrFilter can actually see the bank's true correlation, it cuts false "
        "retention sharply: position bias drops from 0.32 to 0.18, well below supermajority. So "
        "the method is sound; the bottleneck is purely 'where does the right R come from at "
        "deployment, with no gold labels?' That question is the entire next section, and it's the "
        "most scientifically interesting part of the project.")

    # ===================================================== SECTION 5
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s)
    section_tag(s, "Section 5 · Adaptive CorrFilter  (COMPLETED)")
    title_block(s, "Adaptive R: closing the gap without gold",
                "Recover the oracle-R gain from the live bank's votes")
    big_stat(s, Inches(0.7), Inches(2.0), "0.286", "naive majority FRR", GREY, 40, w=Inches(2.8))
    tb, tf = textbox(s, Inches(3.2), Inches(2.2), Inches(0.8), Inches(0.9))
    set_run(tf.paragraphs[0].add_run(), "→", 36, SLATE, bold=True)
    big_stat(s, Inches(4.0), Inches(2.0), "0.205", "adaptive CorrFilter FRR", GREEN, 40, w=Inches(2.8))
    bullets(s, [
        ("Adaptive R (estimated from live votes) matches the oracle, gap −0.002 FRR.", 0, NAVY, True),
        "Beats naive majority, supermajority, and clean-R alike.",
        ("Caveat (next slide): fixed-direction gold makes vote-corr = error-corr exactly here,", 0, AMBER, True),
        ("so the gold-free estimator is the algebraic twin of the oracle in this benchmark.", 1, AMBER, False),
    ], l=Inches(0.6), t=Inches(3.5), w=Inches(6.0), size=15.5)
    add_image_fit(s, ADA / "false_retention_by_r_mode.png", Inches(6.85), Inches(2.1),
                  Inches(6.1), Inches(4.5))
    footer_tagline(s)
    speaker_notes(s, "First-pass result looked spectacular: a fully gold-free estimator (just "
        "correlate the live votes) matches the oracle and closes the whole gap. But flag the "
        "structural reason it worked so well: because gold is fixed-direction, per-item error = "
        "1 - vote, so vote-correlation EQUALS error-correlation exactly. disagreement_R is the "
        "algebraic twin of the oracle in this benchmark. That's a benchmark artifact, not a free "
        "lunch, which is exactly why we ran the next validation.")

    content_slide(prs, "Section 5 · Adaptive CorrFilter  (COMPLETED)",
        "R accuracy predicts performance, and who needs it",
        [
            "The closer R is to the true (oracle) R, the lower the false retention.",
            "Most R-drift-sensitive mechanisms:",
            ("position_bias (‖ΔR‖_F = 2.73, gap +0.122)", 1, CRIMSON, True),
            ("polite_hallucination (2.52, +0.044)", 1, CRIMSON, True),
            "All others: small drift (<0.75), small gap, clean R is fine.",
            "~25 labels gets most of small_gold recovery; λ→0 (pure adaptive) best under drift.",
        ],
        ADA / "frobenius_vs_performance.png",
        "Two payoffs. (1) We validate the MECHANISM of CorrFilter: performance tracks R-accuracy, "
        "so 'get R right' is provably the lever. (2) We can predict WHEN adaptation matters: only "
        "high-drift mechanisms (position bias, polite hallucination) need it; for low-drift ones "
        "clean R is fine. This tells a practitioner when to bother. Position bias is the strongest "
        "case: adaptive R cuts its false retention by 12 points over clean R.")

    content_slide(prs, "Section 5 · Adaptive CorrFilter  (COMPLETED)",
        "Direction-randomized validation: the real stress test",
        [
            "Break the fixed-direction artifact: flip the presentation FRAME on ~50% of items.",
            ("(invert vote AND gold so per-item error is invariant)", 1, GREY, False),
            "Preserves oracle R & achievable benefit; scrambles the gold-free observable.",
            ("Gold-free disagreement_R does NOT fully survive:", 0, CRIMSON, True),
            ("oracle gap −0.002 → +0.020 FRR (now worse than clean_R)", 1, CRIMSON, True),
            ("Recovery: ~25–100 labels, or hybrid λ≈0.5, restores the oracle.", 0, GREEN, True),
        ],
        DIR / "fixed_vs_randomized_false_retention.png",
        "The experiment that keeps us honest. We deliberately destroy the benchmark artifact from "
        "the previous slide by randomizing which response is 'option A,' so vote-correlation no "
        "longer equals error-correlation. The gold-free estimator loses ~2 false-retention points. "
        "It degrades GRACEFULLY, not catastrophically (eigenstructure overlap stays 0.97), but no "
        "longer matches the oracle. The fix is cheap: 25-100 labeled items, or a 50/50 blend of "
        "clean and adaptive R.")

    content_slide(prs, "Section 5 · Regime A mitigation: CorrFilter",
        "Adaptive CorrFilter helps under globally correlated co-failure",
        [
            ("CorrFilter is not the final solution: it is the mitigation for ONE regime.", 0, NAVY, True),
            ("CFI (globally correlated co-failure):  CorrFilter is the best method.", 0, GREEN, True),
            ("position_bias false retention 0.32 to 0.18; clear gain over consensus.", 1, SLATE, False),
            ("Position attack (vulnerable subgroup):  CorrFilter fails.", 0, CRIMSON, True),
            ("at/below consensus (0.820 vs 0.824 at 10%); even oracle-R does not help.", 1, SLATE, False),
            ("The benefit depends on the dependence regime, not on the method alone.", 0, NAVY, True),
        ],
        DIR / "small_gold_recovery_curve.png",
        "Reframe CorrFilter as a regime-specific tool, not the headline method. Where failures are "
        "globally correlated (the CFI injection), CorrFilter is the best method: position-bias false "
        "retention drops from 0.32 to 0.18. But on the position attack, where the failure comes from "
        "a vulnerable subgroup rather than global correlation, CorrFilter fails, it matches "
        "consensus and even the oracle correlation matrix does not help. So the benefit depends on "
        "the dependence regime, not on the method in isolation. That sets up the taxonomy.",
        sub="CorrFilter is the Regime A tool, not a universal solution")

    # ===================================================== SECTION 6: TAXONOMY OF CONSENSUS FAILURE
    # ----- Regime C: weak / absent dependence (H2b)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s, SLATE)
    section_tag(s, "Section 6 · Regime C: weak / absent dependence", SLATE)
    title_block(s, "H2b Result: Synthetic-Poisoned UltraFeedback")
    status_chip(s, Inches(10.7), Inches(0.55), "CONSENSUS WORKS", SLATE)
    bullets(s, [
        ("Precision at matched retention (higher = better):", 0, NAVY, True),
        ("5% poison:   Majority 0.980   vs   Adaptive 0.978", 1, SLATE, False),
        ("10% poison:  Majority 0.956   vs   Adaptive 0.949", 1, SLATE, False),
        ("20% poison:  Majority 0.901   vs   Adaptive 0.895", 1, SLATE, False),
        ("Adaptive CorrFilter ~= Consensus", 0, CRIMSON, True),
        ("No meaningful filtering gain observed (gains -0.2 to -0.6 pts).", 0, CRIMSON, True),
        ("Oracle-R also fails to outperform consensus.", 0, SLATE, False),
    ], l=Inches(0.55), t=Inches(2.0), w=Inches(6.0), size=15.5)
    add_image_fit(s, f_h2b_bars, Inches(6.85), Inches(1.95), Inches(6.1), Inches(4.7))
    footer_tagline(s)
    speaker_notes(s, "This is a negative result, and I am reporting it as one. The full run "
        "completed (2000 pairs, ~99% vote coverage). At every poison level Adaptive CorrFilter "
        "lands within a few tenths of a point of naive consensus, and on the wrong side of it: "
        "gains are -0.2 to -0.6 points, i.e. essentially zero. Even Oracle-R, which is handed the "
        "true correlation matrix, does not beat consensus. The important finding is NOT that the "
        "method is broken; it is that correlation-aware filtering provides no benefit when the "
        "poisoning attack does not induce correlated co-failure. The next two slides explain why, "
        "and why this makes the project scientifically stronger.")

    # ----- Why H2b Failed (regime C explainer)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s, SLATE)
    section_tag(s, "Section 6 · Regime C: weak / absent dependence", SLATE)
    title_block(s, "Why H2b Failed",
                "The attack never created the correlation CorrFilter is designed to remove")
    add_image_fit(s, f_h2b_failed, Inches(0.6), Inches(1.85), Inches(12.1), Inches(4.2))
    tb, tf = textbox(s, Inches(0.6), Inches(6.15), Inches(12.1), Inches(0.7))
    set_run(tf.paragraphs[0].add_run(),
            "Key finding: correlation-aware filtering only helps when correlation actually exists.",
            18, CRIMSON, bold=True)
    footer_tagline(s)
    speaker_notes(s, "This is now one of the most important slides in the deck. We expected the "
        "chain on top: structured poisoning induces shared judge failures, those produce a "
        "correlated consensus error, and CorrFilter then has an edge to exploit. What we actually "
        "observed is the bottom chain: verbosity/length/polite/style poisoning barely shifted the "
        "deployment correlation structure, the judges stayed largely independent on these items, so "
        "naive consensus already filtered the attack well, and CorrFilter had nothing left to "
        "correct. The deployment correlation matrix was nearly unchanged from calibration. The key "
        "finding is the one-liner at the bottom: correlation-aware filtering only helps when "
        "correlation actually exists.")

    # ----- Regime B: Position-Aligned Poisoning (Completed)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s, AMBER)
    section_tag(s, "Section 6 · Regime B: dominant vulnerable subgroup", AMBER)
    title_block(s, "Position-Aligned Poisoning (Completed)",
                "Poison the bank's strongest measured shared bias (position) instead of content")
    status_chip(s, Inches(10.7), Inches(0.55), "COMPLETED", GREEN)
    bullets(s, [
        ("Consensus precision (naive majority):", 0, NAVY, True),
        ("5% poison  =  0.897", 1, SLATE, False),
        ("10% poison =  0.824", 1, SLATE, False),
        ("20% poison =  0.791", 1, SLATE, False),
        ("Key observation: ~100% of poisoned items were majority-affirmed.", 0, CRIMSON, True),
        ("Conclusion: consensus can confidently retain bad labels.", 0, CRIMSON, True),
    ], l=Inches(0.55), t=Inches(2.0), w=Inches(6.0), size=16)
    add_image_fit(s, POS / "correlation_drift.png", Inches(6.9), Inches(2.0),
                  Inches(6.0), Inches(4.5))
    footer_tagline(s)
    speaker_notes(s, "Regime B. Instead of content poisoning, we poison along the bank's strongest "
        "measured shared bias from H1, position bias. Now consensus genuinely fails: precision "
        "falls to 0.82 at 10% and 0.79 at 20%, and crucially ~100% of the poisoned items are "
        "affirmed by the majority. So consensus does not just slip, it CONFIDENTLY retains bad "
        "labels. The figure shows the attack did move the correlation structure (unlike H2b). This "
        "is the regime where we most want a fix.")

    # ----- Regime B: CorrFilter's blind spot
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s, AMBER)
    section_tag(s, "Section 6 · Regime B: dominant vulnerable subgroup", AMBER)
    title_block(s, "CorrFilter's Blind Spot",
                "Position attack, 10% poisoning, precision at matched retention")
    bullets(s, [
        ("Majority             =  0.824", 0, SLATE, False),
        ("CorrFilter (small-gold) =  0.820", 0, SLATE, False),
        ("CorrFilter (clean-R)    =  0.817", 0, SLATE, False),
        ("Oracle (perfect)     =  1.000", 0, GREEN, False),
        ("The wrong majority comes from a vulnerable subgroup.", 0, CRIMSON, True),
        ("Global correlation is low, so CorrFilter cannot detect this failure mode.", 0, CRIMSON, True),
    ], l=Inches(0.55), t=Inches(2.0), w=Inches(6.0), size=16)
    add_image_fit(s, POS / "precision_by_method.png", Inches(6.9), Inches(2.0),
                  Inches(6.0), Inches(4.5))
    footer_tagline(s)
    speaker_notes(s, "Here is why CorrFilter cannot save regime B. Even with the oracle correlation "
        "matrix, CorrFilter stays at consensus level (0.82-0.84) versus a perfect oracle of 1.000. "
        "The reason is structural: the wrong majority is produced by a specific vulnerable subgroup "
        "of judges (the position-biased ones), but the overall pairwise correlation is actually LOW "
        "on these items, so a correlation-aware score has no global signal to latch onto. CorrFilter "
        "is the right tool for regime A, but it is blind to regime B.")

    # ----- Regime B: Deployable bias-aware filtering (replaces old bias-cluster slide)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s, GREEN)
    section_tag(s, "Section 6 · Regime B: dominant vulnerable subgroup", GREEN)
    title_block(s, "Deployable Vulnerability-Aware Filtering",
                "Down-weight the vulnerable subgroup instead of the global correlation")
    bullets(s, [
        ("Position attack gain over majority (10% poison):", 0, NAVY, True),
        ("H1 imported cluster:  +4.8", 1, GREEN, True),
        ("Learned cluster:  +4.9", 1, GREEN, True),
        ("Label-free position-sensitive cluster:  +4.8", 1, GREEN, True),
        ("The vulnerable subgroup is recoverable from deployment-time behavior.", 0, CRIMSON, True),
        ("No prior measurement study is required.", 0, CRIMSON, True),
    ], l=Inches(0.55), t=Inches(2.05), w=Inches(6.1), size=16)
    add_image_fit(s, LBC / "calibration_size_curve.png", Inches(6.95), Inches(2.0),
                  Inches(6.0), Inches(4.5))
    footer_tagline(s)
    speaker_notes(s, "This is one of the strongest new results. The right mitigation for regime B is "
        "not correlation-aware scoring, it is down-weighting the vulnerable SUBGROUP of judges. A "
        "small labeled calibration set learns that subgroup and recovers a real gain: about +3.8 at "
        "25 labels rising to +4.9 at 100. Importantly, importing the cluster from the H1 study "
        "(+4.8) and a fully label-free position-sensitivity cluster (+4.8) match that. So the "
        "vulnerable subgroup is recoverable directly from deployment-time behavior, with no prior "
        "measurement study required. Honest caveat: the best gain is ~+4.9, just under our +5 "
        "target, so it is a partial repair, not a full fix.")

    # ----- Regime B: Most surprising result
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s, GREEN)
    section_tag(s, "Section 6 · Regime B: dominant vulnerable subgroup", GREEN)
    title_block(s, "Most Surprising Result")
    bullets(s, [
        ("0 labels   ≈   100 labels   ≈   H1 imported cluster", 0, NAVY, True),
        ("All recover roughly the same gain.", 0, SLATE, False),
        ("Interpretation:", 0, CRIMSON, True),
        ("the vulnerability signal is observable directly from judge behavior.", 1, SLATE, False),
        ("This makes the mitigation deployable.", 0, CRIMSON, True),
    ], l=Inches(0.55), t=Inches(2.0), w=Inches(6.0), size=17)
    add_image_fit(s, f_surprising, Inches(6.9), Inches(2.0), Inches(6.0), Inches(4.5))
    footer_tagline(s)
    speaker_notes(s, "The surprise: you barely need labels. A fully label-free cluster (0 labels), a "
        "learned cluster from 100 labels, and the cluster imported from the separate H1 measurement "
        "study all recover essentially the same gain (~+4.8). That tells us the vulnerability signal "
        "is observable directly from how the judges behave at deployment time, so the mitigation is "
        "deployable without a prior measurement campaign. This is the most practically important "
        "finding in the update.")

    # ----- What Failed? What Worked? (taxonomy results matrix)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s)
    section_tag(s, "Section 6 · A taxonomy of consensus failure")
    title_block(s, "What Failed? What Worked?")
    matrix = [
        ("Regime", "Consensus", "CorrFilter", "Bias Cluster"),
        ("H2b (weak dependence)", "Works", "Neutral", "Neutral"),
        ("CFI (global co-failure)", "Fails", "Works", "Hurts"),
        ("Position attack (subgroup)", "Fails", "Fails", "Helps"),
    ]
    cell_color = {"Works": GREEN, "Helps": GREEN, "Neutral": GREY, "Fails": CRIMSON, "Hurts": CRIMSON}
    tbl = s.shapes.add_table(len(matrix), 4, Inches(0.7), Inches(2.1),
                             Inches(11.9), Inches(2.9)).table
    for j, wd in enumerate((Inches(4.1), Inches(2.6), Inches(2.6), Inches(2.6))):
        tbl.columns[j].width = wd
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = NAVY; col = WHITE; bold = True
            elif j == 0:
                cell.fill.fore_color.rgb = LIGHT; col = NAVY; bold = True
            else:
                cell.fill.fore_color.rgb = cell_color.get(val, LIGHT); col = WHITE; bold = True
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            set_run(p.add_run(), val, 15, col, bold=bold)
    band = s.shapes.add_shape(1, Inches(0.7), Inches(5.35), Inches(11.9), Inches(0.95))
    band.fill.solid(); band.fill.fore_color.rgb = CRIMSON; band.line.fill.background()
    bt = band.text_frame; bt.vertical_anchor = MSO_ANCHOR.MIDDLE
    bp = bt.paragraphs[0]; bp.alignment = PP_ALIGN.CENTER
    set_run(bp.add_run(), "Consensus failure is heterogeneous.", 20, WHITE, bold=True)
    footer_tagline(s)
    speaker_notes(s, "This is the results matrix behind the taxonomy. Read it by row. H2b (weak "
        "dependence): consensus works, CorrFilter and bias-cluster are neutral, do no harm. CFI "
        "(global co-failure): consensus fails, CorrFilter works, and bias-cluster actually hurts. "
        "Position attack (vulnerable subgroup): consensus fails, CorrFilter also fails, and only "
        "bias-cluster helps. No column is good everywhere; each mitigation has exactly one regime "
        "where it is the right tool. The bottom line is the banner: consensus failure is "
        "heterogeneous, which is why the contribution is a diagnosis story. Next: can we infer the "
        "regime automatically?")

    # ----- Regime routing: can we detect the regime automatically?
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s, ACCENT)
    section_tag(s, "Section 6 · Regime routing", ACCENT)
    title_block(s, "Can We Detect the Regime Automatically?")
    big_stat(s, Inches(0.8), Inches(2.0), "78%", "regime classification accuracy", ACCENT, 54)
    bullets(s, [
        ("Signals:", 0, NAVY, True),
        ("mean correlation (ρ̄)", 1, SLATE, False),
        ("eigenvector drift", 1, SLATE, False),
        ("leave-cluster-out flip score", 1, SLATE, False),
        ("Weak regime:  detected reliably", 0, GREEN, True),
        ("Global regime:  detected reliably", 0, GREEN, True),
        ("Subgroup regime:  detected only at strong attack strength", 0, AMBER, True),
    ], l=Inches(0.7), t=Inches(3.5), w=Inches(12.0), size=16)
    footer_tagline(s)
    speaker_notes(s, "The router is the headline new experiment. From three cheap diagnostics, mean "
        "correlation, eigenvector drift versus the clean bank, and a leave-cluster-out flip score, "
        "we classify which regime a bank is in with 78% accuracy. Weak and global regimes are "
        "detected reliably; the subgroup regime is only detected at strong attack strength, because "
        "at low poison rates the subgroup signature is faint. The point of this slide is evidence "
        "that the regimes are diagnostically identifiable, not just conceptually distinct.")

    # ----- Regime routing: results (feasible, not decisive)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s, ACCENT)
    section_tag(s, "Section 6 · Regime routing", ACCENT)
    title_block(s, "Routing Is Feasible, But Not Yet Decisive")
    rows = [
        ("Method", "Overall Precision"),
        ("Naive", "0.838"),
        ("CorrFilter (best single)", "0.842"),
        ("Bias Cluster", "0.841"),
        ("Soft router", "0.840"),
        ("Hard router", "0.834"),
        ("Oracle Router", "0.854"),
    ]
    tbl = s.shapes.add_table(len(rows), 2, Inches(0.7), Inches(2.0),
                             Inches(5.7), Inches(3.4)).table
    for j, wd in enumerate((Inches(3.4), Inches(2.3))):
        tbl.columns[j].width = wd
    for i, (a, b) in enumerate(rows):
        for j, val in enumerate((a, b)):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = NAVY; col = WHITE; bold = True
            elif a == "CorrFilter (best single)":
                cell.fill.fore_color.rgb = RGBColor(0xE3, 0xEC, 0xF5); col = ACCENT; bold = True
            elif a == "Oracle Router":
                cell.fill.fore_color.rgb = LIGHT; col = GREY; bold = False
            else:
                cell.fill.fore_color.rgb = WHITE if i % 2 else LIGHT; col = SLATE; bold = False
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            set_run(p.add_run(), val, 14, col, bold=bold)
    add_image_fit(s, RTR / "router_vs_filters.png", Inches(6.7), Inches(2.0),
                  Inches(6.2), Inches(3.4))
    tb, tf = textbox(s, Inches(0.7), Inches(5.7), Inches(12.0), Inches(1.1))
    set_run(tf.paragraphs[0].add_run(),
            "No router beats the best single filter beyond noise; only the oracle (told the regime) gains.",
            16, CRIMSON, bold=True)
    footer_tagline(s)
    speaker_notes(s, "Here are the routing results, reported honestly. The soft router reaches 0.840 "
        "overall precision, versus 0.838 for naive consensus and 0.842 for the best fixed single "
        "filter (CorrFilter); hard routing is 0.834. A paired bootstrap puts the soft-router edge "
        "over the best single filter WITHIN NOISE (95% CI of the gap [-0.3, +0.4] points), so no "
        "router beats the best single filter. Only the oracle router, told the true regime, gains "
        "(0.854). This is exactly the non-dominance result (Theorem 1): the honest takeaway is that "
        "routing is feasible and the regimes are real, but inferring the regime well enough to route "
        "correctly is not yet a decisive unified solution. The paper stays a diagnostic taxonomy.")

    # ----- The most important scientific result
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s, CRIMSON)
    section_tag(s, "Section 6 · A taxonomy of consensus failure", CRIMSON)
    title_block(s, "The Most Important Scientific Result")
    bullets(s, [
        ("Dependence is not one phenomenon but a taxonomy of failure regimes.", 0, NAVY, True),
        ("Globally correlated co-failure: rho-bar RISES, CorrFilter helps.", 0, GREEN, True),
        ("Dominant vulnerable subgroup: rho-bar FALLS, CorrFilter hurts.", 0, AMBER, True),
        ("Same diagnostics, OPPOSITE directions: a taxonomy, not a severity continuum.", 0, CRIMSON, True),
    ], l=Inches(0.7), t=Inches(2.1), w=Inches(12.0), size=20)
    band = s.shapes.add_shape(1, Inches(0.7), Inches(5.0), Inches(11.9), Inches(1.4))
    band.fill.solid(); band.fill.fore_color.rgb = NAVY; band.line.fill.background()
    bt = band.text_frame; bt.vertical_anchor = MSO_ANCHOR.MIDDLE; bt.word_wrap = True
    bp = bt.paragraphs[0]; bp.alignment = PP_ALIGN.CENTER
    set_run(bp.add_run(), "The problem is not filtering but understanding dependence.", 20, WHITE, bold=True)
    p = bt.add_paragraph(); p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), "Theorem 1: no fixed filter dominates both regimes (non-dominance).", 18, RGBColor(0xF4, 0xC4, 0x30), bold=True)
    footer_tagline(s)
    speaker_notes(s, "If the audience takes one scientific point away, it is this. Dependence is not "
        "one phenomenon: globally correlated co-failure RAISES the mean correlation and CorrFilter "
        "helps, while a dominant vulnerable subgroup LOWERS the mean correlation (the bank looks more "
        "independent) yet consensus gets worse and CorrFilter hurts. The two regimes move the same "
        "diagnostics in opposite directions, so no scalar dependence-strength account can order them. "
        "That sign reversal is exactly why this is a taxonomy, not a severity continuum, and Theorem 1 "
        "makes the non-dominance formal: no fixed filter is optimal in both. The real problem is "
        "understanding the dependence structure, not designing one more filter.")

    # ----- NEW: Taxonomy, not a severity continuum (quantitative directions)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s, CRIMSON)
    section_tag(s, "Section 6 · A taxonomy of consensus failure", CRIMSON)
    title_block(s, "Taxonomy, Not a Severity Continuum")
    rows_data = [
        ("Regime", "rho-bar vs clean", "n_eff", "CorrFilter", "Best filter"),
        ("Weak (H2b)", "~ clean", "~ clean (3.3)", "neutral", "supermajority"),
        ("Global co-failure (CFI)", "RISES  (+0.11)", "FALLS  (2.6)", "helps", "CorrFilter"),
        ("Vulnerable subgroup (position)", "FALLS  (-0.08)", "RISES  (4.6)", "hurts", "bias-cluster"),
    ]
    row_colors = [NAVY, RGBColor(0x6B, 0x72, 0x80), GREEN, AMBER]
    gf = s.shapes.add_table(4, 5, Inches(0.6), Inches(1.95), Inches(12.1), Inches(2.3))
    tbl = gf.table
    widths = [Inches(3.6), Inches(2.5), Inches(2.2), Inches(1.9), Inches(1.9)]
    for c, wd in enumerate(widths):
        tbl.columns[c].width = wd
    for r, row in enumerate(rows_data):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = row_colors[r] if r == 0 else (LIGHT if c == 0 else WHITE)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            color = WHITE if r == 0 else (NAVY if c == 0 else row_colors[r])
            set_run(para.add_run(), val, 12.5 if r else 13, color, bold=(r == 0 or c == 0))
    band = s.shapes.add_shape(1, Inches(0.6), Inches(4.7), Inches(12.1), Inches(1.5))
    band.fill.solid(); band.fill.fore_color.rgb = NAVY; band.line.fill.background()
    bt = band.text_frame; bt.vertical_anchor = MSO_ANCHOR.MIDDLE; bt.word_wrap = True
    bp = bt.paragraphs[0]; bp.alignment = PP_ALIGN.CENTER
    set_run(bp.add_run(), "Global and subgroup move the SAME diagnostics in OPPOSITE directions.", 19, WHITE, bold=True)
    p = bt.add_paragraph(); p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), "A scalar dependence-strength continuum cannot produce this sign reversal.", 17, RGBColor(0xF4, 0xC4, 0x30), bold=True)
    footer_tagline(s)
    speaker_notes(s, "This is the quantitative rebuttal to the obvious reviewer objection, 'this is "
        "just a continuous dependence spectrum.' A scalar continuum predicts that more dependence "
        "always means worse consensus and a stronger need for correlation discounting. But the data "
        "break that: CFI raises rho-bar and CorrFilter helps, while position-aligned poisoning LOWERS "
        "rho-bar (n_eff rises from ~3.3 to 4.6) and yet consensus precision drops and CorrFilter "
        "HURTS. The two dangerous regimes sit on opposite sides of rho-bar but demand opposite "
        "treatments, so no monotone function of dependence strength can rank them. Numbers are the "
        "cached diagnostics from the taxonomy-validation table; all real, no new inference.")

    # ===================================================== SECTION 7
    content_slide(prs, "Section 7 · Remaining work",
        "Roadmap to completion",
        [
            ("All three diagnostic regimes are now measured (A, B, C).", 0, GREEN, True),
            ("Next: regime routing, can we auto-detect the regime and pick the mitigation?", 0, AMBER, True),
            ("Unify CorrFilter + Bias-Cluster into one adaptive framework.", 0, NAVY, True),
            ("C4: DPO / IPO downstream, do filtering gains translate into better policies?", 0, NAVY, True),
            "Helpfulness (AlpacaEval) and safety (PKU-SafeRLHF) evaluation.",
            "Paper write-up: lead with the taxonomy of consensus failure.",
            ("DPO is GPU-bound on one TITAN RTX, the long pole.", 0, CRIMSON, True),
        ],
        f_time,
        "The diagnostic phase is essentially complete: we have characterized all three regimes. The "
        "forward work is no longer 'run the next attack', it is (1) regime routing, automatically "
        "detecting which regime you are in and applying the matching mitigation; (2) unifying "
        "CorrFilter and bias-cluster filtering into one adaptive framework; and (3) C4, whether the "
        "filtering gains translate into better DPO/IPO policies, evaluated on AlpacaEval and "
        "PKU-SafeRLHF. DPO is GPU-bound on the single TITAN RTX, so it is the long pole.")

    # ===================================================== SECTION 8
    # ----- NEW SLIDE: What Did We Learn? (summary table) before conclusions
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s)
    section_tag(s, "Section 8 · Conclusions")
    title_block(s, "What Did We Learn?")
    learn = [
        ("Judge dependence exists", "Confirmed", GREEN),
        ("Consensus can fail badly", "Confirmed", GREEN),
        ("CorrFilter helps in some regimes", "Confirmed", GREEN),
        ("CorrFilter solves all failures", "Rejected", CRIMSON),
        ("Bias-aware filtering helps subgroup failures", "Confirmed", GREEN),
        ("Regimes are diagnosable (opposite signatures)", "Confirmed", GREEN),
        ("Router is feasible", "Confirmed", GREEN),
        ("A fixed filter dominates all regimes (Thm 1)", "Disproven", CRIMSON),
    ]
    y = Inches(1.62)
    rh = Inches(0.48)
    for finding, status, color in learn:
        cell = s.shapes.add_shape(1, Inches(0.7), y, Inches(9.4), rh)
        cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT; cell.line.color.rgb = WHITE
        ct = cell.text_frame; ct.vertical_anchor = MSO_ANCHOR.MIDDLE; ct.margin_left = Inches(0.2)
        set_run(ct.paragraphs[0].add_run(), finding, 14.5, NAVY, bold=True)
        chip = s.shapes.add_shape(5, Inches(10.3), y + Inches(0.03), Inches(2.3), Inches(0.46))
        chip.fill.solid(); chip.fill.fore_color.rgb = color; chip.line.fill.background()
        cf = chip.text_frame; cf.vertical_anchor = MSO_ANCHOR.MIDDLE
        cp = cf.paragraphs[0]; cp.alignment = PP_ALIGN.CENTER
        set_run(cp.add_run(), status, 13, WHITE, bold=True)
        y = Emu(int(y) + int(rh) + Inches(0.06))
    footer_tagline(s)
    speaker_notes(s, "This is the ledger of what is confirmed and what is rejected. Confirmed: "
        "judge dependence exists; consensus can fail badly; CorrFilter helps under globally "
        "correlated failures; vulnerable subgroups matter; bias-aware filtering partially recovers "
        "subgroup failures; label-free recovery is possible. Rejected: the original over-claim that "
        "CorrFilter solves ALL failures. Reporting a rejected line next to the confirmed ones is "
        "what makes the taxonomy credible.")

    # open problems (custom) -- position-aligned poisoning now completed
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s)
    section_tag(s, "Section 8 · Conclusions")
    title_block(s, "Open Problems")
    problems = [
        ("Open Problem 1", "Automatic regime detection at low signal strength."),
        ("Open Problem 2", "Unified filtering across regimes."),
        ("Open Problem 3", "Downstream DPO evaluation."),
        ("Open Problem 4", "Generalization to larger judge banks."),
    ]
    y = Inches(1.95)
    for name, body in problems:
        box = s.shapes.add_shape(1, Inches(0.7), y, Inches(11.9), Inches(1.0))
        box.fill.solid(); box.fill.fore_color.rgb = LIGHT
        box.line.color.rgb = AMBER; box.line.width = Pt(2)
        bt = box.text_frame; bt.word_wrap = True
        bt.margin_left = Inches(0.3); bt.vertical_anchor = MSO_ANCHOR.MIDDLE
        set_run(bt.paragraphs[0].add_run(), name + ":  ", 17, CRIMSON, bold=True)
        set_run(bt.paragraphs[0].add_run(), body, 16, NAVY, bold=True)
        y = Emu(int(y) + int(Inches(1.18)))
    footer_tagline(s)
    speaker_notes(s, "The open problems are now higher-level, since the diagnostic program is done. "
        "One: automatic regime detection at LOW signal strength, our router is reliable for weak and "
        "global regimes but only catches the subgroup regime at strong attack, so the low-signal "
        "case is open. Two: a unified filter that works across regimes rather than three separate "
        "tools. Three: the downstream DPO evaluation, the remaining gate for the alignment claim. "
        "Four: generalization to larger judge banks than our 10-judge setup.")

    # ===================================================== status table toward ICLR
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s); accent_bar(s)
    section_tag(s, "Section 8 · Conclusions")
    title_block(s, "Current Status Toward ICLR Submission")
    status_rows = [
        ("Measurement study", "Done", GREEN),
        ("Position bias analysis", "Done", GREEN),
        ("CFI", "Done", GREEN),
        ("H2b", "Done", GREEN),
        ("Position-Aligned Poisoning", "Done", GREEN),
        ("Bias Cluster Filtering", "Done", GREEN),
        ("Learned Bias Cluster", "Done", GREEN),
        ("Regime Router", "Done", GREEN),
        ("DPO / IPO", "Not started (gated)", AMBER),
        ("Paper writing", "In progress", ACCENT),
    ]
    y = Inches(1.6)
    rh = Inches(0.46)
    for comp, stat, color in status_rows:
        row = s.shapes.add_shape(1, Inches(0.7), y, Inches(7.4), rh)
        row.fill.solid(); row.fill.fore_color.rgb = LIGHT; row.line.color.rgb = WHITE
        tf = row.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE; tf.word_wrap = True
        tf.margin_left = Inches(0.2)
        set_run(tf.paragraphs[0].add_run(), comp, 14, NAVY, bold=True)
        chip = s.shapes.add_shape(5, Inches(8.3), y + Inches(0.03), Inches(4.3), Inches(0.44))
        chip.fill.solid(); chip.fill.fore_color.rgb = color; chip.line.fill.background()
        ctf = chip.text_frame; ctf.vertical_anchor = MSO_ANCHOR.MIDDLE
        cp = ctf.paragraphs[0]; cp.alignment = PP_ALIGN.CENTER
        set_run(cp.add_run(), stat, 12, WHITE, bold=True)
        y = Emu(int(y) + int(rh) + Inches(0.06))
    footer_tagline(s)
    speaker_notes(s, "One-glance status board. The whole diagnostic program is now done: measurement, "
        "position-bias analysis, CFI, H2b, position-aligned poisoning, bias-cluster filtering, the "
        "learned bias cluster, and the regime router. The only items not done are DPO/IPO downstream, "
        "which is not started and remains gated, and paper writing, in progress. The honest line: "
        "the diagnostic taxonomy and the routing study are complete; the downstream alignment impact "
        "remains to be proven.")

    # ===================================================== closing / final takeaway
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    bar = s.shapes.add_shape(1, 0, Inches(1.5), SW, Inches(0.06))
    bar.fill.solid(); bar.fill.fore_color.rgb = CRIMSON; bar.line.fill.background()
    tb, tf = textbox(s, Inches(0.7), Inches(0.5), Inches(12.0), Inches(1.0))
    set_run(tf.paragraphs[0].add_run(), "Final Takeaways", 34, WHITE, bold=True)
    tb, tf = textbox(s, Inches(0.7), Inches(1.7), Inches(12.0), Inches(5.6))
    lines = [
        ("Consensus is not reliability under dependence.", True),
        ("Dependence is a taxonomy of failure regimes, not a severity continuum.", True),
        ("CorrFilter helps under globally correlated co-failure.", False),
        ("Bias-cluster filtering helps under vulnerable-subgroup failures.", False),
        ("Regimes move the same diagnostics in opposite directions (Thm 1: non-dominance).", False),
        ("Understanding dependence matters more than designing one universal filter.", True),
    ]
    for i, (ln, strong) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(13)
        set_run(p.add_run(), "•  " + ln, 19 if strong else 17,
                RGBColor(0xF4, 0xC4, 0x30) if strong else RGBColor(0xE6, 0xE9, 0xEF),
                bold=strong)
    speaker_notes(s, "The final message. Consensus is not reliability. LLM judge banks exhibit "
        "multiple dependence regimes. CorrFilter helps under globally correlated co-failure; "
        "bias-cluster filtering helps under vulnerable-subgroup failures; no single filtering "
        "strategy dominates all regimes. The deepest point, and the one to leave the room with: "
        "understanding the dependence structure matters more than designing one more universal "
        "filter. That is the contribution of this work.")

    out = root / "reports" / "lab_meeting_deck.pptx"
    prs.save(str(out))
    print(f"Wrote {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
