#!/usr/bin/env python3
"""Generate HD architecture diagram for widemem."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib import font_manager
import numpy as np

FONTS = "/Users/radu/.claude/skills/canvas-design/canvas-fonts"
font_manager.fontManager.addfont(f"{FONTS}/GeistMono-Regular.ttf")
font_manager.fontManager.addfont(f"{FONTS}/GeistMono-Bold.ttf")
font_manager.fontManager.addfont(f"{FONTS}/Outfit-Regular.ttf")
font_manager.fontManager.addfont(f"{FONTS}/Outfit-Bold.ttf")
font_manager.fontManager.addfont(f"{FONTS}/JetBrainsMono-Regular.ttf")
font_manager.fontManager.addfont(f"{FONTS}/JetBrainsMono-Bold.ttf")

MONO = "Geist Mono"
SANS = "Geist Mono"
CODE = "Geist Mono"

BG = "#0D1117"
CARD_BG = "#161B22"
CARD_BORDER = "#30363D"
ACCENT_BLUE = "#58A6FF"
ACCENT_GREEN = "#3FB950"
ACCENT_PURPLE = "#BC8CFF"
ACCENT_ORANGE = "#D29922"
ACCENT_RED = "#F85149"
ACCENT_TEAL = "#39D2C0"
TEXT_PRIMARY = "#E6EDF3"
TEXT_SECONDARY = "#8B949E"
TEXT_DIM = "#484F58"
ARROW_COLOR = "#484F58"


def rounded_box(ax, x, y, w, h, label, sublabel=None, color=ACCENT_BLUE,
                bg=CARD_BG, border=CARD_BORDER, fontsize=11, icon=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                         facecolor=bg, edgecolor=border, linewidth=1.2,
                         zorder=2)
    ax.add_patch(box)

    accent = FancyBboxPatch((x, y + h - 0.004), w, 0.004,
                            boxstyle="round,pad=0.001",
                            facecolor=color, edgecolor="none", zorder=3)
    ax.add_patch(accent)

    cx, cy = x + w / 2, y + (h - 0.008) / 2
    if sublabel:
        cy += 0.012
    if icon:
        ax.text(cx, cy + 0.003, icon, fontsize=fontsize + 4,
                ha="center", va="center", color=color, fontfamily=SANS, zorder=4)
        cy -= 0.018
    ax.text(cx, cy, label, fontsize=fontsize, fontweight="bold",
            ha="center", va="center", color=TEXT_PRIMARY, fontfamily=SANS, zorder=4)
    if sublabel:
        ax.text(cx, cy - 0.018, sublabel, fontsize=8,
                ha="center", va="center", color=TEXT_SECONDARY, fontfamily=MONO, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR, style="-|>", lw=1.5):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color,
                            linewidth=lw, mutation_scale=12, zorder=1)
    ax.add_patch(arrow)


def section_label(ax, x, y, text, color=TEXT_DIM):
    ax.text(x, y, text, fontsize=7, fontweight="bold",
            ha="left", va="center", color=color, fontfamily=MONO, zorder=4)


fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Title
ax.text(0.5, 0.96, "widemem", fontsize=32, fontweight="bold",
        ha="center", va="center", color=TEXT_PRIMARY, fontfamily=SANS)
ax.text(0.5, 0.925, "Architecture Overview", fontsize=13,
        ha="center", va="center", color=TEXT_SECONDARY, fontfamily=MONO)

# Horizontal rule
ax.plot([0.1, 0.9], [0.905, 0.905], color=CARD_BORDER, linewidth=0.8, zorder=1)

# ── INPUT ──
section_label(ax, 0.06, 0.88, "INPUT")
rounded_box(ax, 0.06, 0.83, 0.12, 0.04, "Text Input", "memory.add()", ACCENT_BLUE)
rounded_box(ax, 0.20, 0.83, 0.12, 0.04, "Batch Input", "add_batch()", ACCENT_BLUE)

# Arrow down from input
draw_arrow(ax, 0.18, 0.83, 0.18, 0.795, ACCENT_BLUE)

# ── EXTRACTION ──
section_label(ax, 0.06, 0.79, "EXTRACTION")
rounded_box(ax, 0.06, 0.735, 0.26, 0.045, "LLM Fact Extraction",
            "importance 1-10 · YMYL detection", ACCENT_GREEN)

# Arrow down
draw_arrow(ax, 0.18, 0.735, 0.18, 0.695, ACCENT_GREEN)

# ── CONFLICT RESOLUTION ──
section_label(ax, 0.06, 0.69, "CONFLICT RESOLUTION")
rounded_box(ax, 0.06, 0.625, 0.26, 0.055, "Batch Resolver",
            "ADD · UPDATE · DELETE · NONE", ACCENT_ORANGE)

# Small boxes for conflict actions
actions_y = 0.575
action_w = 0.058
gap = 0.008
start_x = 0.06
for i, (label, color) in enumerate([("ADD", ACCENT_GREEN), ("UPDATE", ACCENT_BLUE),
                                      ("DELETE", ACCENT_RED), ("SKIP", TEXT_DIM)]):
    bx = start_x + i * (action_w + gap)
    box = FancyBboxPatch((bx, actions_y), action_w, 0.025,
                         boxstyle="round,pad=0.005",
                         facecolor=BG, edgecolor=color, linewidth=1, zorder=2)
    ax.add_patch(box)
    ax.text(bx + action_w / 2, actions_y + 0.0125, label, fontsize=7,
            fontweight="bold", ha="center", va="center", color=color,
            fontfamily=CODE, zorder=4)

# Arrow down to storage
draw_arrow(ax, 0.18, 0.575, 0.18, 0.535, ACCENT_ORANGE)

# ── STORAGE ──
section_label(ax, 0.06, 0.53, "STORAGE")

# FAISS box
rounded_box(ax, 0.06, 0.44, 0.12, 0.075, "FAISS", "vectors · embeddings", ACCENT_PURPLE)
# SQLite box
rounded_box(ax, 0.20, 0.44, 0.12, 0.075, "SQLite", "history · audit trail", ACCENT_PURPLE)

# ── HIERARCHY (right side) ──
section_label(ax, 0.40, 0.88, "HIERARCHICAL MEMORY")

tier_x = 0.40
tier_w = 0.22
rounded_box(ax, tier_x, 0.82, tier_w, 0.045, "Themes", "high-level patterns", ACCENT_TEAL)
rounded_box(ax, tier_x, 0.76, tier_w, 0.045, "Summaries", "grouped related facts", ACCENT_TEAL)
rounded_box(ax, tier_x, 0.70, tier_w, 0.045, "Facts", "atomic memory units", ACCENT_TEAL)

# Arrows between tiers
draw_arrow(ax, tier_x + tier_w / 2, 0.76, tier_x + tier_w / 2, 0.795, ACCENT_TEAL, "<|-|>")
draw_arrow(ax, tier_x + tier_w / 2, 0.70, tier_x + tier_w / 2, 0.735, ACCENT_TEAL, "<|-|>")

# ── SCORING (right side) ──
section_label(ax, 0.40, 0.66, "COMBINED SCORING")

score_x = 0.40
score_w = 0.22
score_items = [
    ("Similarity", "vector cosine distance", ACCENT_BLUE),
    ("Importance", "1-10 LLM rating", ACCENT_GREEN),
    ("Recency", "time decay function", ACCENT_ORANGE),
]
for i, (label, sub, color) in enumerate(score_items):
    sy = 0.595 - i * 0.055
    box = FancyBboxPatch((score_x, sy), score_w, 0.042,
                         boxstyle="round,pad=0.008",
                         facecolor=CARD_BG, edgecolor=CARD_BORDER,
                         linewidth=1, zorder=2)
    ax.add_patch(box)
    # Color accent left edge
    accent = FancyBboxPatch((score_x, sy), 0.005, 0.042,
                            boxstyle="round,pad=0.001",
                            facecolor=color, edgecolor="none", zorder=3)
    ax.add_patch(accent)
    ax.text(score_x + 0.015, sy + 0.025, label, fontsize=9, fontweight="bold",
            ha="left", va="center", color=TEXT_PRIMARY, fontfamily=SANS, zorder=4)
    ax.text(score_x + 0.015, sy + 0.01, sub, fontsize=7,
            ha="left", va="center", color=TEXT_SECONDARY, fontfamily=MONO, zorder=4)

# Formula box
formula_y = 0.42
formula_box = FancyBboxPatch((score_x, formula_y), score_w, 0.05,
                             boxstyle="round,pad=0.01",
                             facecolor="#1A1F2B", edgecolor=ACCENT_BLUE,
                             linewidth=1.2, zorder=2)
ax.add_patch(formula_box)
ax.text(score_x + score_w / 2, formula_y + 0.031, "final_score =", fontsize=8,
        ha="center", va="center", color=TEXT_SECONDARY, fontfamily=CODE, zorder=4)
ax.text(score_x + score_w / 2, formula_y + 0.015, "sim × w₁ + imp × w₂ + rec × w₃",
        fontsize=9, fontweight="bold",
        ha="center", va="center", color=ACCENT_BLUE, fontfamily=CODE, zorder=4)

# ── RETRIEVAL (far right) ──
section_label(ax, 0.70, 0.88, "RETRIEVAL")

ret_x = 0.70
ret_w = 0.24
rounded_box(ax, ret_x, 0.81, ret_w, 0.05, "Query Router",
            "routes to appropriate tier", ACCENT_BLUE)
rounded_box(ax, ret_x, 0.74, ret_w, 0.05, "Temporal Filter",
            "time_after · time_before · TTL", ACCENT_ORANGE)
rounded_box(ax, ret_x, 0.67, ret_w, 0.05, "Active Retrieval",
            "contradiction · ambiguity detection", ACCENT_RED)
rounded_box(ax, ret_x, 0.60, ret_w, 0.05, "Score & Rank",
            "combined scoring + reranking", ACCENT_GREEN)

# Arrows in retrieval pipeline
mid_ret = ret_x + ret_w / 2
draw_arrow(ax, mid_ret, 0.81, mid_ret, 0.79, ACCENT_BLUE)
draw_arrow(ax, mid_ret, 0.74, mid_ret, 0.72, ACCENT_ORANGE)
draw_arrow(ax, mid_ret, 0.67, mid_ret, 0.65, ACCENT_RED)

# ── OUTPUT ──
section_label(ax, 0.70, 0.56, "OUTPUT")
rounded_box(ax, ret_x, 0.49, ret_w, 0.055, "Search Results",
            "score breakdown · clarifications", ACCENT_GREEN)

draw_arrow(ax, mid_ret, 0.60, mid_ret, 0.545, ACCENT_GREEN)

# ── PROVIDERS (bottom) ──
section_label(ax, 0.06, 0.39, "PROVIDERS")

prov_y = 0.33
prov_w = 0.14
prov_h = 0.045
providers = [
    ("LLM Providers", "OpenAI · Claude · Ollama", ACCENT_BLUE),
    ("Embeddings", "OpenAI · sentence-transformers", ACCENT_GREEN),
    ("Vector Stores", "FAISS · Qdrant", ACCENT_PURPLE),
]
for i, (label, sub, color) in enumerate(providers):
    px = 0.06 + i * (prov_w + 0.015)
    rounded_box(ax, px, prov_y, prov_w, prov_h, label, sub, color)

# ── FEATURES (bottom right) ──
section_label(ax, 0.54, 0.40, "FEATURES")

feat_y = 0.30
feat_items = [
    "Retry/Backoff", "Memory TTL", "Batch Add",
    "Export/Import", "YMYL Safety", "Thread Safe",
]
feat_w = 0.076
feat_h = 0.035
cols = 3
for i, feat in enumerate(feat_items):
    row = i // cols
    col = i % cols
    fx = 0.54 + col * (feat_w + 0.008)
    fy = feat_y + (1 - row) * (feat_h + 0.006)
    box = FancyBboxPatch((fx, fy), feat_w, feat_h,
                         boxstyle="round,pad=0.005",
                         facecolor=CARD_BG, edgecolor=CARD_BORDER,
                         linewidth=0.8, zorder=2)
    ax.add_patch(box)
    ax.text(fx + feat_w / 2, fy + feat_h / 2, feat, fontsize=7,
            ha="center", va="center", color=TEXT_SECONDARY,
            fontfamily=MONO, fontweight="bold", zorder=4)

# ── CONNECTION ARROWS between sections ──
# Extraction → Hierarchy (facts feed into hierarchy)
draw_arrow(ax, 0.32, 0.755, 0.40, 0.72, TEXT_DIM, "-|>", 1.0)

# Storage → Retrieval
draw_arrow(ax, 0.32, 0.475, 0.70, 0.635, TEXT_DIM, "-|>", 1.0)

# Hierarchy → Retrieval
draw_arrow(ax, 0.62, 0.78, 0.70, 0.835, TEXT_DIM, "-|>", 1.0)

# Scoring → Retrieval
draw_arrow(ax, 0.62, 0.45, 0.70, 0.52, TEXT_DIM, "-|>", 1.0)

# ── Version badge ──
ax.text(0.94, 0.04, "v1.3.0", fontsize=8, ha="right", va="center",
        color=TEXT_DIM, fontfamily=CODE, zorder=4)
ax.text(0.06, 0.04, "widemem.ai", fontsize=8, ha="left", va="center",
        color=TEXT_DIM, fontfamily=CODE, zorder=4)

plt.tight_layout(pad=0.5)
plt.savefig("/Users/radu/widemem/docs/architecture.png",
            facecolor=BG, edgecolor="none", bbox_inches="tight", dpi=200)
print("Saved: docs/architecture.png")
