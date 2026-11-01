"""Figure 1: EPS workflow (graphical abstract). Deterministic vector output.

Usage:  python figures/make_fig1_eps_workflow.py
Writes figures/Fig1_EPS_workflow.{pdf,svg,png} at 178 mm x 100 mm (PNG 600 dpi).

Worked example = the real QMOF query of Figure 2 (seed 1, contradiction threshold 1.0 eV):
MOF E has the largest HSE06 band gap (4.70 eV, DFT) and is contested by a cross-fitted
ML prediction (3.12 eV); MOF B has 2.85 eV; MOFs A, C, D and F have no HSE06 value in the
snapshot and are covered by closed-world exclusion records.
The LLM query parser and an LLM reader are shown as interfaces; the study itself uses
formal queries and deterministic readers (see caption).
"""
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Polygon

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig1_EPS_workflow")
W, H = 178.0, 100.0
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
    "svg.hashsalt": "eps-fig1", "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
})
COL = dict(M="#3B4252", C="#1B6CA8", E="#2A9D8F", K="#E09F3E", Q="#7B4FB0", P="#7A7F87", D="#9C5B2E",
           X="#B23A2E", red="#C0392B", pale="#E8EAED", paleE="#D0D4D9", ink="#1F2328", mute="#555B66",
           panel="#F5F6F8", ok="#2A7D4F")
FS, FSH, FSL = 8, 10.5, 9

fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, fc="white", ec=COL["paleE"], lw=0.8, r=1.6, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def T(x, y, s, size=FS, weight="normal", color=COL["ink"], ha="left", va="center", style="normal", z=6, rot=0):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, style=style, zorder=z, rotation=rot)


def line(xs, ys, color=COL["ink"], lw=1.1, dashed=False, z=3):
    ax.plot(xs, ys, color=color, lw=lw, ls=(0, (2.4, 1.6)) if dashed else "-", zorder=z, solid_capstyle="butt")


def arrow(p0, p1, color=COL["ink"], lw=1.1, dashed=False, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=1.6,head_width=1.0", mutation_scale=4.5, color=color,
                                 lw=lw, ls=(0, (2.4, 1.6)) if dashed else "-", zorder=z, shrinkA=0, shrinkB=0))


def chip(x, y, t, r=1.35, square=False, z=6, alpha=1.0, letter=True):
    if square:
        ax.add_patch(Rectangle((x - r, y - r), 2 * r, 2 * r, fc=COL[t], ec="white", lw=0.4, zorder=z, alpha=alpha))
    else:
        ax.add_patch(Circle((x, y), r, fc=COL[t], ec="white", lw=0.4, zorder=z, alpha=alpha))
    if letter:
        T(x, y - 0.05, t, size=6.3, weight="bold", color="white", ha="center", z=z + 1)


def pale(x, y, r=1.0):
    ax.add_patch(Circle((x, y), r, fc=COL["pale"], ec=COL["paleE"], lw=0.4, zorder=4))


def check(x, y, s=1.0, color=COL["ok"]):
    ax.plot([x - 0.9 * s, x - 0.25 * s, x + 1.0 * s], [y, y - 0.7 * s, y + 0.9 * s], color=color, lw=1.2, zorder=7,
            solid_capstyle="round", solid_joinstyle="round")


def warn(x, y, s=1.5, color=COL["red"]):
    ax.add_patch(Polygon([[x - s, y - 0.9 * s], [x + s, y - 0.9 * s], [x, y + 0.95 * s]], closed=True, fc=color, ec="none", zorder=7))
    T(x, y - 0.25 * s, "!", size=6.2, weight="bold", color="white", ha="center", z=8)


# ---------------------------------------------------------------- panels
PA, PB, PC = (2, 13.5, 46, 84.5), (50, 13.5, 70, 84.5), (122, 13.5, 54, 84.5)
for (x, y, w, h) in (PA, PB, PC):
    box(x, y, w, h, fc=COL["panel"], ec="none", r=2.5, z=0)
for (x, _, _, _), s in zip((PA, PB, PC), ("a   Ask and formalize", "b   Preserve complete audit units", "c   Answer and audit")):
    T(x + 1.8, 94.6, s, size=FSH, weight="bold")

# ---------------------------------------------------------------- panel a
sx, sy = 6.6, 79.5                                   # scientist silhouette
ax.add_patch(Circle((sx, sy + 4.6), 2.0, fc=COL["M"], ec="none", zorder=5))
ax.add_patch(FancyBboxPatch((sx - 3.1, sy - 3.2), 6.2, 5.6, boxstyle="round,pad=0,rounding_size=2.3", fc=COL["M"], ec="none", zorder=5))
bx, by, bw, bh = 11.8, 74.5, 35.4, 15.5               # question bubble
box(bx, by, bw, bh, fc="white", ec=COL["C"], lw=1.1, r=2.2, z=4)
ax.add_patch(Polygon([[bx + 0.3, by + 7.2], [bx - 2.6, by + 5.0], [bx + 0.3, by + 4.2]], closed=True, fc="white", ec=COL["C"], lw=1.1, zorder=4))
ax.add_patch(Rectangle((bx - 0.05, by + 4.35), 0.8, 2.7, fc="white", ec="none", zorder=4.6))
T(bx + bw / 2, by + bh / 2, "“Which shortlisted MOF\nhas the largest HSE06\nband gap?”", size=8.6, weight="bold", color=COL["C"], ha="center")
T(4.0, 68.6, "Candidates C", size=FS, color=COL["mute"])
for i, L in enumerate("ABCDEF"):
    cx = 23.5 + i * 3.95
    ax.add_patch(Circle((cx, 68.6), 1.55, fc=COL["M"], ec="white", lw=0.4, zorder=5))
    T(cx, 68.55, L, size=6.6, weight="bold", color="white", ha="center", z=7)
box(6, 52.6, 38, 11.0, fc="white", ec=COL["mute"], lw=0.8)
T(25, 61.0, "LLM query parser", size=FS, weight="bold", ha="center")
T(25, 57.9, "optional natural-language", size=FS, color=COL["mute"], ha="center")
T(25, 55.0, "interface", size=FS, color=COL["mute"], ha="center")
box(6, 44.0, 38, 6.4, fc="white", ec=COL["mute"], lw=0.8)
T(25, 47.2, "Schema validation", size=FS, weight="bold", ha="center")
arrow((25, 66.3), (25, 63.7)); arrow((25, 52.5), (25, 50.5))
box(4, 16.5, 42, 23.5, fc="white", ec=COL["ink"], lw=1.1, r=2)  # formal query
T(25, 35.6, r"$q = (\mathrm{argmax},\ \mathrm{HSE06},\ C,\ k)$", size=FSL, weight="bold", ha="center")
for j, (k_, v_) in enumerate((("operation", "argmax"), ("condition", "HSE06"), ("candidates", "C (MOFs A–F)"), ("node budget", "k"))):
    T(6.5, 30.6 - j * 3.5, k_, size=FS, color=COL["mute"])
    T(23.5, 30.6 - j * 3.5, v_, size=FS)
arrow((25, 43.9), (25, 40.1))

# ---------------------------------------------------------------- panel b
T(59.6, 57.4, "query\nneighbourhood\nN(q)", size=FS, color=COL["mute"], ha="center")
gx, gy = 60, 71
outer = [(gx + 6.3 * math.cos(2 * math.pi * i / 10 + 0.3), gy + 8.0 * math.sin(2 * math.pi * i / 10 + 0.3)) for i in range(10)]
inner = [(gx - 2.2, gy + 2.6), (gx + 2.4, gy + 3.0), (gx - 1.8, gy - 3.0), (gx + 2.3, gy - 2.7), (gx - 3.6, gy - 0.2)]
for (x0, y0) in outer:
    line([x0, gx], [y0, gy], color=COL["paleE"], lw=0.5)
for (x0, y0) in outer:
    pale(x0, y0, 0.9)
for (x0, y0), t in zip(inner, "CKEQD"):
    chip(x0, y0, t, r=1.0, alpha=0.5, letter=False)
chip(gx, gy, "P", r=1.15, alpha=0.5, letter=False)
# entry of q (from panel a) and arrow to units
arrow((46.1, 28.0), (60, 28.0), lw=1.5); line([60, 60], [28.0, 50.0], lw=1.5)
arrow((60, 49.9), (60, 52.0), lw=1.5)
T(61.3, 36, "q, k", size=FS, color=COL["mute"])
arrow((66.8, 71), (69.6, 71), lw=1.2)
# canonical audit units
cx0, cw = 70, 34
T(cx0 + cw / 2, 89.4, r"Query-specific audit units $U(q)$", size=FS, color=COL["mute"], ha="center")
box(cx0, 74.5, cw, 12.0, fc="white", ec=COL["C"], lw=1.0)
T(cx0 + 1.6, 83.8, "Support unit", size=FS, weight="bold", color=COL["C"])
for i, t in enumerate("MCEKQPD"):
    chip(cx0 + 3.0 + i * 3.45, 78.4, t)
box(cx0, 59.5, cw, 12.0, fc="white", ec=COL["red"], lw=1.0)
T(cx0 + 1.6, 68.8, "Contradiction unit", size=FS, weight="bold", color=COL["red"])
chip(cx0 + 3.0, 63.6, "C"); chip(cx0 + 10.4, 63.6, "C")
line([cx0 + 4.4, cx0 + 9.0], [63.6, 63.6], color=COL["red"], lw=1.1, dashed=True, z=5)
for i, t in enumerate("EKQPD"):
    chip(cx0 + 14.0 + i * 3.45, 63.6, t)
box(cx0, 44.0, cw, 13.0, fc="white", ec=COL["X"], lw=1.0)
T(cx0 + 1.6, 54.3, "Exclusion record", size=FS, weight="bold", color=COL["X"])
chip(cx0 + 3.0, 50.4, "M"); chip(cx0 + 6.5, 50.4, "X", square=True)
T(cx0 + 9.0, 50.4, "no HSE06 value in", size=FS)
T(cx0 + 1.6, 46.6, "snapshot (closed world)", size=FS, color=COL["mute"])
box(cx0, 29.5, cw, 8.6, fc=COL["pale"], ec=COL["paleE"], lw=0.8, ls=(0, (2.2, 1.5)))
T(cx0 + cw / 2, 35.4, "unit that does not fit", size=FS, color=COL["mute"], ha="center")
T(cx0 + cw / 2, 32.0, "omitted, then reported", size=FS, color=COL["mute"], ha="center", style="italic")
# whole-unit budget gate
gx0, gw, gy0, gh = 107.5, 10.5, 44.0, 45.0
box(gx0, gy0, gw, gh, fc=COL["ink"], ec="none", r=2.0, z=2)
T(gx0 + gw / 2, gy0 + gh - 6.5, r"$|S|\leq k$", size=FSL, weight="bold", color="white", ha="center")
T(gx0 + gw / 2, gy0 + gh / 2 - 4.5, "whole-unit budget gate", size=FS, weight="bold", color="white", ha="center", rot=90)
for yy in (80.5, 65.5, 50.5):  # inputs
    arrow((cx0 + cw + 0.2, yy), (gx0 - 0.2, yy))
T(PB[0] + PB[2] / 2, 23.2, "Select complete units; never split an audit unit.", size=FS, weight="bold", ha="center")
T(PB[0] + PB[2] / 2, 18.0, "Compression occurs at audit-unit boundaries.", size=FSL, weight="bold", color=COL["C"], ha="center")

# ---------------------------------------------------------------- panel c
# compact subgraph S (left) and audit certificate (right), side by side
box(124.5, 57.0, 17.0, 34.5, fc="white", ec=COL["C"], lw=1.0)
T(133.0, 87.6, "Compact\nsubgraph S", size=FS, weight="bold", ha="center")
sp = {"M": (127.6, 78.5), "C": (132.2, 80.6), "E": (137.6, 81.8), "K": (131.2, 74.6), "Q": (137.2, 75.9), "P": (138.8, 70.2), "D": (133.2, 68.6)}
for a_, b_ in (("M", "C"), ("C", "E"), ("C", "K"), ("C", "Q"), ("E", "P"), ("E", "D")):
    line(*zip(sp[a_], sp[b_]), color=COL["mute"], lw=0.6, z=4)
for t, (x0, y0) in sp.items():
    chip(x0, y0, t, r=1.25)
T(133.0, 62.6, "+ other\nunits", size=FS, color=COL["mute"], ha="center")
box(143.3, 57.0, 32.0, 34.5, fc="white", ec=COL["ok"], lw=1.0)
T(144.8, 88.6, "Audit certificate", size=FS, weight="bold", color=COL["ok"])
items = ["R1 Support", "R2 Comparability", "R3 Qualification", "R4 Contradiction", "R5 Replay information", "Exclusion coverage"]
for i, it in enumerate(items):
    yy = 84.6 - i * 3.55
    check(145.9, yy); T(147.6, yy, it, size=FS)
T(144.8, 62.0, "per unit: selected\nwhole or omitted", size=FS, color=COL["mute"])
arrow((118.1, 80.5), (124.4, 80.5), lw=1.5)                 # gate -> S
arrow((141.6, 73.8), (143.2, 73.8), lw=1.1)                 # S -> certificate (certify)
# outcome 1: all required units fit
T(146.5, 53.0, "all required\nunits fit", size=FS, weight="bold", color=COL["ok"], ha="center")
box(124.5, 40.0, 49.5, 9.0, fc="white", ec=COL["ink"], lw=1.1)
T(149.25, 46.3, "Grounded reader / LLM", size=FS, weight="bold", ha="center")
T(149.25, 42.8, "uses only S and the certificate", size=FS, color=COL["mute"], ha="center")
arrow((133.0, 56.9), (133.0, 49.1)); arrow((159.0, 56.9), (159.0, 49.1))
box(124.5, 25.0, 49.5, 13.2, fc="white", ec=COL["C"], lw=1.0)
T(126.2, 36.2, "Example: real query of Figure 2", size=FS, color=COL["mute"], style="italic")
T(126.2, 33.0, "MOF E has the largest HSE06", size=FS, weight="bold")
T(126.2, 29.9, "band gap: 4.70 eV (DFT).", size=FS, weight="bold")
warn(127.6, 26.9, s=1.1); T(129.6, 26.9, "Contested by an ML prediction", size=FS, color=COL["red"])
arrow((149.25, 39.9), (149.25, 38.3))
T(125.5, 22.0, "Audit:", size=FS, weight="bold", color=COL["mute"]); T(135.0, 22.0, "inspect sources →", size=FS, color=COL["mute"])
T(125.5, 18.8, "check conditions → trace provenance", size=FS, color=COL["mute"])
T(125.5, 15.6, "→ inspect replay information", size=FS, color=COL["mute"])
# outcome 2: an essential unit is omitted
box(124.5, 0.8, 49.5, 11.4, fc="white", ec=COL["X"], lw=1.1, r=1.8)
T(149.25, 9.9, "essential unit omitted", size=FS, weight="bold", color=COL["X"], ha="center")
T(149.25, 6.6, "→ unmet requirements →", size=FS, color=COL["X"], ha="center")
T(149.25, 3.3, "qualified answer or abstention", size=FS, weight="bold", color=COL["X"], ha="center")
line([cx0 + cw + 0.2, 121.0, 121.0], [33.8, 33.8, 6.6], color=COL["X"], lw=0.9, dashed=True)
arrow((121.0, 6.6), (124.4, 6.6), color=COL["X"], dashed=True, lw=0.9)

# ---------------------------------------------------------------- legend (once), positions from measured text widths
REND = fig.canvas.get_renderer()
def TW(x, y, lab):
    t = ax.text(x, y, lab, fontsize=FS, color=COL["ink"], ha="left", va="center", zorder=6)
    return t.get_window_extent(renderer=REND).width / fig.dpi * 25.4      # width in mm (= data units)
y1, y2, GAP = 9.2, 4.6, 2.5
x = 3.5
for t, lab in (("M", "material"), ("C", "claim"), ("E", "evidence"), ("K", "condition"), ("Q", "qualification"), ("P", "provenance")):
    chip(x, y1, t, r=1.2); x += 1.8 + TW(x + 1.8, y1, lab) + GAP
pale(x, y1, 1.1); TW(x + 1.8, y1, "not retained")
x = 3.5
for t, lab in (("D", "derivation"), ("X", "exclusion record")):
    chip(x, y2, t, r=1.2, square=(t == "X")); x += 1.8 + TW(x + 1.8, y2, lab) + GAP
x -= 1.2
line([x, x + 4.0], [y2, y2], color=COL["red"], lw=1.1, dashed=True); x += 5.0 + TW(x + 5.0, y2, "contradiction") + GAP
line([x, x + 4.0], [y2, y2], color=COL["ink"], lw=1.2); x += 5.0 + TW(x + 5.0, y2, "information flow") + GAP
line([x, x + 4.0], [y2, y2], color=COL["mute"], lw=1.0, dashed=True); x += 5.0 + TW(x + 5.0, y2, "audit / reporting")
assert x < 119.5, f"legend too wide: {x:.1f} mm"

for ext, kw in (("pdf", {"metadata": {"CreationDate": None}}), ("svg", {"metadata": {"Date": None}}), ("png", {"dpi": 600})):
    fig.savefig(f"{OUT}.{ext}", **kw)
print("wrote", OUT + ".{pdf,svg,png}")
