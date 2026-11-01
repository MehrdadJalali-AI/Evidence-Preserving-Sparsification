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
        T(x, y - 0.05, t, size=min(8.0, 6.3 * r / 1.35), weight="bold", color="white", ha="center", z=z + 1)


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


def G(pos, edges, dashed=(), r=1.25, sq=("X",), lw=0.7):
    for a_, b_ in edges:
        line(*zip(pos[a_], pos[b_]), color=COL["mute"], lw=lw, z=4)
    for a_, b_ in dashed:
        line(*zip(pos[a_], pos[b_]), color=COL["red"], lw=1.0, dashed=True, z=4)
    for k_, (x0, y0) in pos.items():
        chip(x0, y0, k_[0], r=r, square=k_[0] in sq)

# ---------------------------------------------------------------- panel a
sx, sy = 6.6, 79.5
ax.add_patch(Circle((sx, sy + 4.6), 2.0, fc=COL["M"], ec="none", zorder=5))
ax.add_patch(FancyBboxPatch((sx - 3.1, sy - 3.2), 6.2, 5.6, boxstyle="round,pad=0,rounding_size=2.3", fc=COL["M"], ec="none", zorder=5))
bx, by, bw, bh = 11.8, 74.5, 35.4, 15.5
box(bx, by, bw, bh, fc="white", ec=COL["C"], lw=1.1, r=2.2, z=4)
ax.add_patch(Polygon([[bx + 0.3, by + 7.2], [bx - 2.6, by + 5.0], [bx + 0.3, by + 4.2]], closed=True, fc="white", ec=COL["C"], lw=1.1, zorder=4))
ax.add_patch(Rectangle((bx - 0.05, by + 4.35), 0.8, 2.7, fc="white", ec="none", zorder=4.6))
T(bx + bw / 2, by + bh / 2, "“Which shortlisted MOF\nhas the largest HSE06\nband gap?”", size=8.6, weight="bold", color=COL["C"], ha="center")
T(4.0, 68.6, "Candidates C", size=FS, color=COL["mute"])
for i, L in enumerate("ABCDEF"):
    cx = 23.5 + i * 3.95
    ax.add_patch(Circle((cx, 68.6), 1.55, fc=COL["M"], ec="white", lw=0.4, zorder=5))
    T(cx, 68.55, L, size=6.6, weight="bold", color="white", ha="center", z=7)
box(6, 50.5, 38, 10.5, fc="white", ec=COL["mute"], lw=0.8)
T(25, 57.6, "Parse + schema check", size=FS, weight="bold", ha="center")
T(25, 53.8, "(LLM parser optional)", size=FS, color=COL["mute"], ha="center")
arrow((25, 66.3), (25, 61.1)); arrow((25, 50.4), (25, 42.1))
box(4, 21.0, 42, 21.0, fc="white", ec=COL["ink"], lw=1.1, r=2)
T(25, 35.0, r"$q = (\mathrm{argmax},\ \mathrm{HSE06},\ C,\ k)$", size=FSL, weight="bold", ha="center")
T(25, 27.0, "operation · condition ·\ncandidates · node budget", size=FS, color=COL["mute"], ha="center")

# ---------------------------------------------------------------- panel b
gx, gy = 60.3, 72.5                                   # query neighbourhood graph (larger, typed)
ring = [(gx + 9.0 * math.cos(2 * math.pi * i / 12 + 0.2), gy + 13.0 * math.sin(2 * math.pi * i / 12 + 0.2)) for i in range(12)]
core = {"M": (gx, gy), "C1": (gx - 4.2, gy + 4.6), "C2": (gx + 4.4, gy + 4.2), "E": (gx - 5.6, gy - 1.6),
        "K": (gx - 1.2, gy + 9.0), "Q": (gx + 5.4, gy - 2.4), "P": (gx + 1.2, gy - 6.8), "D": (gx - 3.6, gy - 7.0)}
for (x0, y0) in ring:
    line([x0, gx], [y0, gy], color=COL["paleE"], lw=0.5)
for (x0, y0) in ring:
    pale(x0, y0, 1.05)
G(core, [("M", "C1"), ("M", "C2"), ("C1", "E"), ("C1", "K"), ("C2", "Q"), ("E", "D"), ("D", "P"), ("Q", "P")], dashed=[("C1", "C2")], r=1.55)
T(gx, 56.6, "N(q)", size=FS, color=COL["mute"], ha="center")
arrow((46.1, 28.0), (60, 28.0), lw=1.5); line([60, 60], [28.0, 52.0], lw=1.5); arrow((60, 51.9), (60, 54.4), lw=1.5)
T(61.3, 36, "q, k", size=FS, color=COL["mute"])
arrow((69.0, 73), (70.0, 73), lw=1.2)
cx0, cw = 70.6, 34
# support unit as a small graph
box(cx0, 71.5, cw, 17.0, fc="white", ec=COL["C"], lw=1.0)
T(cx0 + 1.6, 86.0, "Support unit", size=FS, weight="bold", color=COL["C"])
G({"M": (cx0 + 3.2, 78.5), "C": (cx0 + 9.5, 78.5), "K": (cx0 + 9.5, 82.4), "Q": (cx0 + 9.5, 74.6), "E": (cx0 + 17, 78.5),
   "P": (cx0 + 24.5, 81.9), "D": (cx0 + 24.5, 75.1)},
  [("M", "C"), ("C", "K"), ("C", "Q"), ("C", "E"), ("E", "P"), ("E", "D"), ("D", "P")], r=1.55)
# contradiction unit
box(cx0, 54.0, cw, 16.0, fc="white", ec=COL["red"], lw=1.0)
T(cx0 + 1.6, 67.6, "Contradiction unit", size=FS, weight="bold", color=COL["red"])
G({"C0": (cx0 + 3.2, 61.0), "C": (cx0 + 9.5, 61.0), "K": (cx0 + 9.5, 64.7), "Q": (cx0 + 9.5, 57.3), "E": (cx0 + 17, 61.0),
   "P": (cx0 + 24.5, 64.3), "D": (cx0 + 24.5, 57.7)},
  [("C", "K"), ("C", "Q"), ("C", "E"), ("E", "P"), ("E", "D"), ("D", "P")], dashed=[("C0", "C")], r=1.55)
# exclusion record
box(cx0, 41.5, cw, 10.5, fc="white", ec=COL["X"], lw=1.0)
T(cx0 + 1.6, 49.6, "Exclusion record", size=FS, weight="bold", color=COL["X"])
G({"M": (cx0 + 3.2, 45.0), "X": (cx0 + 9.5, 45.0), "D": (cx0 + 17, 45.0), "P": (cx0 + 24.5, 45.0)},
  [("M", "X"), ("X", "D"), ("D", "P")], r=1.55)
box(cx0, 29.5, cw, 8.6, fc=COL["pale"], ec=COL["paleE"], lw=0.8, ls=(0, (2.2, 1.5)))
T(cx0 + cw / 2, 35.4, "unit that does not fit", size=FS, color=COL["mute"], ha="center")
T(cx0 + cw / 2, 32.0, "omitted, then reported", size=FS, color=COL["mute"], ha="center", style="italic")
gx0, gw, gy0, gh = 107.8, 10.2, 41.5, 47.0
box(gx0, gy0, gw, gh, fc=COL["ink"], ec="none", r=2.0, z=2)
T(gx0 + gw / 2, gy0 + gh - 6.0, r"$|S|\leq k$", size=FSL, weight="bold", color="white", ha="center")
T(gx0 + gw / 2, gy0 + gh / 2 - 4.5, "whole-unit budget gate", size=FS, weight="bold", color="white", ha="center", rot=90)
for yy in (80.0, 62.0, 46.7):
    arrow((cx0 + cw + 0.2, yy), (gx0 - 0.2, yy))
T(PB[0] + PB[2] / 2, 19.5, "Compression occurs at audit-unit boundaries.", size=FSL, weight="bold", color=COL["C"], ha="center")

# ---------------------------------------------------------------- panel c
box(124.5, 61.0, 49.5, 31.0, fc="white", ec=COL["C"], lw=1.0)
T(126.2, 89.3, "Compact subgraph S", size=FS, weight="bold")
sp = {"M": (130.0, 71.0), "C1": (137.0, 77.0), "E1": (143.0, 82.5), "K1": (131.0, 81.5), "Q1": (143.5, 72.5), "D1": (149.0, 79.0),
      "P1": (153.0, 85.5), "C2": (158.5, 74.0), "E2": (164.5, 80.0), "K2": (168.5, 70.5), "Q2": (161.0, 66.5), "D2": (170.0, 77.0),
      "P2": (169.5, 85.5), "X": (136.5, 64.5), "D3": (144.5, 63.5)}
G(sp, [("M", "C1"), ("C1", "E1"), ("C1", "K1"), ("C1", "Q1"), ("E1", "D1"), ("D1", "P1"), ("E1", "P1"), ("M", "C2"),
       ("C2", "E2"), ("C2", "K2"), ("C2", "Q2"), ("E2", "D2"), ("D2", "P2"), ("E2", "P2"), ("M", "X"), ("X", "D3"), ("D3", "P1")],
  dashed=[("C1", "C2")], r=1.65)
arrow((118.1, 80.0), (124.4, 80.0), lw=1.5)
box(124.5, 44.0, 49.5, 14.6, fc="white", ec=COL["ok"], lw=1.0)
T(126.2, 56.3, "Audit certificate", size=FS, weight="bold", color=COL["ok"])
for i, it in enumerate(("Support", "Comparability", "Qualification", "Contradiction", "Exclusion (R2x)", "Replay info")):
    xx = 127.0 + (i % 2) * 24.6; yy = 52.6 - (i // 2) * 3.3
    check(xx, yy); T(xx + 1.6, yy, it, size=FS)
arrow((149.25, 60.9), (149.25, 58.7))
box(124.5, 33.0, 49.5, 8.6, fc="white", ec=COL["ink"], lw=1.1)
T(149.25, 38.9, "Grounded reader / LLM", size=FS, weight="bold", ha="center")
T(149.25, 35.4, "uses only S + certificate", size=FS, color=COL["mute"], ha="center")
arrow((149.25, 43.9), (149.25, 41.7))
box(124.5, 18.5, 49.5, 12.0, fc="white", ec=COL["C"], lw=1.0)
T(126.2, 27.6, "MOF E: largest HSE06 gap,", size=FS, weight="bold")
T(126.2, 24.4, "4.70 eV (DFT)", size=FS, weight="bold")
warn(127.6, 21.1, s=1.1); T(129.6, 21.1, "contested by ML prediction", size=FS, color=COL["red"])
arrow((149.25, 32.9), (149.25, 30.6))
box(124.5, 1.8, 49.5, 9.0, fc="white", ec=COL["X"], lw=1.1, r=1.8)
T(149.25, 8.0, "essential unit omitted →", size=FS, weight="bold", color=COL["X"], ha="center")
T(149.25, 4.6, "qualified answer or abstention", size=FS, color=COL["X"], ha="center")
line([cx0 + cw + 0.2, 121.0, 121.0], [33.8, 33.8, 6.3], color=COL["X"], lw=0.9, dashed=True)
arrow((121.0, 6.3), (124.4, 6.3), color=COL["X"], dashed=True, lw=0.9)

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
