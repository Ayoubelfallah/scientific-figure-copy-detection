"""
10 - Figure-clé de la révision : dégradation du recall@1 quand la région réutilisée
rétrécit (copie partielle dure). Montre que le tiling fixe s'effondre.
Usage: python code/10_hard_figure.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data", "results")
FIG = os.path.join(ROOT, "data", "figures")

df = pd.read_csv(os.path.join(RES, "hard_partial_results.csv"))
ratios = sorted(df["ratio"].unique(), reverse=True)        # 0.5 -> 0.05
x = [f"{int(r*100)}%" for r in ratios]
meth = [("whole", "SSCD (whole)", "#ea4335", "o"),
        ("tile2x2", "SSCD + Tiling 2×2", "#e67e22", "s"),
        ("tile3x3", "SSCD + Tiling 3×3", "#2a9d8f", "^")]

fig, ax = plt.subplots(figsize=(6.6, 4.2))
for key, lab, col, mk in meth:
    y = [100 * df[(df.method == key) & (df.ratio == r)]["recall@1"].iloc[0] for r in ratios]
    ax.plot(x, y, marker=mk, color=col, linewidth=2.2, markersize=7, label=lab)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=col)

ax.set_xlabel("Reused-region area (share of the query)")
ax.set_ylabel("Recall@1 (%)")
ax.set_ylim(-5, 108)
ax.set_title("Fixed tiling degrades as the reused region shrinks (off-grid, cluttered)")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="lower left", fontsize=9)
# x déjà ordonné 50% -> 5% (gauche->droite = la région rétrécit)
fig.tight_layout()
out = os.path.join(FIG, "fig3_hard_partial_degradation.png")
fig.savefig(out, dpi=200)
print("Fig3 ->", os.path.relpath(out, ROOT))
