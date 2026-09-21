"""
06 - Agrège les résultats des 3 méthodes et produit tableau + figures pour l'article.
Lit data/results/{method}_results.csv  ->  master table + 2 figures.
Usage: python code/06_make_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data", "results")
FIG = os.path.join(ROOT, "data", "figures"); os.makedirs(FIG, exist_ok=True)

METHODS = ["phash", "dinov2", "sscd", "sscd_local"]
LABELS  = {"phash": "pHash", "dinov2": "DINOv2", "sscd": "SSCD", "sscd_local": "SSCD+Local"}
COLORS  = {"phash": "#9aa0a6", "dinov2": "#4285f4", "sscd": "#ea4335", "sscd_local": "#8e44ad"}
TF_ORDER = ["jpeg40","rescale70","contrast","flip_h","crop80","rot90","rot180","rot270","partial"]
TF_NICE  = {"jpeg40":"JPEG","rescale70":"Rescale","contrast":"Contrast","flip_h":"Flip",
            "crop80":"Crop","rot90":"Rot90","rot180":"Rot180","rot270":"Rot270","partial":"PARTIAL"}

def load():
    frames = []
    for m in METHODS:
        df = pd.read_csv(os.path.join(RES, f"{m}_results.csv"))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def val(df, method, group, metric="recall@1"):
    r = df[(df.method == method) & (df.group == group)]
    return float(r[metric].iloc[0]) if len(r) else np.nan

def main():
    df = load()

    # ---- Master table (recall@1) : lignes = groupes, colonnes = méthodes ----
    rows = ["TYPE:full", "TYPE:partial"] + [f"TF:{t}" for t in TF_ORDER] + ["MOD:Blot/Gel", "MOD:Microscopy", "GLOBAL"]
    table = {}
    for m in METHODS:
        table[LABELS[m]] = [round(100*val(df, m, g), 1) for g in rows]
    master = pd.DataFrame(table, index=[r.replace("TF:","").replace("TYPE:","") for r in rows])
    master.to_csv(os.path.join(RES, "MASTER_recall@1.csv"))
    print("== MASTER (recall@1, %) =="); print(master.to_string())

    # ---- Figure 1 : Copie TOTALE vs PARTIELLE (le message central) ----
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(METHODS)); w = 0.38
    full = [100*val(df, m, "TYPE:full") for m in METHODS]
    part = [100*val(df, m, "TYPE:partial") for m in METHODS]
    ax.bar(x-w/2, full, w, label="Copie totale", color="#34a853")
    ax.bar(x+w/2, part, w, label="Copie partielle", color="#ea4335")
    for i,(f,p) in enumerate(zip(full,part)):
        ax.text(i-w/2, f+1.5, f"{f:.0f}", ha="center", fontsize=9)
        ax.text(i+w/2, p+1.5, f"{p:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in METHODS])
    ax.set_ylabel("Recall@1 (%)"); ax.set_ylim(0, 108)
    ax.set_title("Full vs partial copy retrieval on scientific figures")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_full_vs_partial.png"), dpi=200)
    print("Fig1 ->", os.path.relpath(os.path.join(FIG,"fig1_full_vs_partial.png"), ROOT))

    # ---- Figure 2 : Heatmap recall@1 par transformation x méthode ----
    M = np.array([[100*val(df, m, f"TF:{t}") for t in TF_ORDER] for m in METHODS])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(TF_ORDER))); ax.set_xticklabels([TF_NICE[t] for t in TF_ORDER], rotation=30, ha="right")
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels([LABELS[m] for m in METHODS])
    for i in range(len(METHODS)):
        for j in range(len(TF_ORDER)):
            ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                    color="black", fontsize=8)
    ax.set_title("Recall@1 (%) per transformation")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_heatmap_transforms.png"), dpi=200)
    print("Fig2 ->", os.path.relpath(os.path.join(FIG,"fig2_heatmap_transforms.png"), ROOT))

if __name__ == "__main__":
    main()
