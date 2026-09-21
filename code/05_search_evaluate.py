"""
05 - Index FAISS + recherche + évaluation pour une méthode.
Charge data/emb/{method}_corpus.npz et {method}_queries.npz,
cherche le top-K, calcule recall@1, recall@10 et mAP (=moyenne de 1/rang),
ventilés par TYPE (copie totale vs partielle) et par MODALITÉ de la source.

Exemple : python code/05_search_evaluate.py --method phash
"""
import os, argparse, collections
import numpy as np
import pandas as pd
import faiss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EMB = os.path.join(DATA, "emb")
RES = os.path.join(DATA, "results"); os.makedirs(RES, exist_ok=True)
K = 1000

def search(method):
    C = np.load(os.path.join(EMB, f"{method}_corpus.npz"), allow_pickle=True)
    Q = np.load(os.path.join(EMB, f"{method}_queries.npz"), allow_pickle=True)
    corpus_ids = C["ids"]; kind = str(C["kind"])
    id2mod = dict(zip(C["ids"], C["modality"]))
    src_ids = Q["source_id"]; qtypes = Q["qtype"]

    if kind == "binary":
        d = C["vecs"].shape[1] * 8
        index = faiss.IndexBinaryFlat(d)
        index.add(C["vecs"])
        _, I = index.search(Q["vecs"], K)         # distances Hamming (croissantes)
    else:
        dim = C["vecs"].shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(C["vecs"].astype(np.float32))
        _, I = index.search(Q["vecs"].astype(np.float32), K)  # IP (décroissant)

    # rang de la vraie source pour chaque requête
    pos = {cid: i for i, cid in enumerate(corpus_ids)}
    ranks = []
    for qi in range(len(src_ids)):
        target = pos.get(src_ids[qi], -1)
        row = I[qi]
        hit = np.where(row == target)[0]
        ranks.append(int(hit[0]) + 1 if len(hit) else 10**9)   # rang (1 = meilleur)
    return np.array(ranks), qtypes, np.array([id2mod.get(s, "?") for s in src_ids])

def metrics(ranks):
    ranks = np.asarray(ranks, float)
    return {
        "n": len(ranks),
        "recall@1":  float(np.mean(ranks <= 1)),
        "recall@10": float(np.mean(ranks <= 10)),
        "mAP":       float(np.mean(1.0 / ranks)),   # 1 seul pertinent -> AP = 1/rang
    }

def group(ranks, labels):
    rows = {}
    for lab in sorted(set(labels)):
        rows[lab] = metrics(ranks[np.array(labels) == lab])
    return rows

def main(method):
    ranks, qtypes, mods = search(method)
    # regrouper copie totale vs partielle
    coarse = np.array(["partial" if t == "partial" else "full" for t in qtypes])

    print(f"\n===== {method.upper()} =====")
    print("GLOBAL:", metrics(ranks))

    print("\n-- par TYPE (total vs partiel) --")
    by_type = group(ranks, coarse)
    for k, v in by_type.items(): print(f"  {k:8} {v}")

    print("\n-- par TRANSFORMATION --")
    by_tf = group(ranks, np.array([str(t).replace('full:','') for t in qtypes]))
    for k, v in by_tf.items(): print(f"  {k:10} {v}")

    print("\n-- par MODALITÉ de la source --")
    by_mod = group(ranks, mods)
    for k, v in by_mod.items(): print(f"  {k:12} {v}")

    # sauvegarde CSV (par type + par transformation)
    df = pd.DataFrame(
        [{"method": method, "group": "TYPE:"+k, **v} for k, v in by_type.items()] +
        [{"method": method, "group": "TF:"+k, **v} for k, v in by_tf.items()] +
        [{"method": method, "group": "MOD:"+k, **v} for k, v in by_mod.items()] +
        [{"method": method, "group": "GLOBAL", **metrics(ranks)}]
    )
    out = os.path.join(RES, f"{method}_results.csv")
    df.to_csv(out, index=False)
    print("\nCSV ->", os.path.relpath(out, ROOT))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    main(ap.parse_args().method)
