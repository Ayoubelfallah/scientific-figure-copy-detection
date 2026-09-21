"""
07 - Re-ranking LOCAL (SIFT + vérification géométrique RANSAC) des candidats SSCD.
Pipeline type DELG : global (SSCD) -> top-K candidats -> re-classement par inliers locaux.
Montre que le matching local récupère des copies PARTIELLES que le global enterre.

Usage: python code/07_local_rerank.py            (K=100 par défaut)
Sortie: data/results/sscd_local_results.csv
"""
import os, json, collections
import numpy as np
import pandas as pd
import cv2
import faiss
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EMB = os.path.join(DATA, "emb")
IMG_ROOT = os.path.join(DATA, "biofors", "biofors_images")
RES = os.path.join(DATA, "results")
K = 100
MAXSIDE = 320
NFEAT = 500

sift = cv2.SIFT_create(nfeatures=NFEAT)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=32))

def corpus_path(cid):
    pid, img = cid.split("_", 1)
    return os.path.join(IMG_ROOT, pid, img)

def feats(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None: return None, None
    h, w = im.shape[:2]
    s = MAXSIDE / max(h, w)
    if s < 1: im = cv2.resize(im, (int(w*s), int(h*s)))
    kp, des = sift.detectAndCompute(im, None)
    if des is None or len(kp) < 4: return None, None
    return kp, des.astype(np.float32)

def inliers(fq, fc):
    kp1, d1 = fq; kp2, d2 = fc
    if d1 is None or d2 is None: return 0
    try:
        matches = flann.knnMatch(d1, d2, k=2)
    except cv2.error:
        return 0
    good = []
    for m in matches:
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            good.append(m[0])
    if len(good) < 4: return len(good)
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else len(good)

def main():
    C = np.load(os.path.join(EMB, "sscd_corpus.npz"), allow_pickle=True)
    Q = np.load(os.path.join(EMB, "sscd_queries.npz"), allow_pickle=True)
    cid = C["ids"]; pos = {c:i for i,c in enumerate(cid)}
    id2mod = dict(zip(C["ids"], C["modality"]))
    index = faiss.IndexFlatIP(C["vecs"].shape[1]); index.add(C["vecs"].astype(np.float32))
    _, I = index.search(Q["vecs"].astype(np.float32), K)

    src = Q["source_id"]; qt = Q["qtype"]; qpaths = [os.path.join(ROOT, p) for p in Q["ids"]]

    # grouper les requêtes par source (les 9 variantes partagent des candidats)
    groups = collections.defaultdict(list)
    for qi, s in enumerate(src): groups[s].append(qi)

    new_ranks = np.full(len(src), 10**9)
    for s, qis in tqdm(groups.items(), desc="re-rank (SIFT)"):
        cand_cache = {}
        cand_ids = set()
        for qi in qis: cand_ids.update(I[qi].tolist())
        for ci in cand_ids: cand_cache[ci] = feats(corpus_path(cid[ci]))
        target = pos.get(s, -1)
        for qi in qis:
            fq = feats(qpaths[qi])
            scores = [(inliers(fq, cand_cache[ci]), -rank, ci)
                      for rank, ci in enumerate(I[qi])]
            scores.sort(reverse=True)                      # inliers desc, puis rang global
            order = [ci for _,_,ci in scores]
            if target in order:
                new_ranks[qi] = order.index(target) + 1

    # évaluation
    def metrics(mask):
        r = new_ranks[mask].astype(float)
        return {"n": int(mask.sum()),
                "recall@1": float(np.mean(r<=1)),
                "recall@10": float(np.mean(r<=10)),
                "mAP": float(np.mean(1.0/r))}
    part = np.array([t=="partial" for t in qt]); full = ~part
    print("\n===== SSCD + LOCAL (SIFT re-rank) =====")
    print("FULL   :", metrics(full))
    print("PARTIAL:", metrics(part))
    print("GLOBAL :", metrics(np.ones(len(src), bool)))

    # sauvegarde même schéma que 05 (pour la figure)
    mods = np.array([id2mod.get(s,"?") for s in src])
    coarse = np.array(["partial" if t=="partial" else "full" for t in qt])
    tf = np.array([str(t).replace("full:","") for t in qt])
    rows = []
    def add(group, mask): rows.append({"method":"sscd_local","group":group, **metrics(mask)})
    add("TYPE:full", full); add("TYPE:partial", part)
    for t in sorted(set(tf)): add("TF:"+t, tf==t)
    for m in sorted(set(mods)): add("MOD:"+m, mods==m)
    add("GLOBAL", np.ones(len(src), bool))
    out = os.path.join(RES, "sscd_local_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nCSV ->", os.path.relpath(out, ROOT))

if __name__ == "__main__":
    main()
