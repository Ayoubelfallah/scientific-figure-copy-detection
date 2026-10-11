"""
08 - Baseline TILING / multi-crop (réponse au reviewer C2).
Pour chaque requête : on encode {image entière + 4 quadrants 2x2} avec SSCD,
on cherche chacun dans le corpus, et on prend le MEILLEUR rang de la source.
-> teste si un simple découpage "résout" la copie partielle.

Usage: python code/08_tiling_baseline.py
Sortie: data/results/sscd_tiled_results.csv  + impression comparative.
"""
import os, json, collections
import numpy as np
import pandas as pd
import faiss
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EMB = os.path.join(DATA, "emb")
RES = os.path.join(DATA, "results")
SSCD_W = os.path.join(ROOT, "models", "sscd_disc_mixup.torchscript.pt")
K = 1000
SIZE = 288
BATCH = 128

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load(SSCD_W, map_location=dev).eval().to(dev)
tf = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def crops(img):
    """image entière + 4 quadrants 2x2."""
    w, h = img.size
    return [
        img,
        img.crop((0, 0, w // 2, h // 2)),
        img.crop((w // 2, 0, w, h // 2)),
        img.crop((0, h // 2, w // 2, h)),
        img.crop((w // 2, h // 2, w, h)),
    ]

@torch.no_grad()
def encode(images):
    out = []
    for i in range(0, len(images), BATCH):
        x = torch.stack([tf(im.convert("RGB")) for im in images[i:i + BATCH]]).to(dev)
        y = model(x)
        if isinstance(y, (list, tuple)): y = y[0]
        out.append(torch.nn.functional.normalize(y, dim=1).cpu().numpy().astype(np.float32))
    return np.vstack(out)

def main():
    C = np.load(os.path.join(EMB, "sscd_corpus.npz"), allow_pickle=True)
    Q = np.load(os.path.join(EMB, "sscd_queries.npz"), allow_pickle=True)
    cid = C["ids"]; pos = {c: i for i, c in enumerate(cid)}
    id2mod = dict(zip(C["ids"], C["modality"]))
    index = faiss.IndexFlatIP(C["vecs"].shape[1]); index.add(C["vecs"].astype(np.float32))

    src = Q["source_id"]; qt = Q["qtype"]
    qpaths = [os.path.join(ROOT, p) for p in Q["ids"]]

    # encoder tous les crops (5 par requête)
    all_imgs, owner = [], []
    print("Préparation des crops (5/requête)...")
    for qi, pth in enumerate(tqdm(qpaths)):
        for cr in crops(Image.open(pth)):
            all_imgs.append(cr); owner.append(qi)
    print(f"Encodage SSCD de {len(all_imgs)} crops...")
    vecs = encode(all_imgs)

    print("Recherche FAISS...")
    _, I = index.search(vecs, K)   # (n_crops, K)
    owner = np.array(owner)

    # pour chaque requête : meilleur rang de la source sur ses 5 crops
    ranks = np.full(len(src), 10**9)
    for qi in range(len(src)):
        tgt = pos.get(src[qi], -1)
        rows = I[owner == qi]            # (5, K)
        best = 10**9
        for row in rows:
            h = np.where(row == tgt)[0]
            if len(h): best = min(best, int(h[0]) + 1)
        ranks[qi] = best

    def metrics(mask):
        r = ranks[mask].astype(float)
        return {"n": int(mask.sum()),
                "recall@1": float(np.mean(r <= 1)),
                "recall@10": float(np.mean(r <= 10)),
                "MRR": float(np.mean(1.0 / r))}
    part = np.array([t == "partial" for t in qt]); full = ~part
    print("\n===== SSCD + TILING (whole + 2x2 quadrants) =====")
    print("FULL   :", metrics(full))
    print("PARTIAL:", metrics(part))
    print("GLOBAL :", metrics(np.ones(len(src), bool)))

    mods = np.array([id2mod.get(s, "?") for s in src])
    tf_lbl = np.array([str(t).replace("full:", "") for t in qt])
    rows = []
    def add(g, m): rows.append({"method": "sscd_tiled", "group": g, **metrics(m)})
    add("TYPE:full", full); add("TYPE:partial", part)
    for t in sorted(set(tf_lbl)): add("TF:" + t, tf_lbl == t)
    for m in sorted(set(mods)): add("MOD:" + m, mods == m)
    add("GLOBAL", np.ones(len(src), bool))
    out = os.path.join(RES, "sscd_tiled_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nCSV ->", os.path.relpath(out, ROOT))

if __name__ == "__main__":
    main()
