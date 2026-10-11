"""
09 - Copie partielle DURE (réponse reviewer P1) : ratio de surface varié + position
aléatoire + clutter (PAS une grille 2x2). Montre que le tiling 2x2 s'effondre quand
le panel n'est plus aligné sur une grille / devient petit.

Génère les requêtes, puis évalue SSCD : whole vs tiling-2x2 vs tiling-3x3, par ratio.
Usage: python code/09_hard_partial.py
Sortie: data/benchmark_hard/ (+ ground_truth) et data/results/hard_partial_results.csv
"""
import os, json, random
import numpy as np, pandas as pd, faiss, torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EMB = os.path.join(DATA, "emb")
RES = os.path.join(DATA, "results")
IMG_ROOT = os.path.join(DATA, "biofors", "biofors_images")
CLASS_JSON = os.path.join(DATA, "classification.json")
HARD = os.path.join(DATA, "benchmark_hard")
SSCD_W = os.path.join(ROOT, "models", "sscd_disc_mixup.torchscript.pt")
SEED = 7; CANVAS = 640; RATIOS = [0.50, 0.25, 0.10, 0.05]; K = 1000; SIZE = 288; BATCH = 128
os.makedirs(HARD, exist_ok=True)

# ---------- génération des requêtes dures ----------
def build_catalog():
    cls = json.load(open(CLASS_JSON, encoding="utf-8"))
    items = []
    for pid, mp in cls.items():
        for img, mod in mp.items():
            items.append((f"{pid}_{img}", os.path.join(IMG_ROOT, pid, img), mod))
    return items

def make_hard(src_img, ratio, distractors, rng):
    """Colle la source (ratio de surface, position aléatoire) sur un fond de clutter."""
    canvas = Image.new("RGB", (CANVAS, CANVAS), (random.randint(235,250),)*3)
    # clutter : 5 distracteurs à positions/tailles aléatoires
    for d in distractors:
        d = d.convert("RGB"); dw = rng.randint(140, 300)
        d.thumbnail((dw, dw), Image.Resampling.LANCZOS)
        canvas.paste(d, (rng.randint(0, CANVAS - d.size[0]), rng.randint(0, CANVAS - d.size[1])))
    # source au ratio de surface voulu, position aléatoire, PAR-DESSUS
    w, h = src_img.size
    scale = (ratio * CANVAS * CANVAS / (w * h)) ** 0.5
    nw, nh = max(16, int(w * scale)), max(16, int(h * scale))
    nw, nh = min(nw, CANVAS), min(nh, CANVAS)
    s = src_img.convert("RGB").resize((nw, nh), Image.Resampling.LANCZOS)
    canvas.paste(s, (rng.randint(0, CANVAS - nw), rng.randint(0, CANVAS - nh)))
    return canvas

def generate():
    gt_path = os.path.join(HARD, "ground_truth.json")
    if os.path.exists(gt_path):
        print("Requêtes dures déjà générées."); return json.load(open(gt_path, encoding="utf-8"))
    items = build_catalog()
    # on réutilise les 500 sources du benchmark principal
    base_gt = json.load(open(os.path.join(DATA, "benchmark_full", "ground_truth.json"), encoding="utf-8"))
    src_ids = sorted({g["source_id"] for g in base_gt})
    id2path = {i: p for i, p, _ in items}
    rng = random.Random(SEED)
    gt = []
    print(f"Génération : {len(src_ids)} sources × {len(RATIOS)} ratios...")
    for sid in tqdm(src_ids):
        src = Image.open(id2path[sid])
        for r in RATIOS:
            distractors = [Image.open(rng.choice(items)[1]) for _ in range(5)]
            q = make_hard(src, r, distractors, rng)
            fn = os.path.join(HARD, f"{sid.replace('/', '_').replace('.png','')}_r{int(r*100)}.png")
            q.save(fn)
            gt.append({"query": os.path.relpath(fn, ROOT), "source_id": sid, "ratio": r})
    json.dump(gt, open(gt_path, "w", encoding="utf-8"), indent=2)
    print("Requêtes dures ->", len(gt))
    return gt

# ---------- encodage SSCD + tiling ----------
dev = "cuda" if torch.cuda.is_available() else "cpu"
def load_model():
    m = torch.jit.load(SSCD_W, map_location=dev).eval().to(dev)
    tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return m, tf

def grid(img, n):
    w, h = img.size; out = []
    for i in range(n):
        for j in range(n):
            out.append(img.crop((j*w//n, i*h//n, (j+1)*w//n, (i+1)*h//n)))
    return out

@torch.no_grad()
def encode(model, tf, imgs):
    out = []
    for i in range(0, len(imgs), BATCH):
        x = torch.stack([tf(im.convert("RGB")) for im in imgs[i:i+BATCH]]).to(dev)
        y = model(x); y = y[0] if isinstance(y,(list,tuple)) else y
        out.append(torch.nn.functional.normalize(y, dim=1).cpu().numpy().astype(np.float32))
    return np.vstack(out)

def best_rank(index, pos, vecs, target):
    _, I = index.search(vecs, K)
    best = 10**9
    for row in I:
        h = np.where(row == target)[0]
        if len(h): best = min(best, int(h[0]) + 1)
    return best

def main():
    gt = generate()
    C = np.load(os.path.join(EMB, "sscd_corpus.npz"), allow_pickle=True)
    pos = {c: i for i, c in enumerate(C["ids"])}
    index = faiss.IndexFlatIP(C["vecs"].shape[1]); index.add(C["vecs"].astype(np.float32))
    model, tf = load_model()

    methods = {"whole": 1, "tile2x2": 2, "tile3x3": 3}
    ranks = {m: [] for m in methods}
    ratios = []
    print("Évaluation (whole / 2x2 / 3x3)...")
    for g in tqdm(gt):
        img = Image.open(os.path.join(ROOT, g["query"]))
        tgt = pos.get(g["source_id"], -1)
        ratios.append(g["ratio"])
        for m, n in methods.items():
            crops = [img] if n == 1 else [img] + grid(img, n)
            v = encode(model, tf, crops)
            ranks[m].append(best_rank(index, pos, v, tgt))
    ratios = np.array(ratios)

    rows = []
    print("\n===== COPIE PARTIELLE DURE — recall@1 par ratio de surface =====")
    print(f"{'ratio':>7} | {'whole':>7} {'tile2x2':>8} {'tile3x3':>8}")
    for r in RATIOS:
        mask = ratios == r
        line = {m: float(np.mean(np.array(ranks[m])[mask] <= 1)) for m in methods}
        print(f"{int(r*100):>6}% | {line['whole']:>7.3f} {line['tile2x2']:>8.3f} {line['tile3x3']:>8.3f}")
        for m in methods:
            rr = np.array(ranks[m])[mask].astype(float)
            rows.append({"ratio": r, "method": m, "recall@1": float(np.mean(rr<=1)),
                         "recall@10": float(np.mean(rr<=10)), "MRR": float(np.mean(1.0/rr)), "n": int(mask.sum())})
    out = os.path.join(RES, "hard_partial_results.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nCSV ->", os.path.relpath(out, ROOT))

if __name__ == "__main__":
    main()
