"""
11 - Statistiques demandées par le reviewer :
  S1  bootstrap 95% CI sur recall@1 (full / partial)
  S2  test de McNemar (paires) : DINOv2 vs SSCD (partial), SSCD vs SSCD+Tiling (partial)
  G1  check de déduplication (BioFors contient de vrais doublons)

Usage: python code/11_stats.py
"""
import os, numpy as np, faiss, torch
from scipy import stats
from torchvision import transforms
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(ROOT, "data", "emb")
SSCD_W = os.path.join(ROOT, "models", "sscd_disc_mixup.torchscript.pt")
K = 1000; SIZE = 288; BATCH = 128; RNG = np.random.default_rng(0)

def load(method, which):
    d = np.load(os.path.join(EMB, f"{method}_{which}.npz"), allow_pickle=True)
    return d

def fast_hits(method):
    """per-query hit@1 pour une méthode à embeddings (phash/dinov2/sscd)."""
    C = load(method, "corpus"); Q = load(method, "queries")
    pos = {c: i for i, c in enumerate(C["ids"])}
    if str(C["kind"]) == "binary":
        idx = faiss.IndexBinaryFlat(C["vecs"].shape[1] * 8); idx.add(C["vecs"])
        _, I = idx.search(Q["vecs"], 1)
    else:
        idx = faiss.IndexFlatIP(C["vecs"].shape[1]); idx.add(C["vecs"].astype(np.float32))
        _, I = idx.search(Q["vecs"].astype(np.float32), 1)
    tgt = np.array([pos.get(s, -1) for s in Q["source_id"]])
    hit = (I[:, 0] == tgt)
    return hit, Q["qtype"], Q["source_id"]

def tiled_hits():
    """per-query hit@1 pour SSCD+Tiling (whole + 2x2) — recalcul GPU."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(SSCD_W, map_location=dev).eval().to(dev)
    tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    C = load("sscd", "corpus"); Q = load("sscd", "queries")
    pos = {c: i for i, c in enumerate(C["ids"])}
    idx = faiss.IndexFlatIP(C["vecs"].shape[1]); idx.add(C["vecs"].astype(np.float32))
    def crops(im):
        w, h = im.size
        return [im, im.crop((0,0,w//2,h//2)), im.crop((w//2,0,w,h//2)),
                im.crop((0,h//2,w//2,h)), im.crop((w//2,h//2,w,h))]
    hits = []
    qpaths = [os.path.join(ROOT, p) for p in Q["ids"]]
    with torch.no_grad():
        for qi, pth in enumerate(qpaths):
            cs = crops(Image.open(pth).convert("RGB"))
            x = torch.stack([tf(c.convert("RGB")) for c in cs]).to(dev)
            y = model(x); y = y[0] if isinstance(y,(list,tuple)) else y
            v = torch.nn.functional.normalize(y, dim=1).cpu().numpy().astype(np.float32)
            _, I = idx.search(v, 1)
            hits.append(pos.get(Q["source_id"][qi], -1) in set(I[:, 0].tolist()))
            if qi % 500 == 0: print(f"  tiling {qi}/{len(qpaths)}")
    return np.array(hits), Q["qtype"], Q["source_id"]

def ci(hit, n=5000):
    hit = np.asarray(hit, float)
    if len(hit) == 0: return (0, 0, 0)
    boots = [RNG.choice(hit, len(hit), replace=True).mean() for _ in range(n)]
    return hit.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)

def mcnemar(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    b_only = int(np.sum(a & ~b)); c_only = int(np.sum(~a & b))
    p = stats.binomtest(min(b_only, c_only), b_only + c_only, 0.5).pvalue if (b_only + c_only) else 1.0
    return b_only, c_only, p

def main():
    print("Calcul des hits (phash/dinov2/sscd)...")
    H = {}
    for m in ["phash", "dinov2", "sscd"]:
        H[m] = fast_hits(m)
    print("Calcul des hits SSCD+Tiling (GPU)...")
    H["sscd_tiled"] = tiled_hits()

    qt = H["sscd"][1]
    part = np.array([t == "partial" for t in qt]); full = ~part

    print("\n===== S1 — Recall@1 avec IC 95% (bootstrap) =====")
    print(f"{'method':>12} | {'FULL (95% CI)':>26} | {'PARTIAL (95% CI)':>26}")
    for m in ["phash", "dinov2", "sscd", "sscd_tiled"]:
        hit = H[m][0]
        mf, lf, hf = ci(hit[full]); mp, lp, hp = ci(hit[part])
        print(f"{m:>12} | {mf*100:5.1f}  [{lf*100:4.1f}, {hf*100:4.1f}]      | {mp*100:5.1f}  [{lp*100:4.1f}, {hp*100:4.1f}]")

    print("\n===== S2 — McNemar (partial) =====")
    for a, b in [("dinov2", "sscd"), ("sscd", "sscd_tiled")]:
        bo, co, p = mcnemar(H[a][0][part], H[b][0][part])
        print(f"  {a} vs {b}:  b={bo}, c={co}, p={p:.2e}  ->  {'significatif' if p < 0.05 else 'NS'}")

    print("\n===== G1 — Check de déduplication (SSCD, cos >= 0.90) =====")
    C = load("sscd", "corpus"); Q = load("sscd", "queries")
    ids = C["ids"]; pos = {c: i for i, c in enumerate(ids)}
    cv = C["vecs"].astype(np.float32)
    idx = faiss.IndexFlatIP(cv.shape[1]); idx.add(cv)
    src_ids = sorted(set(Q["source_id"]))
    dup = {}
    for sid in src_ids:
        D, I = idx.search(cv[pos[sid]:pos[sid]+1], 20)
        dup[sid] = {ids[j] for j, d in zip(I[0], D[0]) if d >= 0.90 and ids[j] != sid}
    n_with = sum(1 for s in src_ids if dup[s])
    print(f"  sources avec >=1 quasi-doublon dans le corpus : {n_with}/{len(src_ids)}")
    # recall relâché (source ∪ doublons) pour SSCD whole
    _, I1 = idx.search(Q["vecs"].astype(np.float32), 1)
    strict = full.copy()*0.0; relax = full.copy()*0.0
    s_hit, r_hit = [], []
    for qi in range(len(Q["source_id"])):
        sid = Q["source_id"][qi]; top = ids[I1[qi, 0]]
        s_hit.append(top == sid)
        r_hit.append(top == sid or top in dup[sid])
    s_hit = np.array(s_hit); r_hit = np.array(r_hit)
    for name, mask in [("FULL", full), ("PARTIAL", part)]:
        print(f"  {name:8} recall@1 strict={s_hit[mask].mean()*100:.1f}%  relâché={r_hit[mask].mean()*100:.1f}%  (Δ={ (r_hit[mask].mean()-s_hit[mask].mean())*100:+.1f})")
    print("\n(Δ faible => la vérité-terrain n'est pas significativement polluée par les doublons.)")

if __name__ == "__main__":
    main()
