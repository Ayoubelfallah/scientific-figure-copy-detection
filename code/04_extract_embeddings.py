"""
04 - Encode les images (corpus OU requêtes) avec une méthode donnée.
Méthodes : phash (offline) | dinov2 (timm) | sscd (TorchScript)
Sorties   : data/emb/{method}_{set}.npz   (ids, vecs, kind, [source_id, qtype])

Exemples :
    python code/04_extract_embeddings.py --method phash  --set corpus
    python code/04_extract_embeddings.py --method phash  --set queries
    python code/04_extract_embeddings.py --method dinov2 --set corpus
"""
import os, json, argparse, glob
# --- Fix SSL : une SSL_CERT_FILE cassée empêche le téléchargement des modèles (HuggingFace) ---
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    os.environ["CURL_CA_BUNDLE"] = certifi.where()
except Exception:
    os.environ.pop("SSL_CERT_FILE", None)  # sinon on retire le chemin invalide
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
IMG_ROOT = os.path.join(DATA, "biofors", "biofors_images")
CLASS_JSON = os.path.join(DATA, "classification.json")
EMB_DIR = os.path.join(DATA, "emb")
os.makedirs(EMB_DIR, exist_ok=True)

# ---------------- manifests ----------------
def corpus_manifest():
    """Tous les panels du corpus : (id, path, modality). id = paperid_imgname."""
    cls = json.load(open(CLASS_JSON, encoding="utf-8"))
    ids, paths, mods = [], [], []
    for pid, mapping in cls.items():
        for img_name, modality in mapping.items():
            ids.append(f"{pid}_{img_name}")
            paths.append(os.path.join(IMG_ROOT, pid, img_name))
            mods.append(modality)
    return ids, paths, mods

def query_manifest():
    """Requêtes depuis la vérité-terrain du benchmark complet."""
    gt = json.load(open(os.path.join(DATA, "benchmark_full", "ground_truth.json"), encoding="utf-8"))
    ids   = [g["query"] for g in gt]                 # chemin relatif = identifiant unique
    paths = [os.path.join(ROOT, g["query"]) for g in gt]
    src   = [g["source_id"] for g in gt]
    typ   = [g["type"] for g in gt]
    return ids, paths, src, typ

# ---------------- encoders ----------------
def encode_phash(paths, hash_size=8):
    import imagehash
    codes = []
    for p in tqdm(paths, desc="pHash"):
        try:
            h = imagehash.phash(Image.open(p).convert("L"), hash_size=hash_size)
            bits = h.hash.flatten().astype(np.uint8)          # 64 bits 0/1
            codes.append(np.packbits(bits))                    # 8 octets
        except Exception:
            codes.append(np.zeros(hash_size*hash_size//8, np.uint8))
    return np.vstack(codes), "binary"

def _deep_loader():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch, dev

def encode_dinov2(paths, batch=64, size=224):
    import torch, timm
    from torchvision import transforms
    torch, dev = _deep_loader()
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                              num_classes=0, dynamic_img_size=True)
    model.eval().to(dev)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return _run_deep(model, tf, paths, batch, dev, torch)

def encode_sscd(paths, batch=64, size=288,
                weights="models/sscd_disc_mixup.torchscript.pt"):
    import torch
    from torchvision import transforms
    torch, dev = _deep_loader()
    wpath = os.path.join(ROOT, weights)
    if not os.path.exists(wpath):
        raise SystemExit(f"Poids SSCD introuvables: {wpath}\n"
                         "Téléchargez le .torchscript.pt (voir instructions) puis relancez.")
    model = torch.jit.load(wpath, map_location=dev).eval().to(dev)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return _run_deep(model, tf, paths, batch, dev, torch)

def _run_deep(model, tf, paths, batch, dev, torch):
    vecs = []
    buf, idx = [], 0
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch), desc="encode"):
            chunk = paths[i:i+batch]
            ims = []
            for p in chunk:
                try: ims.append(tf(Image.open(p).convert("RGB")))
                except Exception: ims.append(torch.zeros(3, tf.transforms[0].size[0], tf.transforms[0].size[0]))
            x = torch.stack(ims).to(dev)
            y = model(x)
            if isinstance(y, (list, tuple)): y = y[0]
            y = torch.nn.functional.normalize(y, dim=1).cpu().numpy().astype(np.float32)
            vecs.append(y)
    return np.vstack(vecs), "float"

ENCODERS = {"phash": encode_phash, "dinov2": encode_dinov2, "sscd": encode_sscd}

# ---------------- main ----------------
def main(method, which):
    if which == "corpus":
        ids, paths, mods = corpus_manifest()
        extra = {"modality": np.array(mods)}
    else:
        ids, paths, src, typ = query_manifest()
        extra = {"source_id": np.array(src), "qtype": np.array(typ)}
    print(f"{method} / {which} : {len(paths)} images")
    vecs, kind = ENCODERS[method](paths)
    out = os.path.join(EMB_DIR, f"{method}_{which}.npz")
    np.savez(out, ids=np.array(ids), vecs=vecs, kind=kind, **extra)
    print("Sauvegardé ->", os.path.relpath(out, ROOT), "| shape", vecs.shape, "| kind", kind)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(ENCODERS))
    ap.add_argument("--set", required=True, choices=["corpus", "queries"])
    args = ap.parse_args()
    main(args.method, args.set)
