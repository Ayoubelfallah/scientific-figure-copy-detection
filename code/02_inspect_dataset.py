"""
02 - Décompresse (si besoin) et inspecte la structure de BioFors.
Usage: python code/02_inspect_dataset.py
"""
import os, zipfile, tarfile, json, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ARCHIVE = os.path.join(DATA, "biofors.zip")  # nom du fichier téléchargé (contenu = TAR)
OUT = os.path.join(DATA, "biofors")

# --- 1. Décompression (auto-détection ZIP vs TAR) ---
if not os.path.isdir(OUT) or not os.listdir(OUT):
    os.makedirs(OUT, exist_ok=True)
    print(f"Décompression de {ARCHIVE} ...")
    if zipfile.is_zipfile(ARCHIVE):
        with zipfile.ZipFile(ARCHIVE) as z:
            z.extractall(OUT)
    elif tarfile.is_tarfile(ARCHIVE):
        with tarfile.open(ARCHIVE) as t:
            t.extractall(OUT)
    else:
        raise SystemExit("Format d'archive non reconnu (ni ZIP ni TAR).")
    print("Décompression terminée.")
else:
    print("Déjà décompressé.")

# --- 2. Arborescence (2 niveaux) ---
print("\n== Arborescence (2 niveaux) ==")
for name in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, name)
    if os.path.isdir(p):
        subs = sorted(os.listdir(p))[:12]
        print(f"[DIR] {name}/  ({len(os.listdir(p))} éléments)")
        for s in subs:
            sp = os.path.join(p, s)
            tag = "DIR" if os.path.isdir(sp) else "file"
            print(f"       - [{tag}] {s}")
    else:
        print(f"[file] {name}  ({os.path.getsize(p)/1024:.0f} Ko)")

# --- 3. Compter les images par extension et par dossier de 1er niveau ---
print("\n== Comptage des fichiers ==")
ext_counter = collections.Counter()
dir_img_counter = collections.Counter()
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}
for dirpath, _, files in os.walk(OUT):
    top = os.path.relpath(dirpath, OUT).split(os.sep)[0]
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        ext_counter[ext] += 1
        if ext in IMG_EXT:
            dir_img_counter[top] += 1
print("Extensions:", dict(ext_counter.most_common()))
print("Images par dossier de 1er niveau:", dict(dir_img_counter.most_common()))

# --- 4. Aperçu des fichiers d'annotation (json/csv/txt) ---
print("\n== Fichiers d'annotation ==")
for dirpath, _, files in os.walk(OUT):
    for f in files:
        if f.lower().endswith((".json", ".csv", ".txt")):
            fp = os.path.join(dirpath, f)
            rel = os.path.relpath(fp, OUT)
            size = os.path.getsize(fp) / 1024
            print(f"  {rel}  ({size:.0f} Ko)")
            if f.lower().endswith(".json"):
                try:
                    with open(fp, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, dict):
                        keys = list(data.keys())
                        print(f"     JSON dict, {len(keys)} clés. Ex: {keys[:5]}")
                        k0 = keys[0]
                        print(f"     data['{k0}'] = {json.dumps(data[k0])[:300]}")
                    elif isinstance(data, list):
                        print(f"     JSON list, {len(data)} éléments. Ex[0]: {json.dumps(data[0])[:300]}")
                except Exception as e:
                    print(f"     (lecture JSON échouée: {e})")
