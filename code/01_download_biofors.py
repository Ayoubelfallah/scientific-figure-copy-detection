"""
01 - Télécharge le dataset BioFors depuis Google Drive (licence CC0-1.0).
Source: https://github.com/vimal-isi-edu/BioFors
Usage:  python code/01_download_biofors.py
"""
import os
import gdown

# ID du fichier Google Drive (depuis le lien officiel du dépôt BioFors)
FILE_ID = "1UVSJ6h7r8pmOWYZkqWeAZ_YvwbFr1wV3"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_ZIP = os.path.join(OUT_DIR, "biofors.zip")

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Téléchargement de BioFors -> {OUT_ZIP}")
print("(fichier volumineux ; sur connexion lente cela peut prendre du temps ; gdown reprend en cas de coupure)")

gdown.download(id=FILE_ID, output=OUT_ZIP, quiet=False, resume=True)

print("\nTéléchargement terminé.")
print(f"Fichier : {OUT_ZIP}")
if os.path.exists(OUT_ZIP):
    size_mb = os.path.getsize(OUT_ZIP) / (1024 * 1024)
    print(f"Taille  : {size_mb:.1f} Mo")
print("\nProchaine étape : décompresser puis inspecter la structure (script 02).")
