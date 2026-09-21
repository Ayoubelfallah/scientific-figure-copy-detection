# Code — Short Paper : Copy-Detection sur figures scientifiques

Pipeline du benchmark (voir `../00_CADRAGE_short_paper.md` pour le cadrage).

## Environnement
- Python 3.11 + venv (`../venv`)
- GPU : RTX 5060 (cu128) ✅
- Paquets : torch, torchvision, faiss-cpu, timm, imagehash, scikit-image, gdown, tqdm, opencv-python, pandas, scikit-learn, matplotlib

## Étapes (scripts numérotés)
| Script | Rôle | Statut |
|--------|------|--------|
| `01_download_biofors.py` | Télécharger BioFors (CC0) | ▶ à lancer |
| `02_inspect_dataset.py` | Inspecter structure, compter par modalité | à venir |
| `03_build_queries.py` | Générer variantes (totale + partielle) | à venir |
| `04_extract_embeddings.py` | Encoder avec SSCD / DINOv2 / pHash | à venir |
| `05_build_index_search.py` | Index FAISS + recherche top-k | à venir |
| `06_evaluate.py` | Métriques µAP, recall@k (total vs partiel) | à venir |

## Données
- `../data/biofors.zip` (téléchargé)
- `../data/biofors/` (décompressé)
