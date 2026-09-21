"""
03 - Génère les variantes de requêtes (COPIE TOTALE + COPIE PARTIELLE) à partir de BioFors.
Mode mini-test : quelques sources + une planche-contact visuelle pour valider.
Usage:
    python code/03_build_queries.py            # mini-test (10 sources) + montage
    python code/03_build_queries.py --full     # benchmark complet (500 sources)
"""
import os, json, random, argparse
from PIL import Image, ImageEnhance, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
IMG_ROOT = os.path.join(DATA, "biofors", "biofors_images")
CLASS_JSON = os.path.join(DATA, "classification.json")

SEED = 42
MIN_SIZE = 128          # on écarte les panels trop petits (vérité-terrain propre)
QUERY_CLASSES = ["Blot/Gel", "Microscopy"]

# ---------- variantes COPIE TOTALE ----------
def full_variants(img):
    """Retourne {nom: image transformée} — transformations typiques du plagiat."""
    v = {}
    v["rot90"]    = img.rotate(90, expand=True)
    v["rot180"]   = img.rotate(180, expand=True)
    v["rot270"]   = img.rotate(270, expand=True)
    v["flip_h"]   = img.transpose(Image.FLIP_LEFT_RIGHT)
    # crop central 80%
    w, h = img.size
    cw, ch = int(w*0.8), int(h*0.8)
    left, top = (w-cw)//2, (h-ch)//2
    v["crop80"]   = img.crop((left, top, left+cw, top+ch))
    # rescale 0.7x (puis on garde la petite taille)
    v["rescale70"] = img.resize((max(1,int(w*0.7)), max(1,int(h*0.7))), Image.Resampling.LANCZOS)
    # contraste + luminosité (beautification)
    v["contrast"] = ImageEnhance.Brightness(ImageEnhance.Contrast(img).enhance(1.4)).enhance(1.15)
    return v  # (le JPEG est géré à la sauvegarde via quality=40)

# ---------- variante COPIE PARTIELLE ----------
def partial_variant(src_img, distractors, cell=256):
    """Colle la source dans UNE case d'une grille 2x2, les autres = distracteurs.
    -> la source n'occupe qu'~1/4 de l'image composite (le cas critique)."""
    canvas = Image.new("RGB", (cell*2, cell*2), (245, 245, 245))
    def fit(im):
        im = im.convert("RGB")
        im.thumbnail((cell-8, cell-8), Image.Resampling.LANCZOS)
        return im
    positions = [(0,0),(cell,0),(0,cell),(cell,cell)]
    random.shuffle(positions)
    src_pos = positions[0]
    tiles = [fit(src_img)] + [fit(d) for d in distractors[:3]]
    for (px,py), tile in zip(positions, tiles):
        ox = px + (cell - tile.size[0])//2
        oy = py + (cell - tile.size[1])//2
        canvas.paste(tile, (ox, oy))
    # bordures de grille
    d = ImageDraw.Draw(canvas)
    d.line([(cell,0),(cell,cell*2)], fill=(180,180,180), width=2)
    d.line([(0,cell),(cell*2,cell)], fill=(180,180,180), width=2)
    return canvas

# ---------- utilitaires ----------
def load_catalog():
    cls = json.load(open(CLASS_JSON, encoding="utf-8"))
    items = []  # (paper_id, img_name, modality, path)
    for pid, mapping in cls.items():
        for img_name, modality in mapping.items():
            items.append((pid, img_name, modality, os.path.join(IMG_ROOT, pid, img_name)))
    return items

def pick_sources(items, n_per_class):
    rng = random.Random(SEED)
    chosen = []
    for modality in QUERY_CLASSES:
        pool = [it for it in items if it[2] == modality]
        rng.shuffle(pool)
        got = 0
        for it in pool:
            if got >= n_per_class: break
            try:
                with Image.open(it[3]) as im:
                    if min(im.size) >= MIN_SIZE:
                        chosen.append(it); got += 1
            except Exception:
                continue
    return chosen

def make_contact_sheet(rows, tile=150, pad=6, label_h=16):
    """rows = liste de (titre_ligne, [(label, PIL.Image), ...]) -> une grande planche."""
    ncol = max(len(r[1]) for r in rows)
    W = pad + ncol*(tile+pad)
    H = pad + len(rows)*(tile+label_h+pad)
    sheet = Image.new("RGB", (W, H), (255,255,255))
    draw = ImageDraw.Draw(sheet)
    try: font = ImageFont.truetype("arial.ttf", 11)
    except Exception: font = ImageFont.load_default()
    y = pad
    for title, cells in rows:
        x = pad
        for label, im in cells:
            thumb = im.convert("RGB").copy()
            thumb.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            cx = x + (tile - thumb.size[0])//2
            cy = y + label_h + (tile - thumb.size[1])//2
            sheet.paste(thumb, (cx, cy))
            draw.text((x, y), label, fill=(0,0,0), font=font)
            x += tile + pad
        y += tile + label_h + pad
    return sheet

# ---------- main ----------
def main(full=False):
    n_per_class = 250 if full else 5
    print(f"Mode: {'COMPLET (500)' if full else 'MINI-TEST (10)'}")
    items = load_catalog()
    print("Catalogue total:", len(items), "panels")
    sources = pick_sources(items, n_per_class)
    print("Sources sélectionnées:", len(sources))

    out_dir = os.path.join(DATA, "benchmark_full" if full else "benchmark_test")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(SEED+1)
    gt = []          # vérité-terrain : (query_file, source_id, type)
    rows = []        # pour la planche-contact (mini-test)

    for pid, img_name, modality, path in sources:
        src_id = f"{pid}_{img_name}"
        src = Image.open(path).convert("RGB")
        sdir = os.path.join(out_dir, src_id.replace("/", "_").replace(".png",""))
        os.makedirs(sdir, exist_ok=True)

        cells = [("ORIGINAL", src)]
        # copie totale
        fv = full_variants(src)
        for name, vim in fv.items():
            fp = os.path.join(sdir, f"{name}.png"); vim.convert("RGB").save(fp)
            gt.append((fp, src_id, "full:"+name)); cells.append((name, vim))
        # JPEG q40
        fp = os.path.join(sdir, "jpeg40.jpg"); src.save(fp, "JPEG", quality=40)
        gt.append((fp, src_id, "full:jpeg40")); cells.append(("jpeg40", Image.open(fp)))
        # copie partielle
        distractors = [Image.open(rng.choice(items)[3]) for _ in range(3)]
        pv = partial_variant(src, distractors)
        fp = os.path.join(sdir, "partial.png"); pv.save(fp)
        gt.append((fp, src_id, "partial")); cells.append(("PARTIAL", pv))

        rows.append((src_id, cells))

    # vérité-terrain
    gt_path = os.path.join(out_dir, "ground_truth.json")
    json.dump([{"query": os.path.relpath(q, ROOT), "source_id": s, "type": t} for q,s,t in gt],
              open(gt_path, "w", encoding="utf-8"), indent=2)
    print("Variantes générées:", len(gt), "| vérité-terrain ->", os.path.relpath(gt_path, ROOT))

    if not full:
        sheet = make_contact_sheet(rows)
        sheet_path = os.path.join(out_dir, "montage_minitest.png")
        sheet.save(sheet_path)
        print("Planche-contact ->", os.path.relpath(sheet_path, ROOT))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    main(full=args.full)
