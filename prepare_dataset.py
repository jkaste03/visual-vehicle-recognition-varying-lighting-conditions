import argparse
import re
from pathlib import Path
from urllib.parse import unquote
from collections import defaultdict

import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _existing_cols(df: pd.DataFrame, cols):
    return [c for c in cols if c in df.columns]


def build_year_column(df: pd.DataFrame) -> pd.DataFrame:
    year_cols = _existing_cols(df, ["year_3", "year_s", "year_x", "year_y"])
    if "year" in df.columns:
        return df
    if not year_cols:
        df["year"] = ""
        return df

    # bfill over the year_* columns and take first non-null
    df["year"] = df[year_cols].bfill(axis="columns").iloc[:, 0]
    return df


def apply_label_fixes(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure strings
    for c in ["model", "year"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    # 3/Y -> "3 2017–2023" etc.
    for m in ["Y", "3"]:
        mask = df["model"].eq(m) & df["year"].ne("")
        df.loc[mask, "model"] = df.loc[mask, "model"] + " " + df.loc[mask, "year"]

    # S 2012–2015 stays explicit, remaining S -> S 2016–nå
    mask_early_s = df["model"].eq("S") & df["year"].eq("2012–2015")
    df.loc[mask_early_s, "model"] = "S 2012–2015"

    mask_s_rest = df["model"].eq("S") & ~mask_early_s
    df.loc[mask_s_rest, "model"] = "S 2016–nå"

    return df


def normalize_labelstudio_image(value: str) -> str:
    """
    Handles typical Label Studio values such as:
      /data/local-files/?d=Users%5C...\%25~n1-192.jpg
      /data/upload/3/c862f010-87d5ed96-...-homepage-model3.jpg

    Returns basename (filename only), URL-decoded.
    """
    if pd.isna(value):
        return ""

    s = str(value).strip()

    # remove local-files prefix if present
    if "local-files" in s and "d=" in s:
        s = s.split("d=", 1)[1]

    # URL decode and remove query
    s = unquote(s)
    s = s.split("?", 1)[0]

    # normalize slashes and take basename
    s = s.replace("\\", "/")
    return Path(s).name


def strip_upload_prefixes(name: str):
    """
    Label Studio sometimes prefixes basenames with IDs, e.g.
      9ba63789-IMG_4105__Kopi.jpeg
      c862f010-87d5ed96-774d-4d07-b337-631ada5c4bcd-homepage-model3.jpg
      bbf07d77-a9d1b6ea-tesla-model-e-125255b425255d.jpg

    We return a list of candidate basenames to try (best-effort).
    """
    if not name:
        return []

    candidates = []
    seen = set()

    def add(x):
        if x and x not in seen:
            seen.add(x)
            candidates.append(x)

    add(name)

    # common IMG/TMG typo fallback (seen in your examples)
    if re.match(r"(?i)^tmg_", name):
        add(re.sub(r"(?i)^tmg_", "IMG_", name))
    if re.match(r"(?i)^img_", name):
        add(re.sub(r"(?i)^img_", "TMG_", name))

    # 8-hex prefix
    add(re.sub(r"^[0-9a-fA-F]{8}-", "", name))

    # 8hex-8hex- prefix
    add(re.sub(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{8}-", "", name))

    # Standard GUID prefix (8-4-4-4-12) + hyphen
    add(re.sub(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}-", "", name))

    # “Weird GUID” prefix (8-8-4-4-4-12) + hyphen (matches your c862f010-87d5ed96-...)
    add(re.sub(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}-", "", name))

    return candidates


def stem_key(filename: str) -> str:
    """
    Loose key: remove separators/punct so that:
      'IMG_4105 - Kopi.jpeg' and 'IMG_4105__Kopi.jpeg' -> same key
    """
    p = Path(filename)
    stem = p.stem.lower()
    return re.sub(r"[^a-z0-9]+", "", stem)


def build_file_index(img_root: Path):
    files = []
    for p in img_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)

    exact = defaultdict(list)      # name.lower -> [paths]
    by_stem = defaultdict(list)    # stem_key -> [paths]

    for p in files:
        n = p.name.lower()
        exact[n].append(p)
        by_stem[stem_key(p.name)].append(p)

    return files, exact, by_stem


def pick_one(paths):
    """
    Pick a deterministic 'best' path among candidates.
    Prefer .jpg/.jpeg over other formats when multiple matches exist.
    """
    ext_prio = {".jpg": 0, ".jpeg": 1, ".png": 2, ".webp": 3}

    def key(p):
        return (ext_prio.get(p.suffix.lower(), 99), len(str(p)), str(p))

    return sorted(paths, key=key)[0]

def ensure_jpg(path: Path, dry_run: bool = False) -> Path:
    """
    Ensure the chosen image exists as .jpg and return the .jpg path.

    - .jpeg -> rename to .jpg (only if .jpg doesn't already exist)
    - .png/.webp -> convert to .jpg (keeps original)
    - dry_run=True: do not touch disk
    """
    if dry_run:
        return path

    sfx = path.suffix.lower()
    if sfx == ".jpg":
        return path

    jpg_path = path.with_suffix(".jpg")

    # If a .jpg already exists next to it, use that.
    if jpg_path.exists():
        return jpg_path

    if sfx == ".jpeg":
        path.rename(jpg_path)
        return jpg_path

    if sfx in {".png", ".webp"}:
        from PIL import Image, ImageOps

        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        jpg_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(jpg_path, quality=95, optimize=True)
        return jpg_path

    # Unknown extension – leave as-is
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", required=True, help="Root folder with images (can contain subfolders)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--unmatched-out", default="", help="Optional CSV for unmatched rows")
    ap.add_argument("--ambiguous-out", default="", help="Optional CSV for ambiguous matches")
    ap.add_argument("--internal-csv", default="annotations/internal.csv")
    ap.add_argument("--external-csv", default="annotations/external.csv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-jpg", action="store_true", help="Convert/rename matched images so output CSV always points to .jpg")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    img_root = Path(args.img_root).resolve()

    internal_path = (base_dir / args.internal_csv).resolve()
    external_path = (base_dir / args.external_csv).resolve()

    internal = pd.read_csv(internal_path)
    external = pd.read_csv(external_path)

    wanted = ["color", "image", "lighting", "model", "year_3", "year_s", "year_x", "year_y"]
    internal = internal[_existing_cols(internal, wanted)].copy()
    external = external[_existing_cols(external, wanted)].copy()

    internal["source"] = "internal"
    external["source"] = "external"

    internal = build_year_column(internal)
    external = build_year_column(external)

    # Drop year_* cols after we made year
    drop_cols = [c for c in ["year_3", "year_s", "year_x", "year_y"] if c in internal.columns]
    internal.drop(columns=drop_cols, inplace=True, errors="ignore")
    drop_cols = [c for c in ["year_3", "year_s", "year_x", "year_y"] if c in external.columns]
    external.drop(columns=drop_cols, inplace=True, errors="ignore")

    internal = apply_label_fixes(internal)
    external = apply_label_fixes(external)

    combined = pd.concat([internal, external], ignore_index=True)

    # Index image files
    all_files, exact_map, stem_map = build_file_index(img_root)

    matched_rows = []
    unmatched_rows = []
    ambiguous_rows = []

    for _, row in combined.iterrows():
        original = row.get("image", "")
        base = normalize_labelstudio_image(original)
        candidates = strip_upload_prefixes(base)

        found_paths = []
        found_via = None

        # Try exact by candidate names first
        for cand in candidates:
            paths = exact_map.get(cand.lower(), [])
            if len(paths) == 1:
                found_paths = paths
                found_via = f"exact:{cand}"
                break
            if len(paths) > 1:
                found_paths = paths
                found_via = f"exact_ambiguous:{cand}"
                break

        # If not resolved, try loose stem match
        if not found_paths:
            for cand in candidates:
                sk = stem_key(cand)
                paths = stem_map.get(sk, [])
                if len(paths) == 1:
                    found_paths = paths
                    found_via = f"stem:{cand}"
                    break
                if len(paths) > 1:
                    found_paths = paths
                    found_via = f"stem_ambiguous:{cand}"
                    break

        if not found_paths:
            u = row.to_dict()
            u["image_original"] = original
            u["image_normalized"] = base
            u["candidates"] = "|".join(candidates[:10])
            unmatched_rows.append(u)
            continue

        chosen = pick_one(found_paths)
        if args.force_jpg:
            chosen = ensure_jpg(chosen, dry_run=args.dry_run)
        rel = chosen.relative_to(img_root).as_posix()

        out_row = row.to_dict()
        out_row["image"] = rel  # store relative path (safe if duplicates exist)
        matched_rows.append(out_row)

        if len(found_paths) > 1:
            a = {
                "image_original": original,
                "image_normalized": base,
                "chosen": rel,
                "num_matches": len(found_paths),
                "matches": "|".join([p.relative_to(img_root).as_posix() for p in found_paths]),
                "via": found_via or "",
            }
            ambiguous_rows.append(a)

    # Report
    print("\n=== MATCH-RAPPORT ===")
    print(f"Bildefiler i mappe: {len(all_files)}")
    print(f"CSV-rader:          {len(combined)}")
    print(f"Match funnet:       {len(matched_rows)}")
    print(f"Umatchet:           {len(unmatched_rows)}")
    print(f"Ambigue (flere):    {len(ambiguous_rows)}")
    print(f"Bildemappe:         {img_root}")
    print(f"Output:             {Path(args.out).resolve()}")

    if unmatched_rows:
        print("\nFørste 25 umatchetede (image_original):")
        for x in unmatched_rows[:25]:
            print(x.get("image_original", ""))

    if args.dry_run:
        print("\n(DRY RUN) Skrev ikke fil.")
        return

    # Write outputs
    out_df = pd.DataFrame(matched_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8")

    if args.unmatched_out:
        um_df = pd.DataFrame(unmatched_rows)
        Path(args.unmatched_out).parent.mkdir(parents=True, exist_ok=True)
        um_df.to_csv(args.unmatched_out, index=False, encoding="utf-8")

    if args.ambiguous_out:
        am_df = pd.DataFrame(ambiguous_rows)
        Path(args.ambiguous_out).parent.mkdir(parents=True, exist_ok=True)
        am_df.to_csv(args.ambiguous_out, index=False, encoding="utf-8")

    print("\nOK: Skrev renset CSV.")


if __name__ == "__main__":
    main()