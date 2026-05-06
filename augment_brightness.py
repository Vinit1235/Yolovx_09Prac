"""
YOLOv9 Brightness Augmentation Script (Google Colab)
════════════════════════════════════════════════════════
Randomly adjusts brightness on a subset of images by ±2-3%.
Labels are copied as-is (brightness changes don't affect bounding boxes).

Usage (Colab):
  !python augment_brightness.py                  # Preview mode (no changes)
  !python augment_brightness.py --apply          # Apply augmentation + create zip
  !python augment_brightness.py --apply --pct 5  # Augment 5% of images instead

Output: augmented_dataset.zip
"""

import os
import sys
import random
import shutil
import zipfile
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from PIL import Image, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

# ═════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════

BASE_DIR       = Path("Package V0.5.yolov9")
IMAGE_DIR      = BASE_DIR / "train" / "images"
LABEL_DIR      = BASE_DIR / "train" / "labels"
OUTPUT_DIR     = Path("augmented_output")
OUTPUT_ZIP     = Path("augmented_dataset.zip")
IMG_EXTS       = {'.jpg', '.jpeg', '.png'}

# Augmentation settings
DEFAULT_PCT    = 3       # % of images to augment (2-3%)
BRIGHTNESS_LO  = 0.85   # min brightness factor (darker)
BRIGHTNESS_HI  = 1.15   # max brightness factor (brighter)
SEED           = 42      # reproducibility


# ═════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═════════════════════════════════════════════════════════════════

def adjust_brightness_pil(src_path, dst_path, factor):
    """Adjust image brightness using PIL. factor < 1 = darker, > 1 = brighter."""
    img = Image.open(src_path).convert("RGB")
    enhancer = ImageEnhance.Brightness(img)
    img_aug = enhancer.enhance(factor)
    # Preserve original format
    ext = src_path.suffix.lower()
    fmt = "JPEG" if ext in (".jpg", ".jpeg") else "PNG"
    img_aug.save(dst_path, format=fmt, quality=95)
    return factor


def adjust_brightness_numpy(src_path, dst_path, factor):
    """Fallback: adjust brightness using raw numpy (no PIL enhance)."""
    img = Image.open(src_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr = arr * factor
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    result = Image.fromarray(arr)
    ext = src_path.suffix.lower()
    fmt = "JPEG" if ext in (".jpg", ".jpeg") else "PNG"
    result.save(dst_path, format=fmt, quality=95)
    return factor


def copy_file(src, dst):
    """Copy a file, creating parent dirs if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def process_one_image(stem, img_name, img_dir, lbl_dir, out_img_dir, out_lbl_dir,
                      selected_stems, apply_mode):
    """Process a single image: augment if selected, otherwise copy as-is."""
    src_img = img_dir / img_name
    dst_img = out_img_dir / img_name

    # Copy label (always unchanged)
    src_lbl = lbl_dir / f"{stem}.txt"
    dst_lbl = out_lbl_dir / f"{stem}.txt"

    if src_lbl.exists():
        copy_file(src_lbl, dst_lbl)
    else:
        # Create empty label (negative sample)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.touch()

    if stem in selected_stems and apply_mode:
        factor = random.uniform(BRIGHTNESS_LO, BRIGHTNESS_HI)
        direction = "🔆 brighter" if factor > 1.0 else "🌑 darker"
        try:
            if HAS_PIL:
                adjust_brightness_pil(src_img, dst_img, factor)
            elif HAS_NP:
                adjust_brightness_numpy(src_img, dst_img, factor)
            else:
                copy_file(src_img, dst_img)
                return stem, None, "skipped (no PIL/numpy)"
            return stem, factor, direction
        except Exception as e:
            copy_file(src_img, dst_img)
            return stem, None, f"error: {e}"
    else:
        copy_file(src_img, dst_img)
        return stem, None, "copied"


# ═════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════

def run(apply_mode=False, pct=DEFAULT_PCT):
    t0 = time.time()

    print("\n┌────────────────────────────────────────────────────────────┐")
    print("│  YOLOv9 Brightness Augmentation                          │")
    print(f"│  Mode: {'🔧 APPLY + ZIP' if apply_mode else '🔍 PREVIEW (dry run)'}                                      │")
    print(f"│  Target: ~{pct}% of images  |  Range: ±15% brightness       │")
    print("└────────────────────────────────────────────────────────────┘\n")

    # ── Validate environment ──
    if not HAS_PIL:
        print("  ⚠ PIL not found. Install: pip install Pillow")
        sys.exit(1)

    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        print(f"  ✗ Dataset not found at {BASE_DIR}")
        print("    Upload & unzip your dataset first.")
        sys.exit(1)

    # ── Scan images ──
    print("  [1/4] Scanning images...")
    img_map = {}  # stem → filename
    for f in sorted(IMAGE_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            img_map[f.stem] = f.name

    total = len(img_map)
    num_augment = max(1, int(total * pct / 100))
    print(f"        Total images: {total}")
    print(f"        Will augment: {num_augment} ({pct}%)")

    # ── Select random images ──
    print("  [2/4] Selecting random images for augmentation...")
    random.seed(SEED)
    all_stems = list(img_map.keys())
    selected = set(random.sample(all_stems, num_augment))

    # Show sample of what will be augmented
    sample_show = list(selected)[:10]
    for s in sample_show:
        factor = random.uniform(BRIGHTNESS_LO, BRIGHTNESS_HI)
        direction = "brighter" if factor > 1.0 else "darker"
        print(f"        • {img_map[s][:55]:<55}  → {factor:.3f}x ({direction})")
    if num_augment > 10:
        print(f"        ... and {num_augment - 10} more")

    if not apply_mode:
        print(f"\n  ℹ Preview complete. Run with --apply to process all images.")
        print(f"    Command: !python augment_brightness.py --apply --pct {pct}")
        return

    # ── Process all images ──
    print(f"\n  [3/4] Processing {total} images ({num_augment} augmented, "
          f"{total - num_augment} copied)...")

    # Reset seed so augmentation factors match preview
    random.seed(SEED)
    _ = random.sample(all_stems, num_augment)  # advance RNG to match

    out_img_dir = OUTPUT_DIR / "train" / "images"
    out_lbl_dir = OUTPUT_DIR / "train" / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    augmented_log = []
    copied_count = 0
    error_count = 0

    # Use threading for I/O-bound work
    random.seed(SEED + 1)  # fresh seed for actual factors
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for stem, img_name in img_map.items():
            fut = pool.submit(
                process_one_image,
                stem, img_name, IMAGE_DIR, LABEL_DIR,
                out_img_dir, out_lbl_dir, selected, apply_mode
            )
            futures[fut] = stem

        done = 0
        for fut in as_completed(futures):
            done += 1
            stem, factor, info = fut.result()
            if factor is not None:
                augmented_log.append((img_map[stem], factor, info))
            elif "error" in str(info):
                error_count += 1
            else:
                copied_count += 1

            if done % 500 == 0 or done == total:
                print(f"        Progress: {done}/{total} "
                      f"({done/total*100:.0f}%)")

    # Copy data.yaml
    yaml_src = BASE_DIR / "data.yaml"
    yaml_dst = OUTPUT_DIR / "data.yaml"
    if yaml_src.exists():
        copy_file(yaml_src, yaml_dst)

    # ── Summary ──
    brighter = sum(1 for _, f, _ in augmented_log if f > 1.0)
    darker = len(augmented_log) - brighter

    print(f"\n  ── Augmentation Summary ──")
    print(f"     Total processed:  {total}")
    print(f"     Augmented:        {len(augmented_log)}")
    print(f"       🔆 Brighter:    {brighter}")
    print(f"       🌑 Darker:      {darker}")
    print(f"     Copied as-is:     {copied_count}")
    print(f"     Errors:           {error_count}")

    # Write augmentation log
    log_path = OUTPUT_DIR / "augmentation_log.txt"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"YOLOv9 Brightness Augmentation Log\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total images: {total}\n")
        f.write(f"Augmented: {len(augmented_log)} ({pct}%)\n")
        f.write(f"Brightness range: [{BRIGHTNESS_LO}, {BRIGHTNESS_HI}]\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"{'Image':<60} {'Factor':<8} {'Direction'}\n")
        f.write(f"{'─' * 80}\n")
        for img_name, factor, direction in sorted(augmented_log):
            f.write(f"{img_name:<60} {factor:<8.4f} {direction}\n")

    # ── Create ZIP ──
    print(f"\n  [4/4] Creating {OUTPUT_ZIP}...")
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for fname in sorted(files):
                fpath = Path(root) / fname
                arcname = fpath.relative_to(OUTPUT_DIR)
                zf.write(fpath, arcname)
    mb = OUTPUT_ZIP.stat().st_size / 1e6
    print(f"  ✅ Done! {OUTPUT_ZIP} ({mb:.1f} MB)")

    # Cleanup temp output dir
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    print(f"  🧹 Cleaned up temporary files")

    elapsed = time.time() - t0
    print(f"  ⏱ Completed in {elapsed:.1f}s")
    print(f"\n  Next step: unzip & run annotation pipeline:")
    print(f"    !unzip augmented_dataset.zip -d Package_augmented")
    print(f"    !python colab_annotation_pipeline.py --fix\n")


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    apply = "--apply" in sys.argv
    pct = DEFAULT_PCT
    if "--pct" in sys.argv:
        idx = sys.argv.index("--pct")
        if idx + 1 < len(sys.argv):
            try:
                pct = int(sys.argv[idx + 1])
                pct = max(1, min(100, pct))
            except ValueError:
                print("  ⚠ Invalid --pct value, using default 3%")
    run(apply_mode=apply, pct=pct)
