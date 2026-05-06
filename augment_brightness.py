"""
YOLOv9 Multi-Augmentation Script (GPU PC / Colab)
═══════════════════════════════════════════════════
Applies 4 augmentation types to random subsets of images:
  • Brightness  (±15%)  — randomly brighter or darker
  • Contrast    (±15%)  — more vivid or flatter
  • Shadow      (darken a random region to simulate cast shadows)
  • Warmth      (shift color temperature warm/cool)

Each augmentation affects ~2-3% of images independently.
Labels are copied unchanged (these transforms don't affect bounding boxes).

Usage:
  python augment_brightness.py                  # Preview (dry run)
  python augment_brightness.py --apply          # Apply all + create zip
  python augment_brightness.py --apply --pct 3  # 3% per augmentation type

Output: augmented_dataset.zip
"""

import os
import sys
import random
import shutil
import zipfile
import time
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
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

# Augmentation parameters
DEFAULT_PCT    = 3        # % of images per augmentation type
SEED           = 42

# Brightness: factor < 1 = darker, > 1 = brighter
BRIGHTNESS_RANGE = (0.85, 1.15)

# Contrast: factor < 1 = flatter, > 1 = more vivid
CONTRAST_RANGE   = (0.85, 1.15)

# Shadow: darken a random rectangular region
SHADOW_DARKNESS  = (0.4, 0.7)   # how dark the shadow is (lower = darker)
SHADOW_SIZE_FRAC = (0.2, 0.5)   # shadow covers 20-50% of image

# Warmth: color temperature shift
WARMTH_RANGE     = (-20, 20)     # negative = cooler (blue), positive = warmer (orange)


# ═════════════════════════════════════════════════════════════════
# AUGMENTATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════

def aug_brightness(img, rng):
    """Adjust overall brightness."""
    factor = rng.uniform(*BRIGHTNESS_RANGE)
    enhanced = ImageEnhance.Brightness(img).enhance(factor)
    direction = "brighter" if factor > 1.0 else "darker"
    return enhanced, f"brightness {factor:.3f}x ({direction})"


def aug_contrast(img, rng):
    """Adjust overall contrast."""
    factor = rng.uniform(*CONTRAST_RANGE)
    enhanced = ImageEnhance.Contrast(img).enhance(factor)
    direction = "more vivid" if factor > 1.0 else "flatter"
    return enhanced, f"contrast {factor:.3f}x ({direction})"


def aug_shadow(img, rng):
    """
    Simulate a cast shadow by darkening a random trapezoidal region.
    Creates a gradient mask for realistic shadow falloff.
    """
    w, h = img.size
    arr = np.array(img, dtype=np.float32)

    # Random shadow region (trapezoid-ish)
    sw = int(w * rng.uniform(*SHADOW_SIZE_FRAC))
    sh = int(h * rng.uniform(*SHADOW_SIZE_FRAC))
    sx = rng.randint(0, max(1, w - sw))
    sy = rng.randint(0, max(1, h - sh))
    darkness = rng.uniform(*SHADOW_DARKNESS)

    # Create gradient mask for smooth shadow edges
    mask = np.ones((h, w), dtype=np.float32)
    # Core shadow region
    mask[sy:sy+sh, sx:sx+sw] = darkness

    # Feather edges (gaussian-like falloff)
    feather = max(10, min(sw, sh) // 4)
    for i in range(feather):
        alpha = darkness + (1.0 - darkness) * (i / feather)
        # Top edge
        if sy - i >= 0:
            mask[sy - i, sx:sx+sw] = alpha
        # Bottom edge
        if sy + sh + i < h:
            mask[sy + sh + i, sx:sx+sw] = alpha
        # Left edge
        if sx - i >= 0:
            mask[sy:sy+sh, sx - i] = alpha
        # Right edge
        if sx + sw + i < w:
            mask[sy:sy+sh, sx + sw + i] = alpha

    # Apply shadow
    mask_3d = mask[:, :, np.newaxis]
    arr = arr * mask_3d
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    result = Image.fromarray(arr)
    pct_covered = (sw * sh) / (w * h) * 100
    return result, f"shadow region ({pct_covered:.0f}% area, {darkness:.2f} darkness)"


def aug_warmth(img, rng):
    """
    Shift color temperature.
    Positive = warmer (boost red, reduce blue).
    Negative = cooler (boost blue, reduce red).
    """
    arr = np.array(img, dtype=np.float32)
    shift = rng.uniform(*WARMTH_RANGE)

    if shift > 0:
        # Warmer: boost red, slightly boost green, reduce blue
        arr[:, :, 0] = np.clip(arr[:, :, 0] + shift * 1.2, 0, 255)  # R
        arr[:, :, 1] = np.clip(arr[:, :, 1] + shift * 0.3, 0, 255)  # G
        arr[:, :, 2] = np.clip(arr[:, :, 2] - shift * 0.8, 0, 255)  # B
        direction = "warmer"
    else:
        # Cooler: boost blue, reduce red
        arr[:, :, 0] = np.clip(arr[:, :, 0] + shift * 1.0, 0, 255)  # R
        arr[:, :, 1] = np.clip(arr[:, :, 1] + shift * 0.1, 0, 255)  # G
        arr[:, :, 2] = np.clip(arr[:, :, 2] - shift * 1.2, 0, 255)  # B
        direction = "cooler"

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    result = Image.fromarray(arr)
    return result, f"warmth {shift:+.1f} ({direction})"


# Map of augmentation types
AUGMENTATIONS = {
    "brightness": aug_brightness,
    "contrast":   aug_contrast,
    "shadow":     aug_shadow,
    "warmth":     aug_warmth,
}


# ═════════════════════════════════════════════════════════════════
# PROCESSING
# ═════════════════════════════════════════════════════════════════

def save_image(img, dst_path, src_ext):
    """Save image preserving format."""
    fmt = "JPEG" if src_ext in (".jpg", ".jpeg") else "PNG"
    img.save(dst_path, format=fmt, quality=95)


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def process_image(stem, img_name, img_dir, lbl_dir, out_img_dir, out_lbl_dir,
                  assignments, apply_mode):
    """
    Process one image.
    assignments: dict of {aug_type: True} for this stem.
    If multiple augmentations hit the same image, they stack.
    """
    src_img = img_dir / img_name
    dst_img = out_img_dir / img_name
    src_lbl = lbl_dir / f"{stem}.txt"
    dst_lbl = out_lbl_dir / f"{stem}.txt"

    # Always copy label
    if src_lbl.exists():
        copy_file(src_lbl, dst_lbl)
    else:
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.touch()

    if not assignments or not apply_mode:
        copy_file(src_img, dst_img)
        return stem, [], "copied"

    # Apply augmentations
    try:
        img = Image.open(src_img).convert("RGB")
        rng = random.Random(hash(stem) + SEED)  # deterministic per image
        applied = []

        for aug_name in sorted(assignments.keys()):
            aug_fn = AUGMENTATIONS[aug_name]
            img, desc = aug_fn(img, rng)
            applied.append(f"{aug_name}: {desc}")

        save_image(img, dst_img, Path(img_name).suffix.lower())
        return stem, applied, "augmented"

    except Exception as e:
        copy_file(src_img, dst_img)
        return stem, [], f"error: {e}"


# ═════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════

def run(apply_mode=False, pct=DEFAULT_PCT):
    t0 = time.time()

    num_types = len(AUGMENTATIONS)
    print("\n┌────────────────────────────────────────────────────────────┐")
    print("│  YOLOv9 Multi-Augmentation Pipeline                      │")
    print(f"│  Mode: {'🔧 APPLY + ZIP' if apply_mode else '🔍 PREVIEW (dry run)'}                                      │")
    print(f"│  Augmentations: brightness, contrast, shadow, warmth     │")
    print(f"│  Each type: ~{pct}% of images (independent random subsets)  │")
    print("└────────────────────────────────────────────────────────────┘\n")

    if not HAS_PIL:
        print("  ✗ Pillow not found. Install: pip install Pillow")
        sys.exit(1)
    if not HAS_NP:
        print("  ✗ NumPy not found. Install: pip install numpy")
        sys.exit(1)
    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        print(f"  ✗ Dataset not found at {BASE_DIR}")
        sys.exit(1)

    # ── Scan ──
    print("  [1/5] Scanning images...")
    img_map = {}
    for f in sorted(IMAGE_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            img_map[f.stem] = f.name

    total = len(img_map)
    num_per_type = max(1, int(total * pct / 100))
    all_stems = list(img_map.keys())
    print(f"        Total images: {total}")
    print(f"        Per augmentation type: ~{num_per_type} images ({pct}%)")

    # ── Select random subsets (independent per type) ──
    print("\n  [2/5] Selecting random subsets...")
    assignments = defaultdict(dict)  # stem → {aug_type: True}

    for i, aug_name in enumerate(sorted(AUGMENTATIONS.keys())):
        rng = random.Random(SEED + i * 1000)
        selected = set(rng.sample(all_stems, num_per_type))
        for stem in selected:
            assignments[stem][aug_name] = True
        print(f"        {aug_name:12s}: {len(selected)} images selected")

    # Count unique affected images
    affected = len(assignments)
    multi_aug = sum(1 for v in assignments.values() if len(v) > 1)
    print(f"\n        Unique images affected: {affected}")
    print(f"        Images with 2+ augmentations: {multi_aug}")

    # ── Preview ──
    print("\n  [3/5] Preview of augmentations:")
    preview_count = 0
    for stem in sorted(assignments.keys()):
        if preview_count >= 12:
            remaining = affected - preview_count
            print(f"        ... and {remaining} more images")
            break
        augs = ", ".join(sorted(assignments[stem].keys()))
        name = img_map[stem][:50]
        print(f"        • {name:<50s}  → [{augs}]")
        preview_count += 1

    if not apply_mode:
        total_ops = sum(len(v) for v in assignments.values())
        print(f"\n  ── Summary ──")
        print(f"     Total images:          {total}")
        print(f"     Unique affected:       {affected} ({affected/total*100:.1f}%)")
        print(f"     Total augment ops:     {total_ops}")
        print(f"     Images unchanged:      {total - affected}")
        print(f"\n  ℹ Run with --apply to process.")
        print(f"    python augment_brightness.py --apply --pct {pct}")
        return

    # ── Process all images ──
    print(f"\n  [4/5] Processing {total} images...")
    out_img_dir = OUTPUT_DIR / "train" / "images"
    out_lbl_dir = OUTPUT_DIR / "train" / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    aug_log = []
    copied_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for stem, img_name in img_map.items():
            stem_augs = assignments.get(stem, {})
            fut = pool.submit(
                process_image,
                stem, img_name, IMAGE_DIR, LABEL_DIR,
                out_img_dir, out_lbl_dir, stem_augs, apply_mode
            )
            futures[fut] = stem

        done = 0
        for fut in as_completed(futures):
            done += 1
            stem, applied, status = fut.result()
            if applied:
                aug_log.append((img_map[stem], applied))
            elif "error" in str(status):
                error_count += 1
            else:
                copied_count += 1

            if done % 500 == 0 or done == total:
                pct_done = done / total * 100
                bar = "█" * int(pct_done / 2.5) + "░" * (40 - int(pct_done / 2.5))
                print(f"        [{bar}] {done}/{total} ({pct_done:.0f}%)")

    # Copy data.yaml
    yaml_src = BASE_DIR / "data.yaml"
    if yaml_src.exists():
        copy_file(yaml_src, OUTPUT_DIR / "data.yaml")

    # ── Stats ──
    aug_type_counts = defaultdict(int)
    for _, applied_list in aug_log:
        for desc in applied_list:
            aug_type = desc.split(":")[0]
            aug_type_counts[aug_type] += 1

    print(f"\n  ── Augmentation Summary ──")
    print(f"     Total processed:     {total}")
    print(f"     Augmented images:    {len(aug_log)}")
    for aug_name in sorted(aug_type_counts):
        print(f"       {aug_name:12s}:     {aug_type_counts[aug_name]}")
    print(f"     Copied unchanged:    {copied_count}")
    print(f"     Errors:              {error_count}")

    # Write log
    log_path = OUTPUT_DIR / "augmentation_log.txt"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("YOLOv9 Multi-Augmentation Log\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total images: {total}\n")
        f.write(f"Augmented: {len(aug_log)}\n")
        f.write(f"Per-type target: {pct}% ({num_per_type} images)\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Augmentation types: {', '.join(sorted(AUGMENTATIONS.keys()))}\n")
        f.write(f"Brightness range: {BRIGHTNESS_RANGE}\n")
        f.write(f"Contrast range: {CONTRAST_RANGE}\n")
        f.write(f"Shadow darkness: {SHADOW_DARKNESS}\n")
        f.write(f"Warmth range: {WARMTH_RANGE}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Image':<55} Augmentations Applied\n")
        f.write("─" * 90 + "\n")
        for img_name, applied_list in sorted(aug_log):
            f.write(f"{img_name:<55}\n")
            for desc in applied_list:
                f.write(f"    ↳ {desc}\n")

    # ── ZIP ──
    print(f"\n  [5/5] Creating {OUTPUT_ZIP}...")
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for fname in sorted(files):
                fpath = Path(root) / fname
                arcname = fpath.relative_to(OUTPUT_DIR)
                zf.write(fpath, arcname)
    mb = OUTPUT_ZIP.stat().st_size / 1e6
    print(f"  ✅ Done! {OUTPUT_ZIP} ({mb:.1f} MB)")

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    print(f"  🧹 Cleaned up temp files")

    elapsed = time.time() - t0
    print(f"  ⏱ Completed in {elapsed:.1f}s")
    print(f"\n  Next: run annotation pipeline on the augmented data:")
    print(f"    unzip augmented_dataset.zip -d augmented_data")
    print(f"    python colab_annotation_pipeline.py --fix\n")


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
