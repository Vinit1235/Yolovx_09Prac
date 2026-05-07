"""
YOLOv9 GPU-Accelerated Annotation Pipeline (Google Colab)
═══════════════════════════════════════════════════════════
Upload your dataset zip to Colab, run this script, and download
a cleaned + validated annotated zip ready for YOLOv9 training.

Usage (Colab cell):
  # Mount drive or upload zip first, then:
  !python colab_annotation_pipeline.py --fix

Output: annotated_dataset_clean.zip
"""

import os
import sys
import shutil
import zipfile
import time
from pathlib import Path
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── GPU imports (graceful fallback to CPU) ──
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ═════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════

BASE_DIR   = Path("Package V0.5.yolov9")
IMAGE_DIR  = BASE_DIR / "train" / "images"
LABEL_DIR  = BASE_DIR / "train" / "labels"
OUTPUT_ZIP = Path("annotated_dataset_clean.zip")
REPORT_FILE = BASE_DIR / "annotation_report.txt"

CLASS_NAMES = [
    'Box', 'Box_broken', 'Boxes', 'Open_package', 'Package',
    'Parcel', 'backpack', 'bench', 'bicycle', 'box',
    'boxes', 'brownbox', 'car', 'cardboard', 'good-parcel',
    'handbag', 'motorbike', 'motorcycle', 'p1', 'package',
    'parcel', 'person', 'refrigerator', 'suitcase', 'truck'
]
NUM_CLASSES = 25
VALID_IDS = set(range(NUM_CLASSES))
IMG_EXTS = {'.jpg', '.jpeg', '.png'}

# ═════════════════════════════════════════════════════════════════
# GPU UTILITIES
# ═════════════════════════════════════════════════════════════════

def get_device():
    """Detect best available device."""
    if HAS_TORCH and torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  🚀 GPU: {torch.cuda.get_device_name(0)} | VRAM: "
              f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        return dev
    print("  ⚡ Running on CPU (no CUDA GPU detected)")
    return torch.device("cpu") if HAS_TORCH else None


def gpu_batch_validate_coords(all_coords, device):
    """
    Validate all annotation coordinates on GPU using batched tensor ops.
    Returns list of row indices where any value is out of [0, 1].
    """
    if not all_coords:
        return []
    t = torch.tensor(all_coords, dtype=torch.float32, device=device)
    # Columns: class_id, x_c, y_c, w, h — check cols 1-4
    coords = t[:, 1:]
    out_of_range = ((coords < 0.0) | (coords > 1.0)).any(dim=1)
    bad_dims = ((coords[:, 2] <= 0) | (coords[:, 3] <= 0))
    flagged = (out_of_range | bad_dims).nonzero(as_tuple=True)[0]
    return flagged.cpu().tolist()


def gpu_validate_images(image_dir, image_files, device):
    """Check image integrity in parallel (PIL-based, threaded)."""
    corrupted = []
    def check_one(img_name):
        try:
            img = Image.open(image_dir / img_name)
            img.verify()
            return None
        except Exception as e:
            return (img_name, str(e))

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(check_one, f): f for f in image_files}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                corrupted.append(result)
    return corrupted


# ═════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ═════════════════════════════════════════════════════════════════

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def parse_line(line):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        return (int(parts[0]), float(parts[1]), float(parts[2]),
                float(parts[3]), float(parts[4]))
    except ValueError:
        return None


def run_pipeline(apply_fix=False, limit=0):
    t0 = time.time()
    print("\n┌────────────────────────────────────────────────────────────┐")
    print("│  YOLOv9 Annotation Pipeline (Colab Edition)               │")
    print(f"│  Mode: {'🔧 FIX + ZIP' if apply_fix else '🔍 DRY RUN'}                                          │")
    if limit > 0:
        print(f"│  Limit: first {limit} images (test mode)                    │")
    print("└────────────────────────────────────────────────────────────┘\n")

    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        print(f"  ✗ Dataset not found at {BASE_DIR}")
        print("  Upload & unzip your dataset first.")
        sys.exit(1)

    device = get_device()

    # ── Scan files ──
    print("\n  [1/5] Scanning directories...")
    img_map = {}
    for f in sorted(IMAGE_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            img_map[f.stem] = f.name
    # Apply limit if set (for testing on a subset)
    if limit > 0:
        all_stems = sorted(img_map.keys())[:limit]
        img_map = {s: img_map[s] for s in all_stems}
        print(f"        ⚠ Limited to first {limit} images for testing")
    lbl_map = {f.stem: f for f in LABEL_DIR.iterdir()
               if f.is_file() and f.suffix == '.txt'}
    # Also limit labels to only those matching selected images
    if limit > 0:
        lbl_map = {s: lbl_map[s] for s in img_map if s in lbl_map}
    print(f"        Images: {len(img_map)} | Labels: {len(lbl_map)}")

    # ── Image integrity (GPU-threaded) ──
    corrupted_imgs = []
    if HAS_PIL:
        print("  [2/5] Validating image integrity (threaded)...")
        corrupted_imgs = gpu_validate_images(IMAGE_DIR, list(img_map.values()), device)
        print(f"        Corrupted: {len(corrupted_imgs)}")
    else:
        print("  [2/5] Skipped image check (PIL not installed)")

    # ── Parse all labels ──
    print("  [3/5] Parsing annotations...")
    stats = {
        'missing': [], 'orphan': [], 'empty': [],
        'bad_format': [], 'bad_class': [], 'bad_range': [],
        'bad_dims': [], 'fixed_files': 0, 'fixed_lines': 0,
        'created': 0, 'total_annots': 0
    }
    class_dist = Counter()
    all_coords = []       # (class_id, x, y, w, h)
    coord_file_map = []   # (file_stem, line_num) parallel to all_coords

    for stem in sorted(img_map):
        if stem not in lbl_map:
            stats['missing'].append(img_map[stem])
            if apply_fix:
                (LABEL_DIR / f"{stem}.txt").touch()
                stats['created'] += 1
            continue

    for stem in sorted(lbl_map):
        if stem not in img_map:
            stats['orphan'].append(lbl_map[stem].name)

    for stem, lbl_path in sorted(lbl_map.items()):
        with open(lbl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            stats['empty'].append(lbl_path.name)
            continue
        for i, line in enumerate(non_empty, 1):
            parsed = parse_line(line)
            if parsed is None:
                stats['bad_format'].append((lbl_path.name, i, line.strip()))
                continue
            cid, x, y, w, h = parsed
            stats['total_annots'] += 1
            if cid not in VALID_IDS:
                stats['bad_class'].append((lbl_path.name, i, cid))
                continue
            class_dist[cid] += 1
            all_coords.append([cid, x, y, w, h])
            coord_file_map.append((stem, i))

    # ── GPU batch coordinate validation ──
    print("  [4/5] GPU batch coordinate validation...")
    if device is not None and all_coords:
        flagged_idx = gpu_batch_validate_coords(all_coords, device)
        for idx in flagged_idx:
            stem, lnum = coord_file_map[idx]
            vals = all_coords[idx]
            stats['bad_range'].append((f"{stem}.txt", lnum, vals))
        print(f"        Out-of-range: {len(flagged_idx)} annotations")
    else:
        # CPU fallback
        for i, (row, (stem, lnum)) in enumerate(zip(all_coords, coord_file_map)):
            _, x, y, w, h = row
            if x < 0 or x > 1 or y < 0 or y > 1 or w <= 0 or w > 1 or h <= 0 or h > 1:
                stats['bad_range'].append((f"{stem}.txt", lnum, row))

    # ── Apply fixes ──
    if apply_fix and stats['bad_range']:
        print("  [4b/5] Clamping out-of-range coordinates...")
        affected_stems = set()
        for fname, lnum, vals in stats['bad_range']:
            affected_stems.add(fname.replace('.txt', ''))
        for stem in affected_stems:
            lbl_path = LABEL_DIR / f"{stem}.txt"
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            fixed = []
            changed = False
            for line in lines:
                p = parse_line(line)
                if p is None:
                    fixed.append(line.rstrip())
                    continue
                cid, x, y, w, h = p
                nx, ny = clamp(x), clamp(y)
                nw, nh = clamp(w, 0.001), clamp(h, 0.001)
                if (nx, ny, nw, nh) != (x, y, w, h):
                    changed = True
                    stats['fixed_lines'] += 1
                fixed.append(f"{cid} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
            if changed:
                with open(lbl_path, 'w') as f:
                    f.write('\n'.join(fixed) + '\n')
                stats['fixed_files'] += 1

    # ── Report ──
    print("  [5/5] Generating report...")
    report = []
    report.append("=" * 65)
    report.append("  YOLOv9 ANNOTATION VALIDATION REPORT")
    report.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 65)
    report.append(f"  Images: {len(img_map)}  |  Labels: {len(lbl_map)}  |  "
                  f"Annotations: {stats['total_annots']}")
    report.append(f"  Missing labels:    {len(stats['missing'])}")
    report.append(f"  Orphan labels:     {len(stats['orphan'])}")
    report.append(f"  Empty labels:      {len(stats['empty'])}")
    report.append(f"  Bad format lines:  {len(stats['bad_format'])}")
    report.append(f"  Bad class IDs:     {len(stats['bad_class'])}")
    report.append(f"  Out-of-range:      {len(stats['bad_range'])}")
    report.append(f"  Corrupted images:  {len(corrupted_imgs)}")
    if apply_fix:
        report.append(f"  Files fixed:       {stats['fixed_files']}")
        report.append(f"  Lines clamped:     {stats['fixed_lines']}")
        report.append(f"  Labels created:    {stats['created']}")
    report.append("")
    report.append("  Class Distribution:")
    report.append(f"  {'ID':<4} {'Name':<18} {'Count':<7}")
    report.append("  " + "─" * 35)
    mx = max(class_dist.values()) if class_dist else 1
    for cid in range(NUM_CLASSES):
        cnt = class_dist.get(cid, 0)
        bar = "█" * int(cnt / mx * 20)
        report.append(f"  {cid:<4} {CLASS_NAMES[cid]:<18} {cnt:<7} {bar}")
    report.append("=" * 65)
    report_text = '\n'.join(report)
    print(report_text)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # ── ZIP output ──
    if apply_fix:
        print(f"\n  📦 Creating {OUTPUT_ZIP}...")
        with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
            for folder in [IMAGE_DIR, LABEL_DIR]:
                for fp in sorted(folder.iterdir()):
                    if fp.is_file():
                        zf.write(fp, fp.relative_to(BASE_DIR))
            # Include data.yaml and report
            yaml_path = BASE_DIR / "data.yaml"
            if yaml_path.exists():
                zf.write(yaml_path, "data.yaml")
            zf.write(REPORT_FILE, "annotation_report.txt")
        mb = OUTPUT_ZIP.stat().st_size / 1e6
        print(f"  ✅ Done! {OUTPUT_ZIP} ({mb:.1f} MB)")
        print("     Download this zip → use for YOLOv9 training.")
    else:
        issues = (len(stats['bad_format']) + len(stats['bad_range']) +
                  len(stats['bad_class']) + len(stats['missing']))
        if issues == 0:
            print("\n  ✅ All annotations are YOLOv9-compliant!")
        else:
            print(f"\n  ⚠ {issues} issues found. Run with --fix to auto-correct.")

    elapsed = time.time() - t0
    print(f"  ⏱ Completed in {elapsed:.1f}s\n")


if __name__ == "__main__":
    apply = "--fix" in sys.argv
    limit = 0
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        if idx + 1 < len(sys.argv):
            try:
                limit = int(sys.argv[idx + 1])
            except ValueError:
                print("  ⚠ Invalid --limit value, processing all images")
    run_pipeline(apply_fix=apply, limit=limit)
