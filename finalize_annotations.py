"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       YOLOv9 Annotation Refinement & Validation Script                     ║
║       Dataset: Package V0.5 — 6,377 images / 25 classes                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Purpose:
  - Validate all 6,377 label files for YOLOv9-compliant formatting.
  - Ensure every image (jpg, jpeg, png) has a matching .txt annotation.
  - Clamp/flag out-of-range normalized coordinates (must be 0–1).
  - Verify class IDs fall within nc=25 (IDs 0–24).
  - Create blank .txt files for images without annotations (negative samples).
  - Generate a detailed report for auditing.

Usage:
  python finalize_annotations.py            # Dry-run (report only, no edits)
  python finalize_annotations.py --fix      # Apply fixes automatically
"""

import os
import sys
import re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Update these paths if your dataset moves
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = Path(r"C:\Users\VINIT\Desktop\Yolov9_w1\Package V0.5.yolov9")
IMAGE_DIR  = BASE_DIR / "train" / "images"
LABEL_DIR  = BASE_DIR / "train" / "labels"
DATA_YAML  = BASE_DIR / "data.yaml"

# From data.yaml — 25 classes (IDs 0–24)
CLASS_NAMES = [
    'Box', 'Box_broken', 'Boxes', 'Open_package', 'Package',
    'Parcel', 'backpack', 'bench', 'bicycle', 'box',
    'boxes', 'brownbox', 'car', 'cardboard', 'good-parcel',
    'handbag', 'motorbike', 'motorcycle', 'p1', 'package',
    'parcel', 'person', 'refrigerator', 'suitcase', 'truck'
]
NUM_CLASSES = len(CLASS_NAMES)  # 25
VALID_CLASS_IDS = set(range(NUM_CLASSES))

# Supported image extensions (case-insensitive matching)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

class AnnotationReport:
    """Tracks all validation issues and statistics."""

    def __init__(self):
        self.total_images = 0
        self.total_labels = 0
        self.total_annotations = 0
        self.missing_labels = []           # Images with no .txt file
        self.empty_labels = []             # .txt files with 0 annotations
        self.orphan_labels = []            # .txt files with no matching image
        self.invalid_format_lines = []     # Lines that don't match YOLO format
        self.out_of_range_coords = []      # Coordinates outside [0, 1]
        self.invalid_class_ids = []        # Class IDs outside 0–24
        self.negative_dimensions = []      # Width or height <= 0
        self.class_distribution = Counter()
        self.files_fixed = 0
        self.lines_fixed = 0
        self.labels_created = 0

    def summary(self):
        lines = [
            "",
            "=" * 72,
            "  YOLOv9 ANNOTATION VALIDATION REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 72,
            "",
            f"  Total images found:         {self.total_images}",
            f"  Total label files found:    {self.total_labels}",
            f"  Total annotation lines:     {self.total_annotations}",
            "",
            "  ─── Issues ───────────────────────────────────────────────────",
            f"  Missing labels (no .txt):   {len(self.missing_labels)}",
            f"  Empty label files:          {len(self.empty_labels)}",
            f"  Orphan labels (no image):   {len(self.orphan_labels)}",
            f"  Invalid format lines:       {len(self.invalid_format_lines)}",
            f"  Out-of-range coordinates:   {len(self.out_of_range_coords)}",
            f"  Invalid class IDs:          {len(self.invalid_class_ids)}",
            f"  Negative/zero dimensions:   {len(self.negative_dimensions)}",
            "",
        ]

        if self.files_fixed or self.lines_fixed or self.labels_created:
            lines += [
                "  ─── Fixes Applied ────────────────────────────────────────────",
                f"  Label files created:        {self.labels_created}",
                f"  Files with coords clamped:  {self.files_fixed}",
                f"  Lines clamped:              {self.lines_fixed}",
                "",
            ]

        # Class distribution table
        lines += [
            "  ─── Class Distribution ───────────────────────────────────────",
            f"  {'ID':<5} {'Class Name':<20} {'Count':<8} {'Bar'}",
            "  " + "─" * 58,
        ]
        max_count = max(self.class_distribution.values()) if self.class_distribution else 1
        for cid in range(NUM_CLASSES):
            count = self.class_distribution.get(cid, 0)
            bar_len = int((count / max_count) * 30) if max_count > 0 else 0
            bar = "█" * bar_len
            name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "UNKNOWN"
            lines.append(f"  {cid:<5} {name:<20} {count:<8} {bar}")

        lines += ["", "=" * 72]

        # Show sample issues (max 5 each)
        if self.missing_labels:
            lines += ["", "  ⚠ Sample missing labels (first 5):"]
            for f in self.missing_labels[:5]:
                lines.append(f"    → {f}")

        if self.orphan_labels:
            lines += ["", "  ⚠ Sample orphan labels (first 5):"]
            for f in self.orphan_labels[:5]:
                lines.append(f"    → {f}")

        if self.invalid_format_lines:
            lines += ["", "  ⚠ Sample invalid format lines (first 5):"]
            for entry in self.invalid_format_lines[:5]:
                lines.append(f"    → {entry['file']} line {entry['line_num']}: \"{entry['content'].strip()}\"")

        if self.out_of_range_coords:
            lines += ["", "  ⚠ Sample out-of-range coords (first 5):"]
            for entry in self.out_of_range_coords[:5]:
                lines.append(f"    → {entry['file']} line {entry['line_num']}: {entry['values']}")

        if self.invalid_class_ids:
            lines += ["", "  ⚠ Sample invalid class IDs (first 5):"]
            for entry in self.invalid_class_ids[:5]:
                lines.append(f"    → {entry['file']} line {entry['line_num']}: class_id={entry['class_id']}")

        lines += ["", ""]
        return "\n".join(lines)


def get_image_stem_map(image_dir: Path) -> dict:
    """
    Build a mapping from stem (filename without extension) -> full filename
    for all supported image files. Handles .jpg, .jpeg, .PNG, etc.
    """
    stem_map = {}
    for f in image_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            stem_map[f.stem] = f.name
    return stem_map


def parse_yolo_line(line: str):
    """
    Parse a single YOLO annotation line.
    Returns (class_id, x_center, y_center, width, height) or None if invalid.
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) != 5:
        return None

    try:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width    = float(parts[3])
        height   = float(parts[4])
        return (class_id, x_center, y_center, width, height)
    except ValueError:
        return None


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def validate_and_fix_label(label_path: Path, report: AnnotationReport, apply_fix: bool) -> None:
    """
    Validate a single label file. Optionally fix clamping issues in-place.
    """
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines or all(l.strip() == '' for l in lines):
        report.empty_labels.append(label_path.name)
        return

    fixed_lines = []
    file_was_fixed = False

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped == '':
            continue  # Skip blank lines within the file

        parsed = parse_yolo_line(stripped)

        if parsed is None:
            report.invalid_format_lines.append({
                'file': label_path.name,
                'line_num': line_num,
                'content': stripped,
            })
            fixed_lines.append(stripped)
            continue

        class_id, x_c, y_c, w, h = parsed
        report.total_annotations += 1

        # ── Check class ID ──
        if class_id not in VALID_CLASS_IDS:
            report.invalid_class_ids.append({
                'file': label_path.name,
                'line_num': line_num,
                'class_id': class_id,
            })
            # Keep line as-is (don't silently drop unknown classes)
            fixed_lines.append(stripped)
            continue

        report.class_distribution[class_id] += 1

        # ── Check for negative/zero dimensions ──
        if w <= 0 or h <= 0:
            report.negative_dimensions.append({
                'file': label_path.name,
                'line_num': line_num,
                'width': w,
                'height': h,
            })

        # ── Coordinate range validation ──
        coords = [x_c, y_c, w, h]
        coord_names = ['x_center', 'y_center', 'width', 'height']
        needs_clamp = False

        for val, name in zip(coords, coord_names):
            if val < 0.0 or val > 1.0:
                needs_clamp = True
                report.out_of_range_coords.append({
                    'file': label_path.name,
                    'line_num': line_num,
                    'values': f"{name}={val}",
                })

        if needs_clamp:
            x_c = clamp(x_c)
            y_c = clamp(y_c)
            w   = clamp(w, 0.001, 1.0)  # Minimum bbox dimension
            h   = clamp(h, 0.001, 1.0)
            file_was_fixed = True
            report.lines_fixed += 1
            fixed_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        else:
            fixed_lines.append(stripped)

    # ── Write fixes if requested ──
    if apply_fix and file_was_fixed:
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines) + '\n')
        report.files_fixed += 1


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def finalize_annotations(apply_fix: bool = False):
    """
    Main entry point. Validates all annotations and optionally applies fixes.

    Args:
        apply_fix: If True, clamp out-of-range coordinates and create
                   missing label files. If False, report-only (dry run).
    """
    print()
    print("┌──────────────────────────────────────────────────────────────────┐")
    print("│  YOLOv9 Annotation Refinement Pipeline                         │")
    print(f"│  Mode: {'🔧 FIX (applying changes)' if apply_fix else '🔍 DRY RUN (report only)'}                          │")
    print("└──────────────────────────────────────────────────────────────────┘")
    print()

    # ── Verify directories exist ──
    if not IMAGE_DIR.exists():
        print(f"  ✗ Image directory not found: {IMAGE_DIR}")
        sys.exit(1)
    if not LABEL_DIR.exists():
        print(f"  ✗ Label directory not found: {LABEL_DIR}")
        sys.exit(1)

    report = AnnotationReport()

    # ── Build image stem map (handles .jpg, .jpeg, .PNG) ──
    print("  [1/4] Scanning image directory...")
    image_stem_map = get_image_stem_map(IMAGE_DIR)
    report.total_images = len(image_stem_map)
    print(f"        Found {report.total_images} images "
          f"(.jpg/.jpeg/.png)")

    # ── Collect all label stems ──
    print("  [2/4] Scanning label directory...")
    label_files = {f.stem: f for f in LABEL_DIR.iterdir()
                   if f.is_file() and f.suffix.lower() == '.txt'}
    report.total_labels = len(label_files)
    print(f"        Found {report.total_labels} label files")

    # ── Find missing labels (images without .txt) ──
    print("  [3/4] Cross-referencing images ↔ labels...")
    for stem, img_name in sorted(image_stem_map.items()):
        if stem not in label_files:
            report.missing_labels.append(img_name)
            if apply_fix:
                # Create empty .txt for negative samples
                new_label = LABEL_DIR / f"{stem}.txt"
                new_label.touch()
                report.labels_created += 1

    # ── Find orphan labels (labels without matching image) ──
    for stem, lbl_file in sorted(label_files.items()):
        if stem not in image_stem_map:
            report.orphan_labels.append(lbl_file.name)

    print(f"        Missing labels: {len(report.missing_labels)} | "
          f"Orphan labels: {len(report.orphan_labels)}")

    # ── Validate each label file ──
    print("  [4/4] Validating annotation formatting...")
    processed = 0
    total = len(label_files)
    for stem, lbl_path in sorted(label_files.items()):
        validate_and_fix_label(lbl_path, report, apply_fix)
        processed += 1
        if processed % 1000 == 0 or processed == total:
            pct = int((processed / total) * 100)
            print(f"        Progress: {processed}/{total} ({pct}%)")

    # ── Print report ──
    report_text = report.summary()
    print(report_text)

    # ── Save report to file ──
    report_path = BASE_DIR / "annotation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  📄 Report saved to: {report_path}")

    # ── Final verdict ──
    total_issues = (
        len(report.invalid_format_lines)
        + len(report.out_of_range_coords)
        + len(report.invalid_class_ids)
        + len(report.negative_dimensions)
    )

    if total_issues == 0 and not report.missing_labels:
        print()
        print("  ✅ All 6,377 annotations are YOLOv9-compliant and training-ready!")
    elif apply_fix:
        print()
        print(f"  ✅ Fixes applied. {report.files_fixed} files corrected, "
              f"{report.labels_created} empty labels created.")
        print("     Re-run without --fix to confirm all issues are resolved.")
    else:
        print()
        print(f"  ⚠  Found {total_issues} annotation issues + "
              f"{len(report.missing_labels)} missing labels.")
        print("     Run with --fix to auto-correct clamping & missing files:")
        print("       python finalize_annotations.py --fix")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    apply = "--fix" in sys.argv
    finalize_annotations(apply_fix=apply)
