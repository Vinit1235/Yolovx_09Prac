# 📦 Package V0.5 — YOLOv9 Dataset Documentation

## 1. Dataset Overview

| Field | Value |
|---|---|
| **Name** | Package V0.5 — Swippe Dataset |
| **Source** | [Roboflow](https://app.roboflow.com/swippe/swippe/dataset) |
| **Exported** | May 6, 2026 at 4:17 AM GMT |
| **Format** | YOLOv9 (YOLO Darknet TXT) |
| **Total Images** | 6,377 |
| **Total Classes** | 25 (nc: 25) |
| **Pre-processing** | None applied |
| **Augmentation** | None applied |
| **License** | Private |
| **Primary Annotations** | Suitcase, person, and package-related objects |

### Source README (from Roboflow)

> *Package V0.5 - vdataset swippe*
>
> This dataset was exported via roboflow.com. The dataset includes 6,377 images.
> Suitcase-suitcase-person-person are annotated in YOLOv9 format.
> No pre-processing or augmentation was applied.

---

## 2. Directory Structure

```
Package V0.5.yolov9/
├── README.roboflow.txt          # Roboflow export metadata
├── data.yaml                    # YOLOv9 training configuration
└── train/
    ├── images/                  # 6,377 images (.jpg, .jpeg, .PNG)
    │   ├── 0a3a22b9b53f81f08b289072_png_jpg.rf.zF862svGejNqaiSXkbjI.jpg
    │   ├── 000001_jpg.rf.FOnuxci3vufKn4LKc6iv.jpg
    │   └── ...
    └── labels/                  # 6,377 annotation files (.txt)
        ├── 0a3a22b9b53f81f08b289072_png_jpg.rf.zF862svGejNqaiSXkbjI.txt
        ├── 000001_jpg.rf.FOnuxci3vufKn4LKc6iv.txt
        └── ...
```

### Image Format Breakdown

| Extension | Count | Notes |
|---|---|---|
| `.jpg` | 6,233 | Majority of images |
| `.jpeg` | 15 | Same format, different extension |
| `.PNG` | 129 | Uppercase extension — requires case-insensitive handling |
| **Total** | **6,377** | |

---

## 3. Classes — The 25 Object Categories

The `data.yaml` defines **25 classes** (IDs 0–24). These fall into **5 logical super-categories** based on what the model is learning to detect:

### 🟫 Category A: Packages & Parcels (Core Detection Targets)

These are the **primary objects** the model is trained to detect — delivery packages in various states.

| Class ID | Class Name | Annotation Count | Description |
|---|---|---|---|
| 0 | `Box` | 538 | Generic box, typically sealed |
| 1 | `Box_broken` | 300 | Damaged/crushed box |
| 2 | `Boxes` | 480 | Multiple boxes grouped together |
| 3 | `Open_package` | 302 | Package that has been opened/unsealed |
| 4 | `Package` | 473 | Wrapped/sealed delivery package |
| 5 | `Parcel` | 199 | Small-to-medium postal parcel |
| 9 | `box` | 4,562 | Generic box (lowercase variant — **most common class**) |
| 10 | `boxes` | 176 | Multiple boxes (lowercase variant) |
| 11 | `brownbox` | 10 | Specifically brown cardboard box |
| 13 | `cardboard` | 7,175 | Cardboard material / flat cardboard — **highest annotation count** |
| 14 | `good-parcel` | 124 | Parcel in good/undamaged condition |
| 18 | `p1` | 171 | Custom label (likely "parcel type 1") |
| 19 | `package` | 218 | Package (lowercase variant) |
| 20 | `parcel` | 499 | Parcel (lowercase variant) |

> **⚠️ Note on duplicate semantics:** Classes like `Box`/`box`, `Package`/`package`, `Parcel`/`parcel`, and `Boxes`/`boxes` appear to be duplicates with different casing. This is common in Roboflow projects that merged multiple annotation sources. The model will learn them as separate classes.

### 👤 Category B: People

| Class ID | Class Name | Annotation Count | Description |
|---|---|---|---|
| 21 | `person` | 1,587 | Human figure (delivery person, recipient, bystander) |

### 🎒 Category C: Personal Items / Luggage

| Class ID | Class Name | Annotation Count | Description |
|---|---|---|---|
| 6 | `backpack` | 46 | Backpack carried by a person |
| 15 | `handbag` | 88 | Handbag / purse |
| 23 | `suitcase` | 1,189 | Travel suitcase / luggage |

### 🚗 Category D: Vehicles

| Class ID | Class Name | Annotation Count | Description |
|---|---|---|---|
| 8 | `bicycle` | 14 | Bicycle |
| 12 | `car` | 107 | Automobile |
| 16 | `motorbike` | 1 | Motorbike (very rare — only 1 annotation) |
| 17 | `motorcycle` | 3 | Motorcycle |
| 24 | `truck` | 29 | Delivery truck / freight vehicle |

### 🪑 Category E: Environment / Context Objects

| Class ID | Class Name | Annotation Count | Description |
|---|---|---|---|
| 7 | `bench` | 4 | Park/street bench |
| 22 | `refrigerator` | 9 | Refrigerator (large appliance) |

### Class Distribution Summary

```
Class ID  Name              Count   ██████████████████████████████
──────────────────────────────────────────────────────────────────
13        cardboard          7175   ██████████████████████████████  ← Dominant
 9        box                4562   ███████████████████
21        person             1587   ██████▋
23        suitcase           1189   █████
 0        Box                 538   ██▎
20        parcel              499   ██
 2        Boxes               480   ██
 4        Package             473   ██
 1        Box_broken          300   █▎
 3        Open_package        302   █▎
19        package             218   █
 5        Parcel              199   ▊
10        boxes               176   ▋
18        p1                  171   ▋
14        good-parcel         124   ▌
12        car                 107   ▍
15        handbag              88   ▎
 6        backpack             46   ▏
24        truck                29   ▏
 8        bicycle              14   ▏
11        brownbox             10   ▏
22        refrigerator          9   ▏
 7        bench                 4   ▏
17        motorcycle            3   ▏
16        motorbike              1   ▏
```

---

## 4. How YOLO Annotation & Labeling Works

### 4.1 The YOLO Annotation Format

Each image has a **corresponding `.txt` file** with the same filename (stem). Every line in the text file represents **one detected object** (bounding box).

```
<class_id> <x_center> <y_center> <width> <height>
```

| Field | Type | Range | Description |
|---|---|---|---|
| `class_id` | Integer | 0–24 | Index into the `names` list in `data.yaml` |
| `x_center` | Float | 0.0–1.0 | Horizontal center of bounding box, **normalized** by image width |
| `y_center` | Float | 0.0–1.0 | Vertical center of bounding box, **normalized** by image height |
| `width` | Float | 0.0–1.0 | Width of bounding box, **normalized** by image width |
| `height` | Float | 0.0–1.0 | Height of bounding box, **normalized** by image height |

### 4.2 Concrete Example

For the image `0a3a22b9b53f81f08b289072_png_jpg.rf.zF862svGejNqaiSXkbjI.jpg`, its label file contains:

```
19 0.50125 0.57685 0.9975 0.84629
```

This means:

```
┌─────────────────────────────────────────────┐
│  Image (normalized 0–1 coordinate space)    │
│                                             │
│        ┌─────────────────────────┐          │
│        │                         │          │
│        │    class 19 (package)   │ ← height │
│        │                         │   0.846  │
│        │     center: (0.50, 0.58)│          │
│        │                         │          │
│        └─────────────────────────┘          │
│              width: 0.998                   │
└─────────────────────────────────────────────┘
```

- **Class 19** = `package` (a shipping label / text block on a parcel)
- **x_center = 0.50125** → the box is horizontally centered
- **y_center = 0.57685** → the box is slightly below the vertical midpoint
- **width = 0.9975** → the box spans nearly the full image width
- **height = 0.84629** → the box covers ~85% of the image height

### 4.3 Why Normalized Coordinates?

YOLO uses **normalized** (0–1) coordinates instead of pixel values so that:
- Annotations are **resolution-independent** — same labels work whether the image is 640×640 or 1920×1080
- The model can be trained on **mixed resolutions** without re-annotating
- **Data augmentation** (resize, crop, pad) can be applied without breaking labels

### 4.4 One-to-One Mapping: Image ↔ Label

```
images/                                    labels/
├── 000001_jpg.rf.FOnuxci3vufKn4LKc6iv.jpg  ├── 000001_jpg.rf.FOnuxci3vufKn4LKc6iv.txt
├── 000002_jpg.rf.gVWcHoBgyOz8rMtTAA3G.jpg  ├── 000002_jpg.rf.gVWcHoBgyOz8rMtTAA3G.txt
└── ...                                     └── ...
```

- Every image **must** have a `.txt` file
- If an image has **no objects**, the `.txt` file is **empty** (0 bytes) — this tells YOLO the image is a **negative sample** (background only)
- Multiple objects → multiple lines in the `.txt` file

### 4.5 The `data.yaml` Configuration

```yaml
train: ../train/images       # Path to training images
val: ../valid/images         # Path to validation images (if exists)
test: ../test/images         # Path to test images (if exists)

nc: 25                       # Number of classes
names: ['Box', 'Box_broken', 'Boxes', 'Open_package', 'Package',
        'Parcel', 'backpack', 'bench', 'bicycle', 'box',
        'boxes', 'brownbox', 'car', 'cardboard', 'good-parcel',
        'handbag', 'motorbike', 'motorcycle', 'p1', 'package',
        'parcel', 'person', 'refrigerator', 'suitcase', 'truck']
```

This file tells YOLOv9:
- **Where** to find train/val/test images
- **How many** classes to detect (`nc: 25`)
- **What** each class ID maps to (index 0 = `Box`, index 19 = `package`, etc.)

### 4.6 The Labeling Pipeline (How Annotations Were Created)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Images  │────▶│  Roboflow    │────▶│  Annotation  │────▶│  Export as   │
│  (6,377)     │     │  Upload      │     │  (Manual +   │     │  YOLOv9 TXT  │
│              │     │              │     │   Auto-label) │     │  format      │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                      │
                                                                      ▼
                                                               ┌──────────────┐
                                                               │  Validation  │
                                                               │  Script      │
                                                               │  (this repo) │
                                                               └──────────────┘
```

1. **Collection** — 6,377 images of packages, parcels, people, and vehicles were collected
2. **Upload** — Images were uploaded to Roboflow's annotation platform
3. **Annotation** — Bounding boxes were drawn around objects, each assigned a class ID
4. **Export** — Roboflow exported the dataset in YOLOv9 format (normalized coordinates)
5. **Validation** — Our `colab_annotation_pipeline.py` script validates and cleans the labels

---

## 5. Validation Checks Performed by Our Script

| Check | What It Does | Action |
|---|---|---|
| **Format Validation** | Ensures every line is `<int> <float> <float> <float> <float>` | Flags malformed lines |
| **Coordinate Clamping** | All values must be in [0.0, 1.0] | Clamps to valid range |
| **Class ID Range** | IDs must be 0–24 | Flags out-of-range IDs |
| **Positive Dimensions** | Width and height must be > 0 | Flags zero/negative sizes |
| **Missing Labels** | Every image needs a `.txt` file | Creates empty file (negative sample) |
| **Orphan Labels** | `.txt` without matching image | Flags for review |
| **Image Integrity** | GPU-accelerated image loading test | Flags corrupted images |

---

## 6. Quick Reference

```bash
# Run validation (Colab — dry run):
!python colab_annotation_pipeline.py

# Run validation + fix + zip:
!python colab_annotation_pipeline.py --fix

# Output:
# → annotated_dataset_clean.zip (ready for YOLOv9 training)
# → annotation_report.txt (audit log)
```
