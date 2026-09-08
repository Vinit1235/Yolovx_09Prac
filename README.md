# YOLOv9 Package Detection Project

This repository contains a complete YOLOv9 pipeline for training and validating a custom object detection model for package-related detection tasks.

The project includes dataset preparation, image augmentation, label verification, annotation workflows, and training scripts for both local and Kaggle/Google Colab usage.

## Project Overview

This project is built around a custom dataset of package, parcel, suitcase, person, and related object classes. It is intended for training a YOLOv9 object detection model to detect shipping and delivery-related objects in images.

### Main goals
- Prepare and split a YOLO dataset into train/val/test sets
- Train a YOLOv9 model on custom labels
- Validate model performance with standard metrics
- Verify annotation correctness and dataset quality
- Support Kaggle and Colab-based training workflows

## Repository Contents

- `split_dataset.py` — splits images and labels into train/val/test folders
- `kaggle_train_yolov9.py` — Kaggle training notebook/script for YOLOv9
- `augment_brightness.py` — brightness augmentation utilities
- `colab_annotation_pipeline.py` — annotation pipeline for Google Colab workflows
- `finalize_annotations.py` — final cleanup/standardization for labels
- `verify_label.py` — checks and validates label files
- `DATASET_DOCUMENTATION.md` — detailed dataset description and class references
- `dataset_split/` — output dataset with train/val/test splits
- `Package V0.5.yolov9/` — packaged dataset export
- `images/` and `labels/` — source image and annotation folders

## Dataset Summary

The dataset used in this project is a YOLO-formatted object detection dataset with:

- 25 classes
- YOLO Darknet TXT format
- Training, validation, and test splits
- Package and delivery-related object categories such as:
  - `Box`
  - `Package`
  - `Parcel`
  - `person`
  - `suitcase`
  - `cardboard`
  - `truck`
  - and other related classes

A full class list and annotation details are documented in [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md).

## Typical Workflow

### 1. Prepare dataset
Place your raw images in `images/` and corresponding labels in `labels/`.

### 2. Split dataset
Run:

```bash
python split_dataset.py
```

This creates a `dataset_split/` structure with:

```text
dataset_split/
├── data.yaml
├── train/
├── val/
└── test/
```

### 3. Train model
For Kaggle training, use the instructions in `kaggle_train_yolov9.py` or run a YOLOv9 training workflow from the generated dataset.

Example:

```python
from ultralytics import YOLO

model = YOLO('yolov9c.pt')
model.train(data='dataset_split/data.yaml', epochs=50, imgsz=640)
```

### 4. Validate and evaluate
After training, validate the model on the validation set and review metrics such as:

- Precision
- Recall
- mAP@50
- mAP@50-95
- confusion matrix

### 5. Run inference
Use your trained model weights for detection on new images.

## Recommended Environment

- Python 3.10+
- PyTorch
- Ultralytics
- CUDA-capable GPU recommended for training

Example installation:

```bash
pip install ultralytics
```

## Important Notes

- The project expects YOLO annotation format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

- Labels must match each image file name exactly.
- Data consistency and correct annotation validation are important for stable training.
- The project contains a Roboflow-exported package dataset and also custom training scripts for further processing.

## GitHub Repository

This repository is connected to the GitHub remote:

- `Vinit1235/Yolovx_09Prac`

## License

This repository’s original code, scripts, and documentation are licensed under the MIT License.

The full text is available in [LICENSE](LICENSE).

### Dataset notice

The dataset files in this project were downloaded from Roboflow and are documented as having a license status of `Private` in [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md).

This means the dataset itself is not covered by the MIT License and must remain subject to the original dataset owner’s terms, source restrictions, and any applicable Roboflow usage conditions. The dataset should not be redistributed or publicly shared without explicit permission from the original rights holder.

This project is intended for research and training purposes only. If you plan to publish, deploy, or redistribute the model or dataset publicly, review the original dataset licensing terms and confirm compliance before doing so.

## Acknowledgements

This workflow is designed around YOLOv9 and custom package-detection dataset preparation for computer vision training tasks.
