# LUNA16 Pulmonary Nodule Classification

A deep learning project for **3D pulmonary nodule classification from chest CT scans** using the [LUNA16](https://luna16.grand-challenge.org/) dataset.

The project implements an end-to-end pipeline for:

* Loading LUNA16 CT volumes in `.mhd` format
* Converting real-world patient coordinates to CT voxel coordinates
* Extracting 3D candidate patches from CT scans
* Performing 3D medical-image augmentation
* Training a custom 3D Convolutional Neural Network with PyTorch
* Handling class imbalance between nodules and non-nodules
* Tracking training and validation metrics with TensorBoard
* Saving and resuming training from checkpoints
* Running on CPU, single GPU, or multiple GPUs

> **Current task:** binary classification of a candidate as **nodule** or **non-nodule**.
> This repository does **not currently perform benign vs. malignant classification**.

---

## Dataset

This project uses the **LUNA16 (LUng Nodule Analysis 2016)** dataset, which is based on chest CT scans from the LIDC-IDRI dataset.

The pipeline uses:

* `annotations.csv` — annotated pulmonary nodules
* `candidates.csv` — candidate nodule locations and binary labels
* `.mhd` / `.raw` files — volumetric chest CT scans

Each candidate contains:

* CT series UID
* Candidate center in real-world `(x, y, z)` coordinates
* Nodule / non-nodule label
* Nodule diameter when an annotation is available

The dataset itself is not included in this repository.

---

## Pipeline

```text
LUNA16 CT Scan (.mhd)
        │
        ▼
Load 3D CT volume
        │
        ▼
Clamp Hounsfield Units
[-1000, 1000]
        │
        ▼
XYZ patient coordinates
        │
        ▼
Convert XYZ → IRC voxel coordinates
        │
        ▼
Crop candidate volume
[32 × 48 × 48]
        │
        ▼
3D Data Augmentation
        │
        ▼
3D CNN
        │
        ▼
Nodule / Non-Nodule
```

---

## CT Preprocessing

CT scans are loaded using **SimpleITK** and converted to NumPy arrays.

The voxel intensities are represented in **Hounsfield Units (HU)** and clipped to:

```python
[-1000, 1000]
```

This removes extreme out-of-range values while preserving useful lung and tissue information.

Candidate locations provided by LUNA16 use patient-space coordinates:

```text
(x, y, z)
```

The project converts these coordinates into CT array coordinates:

```text
(index, row, column)
```

using the scan's:

* origin
* voxel spacing
* direction matrix

A fixed 3D region is then extracted around each candidate:

```text
Depth  = 32
Height = 48
Width  = 48
```

The final model input therefore has shape:

```text
[B, 1, 32, 48, 48]
```

where `B` is the batch size.

---

## Data Augmentation

Training samples are augmented using **TorchIO**, a library designed for medical imaging.

The current augmentation pipeline includes:

### Random 3D Flip

Randomly flips the volume across spatial axes.

```python
RandomFlip(
    axes=(0, 1, 2),
    flip_probability=0.5
)
```

### Random Affine Transformation

Applies random scaling, rotation, and translation.

```python
RandomAffine(
    scales=(0.9, 1.1),
    degrees=(10, 0, 0),
    translation=3,
    p=0.75
)
```

### Random Noise

Adds random Gaussian noise to selected samples.

```python
RandomNoise(
    std=(0, 25),
    p=0.25
)
```

These transformations improve model robustness and reduce overfitting.

---

## Model Architecture

The classifier is implemented as a custom **3D Convolutional Neural Network**.

Each `LunaBlock` contains:

```text
Conv3D
  ↓
ReLU
  ↓
Conv3D
  ↓
ReLU
  ↓
MaxPool3D
```

The complete network consists of four convolutional blocks:

```text
Input
[1 × 32 × 48 × 48]

        │
        ▼
BatchNorm3D

        │
        ▼
LunaBlock 1
1 → 8 channels

        │
        ▼
LunaBlock 2
8 → 16 channels

        │
        ▼
LunaBlock 3
16 → 32 channels

        │
        ▼
LunaBlock 4
32 → 64 channels

        │
        ▼
Flatten

        │
        ▼
Linear
1152 → 2

        │
        ▼
Softmax

        │
        ▼
Non-Nodule / Nodule
```

The model returns both:

```python
logits, probabilities
```

The logits are used for calculating the training loss, while the probabilities are used for evaluation and prediction.

Weights for convolutional and linear layers are initialized with **Kaiming initialization** for ReLU-based networks.

---

## Handling Class Imbalance

LUNA16 contains significantly more non-nodule candidates than positive nodule candidates.

The dataset implementation supports balanced sampling using:

```bash
--balanced
```

When enabled, positive and negative samples are selected separately rather than directly following the highly imbalanced original candidate distribution.

Samples are also reshuffled between epochs.

---

## Train / Validation Split

The project performs a deterministic train-validation split using:

```python
val_stride = 10
```

Approximately every 10th candidate is assigned to the validation dataset, while the remaining candidates are used for training.

---

## Training

Training is implemented in:

```text
training.py
```

The current optimizer is:

```python
SGD(
    lr=0.001,
    momentum=0.99
)
```

The classification objective uses:

```python
CrossEntropyLoss
```

The training pipeline automatically detects whether CUDA is available:

```text
CUDA available
    │
    ├── 1 GPU → standard GPU training
    │
    └── >1 GPU → PyTorch DataParallel
    │
    ▼
GPU training

CUDA unavailable
    │
    ▼
CPU training
```

---

## Training Metrics

The pipeline tracks separate statistics for positive and negative samples.

Metrics include:

* Training loss
* Validation loss
* Overall accuracy
* Positive-class accuracy
* Negative-class accuracy
* Precision
* Recall
* F1 score
* True Positives
* True Negatives
* False Positives
* False Negatives

These metrics are written both to log files and TensorBoard.

---

## TensorBoard

Training and validation metrics are written using PyTorch's `SummaryWriter`.

To visualize the results:

```bash
tensorboard --logdir runs-trn_cls
```

or point TensorBoard at the project directory containing both training and validation run folders.

This allows monitoring metrics such as:

```text
loss/all
loss/pos
loss/neg

correct/all
correct/pos
correct/neg

pr/precision
pr/recall
pr/f1_score
```

---

## Checkpointing

The project supports saving and resuming training.

Each checkpoint stores:

```python
{
    "epoch": ...,
    "model_state": ...,
    "optimizer_state": ...,
    "totalTrainingSamples_count": ...
}
```

When a checkpoint already exists, training automatically restores:

* model parameters
* optimizer state
* completed epoch
* total number of processed training samples

and continues from the following epoch.

---

## Project Structure

```text
luna_dataset/
│
├── dsets.py
│   ├── LUNA16 metadata parsing
│   ├── CT loading
│   ├── XYZ ↔ IRC coordinate conversion
│   ├── candidate cropping
│   ├── dataset splitting
│   ├── balanced sampling
│   └── 3D augmentation
│
├── model.py
│   ├── LunaBlock
│   └── LunaModel 3D CNN
│
├── training.py
│   ├── DataLoader creation
│   ├── training loop
│   ├── validation loop
│   ├── loss computation
│   ├── metric calculation
│   ├── TensorBoard logging
│   ├── checkpointing
│   └── GPU / multi-GPU support
│
├── helper.py
│   └── supporting utilities
│
├── test.py
│   └── development / testing utilities
│
├── util/
│   └── caching and utility modules
│
├── logs/
│   └── training and validation logs
│
├── runs-trn_cls/
│   └── TensorBoard training logs
│
├── runs-val_cls/
│   └── TensorBoard validation logs
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/thinh9666/luna_dataset.git
cd luna_dataset
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the main dependencies:

```bash
pip install torch torchvision
pip install torchio
pip install SimpleITK
pip install numpy
pip install tqdm
pip install tensorboard
```

Additional dependencies may be required by the utilities under `util/`.

---

## Dataset Setup

Download the LUNA16 dataset and organize the required files.

A simplified layout is:

```text
data/
└── luna/
    ├── annotations.csv
    ├── candidates.csv
    └── subset0/
        ├── <series_uid>.mhd
        ├── <series_uid>.raw
        └── ...
```

### Important

The current implementation contains several **Google Colab-specific absolute paths**, for example paths under:

```text
/content/...
```

Before running the project locally, update the dataset, TensorBoard, and checkpoint paths in the source code to match your environment.

For example:

```python
mhd_data_folder = "/path/to/luna/subset0"
```

and update the paths to:

```text
annotations.csv
candidates.csv
checkpoint directory
TensorBoard directory
```

accordingly.

---

## Running Training

Basic training:

```bash
python training.py
```

Specify the number of epochs:

```bash
python training.py --epochs 10
```

Set the batch size:

```bash
python training.py --epochs 10 --batch-size 32
```

Configure DataLoader workers:

```bash
python training.py \
    --epochs 10 \
    --batch-size 32 \
    --num-workers 4
```

Train using balanced positive / negative samples:

```bash
python training.py \
    --epochs 10 \
    --batch-size 32 \
    --num-workers 4 \
    --balanced
```

---

## Technologies

This project uses:

* **Python**
* **PyTorch**
* **3D Convolutional Neural Networks**
* **TorchIO**
* **SimpleITK**
* **NumPy**
* **TensorBoard**
* **CUDA**
* **PyTorch DataParallel**
* **Medical CT Image Processing**

---

## What I Learned

This project was developed as part of my study of deep learning and medical computer vision.

Through the project, I practiced implementing:

* 3D convolutional neural networks from scratch with PyTorch
* Processing volumetric medical imaging data
* Understanding CT scans and Hounsfield Units
* Converting between physical coordinates and voxel coordinates
* Designing custom PyTorch `Dataset` and `DataLoader` pipelines
* Performing 3D medical-image augmentation
* Handling highly imbalanced datasets
* GPU and multi-GPU model training
* Checkpoint-based training recovery
* TensorBoard experiment tracking
* Precision, recall, F1 score, and confusion-matrix statistics
* Efficient CT loading and caching

---

## Current Limitations

This repository is primarily an experimental / educational implementation.

Some areas that can be improved include:

* Replace hard-coded paths with a configuration file or CLI arguments
* Add a `requirements.txt`
* Add a dedicated inference script
* Add unit tests
* Improve experiment reproducibility with deterministic random seeds
* Replace `DataParallel` with `DistributedDataParallel` for scalable multi-GPU training
* Add ROC-AUC and FROC evaluation
* Add model export for deployment
* Separate data preparation, training, evaluation, and inference into independent modules

---

## Future Work

Potential extensions include:

* Pulmonary nodule detection on complete CT volumes
* False-positive reduction
* Benign vs. malignant nodule classification
* Stronger 3D CNN architectures
* 3D ResNet-based models
* Distributed training with PyTorch DDP
* Mixed-precision training
* ONNX model export
* Inference API deployment
* Automated LUNA16 evaluation pipeline

---

## Disclaimer

This project is intended for **educational and research purposes only**.

It is **not a medical diagnostic system** and should not be used for clinical decision-making.

---

## Author

**Nguyen Quoc Thinh**

GitHub: [thinh9666](https://github.com/thinh9666)
