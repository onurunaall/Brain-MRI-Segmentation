# Brain MRI Segmentation

A PyTorch implementation of U-Net for automated segmentation of FLAIR signal abnormalities in brain MRI volumes. This project addresses the clinical challenge of identifying tumor-related hyperintensities in lower-grade glioma patients, providing a reproducible deep learning pipeline from raw TIFF stacks to per-patient Dice evaluation.

---

## Table of Contents

- [Background and Motivation](#background-and-motivation)
- [Dataset](#dataset)
- [Method](#method)
  - [Preprocessing](#preprocessing)
  - [Model Architecture](#model-architecture)
  - [Loss Function](#loss-function)
  - [Training Strategy](#training-strategy)
  - [Post-processing](#post-processing)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference and Evaluation](#inference-and-evaluation)
  - [PyTorch Hub](#pytorch-hub)
- [Evaluation](#evaluation)
- [References](#references)

---

## Background and Motivation

Fluid-attenuated inversion recovery (FLAIR) MRI is a standard neuroimaging protocol that suppresses cerebrospinal fluid signal while accentuating pathological tissue such as edema and glioma infiltration. Manual delineation of FLAIR abnormalities is time-consuming, subject to inter-rater variability, and impractical at the scale required for large clinical trials or population studies.

Convolutional encoder-decoder networks — in particular the U-Net family introduced by Ronneberger et al. (2015) — have become the de facto standard for volumetric medical image segmentation owing to their ability to integrate global context with fine-grained spatial resolution through hierarchical skip connections. This repository implements a modern U-Net baseline with best-practice training techniques (mixed-precision training, cosine annealing, weighted slice sampling, and online augmentation) and provides a complete, reproducible pipeline suitable as a reference implementation or a starting point for graduate-level research.

---

## Dataset

**Kaggle Brain MRI Segmentation** (LGG Segmentation Dataset)

- **Source:** Mateusz Buda, Ashirbani Saha, Maciej A. Mazurowski — "Association of genomic subtypes of lower-grade gliomas with shape features automatically measured on MRI", *Computers in Biology and Medicine*, 2019. Data is hosted on [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).
- **Subjects:** 110 patients from The Cancer Imaging Archive (TCIA), diagnosed with lower-grade glioma (LGG).
- **Modality:** Three-channel FLAIR-derived images stored as per-slice TIFF files.
- **Ground truth:** Binary masks delineating FLAIR signal abnormality, produced by board-certified radiologists.

**Expected directory layout:**

```
kaggle_3m/
├── TCGA_CS_4941_19960909/
│   ├── TCGA_CS_4941_19960909_1.tif
│   ├── TCGA_CS_4941_19960909_1_mask.tif
│   ├── TCGA_CS_4941_19960909_2.tif
│   ├── TCGA_CS_4941_19960909_2_mask.tif
│   └── ...
├── TCGA_CS_4942_19970222/
│   └── ...
└── ...
```

Each patient subdirectory contains matched image/mask pairs indexed by axial slice number. The first and last slices of each volume are discarded to exclude coverage artifacts common in clinical acquisitions.

---

## Method

### Preprocessing

All preprocessing is applied consistently at both training and inference time:

| Step | Details |
|------|----------|
| **Slice selection** | First and last axial slices dropped per volume |
| **Brain extraction** | Bounding-box crop retaining voxels above 10 % of the per-channel maximum, removing dark background |
| **Spatial padding** | Zero-pad to square spatial dimensions (preserves aspect ratio) |
| **Resizing** | Bicubic (order-2) interpolation to 256 × 256; nearest-neighbor for masks |
| **Intensity normalization** | Per-channel: clip to [10th, 99th] percentile, then z-score standardize |

### Data Augmentation

Online augmentation is applied to training slices only:

- **Random scaling** — isotropic scale factor sampled from U(1 − s, 1 + s), default s = 0.05, followed by center-crop or zero-pad to restore target resolution.
- **Random rotation** — in-plane rotation sampled from U(−θ, θ), default θ = 15°.
- **Random horizontal flip** — applied with probability 0.5.

Bilinear interpolation is used for image channels; nearest-neighbor is used for binary masks to preserve label integrity.

### Model Architecture

The segmentation network follows the original U-Net design with the following configuration:

```
Input  (B × 3 × 256 × 256)
  │
  ├─ Encoder Block 1  ─────────────────────────────────────── skip₁
  │   [Conv3×3 → BN → ReLU] × 2  |  32 filters  |  MaxPool 2×2
  │
  ├─ Encoder Block 2  ─────────────────────────────────────── skip₂
  │   [Conv3×3 → BN → ReLU] × 2  |  64 filters  |  MaxPool 2×2
  │
  ├─ Encoder Block 3  ─────────────────────────────────────── skip₃
  │   [Conv3×3 → BN → ReLU] × 2  | 128 filters  |  MaxPool 2×2
  │
  ├─ Encoder Block 4  ─────────────────────────────────────── skip₄
  │   [Conv3×3 → BN → ReLU] × 2  | 256 filters  |  MaxPool 2×2
  │
  ├─ Bottleneck
  │   [Conv3×3 → BN → ReLU] × 2  | 512 → 1024 filters
  │
  ├─ Decoder Block 4  ← ConvTranspose2×2 + cat(skip₄)
  │   [Conv3×3 → BN → ReLU] × 2  | 512 filters
  │
  ├─ Decoder Block 3  ← ConvTranspose2×2 + cat(skip₃)
  │   [Conv3×3 → BN → ReLU] × 2  | 256 filters
  │
  ├─ Decoder Block 2  ← ConvTranspose2×2 + cat(skip₂)
  │   [Conv3×3 → BN → ReLU] × 2  | 128 filters
  │
  ├─ Decoder Block 1  ← ConvTranspose2×2 + cat(skip₁)
  │   [Conv3×3 → BN → ReLU] × 2  |  64 filters
  │
  └─ Output Conv 1×1 → Sigmoid
Output (B × 1 × 256 × 256)
```

Skip connections concatenate encoder feature maps with decoder feature maps at matching spatial resolutions, allowing the network to recover fine-grained spatial detail that is lost during downsampling.

**Parameter count:** ~31 M (with base_filters = 32, standard U-Net depth).

### Loss Function

Training minimizes the **soft Dice loss**:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{N} \sum_{i=1}^{N} \frac{2 \sum_j p_{ij} \, g_{ij} + \epsilon}{\sum_j p_{ij} + \sum_j g_{ij} + \epsilon}$$

where $p_{ij} \in [0,1]$ is the predicted probability for sample $i$ at voxel $j$, $g_{ij} \in \{0,1\}$ is the binary ground-truth label, and $\epsilon = 1$ is a Laplace smoothing term that prevents division by zero and ensures gradient stability when both the prediction and ground truth are empty.

The Dice loss is preferred over binary cross-entropy for this task because the foreground (abnormality) class is strongly imbalanced relative to the background brain tissue; Dice directly optimizes the overlap metric used for evaluation.

### Training Strategy

| Hyperparameter | Default |
|----------------|---------|
| Optimizer | Adam (β₁ = 0.9, β₂ = 0.999) |
| Learning rate | 1 × 10⁻⁴ |
| LR schedule | Cosine annealing (T_max = epochs, η_min = 0) |
| Batch size | 16 |
| Epochs | 100 |
| Mixed precision | FP16 via `torch.cuda.amp` |
| Compilation | Optional `torch.compile` (PyTorch ≥ 2.0) |

**Weighted slice sampling** — the training `DataLoader` uses a `WeightedRandomSampler` that assigns each 2-D slice a sampling weight proportional to its foreground area, counteracting the dominance of near-empty slices in volumetric neuroimaging data.

**Validation** — a held-out set of 10 patients (configurable) is reserved for model selection. Validation Dice is computed per volume (aggregating all slices for a patient) to match the clinical evaluation protocol. The checkpoint with the highest validation Dice is retained.

**Logging** — training progress (loss, Dice, learning rate, and prediction overlays) is written to TensorBoard event files at every epoch; visualization examples are saved at a configurable frequency.

### Post-processing

At inference time, the binary prediction for each patient volume is post-processed by retaining only the **largest connected component** (LCC). This removes isolated false-positive predictions that are spatially disconnected from the main lesion region, a standard heuristic in brain tumor segmentation.

---

## Repository Structure

```
Brain-MRI-Segmentation/
│
├── network.py          # U-Net encoder–decoder definition (UNetModel)
├── dataset.py          # MRISegmentationDataset — volume loading and preprocessing
├── augmentations.py    # Online augmentation transforms
├── losses.py           # SoftDiceLoss
├── train.py            # Training entry point
├── predict.py          # Inference and per-patient evaluation entry point
├── utils.py            # Preprocessing, normalization, DSC, and visualization helpers
├── tb_logger.py        # Thin TensorBoard SummaryWriter wrapper
├── hubconf.py          # PyTorch Hub entry point (unet_segmentation)
└── requirements.txt    # Pinned dependencies
```

---

## Installation

**Requirements:** Python ≥ 3.9, CUDA-capable GPU recommended.

```bash
git clone https://github.com/onurunaall/Brain-MRI-Segmentation.git
cd Brain-MRI-Segmentation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the Kaggle LGG dataset and place it in `./kaggle_3m/` (or pass a custom path via `--data-dir`):

```bash
# Requires a Kaggle API token (~/.kaggle/kaggle.json)
kaggle datasets download mateuszbuda/lgg-mri-segmentation
unzip lgg-mri-segmentation.zip -d kaggle_3m
```

---

## Usage

### Training

```bash
python train.py \
    --data-dir ./kaggle_3m \
    --checkpoint-dir ./checkpoints \
    --log-dir ./tb_logs \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-4 \
    --image-size 256 \
    --aug-scale 0.05 \
    --aug-angle 15.0 \
    --device cuda:0
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir ./tb_logs
```

All command-line arguments and their defaults:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `./kaggle_3m` | Root directory of the dataset |
| `--checkpoint-dir` | `./checkpoints` | Directory for saving `.pt` checkpoints |
| `--log-dir` | `./tb_logs` | TensorBoard log directory |
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `16` | Slices per mini-batch |
| `--lr` | `1e-4` | Initial Adam learning rate |
| `--image-size` | `256` | Spatial resolution after resizing |
| `--aug-scale` | `0.05` | Scale jitter magnitude |
| `--aug-angle` | `15.0` | Rotation range in degrees |
| `--device` | `cuda:0` | PyTorch device string |
| `--num-workers` | `4` | DataLoader worker processes |
| `--vis-frequency` | `10` | Epoch interval for saving visualizations |
| `--n-vis-images` | `200` | Number of overlay images saved per visualization pass |

### Inference and Evaluation

```bash
python predict.py \
    --model-path ./checkpoints/best_model.pt \
    --data-dir ./kaggle_3m \
    --output-dir ./predictions \
    --figure-path ./dice_distribution.png \
    --device cuda:0
```

`predict.py` will:

1. Load the validation split (same patient list as used during training).
2. Run forward passes in batches, applying the same preprocessing pipeline.
3. Post-process predictions with largest-connected-component filtering.
4. Compute per-patient Dice Similarity Coefficient.
5. Save side-by-side overlay images (prediction contour in red, ground-truth contour in green) to `--output-dir`.
6. Save a bar chart of the Dice distribution to `--figure-path`.

### PyTorch Hub

The model can be loaded directly via the PyTorch Hub API without cloning the repository:

```python
import torch

model = torch.hub.load(
    "onurunaall/Brain-MRI-Segmentation",
    "unet_segmentation",
    pretrained=True,
    weights_path="./checkpoints/best_model.pt",
)
model.eval()

# Example forward pass
x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    mask = model(x)   # shape: (1, 1, 256, 256), values in [0, 1]
```

---

## Evaluation

Segmentation performance is quantified using the **Dice Similarity Coefficient (DSC)**:

$$\text{DSC} = \frac{2 |P \cap G|}{|P| + |G|}$$

where $P$ is the set of predicted foreground voxels and $G$ is the set of ground-truth foreground voxels. DSC ranges from 0 (no overlap) to 1 (perfect overlap). When both $P$ and $G$ are empty (i.e., a patient with no detectable abnormality), DSC is defined as 1.0 by convention.

Metrics are computed **per patient** (i.e., all 2-D slices for a subject are combined into a 3-D volume before evaluation) to avoid the overrepresentation of patients with many normal slices.

---

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*, LNCS 9351, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28

2. Buda, M., Saha, A., & Mazurowski, M. A. (2019). Association of genomic subtypes of lower-grade gliomas with shape features automatically measured on MRI. *Computers in Biology and Medicine*, 109, 218–225. https://doi.org/10.1016/j.compbiomed.2019.05.002

3. Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. *3DV*, 565–571. https://doi.org/10.1109/3DV.2016.79

4. Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18, 203–211. https://doi.org/10.1038/s41592-020-01008-z

5. Micikevicius, P., et al. (2018). Mixed precision training. *ICLR 2018*. https://arxiv.org/abs/1710.03740
