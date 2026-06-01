# BrainNet — Brain Tumor MRI Classification

A full end-to-end deep learning pipeline for classifying brain MRI scans into 4 categories using a custom CNN architecture with Residual blocks and Squeeze-and-Excitation attention.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20T4-yellow)
![Accuracy](https://img.shields.io/badge/Val%20Accuracy-96.52%25-brightgreen)

---

## Classes

| Class | Description |
|---|---|
| `glioma_tumor` | Malignant tumors originating in glial cells |
| `meningioma_tumor` | Tumors arising from the meninges |
| `pituitary_tumor` | Tumors of the pituitary gland |
| `no_tumor` | Healthy MRI scans |

---

##  Architecture — BrainNet

Custom ResNet-style CNN built from scratch with:

- **Residual (skip) connections** — prevents vanishing gradients, lets the network learn incremental improvements
- **Squeeze-and-Excitation (SE) blocks** — channel-wise attention that recalibrates which feature maps matter for each image
- **3 progressive stages** — 64 → 128 → 256 filters with spatial downsampling between stages
- **Global Average Pooling** — replaces Flatten, reduces overfitting

```
Input (128×128×3)
    │
    ▼
Stem: Conv 7×7, BN, ReLU → MaxPool       [64×64×32]
    │
    ▼
Stage 1: ResBlock×2 + SE  (64 filters)   [32×32×64]
Stage 2: ResBlock×2 + SE  (128 filters)  [16×16×128]
Stage 3: ResBlock×2 + SE  (256 filters)  [8×8×256]
    │
    ▼
Global Average Pooling → Dense(256) → Dropout(0.4) → Dense(4, softmax)

Total parameters: ~2.8M
```

---

##  Results

| Metric | Value |
|---|---|
| Validation Accuracy | **96.52%** |
| Validation AUC | **0.9967** |
| Test Accuracy | 71.32% |
| Best Epoch | 31 / 40 |

### Per-class Test Performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| glioma_tumor | 1.00 | 0.19 | 0.32 |
| meningioma_tumor | 0.71 | 0.90 | 0.79 |
| no_tumor | 0.62 | 1.00 | 0.76 |
| pituitary_tumor | 0.91 | 0.72 | 0.80 |

---

##  Key Finding — Domain Shift

The model achieves 96.5% validation accuracy but 71% test accuracy, with glioma recall dropping to 19% on the test set. Visual inspection revealed that **glioma training and test images look significantly different** — different scanner characteristics, contrast levels, and orientations.

This is a known challenge in medical AI called **domain shift**: models trained on data from one distribution fail to generalize to another. The other 3 classes (meningioma, no_tumor, pituitary) perform well on test, confirming the architecture and pipeline are sound — the problem is dataset-specific.

This finding motivates transfer learning approaches (e.g. EfficientNet pretrained on ImageNet) for better cross-distribution generalization.

---

##  Pipeline

```
1. EDA                  → class distribution, sample visualization
2. Data Pipeline        → tf.data with stratified split, zero-leakage guarantee
3. Augmentation         → flip, brightness, contrast, rotation (MRI-specific, mild)
4. Class Weights        → compensate for no_tumor underrepresentation
5. Training             → Adam(1e-3), EarlyStopping, ReduceLROnPlateau
6. Evaluation           → confusion matrix, ROC curves, per-class metrics
7. Explainability       → Grad-CAM heatmaps
8. Prediction Widget    → upload any MRI and get confidence breakdown
```

---

## Usage

**1. Clone and open in Colab**
```bash
git clone https://github.com/yourusername/brainnet-mri-classification
```

**2. Upload your dataset to Google Drive with this structure:**
```
dataset-Brain/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

**3. Update `BASE_DIR` in Cell 1, then Run All**

---

##  Requirements

```
tensorflow >= 2.15
scikit-learn
matplotlib
seaborn
pandas
pillow
opencv-python
ipywidgets
```

---

##  Dataset

[Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

3,264 MRI images across 4 classes.

---

##  Author

**Takoua Belkis Hanchi**  
M.Sc. Data Science & Intelligent Systems  
[LinkedIn](www.linkedin.com/in/belkis-takoua-hanchi-445274250) · [GitHub](https://github.com/belkisKOS)
