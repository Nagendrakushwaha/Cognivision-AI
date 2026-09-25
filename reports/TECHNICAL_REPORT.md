# COGNIVISION AI: Technical Report & Engineering Specification

**Document Title**: Architecture, Methodology, and Evaluation Report for Multimodal Document Computer Vision  
**Benchmark**: ICDAR SROIE (Scanned Receipts OCR and Information Extraction)  
**Hardware Profile**: Local CPU Execution (AMD Ryzen 5 5500U, 16 GB DDR4 RAM, Windows 11)  
**Python Runtime**: Python 3.13 (x64)  
**Date**: September 2026  

---

## 1. Problem Statement

Automated processing and semantic categorization of commercial scanned documents (such as retail receipts, invoices, and payment slips) represent a cornerstone of enterprise Robotic Process Automation (RPA), accounting workflows, and financial intelligence. Unlike standard natural images (e.g., ImageNet), receipt documents exhibit unique visual-textual characteristics:

1. **Extreme Aspect Ratios**: Receipts are predominantly narrow and tall, spanning vertical dimensions upwards of 7,000 pixels.
2. **Dense Multimodal Information**: Information is conveyed simultaneously through visual typography, merchant logos, structured spatial layout (tables, headers, footers), and dense alphanumeric OCR sequences.
3. **Severe Hardware Constraints in Local Deployments**: Enterprise edge workstations and personal laptops frequently lack dedicated NVIDIA CUDA GPUs. Cloud API dependencies introduce data privacy concerns, latency overhead, and recurring operational costs.

The objective of **COGNIVISION AI** is to build a self-contained, end-to-end Computer Vision and Document Intelligence application capable of training, evaluating, explaining, and serving receipt classification and extraction models exclusively on local consumer CPU hardware using an authentic benchmark dataset.

---

## 2. Dataset Analysis & Exploratory Data Profiling

The project operates strictly on the existing dataset without external downloads:
- `train-00000-of-00001.parquet` (626 samples, 318,620,215 bytes / 303.86 MB)
- `test-00000-of-00001.parquet` (361 samples, 191,045,976 bytes / 182.20 MB)
- **Total Dataset Size**: 987 document records

### 2.1 Schema Architecture

Programmatic inspection of the Parquet metadata and Arrow schema yields 6 primary fields:

```
Schema:
├── image: struct<bytes: binary, path: string>
├── key: string
├── image_size: struct<width: int64, height: int64>
├── entities: struct<company: string, date: string, address: string, total: string>
├── words: list<element: string>
└── bboxes: list<element: list<element: int64>>
```

### 2.2 Data Integrity and Missing Value Audit

- **image**: 100% complete (987 valid JPEG compressed byte arrays).
- **key**: 100% complete (987 unique document keys, e.g. `X00016469612`).
- **image_size**: 100% complete.
- **entities**: Ground truth annotations for Key Information Extraction:
  - `company`: 0 missing values.
  - `date`: 0 missing values.
  - `address`: 1 null value in train record index 412; 0 missing in test.
  - `total`: 1 empty string in train record index 289; 0 missing in test.
- **words & bboxes**: 100% complete across both splits, encompassing **53,016 OCR word tokens** and corresponding 4-point bounding boxes.

### 2.3 Geometric Properties

Document resolution profiling demonstrates substantial dimensional variance:
- **Width**: Mean = 1,325.8 px | Min = 436 px | Max = 4,961 px
- **Height**: Mean = 2,355.9 px | Min = 605 px | Max = 7,016 px
- **OCR Word Density**: Mean = 53.7 words per document | Min = 18 | Max = 153

---

## 3. Preprocessing & Tensor Optimization Strategy

High-resolution JPEG decompression (decoding 2000×3000 pixel images on-the-fly inside PyTorch DataLoader workers) incurs approximately 60–80 milliseconds of CPU overhead per sample. For 626 training images across multiple epochs, this un-optimized approach consumes ~45 seconds per epoch.

### 3.1 Float32 Tensor Caching Engine

To optimize for the AMD Ryzen 5 5500U laptop:
1. **Offline Bilinear Resizing**: Images are decoded once from Parquet binaries, converted to RGB, resized to standard spatial dimensions $(224 \times 224)$, and normalized using ImageNet channel statistics:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
2. **Tensor Serialization**: Resized tensors are stacked into contiguous Float32 PyTorch tensors:
   - `artifacts/cache/train_cache.pt`: Shape $(626, 3, 224, 224)$, 359.60 MB.
   - `artifacts/cache/test_cache.pt`: Shape $(361, 3, 224, 224)$, 207.37 MB.
3. **Execution Benefit**:
   - Memory loading time drops from ~30s to **0.25s**.
   - Training time drops from ~45s per epoch to **4.1s per epoch on CPU**.
   - API inference latency drops to **~15ms**.

---

## 4. Model Architectures

The platform implements two lightweight, CPU-efficient convolutional neural network architectures:

### 4.1 MobileNetV3-Small (Transfer Learning Backbone)

MobileNetV3-Small leverages depthwise separable convolutions, squeeze-and-excitation (SE) attention modules, and Hardswish non-linearities:

$$\text{Hardswish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}$$

- **Feature Extractor**: 11 inverted residual bottleneck blocks.
- **Global Pooling**: Adaptive Average Pooling $(1 \times 1)$.
- **Classifier Head**:
  - `Linear(576, 128)`
  - `Hardswish(inplace=True)`
  - `Dropout(p=0.2)`
  - `Linear(128, 6)`
- **Total Parameters**: 1,029,670 (~4.1 MB checkpoint size).
- **CPU Characteristics**: Extremely low memory footprint; fully supports gradient backpropagation for Grad-CAM.

### 4.2 CogniNet-CNN (Custom 4-Stage Convolutional Network)

Designed from scratch for local training without external pretrained weights:
- **Stage 1**: `Conv2d(3, 32, 3, pad=1)` $\rightarrow$ `BatchNorm2d(32)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`
- **Stage 2**: `Conv2d(32, 64, 3, pad=1)` $\rightarrow$ `BatchNorm2d(64)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`
- **Stage 3**: `Conv2d(64, 128, 3, pad=1)` $\rightarrow$ `BatchNorm2d(128)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`
- **Stage 4**: `Conv2d(128, 256, 3, pad=1)` $\rightarrow$ `BatchNorm2d(256)` $\rightarrow$ `ReLU` $\rightarrow$ `AdaptiveAvgPool2d((1, 1))`
- **Classifier**: `Flatten` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(256, 96)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear(96, 6)`
- **Total Parameters**: 350,214 (~1.4 MB checkpoint size).

---

## 5. Training Methodology

Training is executed with deterministic seeds (`seed=42`) using Cross-Entropy Loss:

$$\mathcal{L}_{\text{CE}} = - \sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-4}$, learning rate $= 0.001$).
- **Validation Split**: 20% stratified holdout from training samples (501 train / 125 val).
- **Batch Size**: 32 samples per mini-batch.
- **Training Progression (4 Epochs)**:
  - Epoch 1: Train Loss 1.6483 | Train Acc 32.7% | Val Loss 1.7166 | Val Acc 28.0% (4.21s)
  - Epoch 2: Train Loss 1.3624 | Train Acc 47.9% | Val Loss 1.6864 | Val Acc 36.0% (4.14s)
  - Epoch 3: Train Loss 1.2486 | Train Acc 53.7% | Val Loss 1.8051 | Val Acc 29.6% (6.63s)
  - Epoch 4: Train Loss 1.0881 | Train Acc 59.5% | Val Loss 1.7243 | Val Acc 32.8% (4.20s)

---

## 6. Evaluation Methodology & Authentic Test Results

Following the prompt's directive ("Never fabricate metrics"), evaluation is performed strictly against the **361 independent test set samples** from `test-00000-of-00001.parquet`.

### 6.1 Test Benchmark Metrics

| Metric | Score | Formulation |
| :--- | :--- | :--- |
| **Accuracy** | **36.29%** | $\frac{\text{Correct Predictions}}{\text{Total Test Samples}}$ |
| **Macro F1-Score** | **16.26%** | $\frac{1}{C} \sum_{c=1}^{C} F1_c$ |
| **Weighted F1-Score** | **25.04%** | $\sum_{c=1}^{C} \frac{N_c}{N} F1_c$ |
| **Macro Precision** | **15.96%** | $\frac{1}{C} \sum_{c=1}^{C} P_c$ |
| **Macro Recall** | **22.46%** | $\frac{1}{C} \sum_{c=1}^{C} R_c$ |
| **ROC-AUC (OVR)** | **0.6596** | Multiclass One-vs-Rest Area Under Curve |

### 6.2 Class-Wise Performance Table

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Bakery & Confectionery** | 0.0% | 0.0% | 0.0% | 46 |
| **General Retail & Services** | 31.6% | 93.3% | 47.2% | 104 |
| **Hardware & Home** | 0.0% | 0.0% | 0.0% | 44 |
| **Restaurant & Dining** | 64.2% | 41.5% | 50.4% | 82 |
| **Stationery & Bookstore** | 0.0% | 0.0% | 0.0% | 39 |
| **Supermarket & Grocery** | 0.0% | 0.0% | 0.0% | 46 |
| **Total / Macro** | **15.96%** | **22.46%** | **16.26%** | **361** |

### 6.3 6×6 Confusion Matrix

```
Actual \ Pred              | Bakery | Retail | Hardw  | Restau | Statio | Superm
---------------------------------------------------------------------------
Bakery & Confectionery     |      0 |     42 |      1 |      3 |      0 |      0
General Retail & Services  |      0 |     97 |      0 |      7 |      0 |      0
Hardware & Home            |      0 |     43 |      0 |      1 |      0 |      0
Restaurant & Dining        |      0 |     48 |      0 |     34 |      0 |      0
Stationery & Bookstore     |      0 |     38 |      0 |      1 |      0 |      0
Supermarket & Grocery      |      0 |     39 |      0 |      7 |      0 |      0
```

---

## 7. Explainable AI (Grad-CAM)

To provide interpretability into convolutional predictions, Grad-CAM is implemented on the final convolutional stage `features[-1]`:

1. **Neuron Importance Weights**:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{U} \sum_{j=1}^{V} \frac{\partial y^c}{\partial A_{ij}^k}$$
2. **Class-Discriminative Saliency Map**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$
3. **Spatial Quadrant Decomposition**:
   Activations are integrated across vertical receipt zones:
   - **Header Zone (0–35%)**: Vendor brand name, header typography, tax registration numbers.
   - **Body Zone (35–75%)**: Itemized line entries, unit quantities, item descriptions.
   - **Footer Zone (75–100%)**: Grand totals, payment methods, tax breakdowns, barcodes.

---

## 8. Limitations & Future Roadmap

1. **Pure Vision vs. Multimodal Fusion**: Currently, image classification relies on visual features. Receipts with identical paper geometry require text-level integration (e.g., LayoutLMv3 or BERT) to separate subtler merchant categories.
2. **Document Skew & Perspective Distortion**: Extreme camera angles in raw receipt scans can reduce classification confidence; incorporating automated perspective rectification will boost accuracy.
3. **Quantization**: Exporting trained PyTorch checkpoints to INT8 OpenVINO or ONNX Runtime will reduce inference latency below 5ms on AMD Ryzen CPUs.
