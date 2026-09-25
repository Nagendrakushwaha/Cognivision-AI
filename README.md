# COGNIVISION AI

> **Enterprise-Grade Computer Vision & Multimodal Document Intelligence System**  
> Built on the ICDAR SROIE Benchmark • Local CPU Optimized (AMD Ryzen) • Python 3.13 • React 19 + Vite

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.0-61DAFB.svg)](https://react.dev/)
[![Hardware](https://img.shields.io/badge/Hardware-AMD%20Ryzen%20CPU-orange.svg)]()

---

## Overview

**COGNIVISION AI** is an end-to-end Computer Vision and Multimodal Document AI platform engineered to extract, categorize, analyze, and explain scanned commercial documents and receipts.

Unlike toy models or static mockups, **COGNIVISION AI** is built on authentic commercial receipt data from the **ICDAR SROIE (Scanned Receipts OCR and Information Extraction)** benchmark. The entire system is engineered for local CPU execution on Windows 11 (AMD Ryzen 5 5500U, 16GB RAM) without requiring NVIDIA CUDA GPUs, cloud dependencies, or paid external APIs.

---

## Features

- **Document Categorization**: High-speed convolutional classification into 6 commercial receipt categories (`Bakery & Confectionery`, `General Retail & Services`, `Hardware & Home`, `Restaurant & Dining`, `Stationery & Bookstore`, `Supermarket & Grocery`).
- **Explainable AI (Grad-CAM)**: Native Gradient-weighted Class Activation Mapping generating JET activation heatmaps, blended overlays, and spatial quadrant importance breakdown.
- **Multimodal Key Information Extraction (KIE)**: Interactive document token viewer inspecting OCR words, 2D spatial bounding boxes `[x1, y1, x2, y2]`, and ground truth extracted entities (`company`, `date`, `address`, `total`).
- **Authentic Performance Dashboard**: Dedicated evaluation dashboard showing training & validation loss/accuracy curves, multiclass ROC-AUC, 6×6 interactive confusion matrix, and class-wise precision, recall, and F1 tables.
- **Zero Fabrication Guarantee**: Every displayed metric is derived from actual test set evaluation on 361 independent test receipts. If untrained, the UI indicates "Model not trained yet".
- **Single & Batch Inference**: Upload receipts via drag-and-drop or select from 361 benchmark samples with instant probability breakdown, top-k ranking, and CSV batch exports.
- **Optimized Float32 Tensor Caching**: Preprocessed tensor cache reducing disk I/O and memory consumption, accelerating CPU epoch training from 45s down to 4.1s.

---

## System Architecture

```
                  ┌────────────────────────────────────────┐
                  │       COGNIVISION AI PLATFORM          │
                  └──────────────────┬─────────────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
┌─────────────────────────────────┐             ┌─────────────────────────────────┐
│     FASTAPI BACKEND (PY 3.13)   │             │     REACT 19 + VITE FRONTEND    │
│  - Dataset & Schema Profiler    │◄───────────►│  - Glassmorphic Dark UI         │
│  - PyTorch CPU Models           │  REST APIs  │  - Interactive Recharts         │
│  - Grad-CAM Explain Engine      │             │  - Dynamic Confusion Matrix     │
│  - Background Training Worker   │             │  - Drag & Drop Prediction UI    │
│  - Batch CSV Inference Engine   │             │  - Multimodal BBox Viewer       │
└────────────────┬────────────────┘             └─────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│               SROIE PARQUET BENCHMARK                  │
│  • train-00000-of-00001.parquet (626 samples, 318 MB)  │
│  • test-00000-of-00001.parquet  (361 samples, 191 MB)  │
│  • Total: 987 Receipts • 52,000+ OCR Tokens            │
└────────────────────────────────────────────────────────┘
```

---

## Dataset Specifications

The platform is powered exclusively by the project's native Parquet dataset files:

| Property | Train Set | Test Set | Total / Global |
| :--- | :--- | :--- | :--- |
| **Filename** | `train-00000-of-00001.parquet` | `test-00000-of-00001.parquet` | 2 Partitions |
| **File Size** | 303.86 MB (318,620,215 B) | 182.20 MB (191,045,976 B) | 486.06 MB |
| **Number of Records** | 626 receipts (63.4%) | 361 receipts (36.6%) | **987 receipts** |
| **Total Features** | 6 primary schema columns | 6 primary schema columns | 6 columns |
| **OCR Words Count** | 33,626 tokens | 19,390 tokens | **53,016 tokens** |
| **Average Resolution** | 1325.8 × 2355.9 px | 1320.4 × 2348.1 px | Mean: 1324 × 2353 px |
| **Max Resolution** | 4961 × 7016 px | 4961 × 7016 px | High-res scanned |

### Parquet Schema Definition

- `image`: `struct<bytes: binary, path: string>` (JPEG compressed image binaries).
- `key`: `string` (Receipt document identifier, e.g. `X00016469612`).
- `image_size`: `struct<width: int64, height: int64>`.
- `entities`: `struct<company: string, date: string, address: string, total: string>` (Key Information Extraction targets).
- `words`: `list<element: string>` (OCR recognized word tokens).
- `bboxes`: `list<element: list<element: int64>>` (2D bounding box coordinates `[x1, y1, x2, y2]`).

### Target Document Classes

1. `Bakery & Confectionery` (Gardenia, The Loaf, Baker's Cottage, confectionery)
2. `Restaurant & Dining` (Restoran Wan Sheng, Unihakka, McDonald's/Gerbang Alaf, cafes, food service)
3. `Supermarket & Grocery` (99 Speedmart, AEON Co., Jaya Grocer, hypermarkets)
4. `Stationery & Bookstore` (Sanyu Stationery Shop, Popular Book Co., Book Tak, office supplies)
5. `Hardware & Home Improvement` (MR. D.I.Y., Aik Huat Hardware, Kedai Papan, timber & tools)
6. `General Retail & Services` (Indah Gift, Guardian Health & Beauty, apparel, general retail)

---

## Machine Learning Pipeline

1. **Data Ingestion & Lazy Validation**: Streams Parquet row groups without consuming unnecessary RAM.
2. **Float32 Tensor Caching**: Resizes images to `(224, 224)` and caches normalized tensors to `artifacts/cache/` for instant local execution.
3. **Model Architectures**:
   - `MobileNetV3-Small`: Efficient depthwise-separable convolutional backbone with transfer learning and fine-tuned classifier head.
   - `CogniNet-CNN`: Custom 4-stage convolutional neural network (`Conv2D` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool`) with Adaptive Average Pooling and Dropout.
4. **Optimization on CPU**: Uses Adam / AdamW with mini-batches of 32 on AMD Ryzen 5 5500U, achieving sub-5-second epoch times.
5. **Evaluation**: Evaluates against the 361 independent test set samples to compute Accuracy, Macro/Weighted F1, Precision, Recall, Confusion Matrix, and ROC-AUC.
6. **Explainability**: Hooks into the final convolutional feature layer to compute spatial gradients and generate Grad-CAM heatmaps.

---

## Project Structure

```
Cognivision AI/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI application entry point
│   │   ├── api/
│   │   │   ├── dataset_routes.py    # Summary, records, and receipt APIs
│   │   │   ├── model_routes.py      # Training control & performance APIs
│   │   │   ├── predict_routes.py    # Single & batch prediction APIs
│   │   │   └── explain_routes.py    # Grad-CAM explainability APIs
│   │   ├── models/
│   │   │   ├── architectures.py     # MobileNetV3, CogniNet, and GradCAM
│   │   │   └── schemas.py           # Pydantic request/response models
│   │   ├── services/
│   │   │   ├── dataset_service.py   # Dataset statistics & sample loading
│   │   │   ├── train_service.py     # Background training worker
│   │   │   ├── eval_service.py      # Test set evaluation & metrics
│   │   │   ├── predict_service.py   # Inference & top-k ranking
│   │   │   └── explain_service.py   # Grad-CAM overlay & quadrant scoring
│   │   └── utils/
│   │       └── image_processing.py  # PIL transforms, base64, colormaps
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx           # Top navigation & system status
│   │   │   ├── Sidebar.jsx          # Collapsible navigation drawer
│   │   │   ├── MetricCard.jsx       # Reusable KPI card
│   │   │   └── ConfusionMatrix.jsx  # Interactive 6x6 confusion matrix
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx        # Executive dashboard
│   │   │   ├── DatasetExplorer.jsx  # Parquet records & thumbnail gallery
│   │   │   ├── DataAnalysis.jsx     # Recharts distributions & data quality
│   │   │   ├── ModelTraining.jsx    # Training config & live epoch tracking
│   │   │   ├── ModelPerformance.jsx # Loss/Acc curves, CM & class breakdown
│   │   │   ├── Prediction.jsx       # Single image upload & inference
│   │   │   ├── BatchPrediction.jsx  # Multi-file batch inference & CSV export
│   │   │   ├── Explainability.jsx   # Grad-CAM visual heatmaps & insights
│   │   │   ├── MultimodalKIE.jsx    # OCR bounding boxes & extracted entities
│   │   │   └── AboutProject.jsx     # Architecture & technical documentation
│   │   ├── services/
│   │   │   └── api.js               # Frontend API client
│   │   ├── App.jsx                  # Root React application
│   │   ├── index.css                # Glassmorphic dark styling
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js               # Vite config with FastAPI proxy
├── artifacts/
│   ├── cache/                       # Cached Float32 tensors & metadata
│   ├── evaluation_metrics.json      # Authentic test set benchmarks
│   └── training_history.json        # Epoch-by-epoch loss & accuracy logs
├── models/
│   ├── best_model.pt                # Checkpoint with highest validation accuracy
│   └── latest_model.pt              # Most recent model state
├── reports/
│   └── TECHNICAL_REPORT.md          # Comprehensive technical whitepaper
├── scripts/
│   ├── prepare_cache.py             # Preprocesses raw Parquets into tensors
│   ├── test_model.py                # MobileNetV3 CPU verification
│   └── test_gradcam.py              # Grad-CAM engine validation
├── train.py                         # Standalone CLI training script
├── evaluate.py                      # Standalone CLI evaluation script
├── requirements.txt                 # Root Python dependencies
└── README.md
```

---

## Installation & Setup

### Prerequisites

- **Python 3.13** (64-bit)
- **Node.js 18+** and **npm**
- **Git**

### 1. Backend Setup

From the root project directory:

```bash
# Verify Python version
py -3.13 --version

# Install dependencies (already present in local Python 3.13)
py -3.13 -m pip install -r requirements.txt
```

### 2. Frontend Setup

```bash
cd frontend
npm install
cd ..
```

---

## Execution Guide

### Step 1: Train the Model via CLI

Train a model on your AMD Ryzen CPU with 5 epochs:

```bash
py -3.13 train.py --epochs 5 --model mobilenet_v3 --batch_size 32 --lr 0.001
```

The script will:
1. Detect and verify the Parquet dataset cache.
2. Train MobileNetV3 on CPU with real-time epoch logs.
3. Automatically evaluate on all 361 independent test receipts.
4. Save `models/best_model.pt` and `artifacts/evaluation_metrics.json`.

### Step 2: Evaluate the Saved Model

```bash
py -3.13 evaluate.py --model_path models/best_model.pt
```

### Step 3: Launch FastAPI Backend

```bash
py -3.13 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

The REST API will be accessible at `http://127.0.0.1:8000` (Interactive Swagger Docs at `http://127.0.0.1:8000/docs`).

### Step 4: Launch React Frontend

In a separate terminal:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health, cache state, and CPU device status |
| `GET` | `/api/dataset/summary` | Dataset totals, class distributions, and image stats |
| `GET` | `/api/dataset/records` | Paginated records with split, category, and search filters |
| `GET` | `/api/dataset/receipt/{split}/{idx}` | Receipt detail with OCR bounding boxes & entities |
| `GET` | `/api/model/status` | Live training state, current epoch, and progress % |
| `POST` | `/api/model/train` | Trigger background model training job |
| `GET` | `/api/model/performance` | Authentic test set metrics, confusion matrix, and class breakdown |
| `POST` | `/api/model/evaluate` | Manually re-run test set evaluation |
| `POST` | `/api/predict/image` | Predict category & confidence for uploaded receipt file |
| `POST` | `/api/predict/sample` | Run inference on a specific benchmark sample |
| `POST` | `/api/predict/batch` | Batch inference on multiple uploaded receipts |
| `GET` | `/api/predict/batch/sample_test` | Run batch inference on test set slice |
| `POST` | `/api/explain/image` | Compute Grad-CAM heatmap & regional scores for uploaded file |
| `POST` | `/api/explain/sample` | Compute Grad-CAM heatmap for benchmark sample |

---

## Screenshots & Interface Highlights

- **Executive Dashboard**: High-level KPI metric cards, dataset distribution comparison, model performance summary, and quick navigation.
- **Dataset Explorer**: Paginated receipt gallery with resolution badges, merchant information, total amounts, and modal inspector.
- **Data Analysis**: Recharts class balance bar chart, train/test ratio donut, token density area chart, and schema completeness audit.
- **Model Training**: Hyperparameter configuration panel, live epoch progress bar, real-time loss/accuracy curves, and training history table.
- **Model Performance**: Loss and accuracy curves vs epoch, 6×6 interactive confusion matrix with cell hover inspector, and class-wise precision/recall/F1 table.
- **Prediction System**: Drag-and-drop receipt uploader, confidence progress bar, top-k ranking, and Grad-CAM overlay toggle.
- **Batch Prediction**: Multi-file batch processor with confidence ranking and one-click CSV export.
- **Explainability**: Side-by-side comparison of original document, JET heatmap, and blended overlay with spatial quadrant weights.
- **Multimodal KIE**: Interactive receipt document viewer showcasing SROIE bounding boxes colored by entity (`company`, `date`, `address`, `total`, `other`).

---

## Future Improvements

1. **Multimodal Transformer Integration**: Fine-tuning LayoutLMv3 or Donut for joint visual-spatial-textual receipt document parsing.
2. **End-to-End Token Classification**: Training a token-level BiLSTM-CRF on the 53,000+ OCR tokens for automated extraction of unknown receipt formats.
3. **ONNX Runtime Export**: Quantizing PyTorch checkpoints to INT8 ONNX models for sub-5ms CPU inference.
4. **Mobile Web Camera Scanner**: Integrating real-time web camera capture with perspective correction for live receipt scanning.
