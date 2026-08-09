---
title: Toxic Comment Classifier
emoji: 🛡️
colorFrom: red
colorTo: yellow
sdk: gradio
python_version: "3.12"
app_file: app.py
pinned: false
---

<div align="center">

# 🛡️ Toxic Comment Classifier

### *AI-Powered Real-Time Content Moderation*

<p>
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=6366F1&center=true&vCenter=true&random=false&width=600&lines=Multi-Label+Toxicity+Detection;Logistic+Regression+%2B+BiLSTM;FastText+300D+Embeddings;Real-Time+Comment+Analysis;97%25+Mean+ROC-AUC+Score" alt="Typing SVG" />
</p>

<p>
  <a href="https://huggingface.co/spaces/YashAI07/Toxic-Comment-Classifier">
    <img src="https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-FFD21E?style=for-the-badge&labelColor=000" alt="HF Live Demo" />
  </a>
  <a href="https://tinyurl.com/2c6bns36">
    <img src="https://img.shields.io/badge/☁️%20Live%20App-AWS%20EC2-FF9900?style=for-the-badge&logo=amazonaws&labelColor=000" alt="AWS Live Demo" />
  </a>
  <a href="https://github.com/vyash0048-bit/Toxic-Comment-Classifier">
    <img src="https://img.shields.io/badge/Source%20Code-GitHub-181717?style=for-the-badge&logo=github&labelColor=000" alt="GitHub" />
  </a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Gradio-5.x-FF7C00?style=flat-square&logo=gradio&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/DVC-Pipeline-13ADC7?style=flat-square&logo=dvc&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-Tracked-0194E2?style=flat-square&logo=mlflow&logoColor=white" />
</p>

<br/>

> Detect **toxic**, **obscene**, **threatening**, and **hateful** language in real-time  
> using a dual-model architecture powered by TF-IDF and FastText embeddings.

<br/>

---

</div>

## ⚡ Highlights

<table>
<tr>
<td width="50%">

### 🎯 Dual-Model Architecture
Choose between a blazing-fast **Logistic Regression** baseline or a deep **Bidirectional LSTM** model — both trained on 6 toxicity categories.

</td>
<td width="50%">

### 📊 97% Mean ROC-AUC
Industry-grade classification performance with per-label optimized decision thresholds and comprehensive evaluation metrics.

</td>
</tr>
<tr>
<td width="50%">

### 🔬 Advanced NLP Features
Dual TF-IDF (word + character n-grams) with 110K features and pre-trained FastText 300D embeddings for deep semantic understanding.

</td>
<td width="50%">

### 🚀 Production-Ready MLOps
End-to-end reproducible pipeline with DVC versioning, MLflow experiment tracking, Docker containerization, and Hugging Face deployment.

</td>
</tr>
</table>

<br/>

## 🏗️ System Architecture

```mermaid
flowchart LR
    subgraph Data["📦 Data Layer"]
        A[(MongoDB)] -->|PyMongo| B[Data Ingestion]
        K -->|Async Store| A
    end

    subgraph Preprocessing["⚙️ Feature Engineering"]
        B --> C[Text Cleaning]
        C --> D["TF-IDF\n(Word + Char)"]
        C --> E["Keras Tokenizer\n+ Padding"]
    end

    subgraph Models["🧠 Model Training"]
        D --> F["Logistic Regression\n(OneVsRest)"]
        E --> G["BiLSTM\n(FastText 300D)"]
    end

    subgraph Evaluation["📊 Evaluation"]
        F --> H[ROC-AUC / F1]
        G --> I["Threshold\nOptimization"]
        I --> H
    end

    subgraph Deployment["🚀 Deployment"]
        H --> J[MLflow + DagsHub]
        H --> K["Gradio UI\n(HF ZeroGPU)"]
    end

    style Data fill:#1e1b4b,stroke:#6366f1,color:#fff
    style Preprocessing fill:#172554,stroke:#3b82f6,color:#fff
    style Models fill:#14532d,stroke:#22c55e,color:#fff
    style Evaluation fill:#713f12,stroke:#eab308,color:#fff
    style Deployment fill:#7f1d1d,stroke:#ef4444,color:#fff
```

<br/>

## 🧠 Models

<details open>
<summary><b>📈 Model 1 — Logistic Regression (Baseline)</b></summary>

<br/>

| Component | Detail |
|:--|:--|
| **Feature Extraction** | `FeatureUnion` of Word TF-IDF + Character TF-IDF |
| **Word TF-IDF** | N-grams `(1, 2)` · 50,000 features · Sublinear TF |
| **Char TF-IDF** | N-grams `(3, 5)` · 60,000 features · Sublinear TF |
| **Classifier** | `OneVsRestClassifier(LogisticRegression)` |
| **Solver** | SAGA · Regularization C=0.5 |
| **Total Features** | ~110,000 sparse dimensions |

</details>

<details open>
<summary><b>🧬 Model 2 — BiLSTM (Deep Learning)</b></summary>

<br/>

| Layer | Configuration |
|:--|:--|
| **Embedding** | FastText 300D · Vocab 20K · Non-trainable |
| **SpatialDropout1D** | Rate: 0.2 |
| **Bidirectional LSTM** | 128 units · Return sequences · Dropout 0.2 |
| **GlobalMaxPool1D** | Temporal max pooling |
| **Dense** | 64 units · ReLU activation |
| **Dropout** | Rate: 0.3 |
| **Output** | 6 units · Sigmoid (multi-label) |
| **Optimizer** | Adam · LR: 0.001 |
| **Callbacks** | ReduceLROnPlateau · EarlyStopping on `val_auc` |

</details>

<br/>

## 📊 Performance Comparison

<div align="center">

| Category | LR ROC-AUC | BiLSTM ROC-AUC | LR F1 | BiLSTM F1 |
|:--|:--:|:--:|:--:|:--:|
| 🔴 **Toxic** | 0.9427 | 0.9426 | 0.6613 | 0.6076 |
| 🟣 **Severe Toxic** | 0.9843 | **0.9860** | 0.3120 | **0.3524** |
| 🟠 **Obscene** | 0.9762 | 0.9744 | 0.6795 | **0.6799** |
| 🔵 **Threat** | **0.9893** | 0.9563 | **0.3765** | 0.1062 |
| 🟡 **Insult** | 0.9699 | 0.9666 | 0.5701 | **0.5925** |
| ⚫ **Identity Hate** | 0.9430 | 0.9214 | 0.3418 | **0.3758** |
| | | | | |
| 🏆 **Mean ROC-AUC** | **0.9676** | 0.9579 | — | — |

</div>

<br/>

## 🔄 ML Pipeline

The project uses **DVC** (Data Version Control) for a fully reproducible, 6-stage pipeline:

```mermaid
flowchart TB
    S1["1️⃣ Data Ingestion\n<i>MongoDB → CSV</i>"]
    S2["2️⃣ Data Preprocessing\n<i>Clean + TF-IDF</i>"]
    S3["3️⃣ LR Training\n<i>OneVsRest SAGA</i>"]
    S4["4️⃣ LR Evaluation\n<i>ROC-AUC + MLflow</i>"]
    S5["5️⃣ BiLSTM Training\n<i>FastText + LSTM</i>"]
    S6["6️⃣ BiLSTM Evaluation\n<i>Threshold Tuning</i>"]

    S1 --> S2
    S2 --> S3
    S3 --> S4
    S2 --> S5
    S5 --> S6

    style S1 fill:#312e81,stroke:#818cf8,color:#fff
    style S2 fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style S3 fill:#14532d,stroke:#4ade80,color:#fff
    style S4 fill:#713f12,stroke:#fbbf24,color:#fff
    style S5 fill:#4c1d95,stroke:#a78bfa,color:#fff
    style S6 fill:#7f1d1d,stroke:#f87171,color:#fff
```

Run the full pipeline:
```bash
dvc repro
```

<br/>

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|:--|:--|
| **ML / DL** | Scikit-Learn · TensorFlow / Keras · Gensim |
| **NLP** | TF-IDF (Word + Char N-grams) · FastText 300D Embeddings |
| **Data** | MongoDB · Pandas · NumPy · SciPy Sparse |
| **Web** | Gradio |
| **MLOps** | DVC · MLflow · DagsHub |
| **Deployment** | Hugging Face Spaces (ZeroGPU) |
| **Monitoring** | Custom Logging · Async MongoDB Prediction Storage |

</div>

<br/>

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/vyash0048-bit/Toxic-Comment-Classifier.git
cd Toxic-Comment-Classifier
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
# Create a .env file
MONGODB_URI=your_mongodb_connection_string
DATABASE_NAME=your_database
PREDICTION_COLLECTION=predictions
TRAIN_COLLECTION=train
TEST_COLLECTION=test
TEST_LABELS_COLLECTION=test_labels
MLFLOW_TRACKING_URI=https://dagshub.com/your_user/your_repo.mlflow
DAGSHUB_TOKEN=your_token
```

### 3. Run the Pipeline

```bash
# Full DVC pipeline
dvc repro

# Or run directly
python main.py
```

### 4. Launch the App

```bash
# Gradio UI with ZeroGPU support
python app.py
```

<br/>

## 📁 Project Structure

```
Toxic-Comment-Classifier/
├── 📄 app.py                    # Gradio web interface (HF Spaces)
├── 📄 flask_app.py              # Flask REST API + dashboard
├── 📄 main.py                   # Full training pipeline runner
├── 📄 Dockerfile                # Container deployment
├── 📄 dvc.yaml                  # DVC pipeline definition
├── 📄 params.yaml               # Hyperparameters config
│
├── 📂 src/ToxicCommentClassifier/
│   ├── 📂 components/           # Core ML components
│   │   ├── data_ingestion.py        # MongoDB data extraction
│   │   ├── data_preprocessing.py    # Text cleaning + TF-IDF
│   │   ├── model_training.py        # Logistic Regression
│   │   ├── model_evaluation.py      # LR metrics + MLflow
│   │   ├── bilstm_training.py       # BiLSTM + FastText
│   │   └── bilstm_evaluation.py     # BiLSTM metrics + thresholds
│   ├── 📂 pipeline/             # Pipeline orchestration
│   │   ├── prediction_pipeline.py   # Inference with lazy loading
│   │   └── stage_01..06_*.py        # Training stage runners
│   ├── 📂 config/               # Configuration management
│   └── 📂 entity/               # Data classes & schemas
│
├── 📂 artifacts/                # Model weights & processed data
├── 📂 config/                   # YAML configuration files
├── 📂 templates/                # Flask HTML dashboard
└── 📂 research/                 # Jupyter notebooks
```

<br/>



## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

<br/>

<div align="center">

---

<p align="center">
  <img src="assets/footer_image.jpg" alt="Toxic Comment Classifier Footer Banner" width="800">
</p>

<p>
  <a href="https://github.com/vyash0048-bit">
    <img src="https://img.shields.io/badge/GitHub-vyash0048--bit-181717?style=flat-square&logo=github" />
  </a>
</p>

</div>