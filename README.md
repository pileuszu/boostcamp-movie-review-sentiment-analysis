# 🎬 Movie Review Sentiment Analysis

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange.svg)](https://huggingface.co/transformers/)

A high-performance Korean movie review sentiment classification project developed as part of the Boostcamp NLP program. This project implements advanced domain adaptation techniques and specialized loss functions to achieve state-of-the-art results in 4-class sentiment analysis.

---

## 🚀 Key Features

- **Domain Adaptation**: Implements **TAPT (Task-Adaptive Pre-Training)** and **Contrastive TAPT** to adapt general-purpose Korean BERT models to the specific nuances of movie review language.
- **Advanced Preprocessing**: A robust pipeline that cleans noise (URLs, metadata) while preserving crucial emotional signals like emoticons (`ㅋ`, `ㅠ`, `❤️`) and punctuation.
- **Handling Imbalance**: Includes a specialized **Focal Loss Pipeline** to effectively handle class imbalance and improve performance on minority sentiment categories.
- **Multi-class Sentiment**: Classifies reviews into 4 distinct levels:
  - `0`: Negative
  - `1`: Neutral
  - `2`: Positive
  - `3`: Strong Positive

---

## 📂 Repository Structure

```text
├── data/                       # Dataset directory (train.csv, test.csv)
├── src/                        # Source code
│   ├── 01_tapt_pretrain.py     # Domain-specific pre-training (TAPT)
│   ├── 03_train.py             # Main fine-tuning script
│   ├── 04_predict.py           # Inference and submission generation
│   ├── focal_loss_pipeline/    # Specialized Trainer with Focal Loss
│   ├── bert_model_comparison/   # Experiments with various BERT variants
│   └── notebooks/              # EDA and experimental analysis
├── model/                      # Saved checkpoints (TAPT, Fine-tuned)
├── output/                     # Prediction results (CSV)
└── requirements.txt            # Project dependencies
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pileuszu/boostcamp-movie-review-sentiment-analysis.git
   cd boostcamp-movie-review-sentiment-analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Usage Guide

### 1. Task-Adaptive Pre-Training (TAPT)
Before classification, adapt the base model to the movie review domain. This script performs Masked Language Modeling (MLM) on the combined train/test corpus.
```bash
python src/01_tapt_pretrain.py
```

### 2. Model Fine-tuning
Fine-tune the TAPT-adapted model for the 4-class sentiment classification task.
```bash
python src/03_train.py
```

### 3. Inference
Generate predictions on the test set using the best checkpoint.
```bash
python src/04_predict.py
```

---

## 🧠 Technical Details

### Task-Adaptive Pre-Training (TAPT)
General models like `kykim/bert-kor-base` are trained on diverse datasets. TAPT allows the model to learn the specific vocabulary and emotional expressions (e.g., "인생영화", "노잼") used in movie reviews before the actual classification task, significantly boosting F1-scores.

### Contrastive TAPT
An experimental pre-training method that uses contrastive learning to pull similar semantic representations together, improving the model's ability to distinguish between subtle sentiment differences.

### Focal Loss
Standard Cross Entropy can be dominated by easy-to-classify samples. Focal Loss applies a modulating factor to the loss, focusing training on hard, misclassified examples which are often found in neutral or mixed reviews.

---

## 📊 Performance Metrics

The project focuses on **Weighted F1-Score** and **Accuracy** as primary metrics to ensure balanced performance across all four classes.

---

## 👥 Contributors

- [pileuszu](https://github.com/pileuszu)

---

## 📄 License
This project is for educational purposes as part of the Boostcamp program.
