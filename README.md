# 📂 NLP Assignments Repository

**Instructor:** Sahil Singh

**Course Module:** Natural Language Processing (NLP)

This repository contains two NLP-focused assignments:

1. **📚 Document Similarity Check** – analyzing and comparing documents using TF-IDF and cosine similarity.
2. **🐦 Twitter Sentiment Analysis** – classifying tweets into emotions like Happy, Sad, Angry, and Neutral.

---

## 📁 Repository Structure

```
nlp-assignments/
│── document_similarity/      # Project 1: Document Similarity
│   ├── data/                 # Input datasets (CSV)
│   ├── outputs/              # Generated outputs (heatmaps)
│   ├── src/                  # Python source code
│   └── README.md             # Project-specific documentation
│
│── twitter_sentiment/        # Project 2: Twitter Sentiment Analysis
│   ├── data/                 # Tweet datasets
│   ├── notebooks/            # Jupyter notebooks
│   ├── src/                  # Source code
│   └── README.md             # Project-specific documentation
│
│── README.md                 # 🔹 This overview file
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone <https://github.com/BluntPhoenix04/NLP-assignment-2301201119_Siddhant.git>
cd nlp-assignments
pip install -r requirements.txt
```

Dependencies include:

* `pandas`, `numpy`
* `scikit-learn`
* `matplotlib`, `seaborn`
* `nltk`

---

## 🚀 Projects

### 1️⃣ Document Similarity Check

* Implements preprocessing (tokenization, stopword removal, lemmatization).
* Computes **TF-IDF vectors** and **cosine similarity**.
* Visualizes document relationships via a **heatmap**.
* Outputs top most similar pairs.

---

### 2️⃣ Twitter Sentiment Analysis

* Preprocesses noisy Twitter data.
* Uses **TF-IDF/BOW** with classifiers like Naive Bayes, Logistic Regression, SVM.
* Evaluates with **accuracy, confusion matrix, and classification report**.
* Provides Jupyter notebook for training & testing.


---

---

