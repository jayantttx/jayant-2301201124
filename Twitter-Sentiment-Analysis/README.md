# Twitter Sentiment Analysis

A small project to experiment with sentiment classification on short comments. It includes a labeled CSV of comments and minimal scripts for preprocessing, feature extraction, training, and visualization.

## Project Structure

```
Twitter-Sentiment-Analysis
├── Comment-Sentiment.csv     # Labeled dataset (Comment, Sentiment)
├── README.md                 # Project documentation (this file)
├── scripts/                  # Suggested scripts folder
│   ├── preprocess.py         # Tokenize, stopword removal, lemmatize
│   ├── features.py           # TF-IDF / feature extraction
│   ├── train.py              # Train and save a classifier
│   └── evaluate.py           # Evaluation and reports
├── notebooks/                # Optional EDA / experiments
│   └── exploration.ipynb
├── outputs/                  # Generated plots, models, logs
│   ├── figures/
│   └── models/
├── requirements.txt          # Python dependencies
└── .gitignore
```

## Installation

Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you don't have requirements.txt, install commonly used packages:

```powershell
pip install pandas scikit-learn nltk matplotlib seaborn
python -c "import nltk; nltk.download('punkt','stopwords','wordnet')"
```

## Usage

Quick examples:

- Inspect dataset:
  ```powershell
  python - <<'PY'
  import pandas as pd
  df = pd.read_csv("Comment-Sentiment.csv")
  print(df.head())
  print(df['Sentiment'].value_counts())
  PY
  ```

- Run a training/evaluation script (if implemented):
  ```powershell
  python scripts/train.py
  python scripts/evaluate.py
  ```

- Run a simple analysis notebook:
  Open notebooks/exploration.ipynb in Jupyter or VS Code.

## Notes & Suggestions

- Dataset is small — use cross-validation and report macro F1 for multi-class evaluation.
- Suggested model: TF-IDF + LogisticRegression (or SVM). Consider class balancing/augmentation if needed.
- Add scripts under scripts/ to keep reproducible workflows: preprocess -> feature extract -> train -> evaluate.

## Quick commands to display project files (PowerShell)

- Recursive list:
  Get-ChildItem -Recurse -Name

- Tree view:
  tree /F /A
