# Document Similarity Finder

This project is a Document Similarity Finder tool that processes a dataset of documents, computes cosine similarity, and visualizes the results. It is designed to help users identify similar documents based on their content.

## Project Structure

```
Document-Similarity-Checker
├── src
│   ├── main.py               # Entry point of the application
│   ├── similarity.py         # Functions to compute cosine similarity
│   ├── preprocessing.py      # Text preprocessing functions
│   ├── visualization.py      # Visualization functions for results
│   └── __init__.py          # Marks the directory as a Python package
├── data
│   ├── raw
│   │   └── documents.csv     # Raw dataset of documents
│   └── processed
│       └── tfidf_matrix.npz  # Processed TF-IDF matrix
├── notebooks
│   └── exploration.ipynb     # Jupyter notebook for exploratory data analysis
├── tests
│   └── test_similarity.py     # Unit tests for similarity functions
├── requirements.txt           # Project dependencies
├── pyproject.toml            # Project configuration and metadata
├── .gitignore                 # Files to ignore in version control
└── README.md                  # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <https://github.com/BluntPhoenix04/NLP-assignment-2301201119_Siddhant.git>
cd document-similarity-finder
pip install -r requirements.txt
```

## Usage

To run the Document Similarity Finder, execute the following command:

```bash
python src/main.py
```

This will preprocess the documents, compute the cosine similarity, and display the top 3 most similar document pairs along with a similarity matrix heatmap.

