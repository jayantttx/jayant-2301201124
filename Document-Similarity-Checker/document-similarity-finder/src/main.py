import pandas as pd
from preprocessing import tokenize, remove_stopwords, lemmatize
from similarity import compute_cosine_similarity
from visualization import plot_heatmap, display_top_pairs
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pathlib import Path

def load_documents(file_path: str | None = None):

    project_root = Path(__file__).parents[1]
    candidates = []

    if file_path:
        candidates.append(Path(file_path))              # as provided (cwd-relative or absolute)
        candidates.append(project_root / file_path)    # project-root relative
    else:
        candidates.append(project_root / "data" / "raw" / "documents.csv")
        candidates.append(project_root / "data" / "documents.csv")

    for p in candidates:
        if p and p.exists():
            df = pd.read_csv(p)
            # CSV in your attachment uses columns: title,content
            if "content" in df.columns:
                return df["content"].astype(str).tolist()
            for col in ("text", "body"):
                if col in df.columns:
                    return df[col].astype(str).tolist()
            # fallback: join all columns per row
            return df.astype(str).agg(" ".join, axis=1).tolist()

    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Documents file not found. Tried these locations:\n{tried}")

def preprocess_documents(documents):
    processed_docs = []
    for doc in documents:
        tokens = tokenize(doc)
        tokens = remove_stopwords(tokens)
        processed_doc = lemmatize(tokens)
        processed_docs.append(' '.join(processed_doc))
    return processed_docs

def main():
    # Load documents
    documents = load_documents('data/raw/documents.csv')
    
    # Preprocess documents
    processed_docs = preprocess_documents(documents)
    
    # Compute TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    
    # Compute cosine similarity
    similarity_matrix = compute_cosine_similarity(tfidf_matrix)
    
    # Display heatmap -- provide labels and output path
    out_plot = Path(__file__).parents[1] / "outputs" / "similarity_heatmap.png"
    labels = [d if len(d) <= 30 else d[:27] + "..." for d in documents]  # short doc names
    plot_heatmap(similarity_matrix, labels, str(out_plot))
    
    # Display top 3 most similar document pairs
    display_top_pairs(similarity_matrix, documents)

if __name__ == "__main__":
    main()