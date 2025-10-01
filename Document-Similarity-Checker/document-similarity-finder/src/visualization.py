import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_heatmap(matrix, labels, output_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="viridis", annot=False)

    if output_path:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(outp), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def display_top_pairs(similarity_matrix, document_names):
    num_docs = similarity_matrix.shape[0]
    similar_pairs = []

    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            similar_pairs.append((document_names[i], document_names[j], similarity_matrix[i, j]))

    # Sort pairs by similarity score in descending order
    similar_pairs.sort(key=lambda x: x[2], reverse=True)

    # Display top 3 most similar pairs
    print("Top 3 Most Similar Document Pairs:")
    for doc1, doc2, score in similar_pairs[:3]:
        print(f"{doc1} - {doc2}: {score:.2f}")