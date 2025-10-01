def compute_cosine_similarity(tfidf_matrix):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix

def get_top_similar_pairs(similarity_matrix, documents, top_n=3):
    # Get the indices of the top N most similar document pairs
    similar_pairs = []
    num_docs = similarity_matrix.shape[0]

    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            similar_pairs.append((i, j, similarity_matrix[i, j]))

    # Sort pairs by similarity score in descending order
    similar_pairs.sort(key=lambda x: x[2], reverse=True)

    # Get the top N pairs
    top_pairs = similar_pairs[:top_n]

    # Format the output
    results = [(documents[i], documents[j], score) for i, j, score in top_pairs]

    return results