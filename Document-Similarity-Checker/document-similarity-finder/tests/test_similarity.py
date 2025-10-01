import unittest
import numpy as np
from src.similarity import compute_cosine_similarity

class TestCosineSimilarity(unittest.TestCase):

    def setUp(self):
        # Sample TF-IDF matrix for testing
        self.tfidf_matrix = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1],
                                       [1, 1, 0]])

    def test_cosine_similarity_shape(self):
        similarity_matrix = compute_cosine_similarity(self.tfidf_matrix)
        self.assertEqual(similarity_matrix.shape, (4, 4), "Similarity matrix shape should be 4x4")

    def test_cosine_similarity_values(self):
        similarity_matrix = compute_cosine_similarity(self.tfidf_matrix)
        expected_values = np.array([[1.0, 0.0, 0.0, 0.70710678],
                                     [0.0, 1.0, 0.0, 0.70710678],
                                     [0.0, 0.0, 1.0, 0.0],
                                     [0.70710678, 0.70710678, 0.0, 1.0]])
        np.testing.assert_almost_equal(similarity_matrix, expected_values, decimal=5)

if __name__ == '__main__':
    unittest.main()