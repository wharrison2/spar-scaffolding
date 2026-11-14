"""Shared, local (no network, no AI) lexical similarity measures for emergent misalignment prompts.

Two measures, both exact over all pairs and both in [0, 1]:
  (a) TF-IDF cosine on word 1-2 grams (sublinear term frequency, lower-cased)
  (b) Jaccard of the sets of character 5-gram shingles (lower-cased)
Used by find_near_duplicate_prompts_with_local_lexical_similarity.py (reporting) and
filter_near_duplicate_and_held_out_similar_prompts_from_combined_snapshot.py (applying thresholds).
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

MEASURE_NAME_TFIDF_COSINE = "tfidf_cosine"
MEASURE_NAME_CHARACTER_FIVE_GRAM_JACCARD = "char5_jaccard"


class LexicalSimilarityModel:
    """Vectorisers fitted on a reference set of texts (the pool); any texts can then be compared to each other."""

    def __init__(self, reference_texts: list[str]):
        self.tfidf_vectoriser = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True).fit(reference_texts)
        self.shingle_vectoriser = CountVectorizer(analyzer="char", ngram_range=(5, 5), binary=True, lowercase=True).fit(reference_texts)

    def vectorise(self, texts: list[str]):
        """(tfidf matrix, shingle matrix, shingle count per text)."""
        tfidf_matrix = self.tfidf_vectoriser.transform(texts)
        shingle_matrix = self.shingle_vectoriser.transform(texts).astype(np.float32).tocsr()
        return tfidf_matrix, shingle_matrix, np.asarray(shingle_matrix.sum(1)).ravel()


def cosine_all_pairs(matrix_a, matrix_b) -> np.ndarray:
    return (matrix_a @ matrix_b.T).toarray()


def jaccard_all_pairs(shingles_a, sizes_a, shingles_b, sizes_b) -> np.ndarray:
    intersection = (shingles_a @ shingles_b.T).toarray()
    union = sizes_a[:, None] + sizes_b[None, :] - intersection
    return intersection / np.maximum(union, 1)


def both_similarity_matrices(model: LexicalSimilarityModel, texts_a: list[str], texts_b: list[str]) -> dict[str, np.ndarray]:
    tfidf_a, shingles_a, sizes_a = model.vectorise(texts_a)
    tfidf_b, shingles_b, sizes_b = model.vectorise(texts_b)
    return {MEASURE_NAME_TFIDF_COSINE: cosine_all_pairs(tfidf_a, tfidf_b),
            MEASURE_NAME_CHARACTER_FIVE_GRAM_JACCARD: jaccard_all_pairs(shingles_a, sizes_a, shingles_b, sizes_b)}
