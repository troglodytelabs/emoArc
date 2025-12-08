from __future__ import annotations

from typing import List, Optional

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_VECTORIZER_KWARGS = dict(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)


class TfidfFeatures:
    def __init__(self, vectorizer: Optional[TfidfVectorizer] = None):
        self.vectorizer = vectorizer or TfidfVectorizer(**DEFAULT_VECTORIZER_KWARGS)
        self.matrix: Optional[sparse.csr_matrix] = None

    def fit(self, documents: List[str]) -> sparse.csr_matrix:
        self.matrix = self.vectorizer.fit_transform(documents)
        return self.matrix

    def transform(self, documents: List[str]) -> sparse.csr_matrix:
        return self.vectorizer.transform(documents)

    def fit_transform(self, documents: List[str]) -> sparse.csr_matrix:
        self.matrix = self.vectorizer.fit_transform(documents)
        return self.matrix

    def feature_names(self) -> List[str]:
        return list(self.vectorizer.get_feature_names_out())
