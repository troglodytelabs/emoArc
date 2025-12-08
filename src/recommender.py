from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel

from .features import TfidfFeatures
from .preprocess import merge_fields, normalize_documents


@dataclass
class Recommendation:
    title: str
    author: str
    score: float


class BookRecommender:
    """Content-based recommender built from scratch with TF-IDF features."""

    def __init__(self, feature_extractor: Optional[TfidfFeatures] = None):
        self.feature_extractor = feature_extractor or TfidfFeatures()
        self.matrix: Optional[sparse.csr_matrix] = None
        self.corpus: Optional[pd.DataFrame] = None
        self.title_index: dict[str, int] = {}

    def fit(self, corpus_df: pd.DataFrame) -> None:
        documents = merge_fields(corpus_df)
        self.matrix = self.feature_extractor.fit_transform(documents)
        self.corpus = corpus_df.reset_index(drop=True)
        self.title_index = {title.lower(): i for i, title in enumerate(self.corpus["title"])}

    def _ensure_fitted(self) -> None:
        if self.matrix is None or self.corpus is None:
            raise RuntimeError("Recommender has not been fitted yet")

    def recommend_by_title(self, title: str, top_n: int = 5) -> List[Recommendation]:
        self._ensure_fitted()
        assert self.matrix is not None and self.corpus is not None

        key = title.lower()
        if key not in self.title_index:
            raise KeyError(f"Title not found in corpus: {title}")

        seed_idx = self.title_index[key]
        seed_vector = self.matrix[seed_idx]
        scores = linear_kernel(seed_vector, self.matrix).flatten()

        ranked_indices = scores.argsort()[::-1]
        recommendations: List[Recommendation] = []
        for idx in ranked_indices:
            if idx == seed_idx:
                continue
            if len(recommendations) >= top_n:
                break
            row = self.corpus.iloc[idx]
            recommendations.append(
                Recommendation(
                    title=str(row.title),
                    author=str(row.author),
                    score=float(scores[idx]),
                )
            )
        return recommendations

    def recommend_for_text(self, text: str, top_n: int = 5) -> List[Recommendation]:
        self._ensure_fitted()
        assert self.matrix is not None and self.corpus is not None

        normalized = normalize_documents([text])
        query_vector = self.feature_extractor.transform(normalized)
        scores = linear_kernel(query_vector, self.matrix).flatten()

        ranked_indices = scores.argsort()[::-1][:top_n]
        recommendations: List[Recommendation] = []
        for idx in ranked_indices:
            row = self.corpus.iloc[idx]
            recommendations.append(
                Recommendation(
                    title=str(row.title),
                    author=str(row.author),
                    score=float(scores[idx]),
                )
            )
        return recommendations

    def attach_additional_features(self, features: sparse.csr_matrix) -> None:
        """Hook for experimental features (e.g., emotion scores) to blend with TF-IDF."""

        self._ensure_fitted()
        assert self.matrix is not None
        self.matrix = sparse.hstack([self.matrix, features]).tocsr()
