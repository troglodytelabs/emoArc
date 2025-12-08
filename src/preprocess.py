from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd


def normalize_text(text: str) -> str:
    """Lightweight normalization: lowercase, strip URLs, punctuation, and collapse spaces."""

    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def merge_fields(df: pd.DataFrame) -> List[str]:
    """Concatenate metadata and text into a single document string per row."""

    documents: List[str] = []
    for row in df.itertuples(index=False):
        parts = [
            getattr(row, "title", ""),
            getattr(row, "author", ""),
            getattr(row, "description", ""),
            getattr(row, "tags", ""),
            getattr(row, "text", ""),
        ]
        merged = "\n".join(str(p) for p in parts if isinstance(p, str) and p.strip())
        documents.append(normalize_text(merged))
    return documents


def normalize_documents(texts: Iterable[str]) -> List[str]:
    return [normalize_text(t) for t in texts]
