from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass
class CorpusRow:
    """Lightweight record representing a single book."""

    title: str
    author: str
    description: str
    tags: str
    text: str
    source_path: Path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def load_corpus(metadata_csv: Path, books_dir: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """Load metadata and book text into a tidy DataFrame.

    Args:
        metadata_csv: CSV containing at least `title` and `path` columns.
        books_dir: Directory where book text files live.
        limit: Optional maximum number of rows to load (useful for quick tests).

    Returns:
        DataFrame with columns [title, author, description, tags, text, source_path].
    """

    books_dir = books_dir.expanduser().resolve()
    df = pd.read_csv(metadata_csv)
    if limit is not None:
        df = df.iloc[:limit].copy()

    required = {"title", "path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    for optional in ["author", "description", "tags"]:
        if optional not in df.columns:
            df[optional] = ""

    texts: list[str] = []
    paths: list[Path] = []
    for rel_path in df["path"]:
        candidate = (books_dir / str(rel_path)).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Book file not found: {candidate}")
        paths.append(candidate)
        texts.append(_read_text(candidate))

    df = df.assign(
        author=df["author"].fillna(""),
        description=df["description"].fillna(""),
        tags=df["tags"].fillna(""),
        text=texts,
        source_path=paths,
    )

    return df


def iter_corpus_rows(df: pd.DataFrame) -> Iterable[CorpusRow]:
    for row in df.itertuples(index=False):
        yield CorpusRow(
            title=str(row.title),
            author=str(row.author),
            description=str(row.description),
            tags=str(row.tags),
            text=str(row.text),
            source_path=Path(row.source_path),
        )
