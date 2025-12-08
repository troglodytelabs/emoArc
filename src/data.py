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


def _find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """Resolve a column name by trying common aliases (case-insensitive)."""

    lowered = {col.lower(): col for col in df.columns}
    for alias in candidates:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    raise ValueError(
        f"Metadata is missing a '{label}' column. Tried aliases: {candidates}. "
        f"Available columns: {sorted(df.columns)}"
    )


def load_corpus(metadata_csv: Path, books_dir: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """Load metadata and book text into a tidy DataFrame.

    Args:
        metadata_csv: CSV containing a title column and a path-like column.
        books_dir: Directory where book text files live.
        limit: Optional maximum number of rows to load (useful for quick tests).

    Returns:
        DataFrame with columns [title, author, description, tags, text, source_path].
    """

    books_dir = books_dir.expanduser().resolve()
    df = pd.read_csv(metadata_csv)
    if limit is not None:
        df = df.iloc[:limit].copy()

    title_col = _find_column(df, ["title", "book_title", "name"], label="title")
    path_col = _find_column(
        df, ["path", "file", "filename", "file_name", "text_path", "relative_path"], label="path"
    )

    author_col = None
    for alias in ["author", "book_author"]:
        if alias in df.columns:
            author_col = alias
            break
    author_col = author_col or "author"
    if author_col not in df.columns:
        df[author_col] = ""

    description_col = None
    for alias in ["description", "summary", "blurb"]:
        if alias in df.columns:
            description_col = alias
            break
    description_col = description_col or "description"
    if description_col not in df.columns:
        df[description_col] = ""

    tags_col = None
    for alias in ["tags", "subjects", "genres", "keywords"]:
        if alias in df.columns:
            tags_col = alias
            break
    tags_col = tags_col or "tags"
    if tags_col not in df.columns:
        df[tags_col] = ""

    texts: list[str] = []
    paths: list[Path] = []
    for rel_path in df[path_col]:
        candidate = (books_dir / str(rel_path)).resolve()
        if not candidate.exists():
            raise FileNotFoundError(
                f"Book file not found: {candidate}. Ensure metadata paths are relative to {books_dir}"
            )
        paths.append(candidate)
        texts.append(_read_text(candidate))

    df = df.assign(
        title=df[title_col].fillna(""),
        author=df[author_col].fillna(""),
        description=df[description_col].fillna(""),
        tags=df[tags_col].fillna(""),
        text=texts,
        source_path=paths,
    )

    return df[["title", "author", "description", "tags", "text", "source_path"]]


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
