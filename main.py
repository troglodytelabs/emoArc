from __future__ import annotations

import argparse
from pathlib import Path

from src.data import load_corpus
from src.recommender import BookRecommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Content-based book recommender")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV with at least title + path columns")
    parser.add_argument("--books-dir", required=True, help="Directory containing book text files")
    parser.add_argument("--top-n", type=int, default=5, help="Number of recommendations to return")
    parser.add_argument("--seed-title", help="Find similar books for this title")
    parser.add_argument("--seed-text", help="Find similar books for this ad-hoc text snippet")
    parser.add_argument("--limit", type=int, help="Limit the number of rows loaded (for quick smoke tests)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    corpus = load_corpus(Path(args.metadata), Path(args.books_dir), limit=args.limit)
    recommender = BookRecommender()
    recommender.fit(corpus)

    if args.seed_title:
        recs = recommender.recommend_by_title(args.seed_title, top_n=args.top_n)
        print(f"Recommendations for '{args.seed_title}':")
        for rec in recs:
            print(f"- {rec.title} (by {rec.author or 'unknown'}) [score={rec.score:.4f}]")
    elif args.seed_text:
        recs = recommender.recommend_for_text(args.seed_text, top_n=args.top_n)
        print("Recommendations for your text snippet:")
        for rec in recs:
            print(f"- {rec.title} (by {rec.author or 'unknown'}) [score={rec.score:.4f}]")
    else:
        print("Model built. Provide --seed-title or --seed-text to generate recommendations.")


if __name__ == "__main__":
    main()
