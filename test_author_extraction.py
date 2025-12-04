#!/usr/bin/env python3
"""
Test author extraction and random sampling functionality
"""

from datasets import load_dataset
from analyze_gutenberg_sample import extract_title_from_text, extract_author_from_text

def test_author_extraction():
    """Test author extraction from sample Project Gutenberg books"""
    print("="*80)
    print("Testing Author Extraction and Random Sampling")
    print("="*80)

    # Load a small sample with shuffling
    print("\nLoading shuffled sample from Project Gutenberg dataset...")
    dataset = load_dataset("manu/project_gutenberg", split="en", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=100)

    print("\nTesting author and title extraction on first 10 books:\n")

    seen_books = set()
    count = 0
    duplicates = 0

    for idx, item in enumerate(dataset):
        if count >= 10:
            break

        text = item.get('text', '')
        fallback_title = item.get('title', 'Unknown')

        if not text or len(text) < 1000:
            continue

        # Extract title and author
        title = extract_title_from_text(text)
        author = extract_author_from_text(text)

        # Use extracted or fallback
        display_title = title if title else fallback_title

        # Check for duplicates
        book_id = (display_title.lower(), author.lower() if author else "unknown")

        if book_id in seen_books:
            duplicates += 1
            print(f"{count + 1}. [DUPLICATE] {display_title}" + (f" by {author}" if author else " (no author found)"))
            continue

        seen_books.add(book_id)
        count += 1

        # Display result
        if author:
            print(f"{count}. {display_title} by {author}")
        else:
            print(f"{count}. {display_title} (no author found)")

    print("\n" + "="*80)
    print(f"Successfully extracted information from {count} books")
    print(f"Found {duplicates} duplicates (skipped)")
    print("="*80)

if __name__ == "__main__":
    test_author_extraction()
