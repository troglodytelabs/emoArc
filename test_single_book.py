#!/usr/bin/env python3
"""
test_single_book.py - Test RoBERTa emotion inference on a single Gutenberg book

This script validates the full pipeline:
1. Download book from Project Gutenberg
2. Strip boilerplate and detect chapters
3. Run emotion predictions on each chapter/segment
4. Generate emotional trajectory output
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import requests
import torch
from transformers import RobertaTokenizer

# Import our emotion predictor
sys.path.append(str(Path(__file__).parent / "src" / "analysis"))
from emoPredict_roberta import PlutchikEmotionClassifier, load_model, predict_emotions


def download_gutenberg_book(book_id, max_retries=3):
    """
    Download a book from Project Gutenberg with fallback URLs.

    Args:
        book_id: Gutenberg book ID
        max_retries: Maximum retry attempts

    Returns:
        Book text as string, or None if failed
    """
    urls = [
        f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt',
        f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt',
        f'https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8'
    ]

    for url in urls:
        for attempt in range(max_retries):
            try:
                print(f"  Downloading from {url} (attempt {attempt + 1}/{max_retries})")
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    # Try UTF-8 first, fall back to latin-1
                    try:
                        text = response.content.decode('utf-8')
                    except UnicodeDecodeError:
                        text = response.content.decode('latin-1')

                    print(f"  ✓ Downloaded {len(text):,} characters")
                    return text

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

    return None


def strip_gutenberg_boilerplate(text):
    """
    Remove Project Gutenberg header and footer boilerplate.
    """
    # Find start marker
    start_patterns = [
        r'\*\*\* START OF TH(IS|E) PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
    ]

    start_pos = 0
    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_pos = match.end()
            break

    # Find end marker
    end_patterns = [
        r'\*\*\* END OF TH(IS|E) PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
    ]

    end_pos = len(text)
    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            end_pos = match.start()
            break

    return text[start_pos:end_pos].strip()


def detect_chapters(text, verbose=False):
    """
    Detect chapters using multiple regex patterns.

    Returns:
        List of (chapter_title, chapter_text, start_pos) tuples
    """
    # Multiple chapter patterns to try
    patterns = [
        r'^CHAPTER [IVXLCDM]+\.?\s*$',  # CHAPTER I, CHAPTER XII, etc.
        r'^CHAPTER \d+\.?\s*$',  # CHAPTER 1, CHAPTER 23, etc.
        r'^Chapter [IVXLCDM]+\.?\s*$',  # Chapter I, Chapter XII, etc.
        r'^Chapter \d+\.?\s*$',  # Chapter 1, Chapter 23, etc.
        r'^[IVXLCDM]+\.?\s*$',  # Just roman numerals
        r'^\d+\.?\s*$',  # Just numbers
    ]

    lines = text.split('\n')
    chapters = []

    # Try each pattern
    for pattern in patterns:
        matches = []

        for i, line in enumerate(lines):
            line = line.strip()
            if re.match(pattern, line, re.MULTILINE):
                # Calculate character position
                char_pos = sum(len(l) + 1 for l in lines[:i])
                matches.append((line, char_pos, i))

        # Use this pattern if we found a reasonable number of chapters
        if 3 <= len(matches) <= 200:
            if verbose:
                print(f"  Using pattern: {pattern}")
                print(f"  Found {len(matches)} chapters")

            # Extract chapter texts
            for i, (title, pos, line_num) in enumerate(matches):
                # Get text until next chapter (or end)
                if i < len(matches) - 1:
                    next_pos = matches[i + 1][1]
                    chapter_text = text[pos:next_pos]
                else:
                    chapter_text = text[pos:]

                chapters.append((title, chapter_text, pos))

            return chapters

    if verbose:
        print("  No reliable chapter pattern found")

    return []


def fixed_chunk_segments(text, chunk_size=500):
    """
    Split text into fixed-size word chunks as fallback.

    Args:
        text: Input text
        chunk_size: Words per chunk

    Returns:
        List of (chunk_name, chunk_text) tuples
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunk_name = f"Segment {i // chunk_size + 1}"
        chunks.append((chunk_name, chunk))

    return chunks


def predict_book_emotions(book_id, title, author, model, tokenizer, thresholds, device,
                         segment_unit="chapter", chunk_size=500, max_segments=None):
    """
    Download and process a book, generating emotional trajectory.

    Args:
        book_id: Gutenberg book ID
        title: Book title
        author: Book author
        model: Loaded emotion model
        tokenizer: RoBERTa tokenizer
        thresholds: Optimal thresholds for predictions
        device: torch device
        segment_unit: "chapter" or "fixed_chunk"
        chunk_size: Words per chunk if using fixed_chunk
        max_segments: Limit number of segments to process (for testing)

    Returns:
        Dictionary with emotional trajectory data
    """
    print(f"\n{'='*60}")
    print(f"Processing: {title} by {author}")
    print(f"Book ID: {book_id}")
    print(f"{'='*60}\n")

    # Download book
    print("Step 1: Downloading book...")
    raw_text = download_gutenberg_book(book_id)
    if not raw_text:
        print("✗ Failed to download book")
        return None

    # Strip boilerplate
    print("\nStep 2: Stripping boilerplate...")
    clean_text = strip_gutenberg_boilerplate(raw_text)
    removed_pct = (len(raw_text) - len(clean_text)) / len(raw_text) * 100
    print(f"  Removed {removed_pct:.1f}% boilerplate")

    # Detect chapters or use fixed chunks
    print("\nStep 3: Segmenting text...")
    if segment_unit == "chapter":
        chapters = detect_chapters(clean_text, verbose=True)
        if chapters:
            segments = [(ch[0], ch[1]) for ch in chapters]
            segment_method = "chapters"
        else:
            print("  No chapters detected, falling back to fixed chunks")
            segments = fixed_chunk_segments(clean_text, chunk_size)
            segment_method = "fixed_chunk"
    else:
        segments = fixed_chunk_segments(clean_text, chunk_size)
        segment_method = "fixed_chunk"

    total_segments = len(segments)
    if max_segments:
        segments = segments[:max_segments]
        print(f"  Limiting to first {max_segments} segments for testing")

    print(f"  Total segments: {total_segments}")
    print(f"  Processing: {len(segments)} segments")

    # Process each segment
    print("\nStep 4: Running emotion predictions...")
    emotional_trajectory = []

    for i, (segment_name, segment_text) in enumerate(segments, 1):
        # Get word count
        word_count = len(segment_text.split())

        # Predict emotions
        results = predict_emotions(segment_text, model, tokenizer, thresholds, device)

        # Format emotion scores
        emotions = {r["emotion"]: r["probability"] for r in results}

        # Find dominant emotion
        dominant = max(results, key=lambda x: x["probability"])

        # Calculate intensity (max probability)
        intensity = dominant["probability"]

        # Add to trajectory
        emotional_trajectory.append({
            "segment": i,
            "segment_name": segment_name,
            "word_count": word_count,
            "emotions": emotions,
            "dominant_emotion": dominant["emotion"],
            "intensity": intensity
        })

        # Progress indicator
        if i % 10 == 0 or i == len(segments):
            print(f"  Processed {i}/{len(segments)} segments...")

    print("  ✓ Emotion predictions complete")

    # Calculate narrative features
    print("\nStep 5: Calculating narrative features...")
    intensities = [seg["intensity"] for seg in emotional_trajectory]
    climax_segment = max(range(len(intensities)), key=lambda i: intensities[i]) + 1
    climax_intensity = max(intensities)

    # Calculate emotional range (variance across emotion dimensions)
    all_emotions = {}
    for seg in emotional_trajectory:
        for emotion, score in seg["emotions"].items():
            if emotion not in all_emotions:
                all_emotions[emotion] = []
            all_emotions[emotion].append(score)

    emotional_range = np.mean([np.var(scores) for scores in all_emotions.values()])

    # Calculate pacing (rate of change in dominant emotion)
    emotion_changes = sum(
        1 for i in range(1, len(emotional_trajectory))
        if emotional_trajectory[i]["dominant_emotion"] != emotional_trajectory[i-1]["dominant_emotion"]
    )
    pacing_score = emotion_changes / len(emotional_trajectory) if emotional_trajectory else 0

    narrative_features = {
        "climax_chapter": climax_segment,
        "climax_intensity": climax_intensity,
        "emotional_range": emotional_range,
        "pacing_score": pacing_score
    }

    print(f"  Climax: Segment {climax_segment} (intensity: {climax_intensity:.2f})")
    print(f"  Emotional range: {emotional_range:.4f}")
    print(f"  Pacing score: {pacing_score:.2f}")

    # Build output
    output = {
        "book_metadata": {
            "book_id": f"gutenberg:{book_id}",
            "title": title,
            "author": author,
            "total_segments": total_segments,
            "segment_method": segment_method,
            "processed_segments": len(segments)
        },
        "emotional_trajectory": emotional_trajectory,
        "narrative_features": narrative_features
    }

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Test RoBERTa emotion inference on a single Gutenberg book"
    )

    parser.add_argument(
        "--book-id",
        type=int,
        default=1342,
        help="Project Gutenberg book ID (default: 1342 = Pride and Prejudice)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Pride and Prejudice",
        help="Book title"
    )
    parser.add_argument(
        "--author",
        type=str,
        default="Jane Austen",
        help="Book author"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pt",
        help="Path to trained model"
    )
    parser.add_argument(
        "--segment-unit",
        choices=["chapter", "fixed_chunk"],
        default="chapter",
        help="Segmentation method"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Words per chunk (if using fixed_chunk)"
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        help="Limit number of segments to process (for testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path"
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.model}...")
    if not Path(args.model).exists():
        print(f"✗ Model not found at {args.model}")
        sys.exit(1)

    model, thresholds = load_model(args.model, device)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Process book
    result = predict_book_emotions(
        args.book_id,
        args.title,
        args.author,
        model,
        tokenizer,
        thresholds,
        device,
        segment_unit=args.segment_unit,
        chunk_size=args.chunk_size,
        max_segments=args.max_segments
    )

    if result:
        # Save output
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"\n✓ Results saved to {output_path}")
        else:
            # Print sample output
            print("\n" + "="*60)
            print("SAMPLE OUTPUT (first 2 segments):")
            print("="*60)
            print(json.dumps({
                **result,
                "emotional_trajectory": result["emotional_trajectory"][:2]
            }, indent=2))
            print("\n... [truncated]")

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Book: {result['book_metadata']['title']}")
        print(f"Segments: {result['book_metadata']['processed_segments']}")
        print(f"Method: {result['book_metadata']['segment_method']}")
        print(f"Climax: Segment {result['narrative_features']['climax_chapter']}")
        print(f"Dominant emotions by segment:")

        emotion_counts = {}
        for seg in result["emotional_trajectory"]:
            emotion = seg["dominant_emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
            pct = count / len(result["emotional_trajectory"]) * 100
            print(f"  {emotion:15s}: {count:3d} segments ({pct:5.1f}%)")

        print("="*60)
        print("\n✓ Test complete!")

        return 0
    else:
        print("\n✗ Failed to process book")
        return 1


if __name__ == "__main__":
    sys.exit(main())
