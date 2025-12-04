#!/usr/bin/env python3
"""
Analyze emotional arcs for a sample of Project Gutenberg books
Tests RoBERTa emotion model on multiple titles efficiently
"""

import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from datasets import load_dataset
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import json
from pathlib import Path

# Emotion labels (adjust based on your model's training)
EMOTION_LABELS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']


class RoBERTaEmotionClassifier(nn.Module):
    """RoBERTa-based emotion classifier"""

    def __init__(self, num_labels=8, dropout=0.1):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def load_model(model_path: str, device: str = 'cpu') -> Tuple[nn.Module, RobertaTokenizer]:
    """Load the trained RoBERTa model and tokenizer"""
    print(f"Loading model from {model_path}...")

    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Initialize model
    model = RoBERTaEmotionClassifier(num_labels=len(EMOTION_LABELS))

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    return model, tokenizer


def extract_metadata_from_text(text: str) -> Tuple[str, str]:
    """
    Extract title and author from Project Gutenberg metadata within text.
    Returns tuple of (title, author)
    """
    # Look for "Title: " pattern in the text
    title_pattern = r'Title:\s*(.+?)(?:\n|\r)'
    title_match = re.search(title_pattern, text, re.IGNORECASE)

    title = None
    if title_match:
        title = title_match.group(1).strip()
        # Remove any trailing punctuation or extra info
        title = re.sub(r'\s*\[.*?\]\s*$', '', title)  # Remove [eBook #...]

    # Look for "Author: " pattern
    author_pattern = r'Author:\s*(.+?)(?:\n|\r)'
    author_match = re.search(author_pattern, text, re.IGNORECASE)

    author = None
    if author_match:
        author = author_match.group(1).strip()
        # Clean up author name
        author = re.sub(r'\s*\[.*?\]\s*$', '', author)  # Remove any bracketed info

    return title, author


def clean_text(text: str, min_length: int = 1000) -> Tuple[str, str, str]:
    """
    Clean and validate text from Project Gutenberg.
    Returns tuple of (cleaned_text, extracted_title, extracted_author)
    """

    # Extract metadata before cleaning
    extracted_title, extracted_author = extract_metadata_from_text(text)

    # Remove Project Gutenberg header - find the actual start of content
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT"
    ]

    # Find header marker
    start_idx = -1
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start_idx = idx
            # Find the end of the header section (usually after "***" line)
            # Skip past the marker line and any metadata
            remaining = text[idx:]
            lines = remaining.split('\n')

            # Skip past metadata section (Title, Author, Release Date, etc.)
            content_start = 0
            for i, line in enumerate(lines):
                # Content typically starts after empty lines following metadata
                # or after a series of asterisks
                if i > 5 and line.strip() == '' and i + 1 < len(lines):
                    # Check if next few lines look like actual content (not metadata)
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith(('Title:', 'Author:', 'Release Date:',
                                                                'Language:', 'Character set:',
                                                                'Produced by', '***', 'www.gutenberg')):
                        content_start = i + 1
                        break

            if content_start > 0:
                text = '\n'.join(lines[content_start:])
            else:
                # Fallback: skip first 20 lines after marker
                text = '\n'.join(lines[20:])
            break

    # Remove Project Gutenberg footer
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook",
        "End of Project Gutenberg's",
        "***END OF THE PROJECT GUTENBERG"
    ]

    for marker in end_markers:
        end_idx = text.find(marker)
        if end_idx != -1:
            text = text[:end_idx]
            break

    # Remove common footer patterns that might remain
    text = re.sub(r'\n\s*End of (?:the )?Project Gutenberg.*$', '', text, flags=re.IGNORECASE | re.DOTALL)

    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    # Return None if text is too short
    if len(text) < min_length:
        return None, None, None

    return text, extracted_title, extracted_author


def split_into_normalized_segments(text: str, num_segments: int = 10) -> List[str]:
    """
    Split text into N equal segments based on character percentage.
    This enables comparing emotional arcs across books of different lengths.

    Args:
        text: The full text to split
        num_segments: Number of segments to divide the text into (default: 10 for 10% increments)

    Returns:
        List of text segments, each representing an equal portion of the book
    """
    text = text.strip()
    text_length = len(text)

    if text_length < num_segments * 100:  # Need at least 100 chars per segment
        return None

    segments = []
    segment_size = text_length // num_segments

    for i in range(num_segments):
        start_idx = i * segment_size
        # For the last segment, go to the end to capture any remainder
        end_idx = (i + 1) * segment_size if i < num_segments - 1 else text_length

        segment = text[start_idx:end_idx]
        if len(segment.strip()) > 50:  # Only add non-empty segments
            segments.append(segment)

    return segments


def analyze_emotion(text: str, model: nn.Module, tokenizer: RobertaTokenizer,
                   device: str = 'cpu', max_length: int = 512) -> Dict[str, float]:
    """Analyze emotions in a text segment"""

    # Tokenize text
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get predictions
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)

    # Convert to emotion scores
    emotion_scores = {}
    for idx, emotion in enumerate(EMOTION_LABELS):
        emotion_scores[emotion] = probs[0, idx].item()

    return emotion_scores


def analyze_book(text: str, fallback_title: str, model: nn.Module,
                tokenizer: RobertaTokenizer, device: str = 'cpu',
                num_segments: int = 10) -> Dict:
    """
    Analyze emotional arc for an entire book using normalized segmentation.

    Args:
        text: Full text of the book
        fallback_title: Fallback title if extraction fails
        model: Trained emotion classification model
        tokenizer: RoBERTa tokenizer
        device: Device to run inference on
        num_segments: Number of equal segments to divide the book into (default: 10)

    Returns:
        Dictionary with title, author, segments, and emotion analysis results
    """

    # Clean the text and extract metadata
    result = clean_text(text)
    if not result or result[0] is None:
        return None

    cleaned_text, extracted_title, extracted_author = result

    # Use extracted metadata if available, otherwise use fallback
    title = extracted_title if extracted_title else fallback_title
    author = extracted_author if extracted_author else "Unknown"

    # Create display name
    display_name = f"{title} by {author}" if author != "Unknown" else title

    # Split into normalized segments
    segments = split_into_normalized_segments(cleaned_text, num_segments=num_segments)

    if not segments or len(segments) < num_segments - 1:  # Allow 1 missing segment
        return None

    print(f"  Title: {display_name}")
    print(f"  Analyzing {len(segments)} segments (each ~{len(cleaned_text)//num_segments} chars)...")

    # Analyze each segment
    segment_emotions = []
    for i, segment in enumerate(segments):
        emotions = analyze_emotion(segment, model, tokenizer, device)
        segment_emotions.append(emotions)

    return {
        'title': title,
        'author': author,
        'display_name': display_name,
        'num_segments': len(segments),
        'segment_emotions': segment_emotions,
        'avg_emotions': calculate_average_emotions(segment_emotions),
        'text_length': len(cleaned_text)
    }


def calculate_average_emotions(segment_emotions: List[Dict[str, float]]) -> Dict[str, float]:
    """Calculate average emotion scores across all segments"""
    avg_emotions = {emotion: 0.0 for emotion in EMOTION_LABELS}

    for emotions in segment_emotions:
        for emotion, score in emotions.items():
            avg_emotions[emotion] += score

    for emotion in avg_emotions:
        avg_emotions[emotion] /= len(segment_emotions)

    return avg_emotions


def fetch_and_analyze_sample(model: nn.Module, tokenizer: RobertaTokenizer,
                             num_books: int = 100, device: str = 'cpu') -> List[Dict]:
    """Fetch and analyze a random sample of books from Project Gutenberg"""
    print(f"\nFetching random sample of {num_books} books from HuggingFace Project Gutenberg dataset...")

    # Load dataset with shuffling for random sampling
    # Use a fixed seed for reproducibility, or remove seed for truly random each time
    dataset = load_dataset("manu/project_gutenberg", split="en", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10000)

    results = []
    processed = 0
    skipped = 0
    seen_titles = set()  # Track titles to avoid duplicates

    for item in dataset:
        if processed >= num_books:
            break

        fallback_title = item.get('title', 'Unknown').strip()
        text = item.get('text', '')

        if not text or len(text) < 1000:
            skipped += 1
            continue

        print(f"\n[{processed + 1}/{num_books}] Processing...")

        try:
            result = analyze_book(text, fallback_title, model, tokenizer, device)

            if result:
                # Check for duplicate titles (using display_name to include author)
                if result['display_name'] in seen_titles:
                    print(f"  ✗ Skipped (duplicate: {result['display_name']})")
                    skipped += 1
                    continue

                results.append(result)
                seen_titles.add(result['display_name'])
                processed += 1
                print(f"  ✓ Completed")
            else:
                skipped += 1
                print(f"  ✗ Skipped (too short)")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            skipped += 1
            continue

    print(f"\n{'='*80}")
    print(f"Processed: {processed} books")
    print(f"Skipped: {skipped} books")
    print(f"{'='*80}\n")

    return results


def save_results(results: List[Dict], output_file: str = 'gutenberg_analysis_results.json'):
    """Save analysis results to JSON"""
    print(f"Saving results to {output_file}...")

    # Ensure directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved successfully")


def plot_emotion_distributions(results: List[Dict], output_file: str = 'emotion_distributions.png'):
    """Plot emotion distributions across all analyzed books"""
    print(f"\nGenerating emotion distribution visualization...")

    # Ensure directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect average emotions for each book
    all_emotions = {emotion: [] for emotion in EMOTION_LABELS}

    for result in results:
        avg_emotions = result['avg_emotions']
        for emotion in EMOTION_LABELS:
            all_emotions[emotion].append(avg_emotions[emotion])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Box plot of emotion distributions
    data_for_boxplot = [all_emotions[emotion] for emotion in EMOTION_LABELS]
    bp = ax1.boxplot(data_for_boxplot, labels=EMOTION_LABELS, patch_artist=True)

    # Color the boxes
    colors = plt.cm.Set3(range(len(EMOTION_LABELS)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_xlabel('Emotion', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title(f'Emotion Distribution Across {len(results)} Books', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Average emotion scores
    avg_scores = [np.mean(all_emotions[emotion]) for emotion in EMOTION_LABELS]
    bars = ax2.bar(EMOTION_LABELS, avg_scores, color=colors)

    ax2.set_xlabel('Emotion', fontsize=12)
    ax2.set_ylabel('Average Score', fontsize=12)
    ax2.set_title('Average Emotion Scores', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")


def plot_comparative_arcs(results: List[Dict], emotion: str = 'joy',
                         num_books: int = 20, output_file: str = 'comparative_emotional_arcs.png'):
    """
    Plot normalized emotional arcs for multiple books on the same chart.
    This shows how emotions evolve across the story arc (beginning to end).
    """
    print(f"\nGenerating comparative emotional arcs for {emotion}...")

    # Ensure directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by average emotion and take top N
    sorted_results = sorted(results,
                          key=lambda x: x['avg_emotions'][emotion],
                          reverse=True)[:num_books]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create color gradient
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_results)))

    for idx, result in enumerate(sorted_results):
        segment_emotions = result['segment_emotions']
        num_segments = len(segment_emotions)

        # Extract emotion scores across segments
        emotion_progression = [seg[emotion] for seg in segment_emotions]

        # X-axis: percentage through the book
        x_values = np.linspace(0, 100, num_segments)

        # Plot with transparency
        label = result['display_name'][:50] + '...' if len(result['display_name']) > 50 else result['display_name']
        ax.plot(x_values, emotion_progression,
               label=label, linewidth=1.5, alpha=0.6, color=colors[idx])

    ax.set_xlabel('Progress Through Book (%)', fontsize=12)
    ax.set_ylabel(f'{emotion.capitalize()} Score', fontsize=12)
    ax.set_title(f'Comparative {emotion.capitalize()} Arcs Across {len(sorted_results)} Books',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    # Place legend outside plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")


def plot_average_emotional_arc(results: List[Dict], output_file: str = 'average_emotional_arc.png'):
    """
    Plot the average emotional arc across all books.
    Shows the typical emotion progression from beginning to end.
    """
    print(f"\nGenerating average emotional arc across all books...")

    # Ensure directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine number of segments (should be consistent, but check)
    num_segments = results[0]['num_segments']

    # Initialize emotion accumulator
    emotion_sums = {emotion: [0.0] * num_segments for emotion in EMOTION_LABELS}
    counts = [0] * num_segments

    # Accumulate emotions across all books
    for result in results:
        segment_emotions = result['segment_emotions']
        for i, seg_emotions in enumerate(segment_emotions):
            if i < num_segments:
                for emotion in EMOTION_LABELS:
                    emotion_sums[emotion][i] += seg_emotions[emotion]
                counts[i] += 1

    # Calculate averages
    emotion_avgs = {}
    for emotion in EMOTION_LABELS:
        emotion_avgs[emotion] = [
            emotion_sums[emotion][i] / counts[i] if counts[i] > 0 else 0
            for i in range(num_segments)
        ]

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    x_values = np.linspace(0, 100, num_segments)
    colors = plt.cm.tab10(range(len(EMOTION_LABELS)))

    for idx, emotion in enumerate(EMOTION_LABELS):
        ax.plot(x_values, emotion_avgs[emotion],
               label=emotion.capitalize(), linewidth=2.5,
               marker='o', markersize=6, color=colors[idx])

    ax.set_xlabel('Progress Through Story (%)', fontsize=12)
    ax.set_ylabel('Average Emotion Score', fontsize=12)
    ax.set_title(f'Average Emotional Arc Pattern Across {len(results)} Books',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")


def plot_top_books(results: List[Dict], emotion: str = 'joy',
                  top_n: int = 10, output_file: str = 'top_books_by_emotion.png'):
    """Plot books with highest scores for a specific emotion"""
    print(f"\nGenerating top {top_n} books by {emotion}...")

    # Ensure directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort books by the specified emotion
    sorted_results = sorted(results,
                          key=lambda x: x['avg_emotions'][emotion],
                          reverse=True)[:top_n]

    titles = [r['display_name'][:60] + '...' if len(r['display_name']) > 60 else r['display_name']
             for r in sorted_results]
    scores = [r['avg_emotions'][emotion] for r in sorted_results]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(titles))
    bars = ax.barh(y_pos, scores, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(titles))))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(titles)
    ax.invert_yaxis()
    ax.set_xlabel(f'{emotion.capitalize()} Score', fontsize=12)
    ax.set_title(f'Top {top_n} Books by {emotion.capitalize()}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")


def plot_individual_book_arcs(results: List[Dict], num_books: int = 12,
                             output_file: str = 'individual_emotional_arcs.png'):
    """
    Create comprehensive infographic showing emotional arcs for multiple books.
    Each subplot shows all 8 emotions over time for one book (similar to Romeo & Juliet style).
    """
    print(f"\nGenerating comprehensive infographic for {num_books} books...")

    # Ensure directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select diverse books by different emotions
    selected_books = []

    # Get top books for different emotions to show variety
    emotions_to_sample = ['joy', 'sadness', 'fear', 'anger']
    books_per_emotion = num_books // len(emotions_to_sample)

    added_titles = set()
    for emotion in emotions_to_sample:
        sorted_by_emotion = sorted(results,
                                  key=lambda x: x['avg_emotions'][emotion],
                                  reverse=True)
        for book in sorted_by_emotion:
            if book['display_name'] not in added_titles and len(selected_books) < num_books:
                selected_books.append(book)
                added_titles.add(book['display_name'])
                if len(selected_books) % books_per_emotion == 0:
                    break

    # Fill remaining slots if needed
    while len(selected_books) < num_books:
        for book in results:
            if book['display_name'] not in added_titles:
                selected_books.append(book)
                added_titles.add(book['display_name'])
                break

    # Create grid layout
    cols = 3
    rows = (num_books + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    fig.suptitle('Emotional Arcs Across Classic Literature', fontsize=18, fontweight='bold', y=0.995)

    # Flatten axes for easier iteration
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    # Color scheme for emotions
    emotion_colors = plt.cm.tab10(range(len(EMOTION_LABELS)))

    for idx, (book, ax) in enumerate(zip(selected_books, axes_flat)):
        segment_emotions = book['segment_emotions']
        num_segments = len(segment_emotions)
        x_values = np.linspace(0, 100, num_segments)

        # Plot each emotion
        for emotion_idx, emotion in enumerate(EMOTION_LABELS):
            emotion_progression = [seg[emotion] for seg in segment_emotions]
            ax.plot(x_values, emotion_progression,
                   label=emotion.capitalize(),
                   linewidth=2,
                   marker='o',
                   markersize=4,
                   color=emotion_colors[emotion_idx],
                   alpha=0.8)

        # Format subplot
        display_name = book['display_name']
        if len(display_name) > 50:
            display_name = display_name[:50] + '...'
        ax.set_title(display_name, fontsize=10, fontweight='bold', pad=10)
        ax.set_xlabel('Progress (%)', fontsize=9)
        ax.set_ylabel('Emotion Score', fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, max(0.5, max([max([seg[e] for e in EMOTION_LABELS]) for seg in segment_emotions]) * 1.1))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7, ncol=2)

    # Hide unused subplots
    for idx in range(len(selected_books), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Infographic saved to {output_file}")


def print_summary(results: List[Dict]):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)

    print(f"\nTotal books analyzed: {len(results)}")

    # Calculate overall averages
    overall_emotions = {emotion: 0.0 for emotion in EMOTION_LABELS}
    for result in results:
        for emotion in EMOTION_LABELS:
            overall_emotions[emotion] += result['avg_emotions'][emotion]

    for emotion in overall_emotions:
        overall_emotions[emotion] /= len(results)

    print("\nOverall Average Emotions:")
    sorted_emotions = sorted(overall_emotions.items(), key=lambda x: x[1], reverse=True)
    for emotion, score in sorted_emotions:
        print(f"  {emotion:15s}: {score:.4f}")

    # Find most emotional book for each emotion
    print("\nMost Emotional Books:")
    for emotion in EMOTION_LABELS:
        top_book = max(results, key=lambda x: x['avg_emotions'][emotion])
        display = top_book['display_name'][:70] + '...' if len(top_book['display_name']) > 70 else top_book['display_name']
        print(f"  {emotion:15s}: {display} ({top_book['avg_emotions'][emotion]:.4f})")


def main():
    """Main execution function"""
    print("="*80)
    print("Project Gutenberg Sample Analysis")
    print("="*80)

    # Configuration
    MODEL_PATH = "/Users/devindyson/Desktop/troglodytelabs/emoWork/emoBERT/models/best_model.pt"
    NUM_BOOKS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Output directories
    OUTPUT_DIR = Path("outputs")
    DATA_DIR = OUTPUT_DIR / "data"
    VIZ_DIR = OUTPUT_DIR / "visualizations"

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Books to analyze: {NUM_BOOKS}")
    print(f"  Device: {DEVICE}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Step 1: Load model
    model, tokenizer = load_model(MODEL_PATH, DEVICE)

    # Step 2: Fetch and analyze sample
    results = fetch_and_analyze_sample(model, tokenizer, NUM_BOOKS, DEVICE)

    if not results:
        print("Error: No books were successfully analyzed!")
        return

    # Step 3: Save results
    save_results(results, str(DATA_DIR / 'gutenberg_sample_analysis.json'))

    # Step 4: Generate visualizations
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)

    # Main infographic showing individual book arcs (NEW!)
    plot_individual_book_arcs(results, num_books=12,
                             output_file=str(VIZ_DIR / 'individual_emotional_arcs.png'))

    # Statistical distributions
    plot_emotion_distributions(results, str(VIZ_DIR / 'emotion_distributions.png'))

    # Average patterns
    plot_average_emotional_arc(results, str(VIZ_DIR / 'average_emotional_arc.png'))

    # Comparative arcs
    plot_comparative_arcs(results, emotion='joy', num_books=20,
                         output_file=str(VIZ_DIR / 'comparative_joy_arcs.png'))
    plot_comparative_arcs(results, emotion='sadness', num_books=20,
                         output_file=str(VIZ_DIR / 'comparative_sadness_arcs.png'))

    # Top books rankings
    plot_top_books(results, emotion='joy', top_n=10,
                  output_file=str(VIZ_DIR / 'top_joyful_books.png'))
    plot_top_books(results, emotion='sadness', top_n=10,
                  output_file=str(VIZ_DIR / 'top_sad_books.png'))

    # Step 5: Print summary
    print_summary(results)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print("\nGenerated files:")
    print("\nData:")
    print(f"  - {DATA_DIR / 'gutenberg_sample_analysis.json'}")
    print("\nVisualizations:")
    print(f"  - {VIZ_DIR / 'individual_emotional_arcs.png'} ⭐ MAIN INFOGRAPHIC")
    print(f"  - {VIZ_DIR / 'emotion_distributions.png'}")
    print(f"  - {VIZ_DIR / 'average_emotional_arc.png'}")
    print(f"  - {VIZ_DIR / 'comparative_joy_arcs.png'}")
    print(f"  - {VIZ_DIR / 'comparative_sadness_arcs.png'}")
    print(f"  - {VIZ_DIR / 'top_joyful_books.png'}")
    print(f"  - {VIZ_DIR / 'top_sad_books.png'}")


if __name__ == "__main__":
    main()
