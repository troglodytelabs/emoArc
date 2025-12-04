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


def clean_text(text: str, min_length: int = 1000) -> str:
    """Clean and validate text from Project Gutenberg"""

    # Remove Project Gutenberg header and footer
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT"
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook",
        "End of Project Gutenberg's"
    ]

    # Find and remove header
    for marker in start_markers:
        start_idx = text.find(marker)
        if start_idx != -1:
            # Skip to next line after marker
            text = text[start_idx:]
            text = '\n'.join(text.split('\n')[1:])
            break

    # Find and remove footer
    for marker in end_markers:
        end_idx = text.find(marker)
        if end_idx != -1:
            text = text[:end_idx]
            break

    # Return None if text is too short
    if len(text.strip()) < min_length:
        return None

    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 3000, overlap: int = 500) -> List[str]:
    """Split text into overlapping chunks for analysis"""
    chunks = []
    words = text.split()

    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 100:  # Only add non-empty chunks
            chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


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


def analyze_book(text: str, title: str, model: nn.Module,
                tokenizer: RobertaTokenizer, device: str = 'cpu') -> Dict:
    """Analyze emotional arc for an entire book"""

    # Clean the text
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return None

    # Split into chunks (representing chapters/sections)
    chunks = split_into_chunks(cleaned_text, chunk_size=3000, overlap=500)

    if len(chunks) < 3:  # Skip very short books
        return None

    print(f"  Analyzing {len(chunks)} chunks...")

    # Analyze each chunk
    chunk_emotions = []
    for i, chunk in enumerate(chunks):
        if i % 10 == 0 and i > 0:
            print(f"    Processed {i}/{len(chunks)} chunks")

        emotions = analyze_emotion(chunk, model, tokenizer, device)
        chunk_emotions.append(emotions)

    return {
        'title': title,
        'num_chunks': len(chunks),
        'chunk_emotions': chunk_emotions,
        'avg_emotions': calculate_average_emotions(chunk_emotions)
    }


def calculate_average_emotions(chunk_emotions: List[Dict[str, float]]) -> Dict[str, float]:
    """Calculate average emotion scores across all chunks"""
    avg_emotions = {emotion: 0.0 for emotion in EMOTION_LABELS}

    for emotions in chunk_emotions:
        for emotion, score in emotions.items():
            avg_emotions[emotion] += score

    for emotion in avg_emotions:
        avg_emotions[emotion] /= len(chunk_emotions)

    return avg_emotions


def fetch_and_analyze_sample(model: nn.Module, tokenizer: RobertaTokenizer,
                             num_books: int = 100, device: str = 'cpu') -> List[Dict]:
    """Fetch and analyze a sample of books from Project Gutenberg"""
    print(f"\nFetching {num_books} books from HuggingFace Project Gutenberg dataset...")

    # Load dataset in streaming mode
    dataset = load_dataset("manu/project_gutenberg", split="en", streaming=True)

    results = []
    processed = 0
    skipped = 0

    for item in dataset:
        if processed >= num_books:
            break

        title = item.get('title', 'Unknown').strip()
        text = item.get('text', '')

        if not text or len(text) < 1000:
            skipped += 1
            continue

        print(f"\n[{processed + 1}/{num_books}] Processing: {title}")

        try:
            result = analyze_book(text, title, model, tokenizer, device)

            if result:
                results.append(result)
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

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved successfully")


def plot_emotion_distributions(results: List[Dict], output_file: str = 'emotion_distributions.png'):
    """Plot emotion distributions across all analyzed books"""
    print(f"\nGenerating emotion distribution visualization...")

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


def plot_top_books(results: List[Dict], emotion: str = 'joy',
                  top_n: int = 10, output_file: str = 'top_books_by_emotion.png'):
    """Plot books with highest scores for a specific emotion"""
    print(f"\nGenerating top {top_n} books by {emotion}...")

    # Sort books by the specified emotion
    sorted_results = sorted(results,
                          key=lambda x: x['avg_emotions'][emotion],
                          reverse=True)[:top_n]

    titles = [r['title'][:50] + '...' if len(r['title']) > 50 else r['title']
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
        print(f"  {emotion:15s}: {top_book['title'][:60]} ({top_book['avg_emotions'][emotion]:.4f})")


def main():
    """Main execution function"""
    print("="*80)
    print("Project Gutenberg Sample Analysis")
    print("="*80)

    # Configuration
    MODEL_PATH = "/Users/devindyson/Desktop/troglodytelabs/emoWork/emoBERT/models/best_model.pt"
    NUM_BOOKS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Books to analyze: {NUM_BOOKS}")
    print(f"  Device: {DEVICE}")

    # Step 1: Load model
    model, tokenizer = load_model(MODEL_PATH, DEVICE)

    # Step 2: Fetch and analyze sample
    results = fetch_and_analyze_sample(model, tokenizer, NUM_BOOKS, DEVICE)

    if not results:
        print("Error: No books were successfully analyzed!")
        return

    # Step 3: Save results
    save_results(results, 'gutenberg_sample_analysis.json')

    # Step 4: Generate visualizations
    plot_emotion_distributions(results, 'emotion_distributions.png')
    plot_top_books(results, emotion='joy', top_n=10, output_file='top_joyful_books.png')
    plot_top_books(results, emotion='sadness', top_n=10, output_file='top_sad_books.png')

    # Step 5: Print summary
    print_summary(results)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
