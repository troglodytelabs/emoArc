#!/usr/bin/env python3
"""
Test RoBERTa model for emotional analysis on Romeo and Juliet
Creates an emotional story arc by analyzing text chapter by chapter
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


def fetch_romeo_and_juliet() -> str:
    """Fetch Romeo and Juliet text from HuggingFace Project Gutenberg dataset"""
    print("Fetching Romeo and Juliet from HuggingFace Project Gutenberg dataset...")

    # Load the Project Gutenberg dataset (English language split)
    dataset = load_dataset("manu/project_gutenberg", split="en", streaming=True)

    # Romeo and Juliet metadata
    # Search for Romeo and Juliet by title or ID (1513)
    target_titles = [
        "Romeo and Juliet",
        "The Tragedy of Romeo and Juliet",
        "romeo and juliet"
    ]

    text = None
    count = 0

    # Stream through dataset to find Romeo and Juliet
    for item in dataset:
        count += 1
        if count % 100 == 0:
            print(f"  Searched through {count} books...")

        # Check title
        title = item.get('title', '').strip()

        # Check if this is Romeo and Juliet
        if any(target.lower() in title.lower() for target in target_titles):
            print(f"  Found: {title}")
            text = item.get('text', '')
            break

        # Also check by Gutenberg ID if available
        if 'id' in item and str(item['id']) == '1513':
            print(f"  Found by ID: {title}")
            text = item.get('text', '')
            break

        # Limit search to first 2000 books to avoid excessive searching
        if count >= 2000:
            print(f"  Searched {count} books, switching to direct URL method...")
            break

    # Fallback: if not found in HuggingFace dataset, use direct URL
    if not text:
        print("  Romeo and Juliet not found in streamed dataset.")
        print("  Using alternative method: direct Project Gutenberg URL...")
        import requests
        url = "https://www.gutenberg.org/cache/epub/1513/pg1513.txt"
        response = requests.get(url)
        response.raise_for_status()
        text = response.text

    print(f"Fetched {len(text)} characters")

    # Remove Project Gutenberg header and footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
        # Remove the marker line itself
        text = '\n'.join(text.split('\n')[1:])

    return text


def split_into_chapters(text: str) -> Dict[str, str]:
    """Split the play into acts and scenes (chapters)"""
    print("Splitting text into acts and scenes...")

    # Clean up the text
    lines = text.split('\n')

    chapters = {}
    current_chapter = None
    current_text = []

    # Pattern to match Act and Scene markers
    act_pattern = re.compile(r'^\s*ACT\s+([IVX]+)', re.IGNORECASE)
    scene_pattern = re.compile(r'^\s*SCENE\s+([IVX]+)', re.IGNORECASE)

    current_act = None

    for line in lines:
        act_match = act_pattern.search(line)
        scene_match = scene_pattern.search(line)

        if act_match:
            # Save previous chapter
            if current_chapter and current_text:
                chapters[current_chapter] = '\n'.join(current_text)

            current_act = act_match.group(1)
            current_chapter = f"Act {current_act}"
            current_text = []

        elif scene_match and current_act:
            # Save previous chapter
            if current_chapter and current_text:
                chapters[current_chapter] = '\n'.join(current_text)

            scene = scene_match.group(1)
            current_chapter = f"Act {current_act}, Scene {scene}"
            current_text = []

        elif current_chapter:
            current_text.append(line)

    # Save last chapter
    if current_chapter and current_text:
        chapters[current_chapter] = '\n'.join(current_text)

    print(f"Found {len(chapters)} chapters")
    return chapters


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


def analyze_chapters(chapters: Dict[str, str], model: nn.Module,
                    tokenizer: RobertaTokenizer, device: str = 'cpu') -> Dict[str, Dict[str, float]]:
    """Analyze emotions for each chapter"""
    print("\nAnalyzing emotions for each chapter...")

    chapter_emotions = {}

    for chapter_name, chapter_text in chapters.items():
        # Skip very short chapters
        if len(chapter_text.strip()) < 100:
            continue

        print(f"  Analyzing: {chapter_name}")
        emotions = analyze_emotion(chapter_text, model, tokenizer, device)
        chapter_emotions[chapter_name] = emotions

    return chapter_emotions


def plot_emotional_arc(chapter_emotions: Dict[str, Dict[str, float]], output_file: str = 'emotional_arc.png'):
    """Create visualization of emotional arc across the story"""
    print(f"\nGenerating emotional arc visualization...")

    # Prepare data
    chapters = list(chapter_emotions.keys())
    emotions = EMOTION_LABELS

    # Create matrix of emotion scores
    emotion_matrix = np.zeros((len(emotions), len(chapters)))

    for i, chapter in enumerate(chapters):
        for j, emotion in enumerate(emotions):
            emotion_matrix[j, i] = chapter_emotions[chapter][emotion]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # Plot 1: Line plot of emotions over chapters
    for i, emotion in enumerate(emotions):
        ax1.plot(range(len(chapters)), emotion_matrix[i, :],
                marker='o', label=emotion, linewidth=2)

    ax1.set_xlabel('Chapter', fontsize=12)
    ax1.set_ylabel('Emotion Score', fontsize=12)
    ax1.set_title('Emotional Arc of Romeo and Juliet', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(len(chapters)))
    ax1.set_xticklabels(chapters, rotation=45, ha='right', fontsize=8)

    # Plot 2: Heatmap
    im = ax2.imshow(emotion_matrix, aspect='auto', cmap='YlOrRd')
    ax2.set_xlabel('Chapter', fontsize=12)
    ax2.set_ylabel('Emotion', fontsize=12)
    ax2.set_title('Emotion Intensity Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(chapters)))
    ax2.set_xticklabels(chapters, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(len(emotions)))
    ax2.set_yticklabels(emotions)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Emotion Score', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")

    return fig


def save_results(chapter_emotions: Dict[str, Dict[str, float]], output_file: str = 'emotion_results.json'):
    """Save emotion analysis results to JSON"""
    print(f"Saving results to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(chapter_emotions, f, indent=2)

    print("Results saved successfully")


def print_summary(chapter_emotions: Dict[str, Dict[str, float]]):
    """Print summary of emotional analysis"""
    print("\n" + "="*80)
    print("EMOTIONAL ANALYSIS SUMMARY")
    print("="*80)

    for chapter, emotions in chapter_emotions.items():
        print(f"\n{chapter}:")
        # Sort emotions by intensity
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        for emotion, score in sorted_emotions[:3]:  # Top 3 emotions
            print(f"  {emotion}: {score:.3f}")


def main():
    """Main execution function"""
    print("="*80)
    print("Romeo and Juliet Emotional Arc Analysis")
    print("="*80)

    # Configuration
    MODEL_PATH = "/Users/devindyson/Desktop/troglodytelabs/emoWork/emoBERT/models/best_model.pt"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nUsing device: {DEVICE}")

    # Step 1: Load model
    model, tokenizer = load_model(MODEL_PATH, DEVICE)

    # Step 2: Fetch Romeo and Juliet
    text = fetch_romeo_and_juliet()

    # Step 3: Split into chapters
    chapters = split_into_chapters(text)

    if not chapters:
        print("Error: No chapters found in the text!")
        return

    # Step 4: Analyze emotions
    chapter_emotions = analyze_chapters(chapters, model, tokenizer, DEVICE)

    if not chapter_emotions:
        print("Error: No emotion analysis results!")
        return

    # Step 5: Print summary
    print_summary(chapter_emotions)

    # Step 6: Save results
    save_results(chapter_emotions, 'romeo_juliet_emotions.json')

    # Step 7: Create visualization
    plot_emotional_arc(chapter_emotions, 'romeo_juliet_emotional_arc.png')

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
