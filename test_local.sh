#!/bin/bash
# Local test script for macOS
# Tests emotion analysis on Pride and Prejudice using your trained model

MODEL_PATH="/Users/devindyson/Desktop/troglodytelabs/emoWork/emoBERT/models/best_model.pt"

echo "Testing RoBERTa emotion inference on Pride and Prejudice..."
echo "Using model: $MODEL_PATH"
echo ""

# Test with just first 5 segments for quick validation
python3 test_single_book.py \
  --book-id 1342 \
  --title "Pride and Prejudice" \
  --author "Jane Austen" \
  --model "$MODEL_PATH" \
  --segment-unit chapter \
  --max-segments 5 \
  --output "output/pride_prejudice_test.json"

echo ""
echo "Test complete! Check output/pride_prejudice_test.json"
