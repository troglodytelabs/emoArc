# Explainable Recommendations Guide

A comprehensive guide to understanding **why** books are recommended and **how** the system makes decisions.

## Overview

Explainability makes recommendations trustworthy and actionable. Users can understand:

- ✅ **Why** a book was recommended
- ✅ **Which** emotions matched
- ✅ **How** emotional arcs compare
- ✅ **What** latent factors mean
- ✅ **Where** in the story similarities occur

## Table of Contents

1. [Quick Start](#quick-start)
2. [Explanation Types](#explanation-types)
3. [Visual Explanations](#visual-explanations)
4. [Latent Factor Interpretation](#latent-factor-interpretation)
5. [Use Cases](#use-cases)
6. [Best Practices](#best-practices)

## Quick Start

```python
from explainable_recommender import ExplainableRecommender

# Initialize
recommender = ExplainableRecommender()
recommender.load_data()
recommender.build_emotion_matrix(method='segment_vectors')
recommender.apply_matrix_factorization(method='svd', n_components=20)

# Get recommendations with explanations
results = recommender.get_recommendation_with_explanation(
    "Pride and Prejudice by Jane Austen",
    n_recommendations=3,
    use_latent=True
)

# Print explanation
for result in results:
    print(result['explanation']['explanation_text'])
```

**Output:**
```
📚 Why we recommended 'Sense and Sensibility by Jane Austen':

✨ Overall Match: 87% similar (using latent factors)

🎭 Emotional Profile Comparison:
  ✓ Excellent match: joy, trust, anticipation
  ✓ Good match: sadness
  ⚠ Different: fear (Δ0.12)

💫 Emotional Tone:
  Your choice emphasizes: joy (0.52)
  This recommendation: joy (0.48)

📖 Story Arc Similarity:
  Found 7 highly similar segments
  Similar at: 10%, 30%, 50%, 70%, 90%
```

## Explanation Types

### 1. Similarity Score Explanation

Shows **overall match percentage** based on emotional arc features.

```python
explanation = recommender.explain_recommendation(
    query_book="Romeo and Juliet by William Shakespeare",
    recommended_book="Hamlet by William Shakespeare",
    use_latent=True
)

print(f"Similarity: {explanation['similarity_score']:.2%}")
# Output: Similarity: 74%
```

**Interpretation:**
- **90-100%**: Nearly identical emotional arcs
- **75-89%**: Very similar patterns
- **60-74%**: Similar with notable differences
- **<60%**: Different emotional journeys

### 2. Emotion-by-Emotion Comparison

Breaks down similarity for each of 8 emotions.

```python
for emotion, data in explanation['emotion_comparison'].items():
    print(f"{emotion}: {data['match_quality']}")
    print(f"  Query: {data['query']:.3f}")
    print(f"  Recommended: {data['recommended']:.3f}")
    print(f"  Difference: {data['difference']:.3f}")
```

**Output:**
```
joy: excellent
  Query: 0.425
  Recommended: 0.418
  Difference: 0.007

sadness: good
  Query: 0.312
  Recommended: 0.289
  Difference: 0.023

fear: different
  Query: 0.156
  Recommended: 0.245
  Difference: 0.089
```

**Match Quality Thresholds:**
- **Excellent**: Difference < 0.05 (emotions nearly identical)
- **Good**: Difference < 0.10 (very similar)
- **Moderate**: Difference < 0.15 (somewhat similar)
- **Different**: Difference ≥ 0.15 (notable variation)

### 3. Segment-Level Similarity

Identifies which parts of the story have similar emotional patterns.

```python
for segment in explanation['matching_segments']:
    print(f"Segment {segment['segment_number']} "
          f"(at {segment['position_percent']}% through story): "
          f"{segment['similarity']:.2%} similar")
```

**Output:**
```
Segment 1 (at 0% through story): 92% similar
Segment 5 (at 50% through story): 89% similar
Segment 10 (at 100% through story): 95% similar
```

**Use Case:** Understand if books share similar:
- **Openings** (segment 1-2): Similar introductions
- **Middle arcs** (segment 4-7): Similar development
- **Endings** (segment 9-10): Similar resolutions

### 4. Dominant Emotion Analysis

Shows the strongest emotion in each book.

```python
query_dom = explanation['dominant_emotions']['query']
rec_dom = explanation['dominant_emotions']['recommended']

print(f"Your book emphasizes: {query_dom['emotion']} ({query_dom['score']:.2f})")
print(f"Recommendation emphasizes: {rec_dom['emotion']} ({rec_dom['score']:.2f})")
```

**Output:**
```
Your book emphasizes: joy (0.52)
Recommendation emphasizes: trust (0.48)
```

**Interpretation:**
- If dominant emotions match → Very consistent tone
- If different but complementary (e.g., joy + trust) → Similar positive feel
- If opposite (e.g., joy vs sadness) → Contrasting experiences

## Visual Explanations

### 1. Emotional Arc Comparison

Generate side-by-side visualization of emotional arcs.

```python
recommender.visualize_comparison(
    query_book="Pride and Prejudice by Jane Austen",
    recommended_books=[
        "Sense and Sensibility by Jane Austen",
        "Emma by Jane Austen"
    ],
    output_file='outputs/visualizations/arc_comparison.png'
)
```

**Visualization includes:**

```
┌─────────────────────────────────────────────────────────────┐
│ Your Interest: Pride and Prejudice by Jane Austen          │
│ [Emotional arc plot with joy, sadness, fear, etc.]         │
└─────────────────────────────────────────────────────────────┘

┌───────────────────────────────────┬─────────────────────────┐
│ Recommendation #1:                 │ Emotion Comparison      │
│ Sense and Sensibility             │ [Bar chart showing]     │
│ [Emotional arc plot]              │ [emotion-by-emotion]    │
│                                    │ [match quality]         │
└───────────────────────────────────┴─────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Emotional Arc Similarity by Segment                         │
│ [Heatmap showing segment-by-segment similarity]             │
│ ★ = Highly similar segment (>90%)                           │
└─────────────────────────────────────────────────────────────┘
```

### 2. Similarity Heatmap

The bottom heatmap shows **where in the story** emotions match:

- **Bright yellow/red**: Highly similar segments
- **Orange**: Moderately similar
- **Dark red**: Less similar
- **★ symbol**: Exceptional match (>90% similar)

**Reading the heatmap:**
```
Recommendation | Beginning → Middle → End
───────────────┼─────────────────────────
Rec #1         │  🟨🟨🟧🟨🟨🟧🟨🟨🟨🟨
Rec #2         │  🟧🟧🟨🟨★🟨🟧🟨🟧🟨
```

### 3. Emotion Comparison Bars

Horizontal bar chart comparing average emotion scores:

```
            joy │ ████████░░ Your choice
                │ █████████░ Recommendation
                │
        sadness │ ████░░░░░░ Your choice
                │ ███░░░░░░░ Recommendation
```

**Use Case:**
- Quickly see which emotions are higher/lower
- Identify complementary patterns
- Spot major differences

## Latent Factor Interpretation

Understanding what the system "learned" about emotional patterns.

### What Are Latent Factors?

Matrix factorization discovers **hidden patterns** in emotional arcs:

- **Original space**: 80 dimensions (8 emotions × 10 segments)
- **Latent space**: 20 dimensions (compressed representation)
- **Latent factors**: Abstract patterns that explain books

### Example Latent Factors

```python
factors = recommender.explain_latent_factors(n_top_features=5)

for factor_idx, interpretation in factors.items():
    print(interpretation['interpretation'])
```

**Output:**
```
Latent Factor #1 (explains 15.3% of variance)
  This factor captures: joy throughout
  Interpretation: Stories with strong joy throughout
  Top Features:
    • joy_segment_5: 0.425 (at 50%)
    • joy_segment_6: 0.398 (at 60%)
    • joy_segment_7: 0.387 (at 70%)

Latent Factor #2 (explains 12.1% of variance)
  This factor captures: fear beginning
  Interpretation: Stories with strong fear beginning
  Top Features:
    • fear_segment_1: 0.512 (at 0%)
    • fear_segment_2: 0.489 (at 10%)
    • anticipation_segment_1: 0.445 (at 0%)

Latent Factor #3 (explains 8.7% of variance)
  This factor captures: sadness ending
  Interpretation: Stories with strong sadness ending
  Top Features:
    • sadness_segment_9: 0.578 (at 90%)
    • sadness_segment_10: 0.551 (at 100%)
    • sadness_segment_8: 0.423 (at 80%)
```

### Common Latent Patterns

1. **"Dramatic Arc" Factor**
   - High: anticipation (beginning), fear (middle), relief/joy (end)
   - Captures: Classic tension → resolution pattern
   - Examples: Thrillers, mysteries, adventure

2. **"Emotional Stability" Factor**
   - High: Consistent trust/joy, low variance
   - Captures: Stories with steady emotional tone
   - Examples: Comedies, light romance

3. **"Tragic Arc" Factor**
   - High: Joy (beginning) → sadness (end)
   - Captures: Fall from grace / tragic hero
   - Examples: Tragedies, dramatic literature

4. **"Redemption Arc" Factor**
   - High: Sadness/anger (beginning) → joy/trust (end)
   - Captures: Character growth, hope
   - Examples: Coming-of-age, redemption stories

5. **"Suspense Factor"**
   - High: Anticipation + fear throughout
   - Low: Joy, trust
   - Captures: Sustained tension
   - Examples: Horror, psychological thrillers

### Using Latent Factors for Recommendations

When `use_latent=True`, the system:

1. Projects books into latent space
2. Compares positions in this abstract space
3. Finds books with similar latent factor patterns

**Advantage:**
- Discovers **deeper similarities** beyond surface-level emotions
- Books may have different specific emotions but similar **narrative structures**

**Example:**
```
Book A: High joy, low sadness
Book B: High trust, low fear

Surface similarity: Low (different emotions)
Latent similarity: High (both "positive, stable" narratives)
```

## Use Cases

### Use Case 1: Understanding Unexpected Recommendations

**Scenario:** System recommends a book that seems unrelated.

```python
explanation = recommender.explain_recommendation(
    query_book="The Great Gatsby by F. Scott Fitzgerald",
    recommended_book="The Picture of Dorian Gray by Oscar Wilde",
    use_latent=True
)

print(explanation['explanation_text'])
```

**Output:**
```
📚 Why we recommended 'The Picture of Dorian Gray by Oscar Wilde':

✨ Overall Match: 78% similar (using latent factors)

🎭 Emotional Profile Comparison:
  ✓ Excellent match: sadness, disgust
  ✓ Good match: fear, trust
  ⚠ Different: joy (Δ0.14)

💫 Emotional Tone:
  Your choice emphasizes: sadness (0.38)
  This recommendation: disgust (0.42)

📖 Story Arc Similarity:
  Found 6 highly similar segments
  Similar at: 0%, 20%, 60%, 80%, 100%
```

**Insight:** Both books share:
- Tragic arc (sadness/disgust dominant)
- Similar beginnings and endings
- Themes of moral decay (disgust)

### Use Case 2: Comparing Multiple Recommendations

**Scenario:** Choose between several recommendations.

```python
query_book = "To Kill a Mockingbird by Harper Lee"

results = recommender.get_recommendation_with_explanation(
    query_book,
    n_recommendations=5,
    use_latent=True
)

# Compare match quality
for i, result in enumerate(results, 1):
    exp = result['explanation']

    # Count excellent matches
    excellent = sum(1 for e, d in exp['emotion_comparison'].items()
                   if d['match_quality'] == 'excellent')

    print(f"{i}. {result['book']}")
    print(f"   Overall: {exp['similarity_score']:.0%}")
    print(f"   Excellent emotion matches: {excellent}/8")
    print(f"   Matching segments: {len(exp['matching_segments'])}")
```

**Output:**
```
1. Go Set a Watchman by Harper Lee
   Overall: 92%
   Excellent emotion matches: 7/8
   Matching segments: 9

2. The Help by Kathryn Stockett
   Overall: 81%
   Excellent emotion matches: 5/8
   Matching segments: 6

3. Invisible Man by Ralph Ellison
   Overall: 76%
   Excellent emotion matches: 4/8
   Matching segments: 5
```

**Decision:** Choose #1 for closest match, or #2 for similar themes with fresh perspective.

### Use Case 3: Query by Mood

**Scenario:** Find books matching a specific emotional profile.

```python
# I want: Uplifting book with some anticipation, minimal fear
desired_emotions = {
    'joy': 0.6,
    'trust': 0.5,
    'anticipation': 0.4,
    'fear': 0.1,
    'sadness': 0.2
}

recommendations = recommender.recommend_by_emotions(
    desired_emotions,
    n_recommendations=5
)

for title, score in recommendations:
    print(f"{title}: {score:.2%} match")
```

Then get explanations:
```python
for title, _ in recommendations[:3]:
    exp = recommender.explain_recommendation(
        query_book=recommendations[0][0],  # Use top match as reference
        recommended_book=title,
        use_latent=False
    )

    print(f"\n{title}:")
    print(f"  Dominant emotion: {exp['dominant_emotions']['recommended']['emotion']}")
```

### Use Case 4: Discovering Book Clusters

**Scenario:** Understand what types of emotional narratives exist.

```python
clusters = recommender.cluster_books(n_clusters=5, use_latent=True)

for cluster_id, books in clusters.items():
    print(f"\n=== Cluster {cluster_id + 1} ===")

    # Analyze cluster characteristics
    cluster_books = [recommender.books[recommender.book_titles.index(b)]
                    for b in books if b in recommender.book_titles]

    # Average emotions in cluster
    avg_emotions = {}
    for emotion in ['joy', 'sadness', 'fear', 'anger']:
        scores = [b['avg_emotions'][emotion] for b in cluster_books]
        avg_emotions[emotion] = np.mean(scores)

    # Identify cluster type
    dominant = max(avg_emotions.items(), key=lambda x: x[1])
    print(f"Type: {dominant[0].capitalize()}-dominant narratives")
    print(f"Sample books:")
    for book in books[:3]:
        print(f"  - {book}")
```

**Output:**
```
=== Cluster 1 ===
Type: Joy-dominant narratives
Sample books:
  - Pride and Prejudice by Jane Austen
  - Emma by Jane Austen
  - The Importance of Being Earnest by Oscar Wilde

=== Cluster 2 ===
Type: Sadness-dominant narratives
Sample books:
  - Wuthering Heights by Emily Brontë
  - Tess of the d'Urbervilles by Thomas Hardy
  - Anna Karenina by Leo Tolstoy
```

## Best Practices

### 1. Start with Latent Factors

```python
# ✅ Better: Use latent factors for deeper patterns
results = recommender.get_recommendation_with_explanation(
    query_book,
    use_latent=True
)

# ❌ Avoid: Only using raw emotions (misses abstract patterns)
results = recommender.get_recommendation_with_explanation(
    query_book,
    use_latent=False
)
```

**Reason:** Latent factors capture **narrative structure**, not just emotion levels.

### 2. Visualize for Complex Cases

For surprising recommendations, create visual comparison:

```python
if result['similarity_score'] < 0.75:  # Lower similarity
    recommender.visualize_comparison(
        query_book,
        [result['book']],
        output_file=f'outputs/visualizations/why_{result["book"]}.png'
    )
```

### 3. Check Segment Similarity for Arc Shape

```python
matching_segments = explanation['matching_segments']

# Analyze where matches occur
positions = [s['position_percent'] for s in matching_segments]

if 0 in positions and 100 in positions:
    print("✓ Similar beginning AND ending")
elif positions and np.mean(positions) < 50:
    print("✓ Similar openings, different endings")
elif positions and np.mean(positions) > 50:
    print("✓ Different openings, similar endings")
```

### 4. Compare Dominant Emotions for Tone

```python
query_dom = explanation['dominant_emotions']['query']['emotion']
rec_dom = explanation['dominant_emotions']['recommended']['emotion']

# Complementary emotions (work well together)
complementary = {
    'joy': ['trust', 'anticipation'],
    'fear': ['anticipation', 'surprise'],
    'sadness': ['trust', 'anticipation'],
    'anger': ['disgust']
}

if rec_dom in complementary.get(query_dom, []):
    print("✓ Complementary emotional tones")
```

### 5. Use Explanations in UI

**Good UI Pattern:**
```
📚 Recommended: "Sense and Sensibility"

Why this match?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 87% similar emotional journey

✨ What matches:
   • Similar joy and trust throughout
   • Both emphasize positive emotions
   • Nearly identical story endings

⚡ Key difference:
   • Your choice has more anticipation
   • This has slightly more sadness

[Show detailed comparison →]
```

### 6. Interpret Latent Factors Periodically

Re-run factor interpretation when adding new books:

```python
# After adding new books to dataset
recommender.apply_matrix_factorization(method='svd', n_components=20)
factors = recommender.explain_latent_factors()

# Check if new patterns emerged
for factor_idx, interp in factors.items():
    print(interp['interpretation'])
```

## Metrics for Explainability

### Coverage
How many recommendations have good explanations?

```python
results = recommender.get_recommendation_with_explanation(query_book, n_recommendations=10)

well_explained = sum(1 for r in results
                    if len(r['explanation']['matching_segments']) >= 3)

coverage = well_explained / len(results)
print(f"Explainability coverage: {coverage:.0%}")
```

**Target:** >80% of recommendations should have ≥3 matching segments.

### Explanation Diversity
Are explanations varied or repetitive?

```python
explanations = [r['explanation']['explanation_text'] for r in results]

# Check if dominant emotions vary
dominant_emotions = [r['explanation']['dominant_emotions']['recommended']['emotion']
                    for r in results]

diversity = len(set(dominant_emotions)) / len(dominant_emotions)
print(f"Explanation diversity: {diversity:.0%}")
```

**Target:** >60% diversity (avoid all recommendations having same dominant emotion).

## Troubleshooting

### "Low similarity but recommended?"

**Cause:** Latent factors found deep pattern not visible in raw emotions.

**Solution:** Check latent factor interpretation:
```python
factors = recommender.explain_latent_factors()
# See which abstract patterns both books score high on
```

### "Explanations too technical?"

**Solution:** Focus on human-readable summary:
```python
print(explanation['explanation_text'])
# Skip detailed emotion_comparison dict
```

### "Visualizations unclear?"

**Solution:** Adjust thresholds or reduce books shown:
```python
recommender.visualize_comparison(
    query_book,
    recommended_books[:2],  # Show only top 2
    output_file='outputs/visualizations/simple_comparison.png'
)
```

## Summary

**Explainable recommendations provide:**

1. **Transparency** - Users understand why books match
2. **Trust** - Clear reasoning builds confidence
3. **Actionability** - Users can refine preferences
4. **Discoverability** - Learn about emotional patterns
5. **Control** - Make informed choices

**Key Features:**
- ✅ Similarity scores with context
- ✅ Emotion-by-emotion breakdowns
- ✅ Segment-level comparisons
- ✅ Visual arc comparisons
- ✅ Latent factor interpretations
- ✅ Human-readable explanations

This makes the recommendation system not just accurate, but **understandable** and **trustworthy**.
