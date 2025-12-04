# Book Recommendation System

A comprehensive recommendation system based on emotional arc analysis of literary works. This system uses **matrix factorization** and **content-based filtering** to discover latent emotional patterns and recommend similar books.

## Overview

The recommendation system consists of three components:

1. **Content-Based Filtering** (`book_recommender.py`) - Uses emotional arc similarity
2. **Collaborative Filtering** (`collaborative_recommender.py`) - Uses user rating data
3. **Hybrid Approach** - Combines both methods for better recommendations

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Emotional Arc Analysis                    │
│              (analyze_gutenberg_sample.py)                   │
│  - Extracts titles + authors from Project Gutenberg         │
│  - Random sampling to avoid duplicates                       │
│  - Analyzes 8 emotions across 10 segments per book          │
│  - Outputs: JSON with emotional arc data                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Content-Based Recommender                       │
│               (book_recommender.py)                          │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Emotional Feature Extraction                        │    │
│  │ • Segment vectors (8 emotions × 10 segments)       │    │
│  │ • Average emotions (8 features)                     │    │
│  │ • Arc statistics (mean, std, trend per emotion)    │    │
│  └────────────────────────────────────────────────────┘    │
│                              │                               │
│                              ▼                               │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Matrix Factorization                                │    │
│  │ • SVD: Discovers latent emotional patterns         │    │
│  │ • NMF: Parts-based representation                   │    │
│  │ • Output: Low-dimensional latent factors           │    │
│  └────────────────────────────────────────────────────┘    │
│                              │                               │
│                              ▼                               │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Recommendation Strategies                           │    │
│  │ • Similar books (cosine similarity)                 │    │
│  │ • Emotion-based matching                            │    │
│  │ • Clustering by emotional patterns                  │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│         Collaborative Filtering Recommender                  │
│          (collaborative_recommender.py)                      │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ User-Book Rating Matrix                             │    │
│  │   Users × Books with ratings (1-5 stars)           │    │
│  └────────────────────────────────────────────────────┘    │
│                              │                               │
│                              ▼                               │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Matrix Factorization                                │    │
│  │ • SVD: Fast, good for dense matrices               │    │
│  │ • ALS: Better for sparse matrices                   │    │
│  │ • Output: User & Item latent factors               │    │
│  └────────────────────────────────────────────────────┘    │
│                              │                               │
│                              ▼                               │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Predictions & Recommendations                       │    │
│  │ • Predict ratings: user_factor @ item_factor       │    │
│  │ • Recommend top N unrated books                     │    │
│  │ • Find similar books via item factors               │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Analyze Books and Extract Emotional Arcs

```bash
python analyze_gutenberg_sample.py
```

This will:
- Download 100 random books from Project Gutenberg
- Extract titles and authors
- Analyze emotional arcs (8 emotions × 10 segments)
- Save results to `outputs/data/gutenberg_sample_analysis.json`

### Step 2: Build Content-Based Recommender

```bash
python book_recommender.py
```

This will:
- Load emotional arc data
- Apply matrix factorization (SVD with 20 components)
- Generate recommendations using multiple strategies
- Create visualizations

### Step 3: (Optional) Collaborative Filtering

```bash
python collaborative_recommender.py
```

This generates sample user ratings and demonstrates collaborative filtering with matrix factorization.

## Usage Examples

### Content-Based Recommendations

```python
from book_recommender import EmotionalArcRecommender

# Initialize
recommender = EmotionalArcRecommender()
recommender.load_data()
recommender.build_emotion_matrix(method='segment_vectors')

# Apply matrix factorization to extract latent emotional patterns
recommender.apply_matrix_factorization(method='svd', n_components=20)

# Find similar books
recommendations = recommender.get_similar_books(
    "Pride and Prejudice by Jane Austen",
    n_recommendations=5,
    use_latent=True  # Use latent factors for deeper similarity
)

for title, score in recommendations:
    print(f"{title}: {score:.3f}")
```

### Emotion-Based Recommendations

```python
# Find books matching a specific emotional profile
desired_emotions = {
    'joy': 0.7,
    'trust': 0.6,
    'anticipation': 0.5,
    'fear': 0.1,
    'sadness': 0.2
}

recommendations = recommender.recommend_by_emotions(
    desired_emotions,
    n_recommendations=5
)
```

### Clustering Books by Emotional Patterns

```python
# Discover groups of books with similar emotional arcs
clusters = recommender.cluster_books(n_clusters=5, use_latent=True)

for cluster_id, books in clusters.items():
    print(f"\nCluster {cluster_id + 1}:")
    for book in books[:5]:
        print(f"  - {book}")
```

### Collaborative Filtering

```python
from collaborative_recommender import CollaborativeFilteringRecommender

# Initialize with user rating data
recommender = CollaborativeFilteringRecommender('outputs/data/ratings.json')
recommender.load_ratings()
recommender.build_rating_matrix()

# Apply matrix factorization (ALS is better for sparse data)
recommender.matrix_factorization_als(n_factors=20, n_iterations=10)

# Get recommendations for a user
recommendations = recommender.recommend_for_user(
    user_id='user_123',
    n_recommendations=10,
    exclude_rated=True
)
```

## Matrix Factorization Methods

### 1. SVD (Singular Value Decomposition)

**When to use:**
- Dense matrices
- Need interpretable components
- Want to capture maximum variance

**How it works:**
```
Emotion Matrix (n_books × n_features) ≈ U @ Σ @ V^T

Where:
- U: Book embeddings in latent space
- Σ: Importance of each latent factor
- V^T: Emotion pattern components
```

**Example:** Discovers latent factors like:
- "Dramatic Arc" (tension → resolution)
- "Light-hearted" (high joy/trust, low fear)
- "Emotional Intensity" (high variance)

### 2. NMF (Non-negative Matrix Factorization)

**When to use:**
- Want parts-based representation
- All features are non-negative
- Need interpretable additive components

**How it works:**
```
Emotion Matrix ≈ W @ H

Where:
- W: Book-to-factor weights (non-negative)
- H: Factor-to-emotion patterns (non-negative)
```

**Example:** Discovers additive patterns like:
- "Adventure" = high anticipation + moderate fear + low sadness
- "Romance" = high joy + high trust + low anger

### 3. ALS (Alternating Least Squares)

**When to use:**
- Sparse user-item rating matrices
- Collaborative filtering
- Large-scale systems

**How it works:**
```
Rating Matrix ≈ User_Factors @ Item_Factors^T

Iteratively optimizes:
1. Fix item factors, solve for user factors
2. Fix user factors, solve for item factors
3. Repeat until convergence
```

## Recommendation Strategies

### 1. Content-Based Similarity

Finds books with similar emotional arcs using cosine similarity:

```
similarity(book_i, book_j) = cos(θ) = (A·B) / (||A|| ||B||)
```

**Pros:**
- No user data needed
- Works for new books immediately
- Explainable (can show which emotions match)

**Cons:**
- Can't discover surprising recommendations
- Limited to emotional features

### 2. Collaborative Filtering

Predicts ratings based on similar users' preferences:

```
predicted_rating(user, book) = user_factor @ book_factor
```

**Pros:**
- Discovers surprising recommendations
- Leverages wisdom of the crowd
- Works even without content features

**Cons:**
- Needs user rating data
- Cold start problem for new users/books
- Less explainable

### 3. Hybrid Approach

Combines both methods:

```
hybrid_score = α × collaborative_score + (1-α) × content_score
```

**Pros:**
- Best of both worlds
- Mitigates cold start problem
- More robust recommendations

**Parameters:**
- α = 0.7: Favor collaborative (when enough user data)
- α = 0.3: Favor content-based (for new books/users)

## Feature Engineering

### Emotional Arc Representations

1. **Segment Vectors** (80 features)
   - Full emotional trajectory
   - 8 emotions × 10 segments = 80-dimensional vector
   - Best for: Capturing narrative arc patterns

2. **Average Emotions** (8 features)
   - Overall emotional tone
   - Simple mean of each emotion across segments
   - Best for: Quick similarity matching

3. **Arc Statistics** (40 features)
   - Per emotion: mean, std, min, max, trend
   - 8 emotions × 5 statistics = 40 features
   - Best for: Capturing emotional dynamics

## Latent Factor Interpretation

After SVD/NMF, examine what each latent factor represents:

```python
# Get top emotions for each factor
for factor_idx in range(5):
    weights = recommender.factorization_model.components_[factor_idx]
    top_features = np.argsort(np.abs(weights))[-5:]

    print(f"Factor {factor_idx + 1}:")
    for feat_idx in top_features:
        print(f"  Feature {feat_idx}: {weights[feat_idx]:.3f}")
```

## Evaluation Metrics

### For Content-Based:
- **Precision@K**: Fraction of recommended books that user likes
- **Recall@K**: Fraction of liked books that appear in recommendations
- **NDCG**: Normalized Discounted Cumulative Gain

### For Collaborative Filtering:
- **RMSE**: Root Mean Squared Error on predicted ratings
- **MAE**: Mean Absolute Error
- **Hit Rate**: Percentage of relevant items in top-K

## Data Format

### Emotional Arc Data (JSON)
```json
{
  "title": "Pride and Prejudice by Jane Austen",
  "author": "Jane Austen",
  "num_segments": 10,
  "segment_emotions": [
    {"joy": 0.45, "sadness": 0.12, ...},
    {"joy": 0.52, "sadness": 0.08, ...},
    ...
  ],
  "avg_emotions": {
    "joy": 0.48,
    "sadness": 0.15,
    ...
  },
  "text_length": 156842
}
```

### User Rating Data (JSON)
```json
[
  {
    "user_id": "user_123",
    "book_title": "Pride and Prejudice by Jane Austen",
    "rating": 4.5,
    "timestamp": "2024-01-15T10:30:00"
  },
  ...
]
```

## Future Enhancements

1. **Deep Learning Approaches**
   - Neural collaborative filtering
   - Sequence models for reading history
   - Attention mechanisms for emotional arc analysis

2. **Additional Features**
   - Genre information
   - Publication year
   - Reading difficulty
   - Length preferences

3. **Real-time Updates**
   - Online learning as new ratings come in
   - Incremental matrix factorization
   - Streaming algorithms

4. **Explainability**
   - Show which emotions matched
   - Display emotional arc comparisons
   - Highlight similar segments

## Performance Tips

1. **For large datasets:**
   - Use sparse matrices
   - Implement mini-batch ALS
   - Consider approximate methods (LSH, ANNOY)

2. **For real-time recommendations:**
   - Pre-compute book embeddings
   - Use approximate nearest neighbors
   - Cache popular recommendations

3. **For better quality:**
   - Tune n_components (more = more detail, less generalization)
   - Try different feature representations
   - Combine multiple similarity metrics

## References

- Koren, Y., Bell, R., & Volinsky, C. (2009). "Matrix Factorization Techniques for Recommender Systems"
- Reagan, A. J., et al. (2016). "The emotional arcs of stories are dominated by six basic shapes"
- Adomavicius, G., & Tuzhilin, A. (2005). "Toward the next generation of recommender systems"

## Contributing

To add new recommendation strategies:

1. Inherit from `EmotionalArcRecommender`
2. Implement new similarity metrics
3. Add visualization methods
4. Update documentation

## License

MIT License - See LICENSE file for details
