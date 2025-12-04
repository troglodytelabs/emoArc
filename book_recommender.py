#!/usr/bin/env python3
"""
Book Recommendation System based on Emotional Arc Analysis

Implements multiple recommendation strategies:
1. Content-based filtering using emotional arc similarity
2. Latent matrix factorization to discover hidden emotional patterns
3. Framework for collaborative filtering (when user data available)
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class EmotionalArcRecommender:
    """
    Recommendation system based on emotional arc patterns in books.

    Features:
    - Content-based filtering using emotional arc similarity
    - Latent factor extraction via matrix factorization (SVD/NMF)
    - Book clustering by emotional patterns
    """

    def __init__(self, results_file: str = 'outputs/data/gutenberg_sample_analysis.json'):
        """
        Initialize recommender with analysis results.

        Args:
            results_file: Path to JSON file with book analysis results
        """
        self.results_file = results_file
        self.books = []
        self.book_titles = []
        self.emotion_matrix = None  # Shape: (n_books, n_features)
        self.latent_factors = None  # Latent representation after factorization
        self.scaler = StandardScaler()

    def load_data(self):
        """Load book analysis data from JSON file"""
        print(f"Loading book data from {self.results_file}...")

        with open(self.results_file, 'r') as f:
            self.books = json.load(f)

        self.book_titles = [book['title'] for book in self.books]
        print(f"Loaded {len(self.books)} books")

    def build_emotion_matrix(self, method: str = 'segment_vectors'):
        """
        Build emotion feature matrix from book data.

        Args:
            method: How to construct features
                - 'segment_vectors': Full emotional arc (n_segments * n_emotions features)
                - 'average_emotions': Average emotion scores (n_emotions features)
                - 'arc_statistics': Statistical features of arcs (mean, std, trend per emotion)
        """
        print(f"\nBuilding emotion matrix using method: {method}")

        if method == 'segment_vectors':
            # Use full emotional arc as feature vector
            # Each book becomes: [joy_seg1, joy_seg2, ..., sadness_seg1, ...]
            feature_vectors = []

            for book in self.books:
                segments = book['segment_emotions']
                # Flatten: for each emotion, concatenate values across all segments
                vector = []
                for emotion in ['anger', 'anticipation', 'disgust', 'fear',
                               'joy', 'sadness', 'surprise', 'trust']:
                    emotion_arc = [seg[emotion] for seg in segments]
                    vector.extend(emotion_arc)
                feature_vectors.append(vector)

            self.emotion_matrix = np.array(feature_vectors)
            print(f"Emotion matrix shape: {self.emotion_matrix.shape}")

        elif method == 'average_emotions':
            # Use average emotion scores
            feature_vectors = []

            for book in self.books:
                avg_emotions = book['avg_emotions']
                vector = [avg_emotions[emotion] for emotion in
                         ['anger', 'anticipation', 'disgust', 'fear',
                          'joy', 'sadness', 'surprise', 'trust']]
                feature_vectors.append(vector)

            self.emotion_matrix = np.array(feature_vectors)
            print(f"Emotion matrix shape: {self.emotion_matrix.shape}")

        elif method == 'arc_statistics':
            # Use statistical features: mean, std, min, max, trend for each emotion
            feature_vectors = []

            for book in self.books:
                segments = book['segment_emotions']
                vector = []

                for emotion in ['anger', 'anticipation', 'disgust', 'fear',
                               'joy', 'sadness', 'surprise', 'trust']:
                    arc = np.array([seg[emotion] for seg in segments])

                    # Calculate statistics
                    mean_val = np.mean(arc)
                    std_val = np.std(arc)
                    min_val = np.min(arc)
                    max_val = np.max(arc)

                    # Calculate trend (slope of linear fit)
                    x = np.arange(len(arc))
                    trend = np.polyfit(x, arc, 1)[0]  # Linear slope

                    vector.extend([mean_val, std_val, min_val, max_val, trend])

                feature_vectors.append(vector)

            self.emotion_matrix = np.array(feature_vectors)
            print(f"Emotion matrix shape: {self.emotion_matrix.shape}")

        # Normalize features
        self.emotion_matrix = self.scaler.fit_transform(self.emotion_matrix)
        print("Features normalized (zero mean, unit variance)")

    def apply_matrix_factorization(self, method: str = 'svd', n_components: int = 20):
        """
        Apply matrix factorization to extract latent factors.

        This discovers hidden emotional patterns that aren't explicitly captured
        by individual emotions. For example, it might discover:
        - "Dramatic arc" factor (high tension -> resolution)
        - "Light-hearted" factor (consistent joy/trust, low fear/sadness)
        - "Emotional rollercoaster" factor (high variance across emotions)

        Args:
            method: 'svd' (Singular Value Decomposition) or 'nmf' (Non-negative Matrix Factorization)
            n_components: Number of latent factors to extract
        """
        print(f"\nApplying {method.upper()} with {n_components} components...")

        if method == 'svd':
            # SVD: Good for capturing variance, can handle negative values
            model = TruncatedSVD(n_components=n_components, random_state=42)
            self.latent_factors = model.fit_transform(self.emotion_matrix)

            explained_variance = np.sum(model.explained_variance_ratio_)
            print(f"Explained variance: {explained_variance:.2%}")

            # Store model for interpretation
            self.factorization_model = model

        elif method == 'nmf':
            # NMF: Parts-based representation, all values non-negative
            # Good for finding additive components

            # NMF requires non-negative values, so shift if needed
            shifted_matrix = self.emotion_matrix - self.emotion_matrix.min() + 0.01

            model = NMF(n_components=n_components, random_state=42, max_iter=500)
            self.latent_factors = model.fit_transform(shifted_matrix)

            # Store model for interpretation
            self.factorization_model = model
            print("NMF factorization complete")

        print(f"Latent factor matrix shape: {self.latent_factors.shape}")

    def get_similar_books(self, book_title: str, n_recommendations: int = 5,
                         use_latent: bool = False) -> List[Tuple[str, float]]:
        """
        Find books with similar emotional arcs.

        Args:
            book_title: Title of the book to find similar books for
            n_recommendations: Number of recommendations to return
            use_latent: If True, use latent factors; if False, use original emotion features

        Returns:
            List of (book_title, similarity_score) tuples
        """
        # Find book index
        if book_title not in self.book_titles:
            # Try partial match
            matches = [t for t in self.book_titles if book_title.lower() in t.lower()]
            if not matches:
                print(f"Book '{book_title}' not found!")
                return []
            book_title = matches[0]
            print(f"Using closest match: {book_title}")

        book_idx = self.book_titles.index(book_title)

        # Choose feature matrix
        if use_latent and self.latent_factors is not None:
            feature_matrix = self.latent_factors
            print(f"\nFinding similar books using latent factors...")
        else:
            feature_matrix = self.emotion_matrix
            print(f"\nFinding similar books using emotional arc features...")

        # Calculate similarity (cosine similarity)
        similarities = cosine_similarity([feature_matrix[book_idx]], feature_matrix)[0]

        # Get top N similar books (excluding the query book itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]

        recommendations = [
            (self.book_titles[idx], similarities[idx])
            for idx in similar_indices
        ]

        return recommendations

    def recommend_by_emotions(self, desired_emotions: Dict[str, float],
                             n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        Recommend books based on desired emotional profile.

        Args:
            desired_emotions: Dict like {'joy': 0.8, 'sadness': 0.2, ...}
            n_recommendations: Number of recommendations

        Returns:
            List of (book_title, match_score) tuples
        """
        print(f"\nFinding books matching emotional profile...")

        # Create query vector from desired emotions
        emotion_order = ['anger', 'anticipation', 'disgust', 'fear',
                        'joy', 'sadness', 'surprise', 'trust']
        query_vector = np.array([desired_emotions.get(e, 0.0) for e in emotion_order])

        # Compare with average emotions of each book
        match_scores = []
        for book in self.books:
            book_emotions = np.array([book['avg_emotions'][e] for e in emotion_order])

            # Calculate similarity
            similarity = cosine_similarity([query_vector], [book_emotions])[0][0]
            match_scores.append(similarity)

        # Get top matches
        top_indices = np.argsort(match_scores)[::-1][:n_recommendations]

        recommendations = [
            (self.book_titles[idx], match_scores[idx])
            for idx in top_indices
        ]

        return recommendations

    def cluster_books(self, n_clusters: int = 5, use_latent: bool = True):
        """
        Cluster books by emotional patterns.

        Args:
            n_clusters: Number of clusters
            use_latent: Use latent factors (True) or original features (False)

        Returns:
            Dict mapping cluster_id -> list of book titles
        """
        from sklearn.cluster import KMeans

        print(f"\nClustering books into {n_clusters} groups...")

        # Choose feature matrix
        feature_matrix = self.latent_factors if use_latent else self.emotion_matrix

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)

        # Organize by cluster
        clusters = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(self.book_titles[idx])

        return clusters

    def visualize_latent_space(self, output_file: str = 'outputs/visualizations/latent_space.png'):
        """
        Visualize books in 2D latent space using first 2 components.
        """
        print(f"\nVisualizing latent space...")

        if self.latent_factors is None:
            print("No latent factors available. Run apply_matrix_factorization() first.")
            return

        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot books in 2D space (first 2 latent factors)
        x = self.latent_factors[:, 0]
        y = self.latent_factors[:, 1]

        # Color by dominant emotion
        colors = []
        for book in self.books:
            dominant_emotion = max(book['avg_emotions'].items(), key=lambda x: x[1])[0]
            emotion_colors = {
                'joy': 'gold', 'sadness': 'blue', 'fear': 'purple',
                'anger': 'red', 'trust': 'green', 'anticipation': 'orange',
                'surprise': 'pink', 'disgust': 'brown'
            }
            colors.append(emotion_colors.get(dominant_emotion, 'gray'))

        ax.scatter(x, y, c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

        # Add labels for a subset of books (to avoid clutter)
        for i in range(min(20, len(self.book_titles))):
            title = self.book_titles[i]
            short_title = title[:30] + '...' if len(title) > 30 else title
            ax.annotate(short_title, (x[i], y[i]), fontsize=7, alpha=0.7)

        ax.set_xlabel('Latent Factor 1', fontsize=12)
        ax.set_ylabel('Latent Factor 2', fontsize=12)
        ax.set_title('Books in Latent Emotional Space', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")


def demo_recommender():
    """Demonstrate the recommendation system"""
    print("="*80)
    print("BOOK RECOMMENDATION SYSTEM - DEMO")
    print("="*80)

    # Initialize recommender
    recommender = EmotionalArcRecommender()

    try:
        # Load data
        recommender.load_data()

        # Build emotion matrix using full emotional arcs
        recommender.build_emotion_matrix(method='segment_vectors')

        # Apply matrix factorization to extract latent factors
        recommender.apply_matrix_factorization(method='svd', n_components=20)

        # Example 1: Find similar books (without latent factors)
        print("\n" + "="*80)
        print("EXAMPLE 1: Content-Based Similarity (Original Features)")
        print("="*80)
        example_book = recommender.book_titles[0]
        print(f"\nBooks similar to: {example_book}")
        recommendations = recommender.get_similar_books(example_book, n_recommendations=5, use_latent=False)
        for i, (title, score) in enumerate(recommendations, 1):
            print(f"  {i}. {title} (similarity: {score:.3f})")

        # Example 2: Find similar books (with latent factors)
        print("\n" + "="*80)
        print("EXAMPLE 2: Latent Factor Similarity")
        print("="*80)
        print(f"\nBooks similar to: {example_book} (using latent factors)")
        recommendations = recommender.get_similar_books(example_book, n_recommendations=5, use_latent=True)
        for i, (title, score) in enumerate(recommendations, 1):
            print(f"  {i}. {title} (similarity: {score:.3f})")

        # Example 3: Recommend by desired emotions
        print("\n" + "="*80)
        print("EXAMPLE 3: Emotion-Based Recommendations")
        print("="*80)
        desired_emotions = {
            'joy': 0.7,
            'trust': 0.6,
            'anticipation': 0.5,
            'fear': 0.1,
            'sadness': 0.2
        }
        print(f"\nFinding books matching profile: {desired_emotions}")
        recommendations = recommender.recommend_by_emotions(desired_emotions, n_recommendations=5)
        for i, (title, score) in enumerate(recommendations, 1):
            print(f"  {i}. {title} (match: {score:.3f})")

        # Example 4: Cluster books
        print("\n" + "="*80)
        print("EXAMPLE 4: Book Clustering")
        print("="*80)
        clusters = recommender.cluster_books(n_clusters=5, use_latent=True)
        for cluster_id, books in clusters.items():
            print(f"\nCluster {cluster_id + 1} ({len(books)} books):")
            for book in books[:3]:  # Show first 3 from each cluster
                print(f"  - {book}")
            if len(books) > 3:
                print(f"  ... and {len(books) - 3} more")

        # Visualize latent space
        recommender.visualize_latent_space()

        print("\n" + "="*80)
        print("RECOMMENDATION SYSTEM DEMO COMPLETE")
        print("="*80)

        return recommender

    except FileNotFoundError:
        print(f"\nError: Analysis results not found at {recommender.results_file}")
        print("Please run analyze_gutenberg_sample.py first to generate the data.")
        print("\nTo generate sample data:")
        print("  python analyze_gutenberg_sample.py")
        return None


if __name__ == "__main__":
    recommender = demo_recommender()
