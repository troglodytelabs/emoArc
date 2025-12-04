#!/usr/bin/env python3
"""
Collaborative Filtering Recommendation System

Uses matrix factorization (SVD, ALS) on user-book rating matrix.
This is the traditional approach when you have user preference data.

Future enhancement: Hybrid system combining:
- Collaborative filtering (user preferences)
- Content-based filtering (emotional arcs)
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


class CollaborativeFilteringRecommender:
    """
    Collaborative filtering recommender using matrix factorization.

    This assumes you have (or will collect) user rating data in format:
    {
        "user_id": "user123",
        "book_title": "Pride and Prejudice by Jane Austen",
        "rating": 4.5
    }
    """

    def __init__(self, ratings_file: Optional[str] = None):
        """
        Initialize collaborative filtering recommender.

        Args:
            ratings_file: Optional JSON file with user ratings
        """
        self.ratings_file = ratings_file
        self.ratings_data = []
        self.user_ids = []
        self.book_titles = []
        self.rating_matrix = None  # Shape: (n_users, n_books)
        self.user_factors = None   # User latent factors
        self.item_factors = None   # Book latent factors

    def load_ratings(self):
        """Load user rating data"""
        if not self.ratings_file or not Path(self.ratings_file).exists():
            print("No rating data available yet.")
            print("\nTo use collaborative filtering, collect user ratings in format:")
            print(json.dumps({
                "user_id": "user123",
                "book_title": "Book Title by Author",
                "rating": 4.5
            }, indent=2))
            return False

        with open(self.ratings_file, 'r') as f:
            self.ratings_data = json.load(f)

        print(f"Loaded {len(self.ratings_data)} ratings")
        return True

    def build_rating_matrix(self):
        """
        Build user-item rating matrix from rating data.

        Matrix shape: (n_users, n_books)
        Missing ratings are filled with 0 (will be masked during training)
        """
        # Extract unique users and books
        users = set()
        books = set()

        for rating in self.ratings_data:
            users.add(rating['user_id'])
            books.add(rating['book_title'])

        self.user_ids = sorted(list(users))
        self.book_titles = sorted(list(books))

        # Create mapping indices
        user_to_idx = {user: idx for idx, user in enumerate(self.user_ids)}
        book_to_idx = {book: idx for idx, book in enumerate(self.book_titles)}

        # Build matrix
        n_users = len(self.user_ids)
        n_books = len(self.book_titles)

        self.rating_matrix = np.zeros((n_users, n_books))

        for rating in self.ratings_data:
            user_idx = user_to_idx[rating['user_id']]
            book_idx = book_to_idx[rating['book_title']]
            self.rating_matrix[user_idx, book_idx] = rating['rating']

        print(f"\nRating matrix shape: {self.rating_matrix.shape}")
        print(f"Sparsity: {(self.rating_matrix == 0).sum() / self.rating_matrix.size:.2%}")

    def matrix_factorization_svd(self, n_factors: int = 50):
        """
        Apply SVD-based matrix factorization.

        Decomposes rating matrix R into:
        R ≈ U @ Σ @ V^T

        Where:
        - U: User latent factors (n_users x n_factors)
        - Σ: Singular values (n_factors)
        - V^T: Item latent factors (n_factors x n_books)
        """
        print(f"\nApplying SVD matrix factorization with {n_factors} factors...")

        svd = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_factors = svd.fit_transform(self.rating_matrix)
        self.item_factors = svd.components_.T

        explained_variance = np.sum(svd.explained_variance_ratio_)
        print(f"Explained variance: {explained_variance:.2%}")

        print(f"User factors shape: {self.user_factors.shape}")
        print(f"Item factors shape: {self.item_factors.shape}")

    def matrix_factorization_als(self, n_factors: int = 50, n_iterations: int = 10,
                                 reg_lambda: float = 0.1):
        """
        Apply Alternating Least Squares (ALS) matrix factorization.

        ALS is better than SVD for sparse matrices with missing values.
        Iteratively optimizes user and item factors.

        Args:
            n_factors: Number of latent factors
            n_iterations: Number of ALS iterations
            reg_lambda: Regularization parameter
        """
        print(f"\nApplying ALS matrix factorization with {n_factors} factors...")

        n_users, n_books = self.rating_matrix.shape

        # Initialize factors randomly
        self.user_factors = np.random.rand(n_users, n_factors) * 0.1
        self.item_factors = np.random.rand(n_books, n_factors) * 0.1

        # Create mask for observed ratings
        mask = self.rating_matrix > 0

        for iteration in range(n_iterations):
            # Fix item factors, update user factors
            for u in range(n_users):
                # Get books rated by this user
                rated_items = np.where(mask[u])[0]
                if len(rated_items) == 0:
                    continue

                # Solve for user factor
                A = self.item_factors[rated_items]
                b = self.rating_matrix[u, rated_items]

                # Regularized least squares: (A^T A + λI)^-1 A^T b
                AtA = A.T @ A + reg_lambda * np.eye(n_factors)
                Atb = A.T @ b
                self.user_factors[u] = np.linalg.solve(AtA, Atb)

            # Fix user factors, update item factors
            for i in range(n_books):
                # Get users who rated this book
                rating_users = np.where(mask[:, i])[0]
                if len(rating_users) == 0:
                    continue

                # Solve for item factor
                A = self.user_factors[rating_users]
                b = self.rating_matrix[rating_users, i]

                # Regularized least squares
                AtA = A.T @ A + reg_lambda * np.eye(n_factors)
                Atb = A.T @ b
                self.item_factors[i] = np.linalg.solve(AtA, Atb)

            # Calculate RMSE on observed ratings
            predictions = self.user_factors @ self.item_factors.T
            errors = (self.rating_matrix - predictions)[mask]
            rmse = np.sqrt(np.mean(errors ** 2))

            if (iteration + 1) % 2 == 0:
                print(f"  Iteration {iteration + 1}/{n_iterations}, RMSE: {rmse:.4f}")

        print("ALS optimization complete")

    def predict_rating(self, user_id: str, book_title: str) -> float:
        """
        Predict rating for a user-book pair.

        Args:
            user_id: User identifier
            book_title: Book title

        Returns:
            Predicted rating (0-5 scale typically)
        """
        if user_id not in self.user_ids or book_title not in self.book_titles:
            return 0.0

        user_idx = self.user_ids.index(user_id)
        book_idx = self.book_titles.index(book_title)

        # Predict: user_factor @ item_factor
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[book_idx])

        # Clip to valid rating range
        return np.clip(prediction, 0, 5)

    def recommend_for_user(self, user_id: str, n_recommendations: int = 10,
                          exclude_rated: bool = True) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a specific user.

        Args:
            user_id: User identifier
            n_recommendations: Number of books to recommend
            exclude_rated: If True, exclude books already rated by user

        Returns:
            List of (book_title, predicted_rating) tuples
        """
        if user_id not in self.user_ids:
            print(f"User {user_id} not found")
            return []

        user_idx = self.user_ids.index(user_id)

        # Predict ratings for all books
        predictions = self.user_factors[user_idx] @ self.item_factors.T

        # Exclude already rated books if requested
        if exclude_rated:
            rated_books = self.rating_matrix[user_idx] > 0
            predictions[rated_books] = -np.inf

        # Get top N
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]

        recommendations = [
            (self.book_titles[idx], predictions[idx])
            for idx in top_indices
        ]

        return recommendations

    def find_similar_books(self, book_title: str, n_similar: int = 5) -> List[Tuple[str, float]]:
        """
        Find books with similar latent factors.

        Args:
            book_title: Query book
            n_similar: Number of similar books to return

        Returns:
            List of (book_title, similarity_score) tuples
        """
        if book_title not in self.book_titles:
            print(f"Book '{book_title}' not found")
            return []

        book_idx = self.book_titles.index(book_title)

        # Calculate cosine similarity with all books
        query_factor = self.item_factors[book_idx]
        similarities = np.dot(self.item_factors, query_factor) / (
            np.linalg.norm(self.item_factors, axis=1) * np.linalg.norm(query_factor)
        )

        # Get top N (excluding query book)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]

        return [
            (self.book_titles[idx], similarities[idx])
            for idx in similar_indices
        ]


class HybridRecommender:
    """
    Hybrid recommender combining:
    - Collaborative filtering (user preferences)
    - Content-based filtering (emotional arcs)

    Provides better recommendations by leveraging both approaches.
    """

    def __init__(self, content_recommender, collaborative_recommender):
        """
        Initialize hybrid recommender.

        Args:
            content_recommender: EmotionalArcRecommender instance
            collaborative_recommender: CollaborativeFilteringRecommender instance
        """
        self.content_rec = content_recommender
        self.collab_rec = collaborative_recommender

    def hybrid_recommend(self, user_id: str, n_recommendations: int = 10,
                        alpha: float = 0.5) -> List[Tuple[str, float]]:
        """
        Generate hybrid recommendations.

        Args:
            user_id: User identifier
            n_recommendations: Number of recommendations
            alpha: Weight for collaborative filtering (1-alpha for content-based)

        Returns:
            List of (book_title, combined_score) tuples
        """
        print(f"\nGenerating hybrid recommendations (alpha={alpha})...")

        # Get collaborative filtering predictions
        collab_scores = {}
        if self.collab_rec.user_factors is not None:
            for book in self.collab_rec.book_titles:
                score = self.collab_rec.predict_rating(user_id, book)
                collab_scores[book] = score

        # Get content-based scores for books user has liked
        # (This is simplified - in practice, build user profile from rated books)
        content_scores = {}

        # Combine scores
        combined_scores = {}
        all_books = set(collab_scores.keys()) | set(content_scores.keys())

        for book in all_books:
            collab_score = collab_scores.get(book, 0.0)
            content_score = content_scores.get(book, 0.0)

            combined_scores[book] = alpha * collab_score + (1 - alpha) * content_score

        # Sort and return top N
        sorted_books = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_books[:n_recommendations]


def generate_sample_ratings(n_users: int = 50, n_ratings: int = 500,
                           books_file: str = 'outputs/data/gutenberg_sample_analysis.json',
                           output_file: str = 'outputs/data/sample_ratings.json'):
    """
    Generate synthetic rating data for testing collaborative filtering.

    Args:
        n_users: Number of synthetic users
        n_ratings: Number of ratings to generate
        books_file: JSON file with book analysis
        output_file: Where to save ratings
    """
    print("Generating sample rating data for demonstration...")

    # Load books
    with open(books_file, 'r') as f:
        books = json.load(f)

    book_titles = [book['title'] for book in books]

    # Generate random ratings
    ratings = []
    for _ in range(n_ratings):
        user_id = f"user_{np.random.randint(1, n_users + 1)}"
        book_title = np.random.choice(book_titles)
        rating = np.random.randint(1, 6)  # 1-5 stars

        ratings.append({
            'user_id': user_id,
            'book_title': book_title,
            'rating': float(rating)
        })

    # Save ratings
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(ratings, f, indent=2)

    print(f"Generated {len(ratings)} ratings and saved to {output_file}")


if __name__ == "__main__":
    print("="*80)
    print("COLLABORATIVE FILTERING RECOMMENDER")
    print("="*80)

    # Check if we have book analysis data
    books_file = 'outputs/data/gutenberg_sample_analysis.json'
    if Path(books_file).exists():
        # Generate sample ratings
        generate_sample_ratings()

        # Demo collaborative filtering
        print("\n" + "="*80)
        print("DEMO: Matrix Factorization Collaborative Filtering")
        print("="*80)

        recommender = CollaborativeFilteringRecommender('outputs/data/sample_ratings.json')

        if recommender.load_ratings():
            recommender.build_rating_matrix()

            # Try both SVD and ALS
            print("\n--- Using SVD ---")
            recommender.matrix_factorization_svd(n_factors=20)

            # Show recommendations for a sample user
            sample_user = recommender.user_ids[0]
            print(f"\nRecommendations for {sample_user}:")
            recs = recommender.recommend_for_user(sample_user, n_recommendations=5)
            for i, (title, score) in enumerate(recs, 1):
                print(f"  {i}. {title} (predicted rating: {score:.2f})")

            print("\n--- Using ALS ---")
            recommender.matrix_factorization_als(n_factors=20, n_iterations=10)

            # Show recommendations
            print(f"\nRecommendations for {sample_user} (ALS):")
            recs = recommender.recommend_for_user(sample_user, n_recommendations=5)
            for i, (title, score) in enumerate(recs, 1):
                print(f"  {i}. {title} (predicted rating: {score:.2f})")

    else:
        print(f"\nBook analysis data not found at {books_file}")
        print("Please run analyze_gutenberg_sample.py first.")
