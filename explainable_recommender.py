#!/usr/bin/env python3
"""
Explainable Book Recommendation System

Provides transparent, human-readable explanations for recommendations:
- Why books were recommended
- Which emotions/features matched
- Visual comparisons of emotional arcs
- Feature contribution analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from book_recommender import EmotionalArcRecommender
from sklearn.metrics.pairwise import cosine_similarity


class ExplainableRecommender(EmotionalArcRecommender):
    """
    Extension of EmotionalArcRecommender with explainability features.

    Provides detailed explanations for:
    - Why books are similar
    - Which emotions contribute most
    - How emotional arcs compare visually
    - What latent factors mean
    """

    def explain_recommendation(self, query_book: str, recommended_book: str,
                               use_latent: bool = False) -> Dict:
        """
        Generate detailed explanation for why a book was recommended.

        Args:
            query_book: Book user is interested in
            recommended_book: Book that was recommended
            use_latent: Whether latent factors were used

        Returns:
            Dictionary with explanation components
        """
        # Find book indices
        if query_book not in self.book_titles:
            return {"error": f"Query book '{query_book}' not found"}
        if recommended_book not in self.book_titles:
            return {"error": f"Recommended book '{recommended_book}' not found"}

        query_idx = self.book_titles.index(query_book)
        rec_idx = self.book_titles.index(recommended_book)

        # Get book data
        query_data = self.books[query_idx]
        rec_data = self.books[rec_idx]

        explanation = {
            'query_book': query_book,
            'recommended_book': recommended_book,
            'similarity_score': 0.0,
            'emotion_comparison': {},
            'arc_similarity': {},
            'matching_segments': [],
            'dominant_emotions': {},
            'explanation_text': ""
        }

        # Choose feature space
        if use_latent and self.latent_factors is not None:
            features = self.latent_factors
            feature_space = "latent factors"
        else:
            features = self.emotion_matrix
            feature_space = "emotional arc features"

        # Calculate similarity
        similarity = cosine_similarity([features[query_idx]], [features[rec_idx]])[0][0]
        explanation['similarity_score'] = float(similarity)

        # Compare average emotions
        emotion_comparison = {}
        for emotion in ['anger', 'anticipation', 'disgust', 'fear',
                       'joy', 'sadness', 'surprise', 'trust']:
            query_val = query_data['avg_emotions'][emotion]
            rec_val = rec_data['avg_emotions'][emotion]
            difference = abs(query_val - rec_val)

            emotion_comparison[emotion] = {
                'query': float(query_val),
                'recommended': float(rec_val),
                'difference': float(difference),
                'match_quality': 'excellent' if difference < 0.05 else
                                'good' if difference < 0.10 else
                                'moderate' if difference < 0.15 else 'different'
            }

        explanation['emotion_comparison'] = emotion_comparison

        # Find most similar segments
        query_segments = query_data['segment_emotions']
        rec_segments = rec_data['segment_emotions']

        matching_segments = []
        for i in range(min(len(query_segments), len(rec_segments))):
            query_vec = np.array([query_segments[i][e] for e in
                                 ['anger', 'anticipation', 'disgust', 'fear',
                                  'joy', 'sadness', 'surprise', 'trust']])
            rec_vec = np.array([rec_segments[i][e] for e in
                               ['anger', 'anticipation', 'disgust', 'fear',
                                'joy', 'sadness', 'surprise', 'trust']])

            segment_similarity = cosine_similarity([query_vec], [rec_vec])[0][0]

            if segment_similarity > 0.85:  # High similarity threshold
                matching_segments.append({
                    'segment_number': i + 1,
                    'position_percent': int((i / len(query_segments)) * 100),
                    'similarity': float(segment_similarity)
                })

        explanation['matching_segments'] = matching_segments

        # Identify dominant emotions
        query_dominant = max(query_data['avg_emotions'].items(), key=lambda x: x[1])
        rec_dominant = max(rec_data['avg_emotions'].items(), key=lambda x: x[1])

        explanation['dominant_emotions'] = {
            'query': {'emotion': query_dominant[0], 'score': float(query_dominant[1])},
            'recommended': {'emotion': rec_dominant[0], 'score': float(rec_dominant[1])}
        }

        # Generate human-readable explanation
        explanation_text = self._generate_explanation_text(explanation, feature_space)
        explanation['explanation_text'] = explanation_text

        return explanation

    def _generate_explanation_text(self, explanation: Dict, feature_space: str) -> str:
        """Generate human-readable explanation text"""
        lines = []

        # Overall similarity
        sim_score = explanation['similarity_score']
        sim_percent = int(sim_score * 100)

        lines.append(f"📚 Why we recommended '{explanation['recommended_book']}':")
        lines.append(f"\n✨ Overall Match: {sim_percent}% similar (using {feature_space})")

        # Emotion matches
        lines.append("\n🎭 Emotional Profile Comparison:")

        excellent_matches = []
        good_matches = []
        differences = []

        for emotion, data in explanation['emotion_comparison'].items():
            quality = data['match_quality']
            if quality == 'excellent':
                excellent_matches.append(emotion)
            elif quality == 'good':
                good_matches.append(emotion)
            elif quality == 'different':
                differences.append((emotion, data['difference']))

        if excellent_matches:
            lines.append(f"  ✓ Excellent match: {', '.join(excellent_matches)}")
        if good_matches:
            lines.append(f"  ✓ Good match: {', '.join(good_matches)}")
        if differences:
            diff_emotions = [f"{e} (Δ{d:.2f})" for e, d in differences[:2]]
            if diff_emotions:
                lines.append(f"  ⚠ Different: {', '.join(diff_emotions)}")

        # Dominant emotions
        query_dom = explanation['dominant_emotions']['query']
        rec_dom = explanation['dominant_emotions']['recommended']

        lines.append(f"\n💫 Emotional Tone:")
        lines.append(f"  Your choice emphasizes: {query_dom['emotion']} ({query_dom['score']:.2f})")
        lines.append(f"  This recommendation: {rec_dom['emotion']} ({rec_dom['score']:.2f})")

        # Matching segments
        if explanation['matching_segments']:
            lines.append(f"\n📖 Story Arc Similarity:")
            lines.append(f"  Found {len(explanation['matching_segments'])} highly similar segments")

            # Show where matches occur
            positions = [s['position_percent'] for s in explanation['matching_segments']]
            if positions:
                lines.append(f"  Similar at: {', '.join([f'{p}%' for p in positions[:5]])}")

        return '\n'.join(lines)

    def visualize_comparison(self, query_book: str, recommended_books: List[str],
                            output_file: str = 'outputs/visualizations/recommendation_explanation.png'):
        """
        Visualize emotional arc comparison between query and recommended books.

        Creates a multi-panel visualization showing:
        1. Emotional arcs for each book
        2. Emotion-by-emotion comparison
        3. Similarity heatmap
        """
        print(f"\nGenerating visual explanation for recommendations...")

        # Find book indices
        if query_book not in self.book_titles:
            print(f"Query book '{query_book}' not found")
            return

        query_idx = self.book_titles.index(query_book)
        query_data = self.books[query_idx]

        # Get recommended book data
        rec_data_list = []
        for rec_book in recommended_books:
            if rec_book in self.book_titles:
                idx = self.book_titles.index(rec_book)
                rec_data_list.append(self.books[idx])

        if not rec_data_list:
            print("No valid recommended books found")
            return

        # Create visualization
        n_recs = len(rec_data_list)
        fig = plt.figure(figsize=(20, 4 + 3 * n_recs))

        # Layout: Top row for query, then one row per recommendation
        gs = fig.add_gridspec(n_recs + 2, 3, hspace=0.4, wspace=0.3)

        # Title
        fig.suptitle(f'Recommendation Explanation: Why These Books Match\n"{query_book}"',
                    fontsize=16, fontweight='bold', y=0.98)

        # Plot query book arc
        ax_query = fig.add_subplot(gs[0, :])
        self._plot_emotional_arc(ax_query, query_data, title_prefix="Your Interest: ")

        # Plot each recommendation with comparison
        for i, rec_data in enumerate(rec_data_list):
            row = i + 1

            # Emotional arc
            ax_arc = fig.add_subplot(gs[row, 0:2])
            self._plot_emotional_arc(ax_arc, rec_data, title_prefix=f"Recommendation #{i+1}: ")

            # Emotion comparison bar chart
            ax_compare = fig.add_subplot(gs[row, 2])
            self._plot_emotion_comparison(ax_compare, query_data, rec_data)

        # Similarity matrix at bottom
        ax_matrix = fig.add_subplot(gs[-1, :])
        self._plot_similarity_matrix(ax_matrix, query_data, rec_data_list)

        # Save
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visual explanation saved to {output_file}")
        plt.close()

    def _plot_emotional_arc(self, ax, book_data: Dict, title_prefix: str = ""):
        """Plot emotional arc for a single book"""
        segments = book_data['segment_emotions']
        n_segments = len(segments)
        x_values = np.linspace(0, 100, n_segments)

        emotions = ['joy', 'sadness', 'fear', 'anger', 'trust', 'anticipation']
        colors = {
            'joy': '#FFD700',      # Gold
            'sadness': '#4169E1',  # Blue
            'fear': '#9370DB',     # Purple
            'anger': '#DC143C',    # Red
            'trust': '#32CD32',    # Green
            'anticipation': '#FF8C00'  # Orange
        }

        for emotion in emotions:
            values = [seg[emotion] for seg in segments]
            ax.plot(x_values, values, label=emotion.capitalize(),
                   linewidth=2.5, marker='o', markersize=4,
                   color=colors[emotion], alpha=0.8)

        title = book_data['title']
        if len(title) > 60:
            title = title[:60] + '...'

        ax.set_title(f"{title_prefix}{title}", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Story Progress (%)', fontsize=9)
        ax.set_ylabel('Emotion Score', fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, None)
        ax.legend(loc='upper right', fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)

    def _plot_emotion_comparison(self, ax, query_data: Dict, rec_data: Dict):
        """Plot bar chart comparing average emotions"""
        emotions = ['joy', 'sadness', 'fear', 'anger', 'trust', 'anticipation']

        query_vals = [query_data['avg_emotions'][e] for e in emotions]
        rec_vals = [rec_data['avg_emotions'][e] for e in emotions]

        x = np.arange(len(emotions))
        width = 0.35

        ax.barh(x - width/2, query_vals, width, label='Your Choice',
               color='steelblue', alpha=0.8)
        ax.barh(x + width/2, rec_vals, width, label='Recommendation',
               color='coral', alpha=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels(emotions)
        ax.set_xlabel('Avg Score', fontsize=9)
        ax.set_title('Emotion Comparison', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')

    def _plot_similarity_matrix(self, ax, query_data: Dict, rec_data_list: List[Dict]):
        """Plot heatmap showing segment-by-segment similarity"""
        n_segments = len(query_data['segment_emotions'])
        n_recs = len(rec_data_list)

        # Build similarity matrix
        similarity_matrix = np.zeros((n_recs, n_segments))

        for i, rec_data in enumerate(rec_data_list):
            for j in range(min(n_segments, len(rec_data['segment_emotions']))):
                query_vec = np.array([query_data['segment_emotions'][j][e] for e in
                                     ['anger', 'anticipation', 'disgust', 'fear',
                                      'joy', 'sadness', 'surprise', 'trust']])
                rec_vec = np.array([rec_data['segment_emotions'][j][e] for e in
                                   ['anger', 'anticipation', 'disgust', 'fear',
                                    'joy', 'sadness', 'surprise', 'trust']])

                similarity_matrix[i, j] = cosine_similarity([query_vec], [rec_vec])[0][0]

        # Plot heatmap
        im = ax.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto', vmin=0.5, vmax=1.0)

        # Labels
        ax.set_xlabel('Story Segment (Beginning → End)', fontsize=10)
        ax.set_ylabel('Recommendation', fontsize=10)
        ax.set_title('Emotional Arc Similarity by Segment', fontsize=11, fontweight='bold')

        # Tick labels
        ax.set_yticks(range(n_recs))
        ax.set_yticklabels([f"Rec #{i+1}" for i in range(n_recs)])

        segment_labels = ['Start'] + ['' for _ in range(n_segments-2)] + ['End']
        ax.set_xticks(range(n_segments))
        ax.set_xticklabels(segment_labels)

        # Colorbar
        plt.colorbar(im, ax=ax, label='Similarity')

        # Add text annotations for high similarity
        for i in range(n_recs):
            for j in range(n_segments):
                if similarity_matrix[i, j] > 0.9:
                    ax.text(j, i, '★', ha='center', va='center',
                           color='white', fontsize=16, fontweight='bold')

    def explain_latent_factors(self, n_top_features: int = 10) -> Dict[int, Dict]:
        """
        Explain what each latent factor represents.

        Args:
            n_top_features: Number of top contributing features to show per factor

        Returns:
            Dictionary mapping factor index to interpretation
        """
        if self.latent_factors is None or not hasattr(self, 'factorization_model'):
            print("No latent factors available. Run apply_matrix_factorization() first.")
            return {}

        print("\n" + "="*80)
        print("LATENT FACTOR INTERPRETATION")
        print("="*80)

        factors_interpretation = {}

        # For each latent factor, find top contributing features
        components = self.factorization_model.components_

        for factor_idx in range(min(5, components.shape[0])):  # Top 5 factors
            weights = components[factor_idx]

            # Get top positive and negative contributors
            top_positive = np.argsort(weights)[-n_top_features:][::-1]
            top_negative = np.argsort(weights)[:n_top_features]

            interpretation = {
                'factor_number': factor_idx + 1,
                'variance_explained': float(self.factorization_model.explained_variance_ratio_[factor_idx]),
                'top_positive_features': [],
                'top_negative_features': [],
                'interpretation': ""
            }

            # Map feature indices to emotion names
            # Assuming segment_vectors: [joy_seg1, joy_seg2, ..., sadness_seg1, ...]
            emotions = ['anger', 'anticipation', 'disgust', 'fear',
                       'joy', 'sadness', 'surprise', 'trust']

            n_segments = 10  # Assuming 10 segments

            for feat_idx in top_positive:
                emotion_idx = feat_idx // n_segments
                segment_idx = feat_idx % n_segments
                emotion_name = emotions[emotion_idx] if emotion_idx < len(emotions) else 'unknown'

                interpretation['top_positive_features'].append({
                    'feature': f"{emotion_name}_segment_{segment_idx+1}",
                    'weight': float(weights[feat_idx]),
                    'position': f"{int((segment_idx/n_segments)*100)}%"
                })

            # Generate human-readable interpretation
            interpretation['interpretation'] = self._interpret_factor(interpretation)

            factors_interpretation[factor_idx] = interpretation

        return factors_interpretation

    def _interpret_factor(self, factor_data: Dict) -> str:
        """Generate human-readable interpretation of a latent factor"""
        lines = []

        variance = factor_data['variance_explained'] * 100
        lines.append(f"Latent Factor #{factor_data['factor_number']} (explains {variance:.1f}% of variance)")

        # Analyze top features
        emotions_mentioned = {}
        for feat in factor_data['top_positive_features'][:5]:
            emotion = feat['feature'].split('_')[0]
            emotions_mentioned[emotion] = emotions_mentioned.get(emotion, 0) + 1

        dominant_emotion = max(emotions_mentioned.items(), key=lambda x: x[1])[0] if emotions_mentioned else "mixed"

        # Check if early or late in story
        early_features = sum(1 for f in factor_data['top_positive_features'][:5]
                            if 'segment_1' in f['feature'] or 'segment_2' in f['feature'])
        late_features = sum(1 for f in factor_data['top_positive_features'][:5]
                           if 'segment_9' in f['feature'] or 'segment_10' in f['feature'])

        if early_features > late_features:
            timing = "beginning"
        elif late_features > early_features:
            timing = "ending"
        else:
            timing = "throughout"

        lines.append(f"  This factor captures: {dominant_emotion} {timing}")
        lines.append(f"  Interpretation: Stories with strong {dominant_emotion} {timing}")

        return '\n'.join(lines)

    def get_recommendation_with_explanation(self, book_title: str,
                                           n_recommendations: int = 3,
                                           use_latent: bool = True) -> List[Dict]:
        """
        Get recommendations with full explanations.

        Args:
            book_title: Query book
            n_recommendations: Number of recommendations
            use_latent: Use latent factors

        Returns:
            List of dictionaries with recommendations and explanations
        """
        # Get base recommendations
        recommendations = self.get_similar_books(book_title, n_recommendations, use_latent)

        if not recommendations:
            return []

        # Add explanations
        results = []
        for rec_title, score in recommendations:
            explanation = self.explain_recommendation(book_title, rec_title, use_latent)
            results.append({
                'book': rec_title,
                'similarity_score': score,
                'explanation': explanation
            })

        return results


def demo_explainable_recommender():
    """Demonstrate explainable recommendations"""
    print("="*80)
    print("EXPLAINABLE BOOK RECOMMENDATION SYSTEM - DEMO")
    print("="*80)

    # Initialize
    recommender = ExplainableRecommender()

    try:
        # Load data
        recommender.load_data()
        recommender.build_emotion_matrix(method='segment_vectors')
        recommender.apply_matrix_factorization(method='svd', n_components=20)

        # Example 1: Get recommendation with explanation
        print("\n" + "="*80)
        print("EXAMPLE 1: Recommendation with Detailed Explanation")
        print("="*80)

        query_book = recommender.book_titles[0]
        print(f"\nQuery: '{query_book}'")

        results = recommender.get_recommendation_with_explanation(
            query_book,
            n_recommendations=3,
            use_latent=True
        )

        for i, result in enumerate(results, 1):
            print(f"\n--- Recommendation #{i} ---")
            print(f"Book: {result['book']}")
            print(f"Similarity: {result['similarity_score']:.3f}")
            print(f"\n{result['explanation']['explanation_text']}")

        # Example 2: Visual explanation
        print("\n" + "="*80)
        print("EXAMPLE 2: Visual Comparison")
        print("="*80)

        recommended_books = [r['book'] for r in results]
        recommender.visualize_comparison(
            query_book,
            recommended_books[:2],  # Show top 2
            output_file='outputs/visualizations/recommendation_explanation.png'
        )

        # Example 3: Latent factor interpretation
        print("\n" + "="*80)
        print("EXAMPLE 3: Latent Factor Interpretation")
        print("="*80)

        factors = recommender.explain_latent_factors(n_top_features=5)

        for factor_idx, interpretation in factors.items():
            print(f"\n{interpretation['interpretation']}")
            print(f"\n  Top Features:")
            for feat in interpretation['top_positive_features'][:3]:
                print(f"    • {feat['feature']}: {feat['weight']:.3f} (at {feat['position']})")

        print("\n" + "="*80)
        print("EXPLAINABLE RECOMMENDATION DEMO COMPLETE")
        print("="*80)

        return recommender

    except FileNotFoundError:
        print(f"\nError: Analysis results not found")
        print("Please run analyze_gutenberg_sample.py first")
        return None


if __name__ == "__main__":
    recommender = demo_explainable_recommender()
