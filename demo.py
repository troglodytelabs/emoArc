"""
Demo script for class presentation - analyzes text/book and provides recommendations.

Workflow:
1. Accepts either a text file or book ID
2. Gets trajectory for the input (from main.py output or by processing)
3. Compares against main.py output for recommendations
4. Generates visualizations
"""

import sys
import os
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from core import (
    create_spark_session,
    load_trajectories,
    get_input_trajectory,
)
from recommender import recommend


def plot_topic_distribution(book_topics_pd, book_title, output_path, num_topics=10):
    """Plot topic distribution for a book."""
    if book_topics_pd is None or len(book_topics_pd) == 0:
        return False

    topics = book_topics_pd.iloc[0]["book_topics"]
    if topics is None:
        return False

    fig, ax = plt.subplots(figsize=(12, 6))

    topic_labels = [f"Topic {i + 1}" for i in range(len(topics))]
    bars = ax.bar(
        topic_labels, topics, color="steelblue", edgecolor="navy", linewidth=1
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Topic", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(f"Topic Distribution: {book_title}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved topic plot to {output_path}")
    return True


def plot_emotion_trajectory(chunk_scores_pd, book_title, output_path):
    """Plot emotion trajectory for a book."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Emotion scores over chunks
    # Use all 8 Plutchik emotions
    emotions = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
    ]
    # Distinct colors for each emotion
    colors = [
        "red",  # anger
        "orange",  # anticipation
        "brown",  # disgust
        "black",  # fear
        "gold",  # joy
        "blue",  # sadness
        "purple",  # surprise
        "green",  # trust
    ]

    has_data = False
    for emotion, color in zip(emotions, colors):
        if emotion in chunk_scores_pd.columns:
            # Check if there's any non-zero data to plot
            if chunk_scores_pd[emotion].any():
                has_data = True

            axes[0].plot(
                chunk_scores_pd["chunk_index"],
                chunk_scores_pd[emotion],
                label=emotion.capitalize(),
                color=color,
                linewidth=2,
                alpha=0.7,
                marker="o",
                markersize=4,
            )

    axes[0].set_xlabel("Chunk Index", fontsize=12)
    axes[0].set_ylabel("Emotion Score", fontsize=12)
    axes[0].set_title(
        f"Emotion Trajectory: {book_title}", fontsize=14, fontweight="bold"
    )
    if has_data:
        axes[0].legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    else:
        axes[0].text(
            0.5,
            0.5,
            "No Emotion Data Available",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
    axes[0].grid(True, alpha=0.3)

    # Plot 2: VAD scores
    vad_has_data = False
    if "avg_valence" in chunk_scores_pd.columns:
        if chunk_scores_pd["avg_valence"].any():
            vad_has_data = True
        axes[1].plot(
            chunk_scores_pd["chunk_index"],
            chunk_scores_pd["avg_valence"],
            label="Valence",
            color="purple",
            linewidth=2,
            linestyle="-",
            marker="o",
            markersize=4,
        )
    if "avg_arousal" in chunk_scores_pd.columns:
        if chunk_scores_pd["avg_arousal"].any():
            vad_has_data = True
        axes[1].plot(
            chunk_scores_pd["chunk_index"],
            chunk_scores_pd["avg_arousal"],
            label="Arousal",
            color="orange",
            linewidth=2,
            linestyle="--",
            marker="o",
            markersize=4,
        )
    if "avg_dominance" in chunk_scores_pd.columns:
        if chunk_scores_pd["avg_dominance"].any():
            vad_has_data = True
        axes[1].plot(
            chunk_scores_pd["chunk_index"],
            chunk_scores_pd["avg_dominance"],
            label="Dominance",
            color="green",
            linewidth=2,
            linestyle=":",
            marker="o",
            markersize=4,
        )

    axes[1].set_xlabel("Chunk Index", fontsize=12)
    axes[1].set_ylabel("VAD Score", fontsize=12)
    axes[1].set_title(
        "Valence-Arousal-Dominance Trajectory", fontsize=14, fontweight="bold"
    )
    if vad_has_data:
        axes[1].legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    else:
        axes[1].text(
            0.5,
            0.5,
            "No VAD Data Available",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved plot to {output_path}")


def demo_analysis(
    spark,
    book_id=None,
    text_file=None,
    output_dir="output",
    books_dir="data/books",
    metadata_path="data/gutenberg_metadata.csv",
    emotion_lexicon="data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    vad_lexicon="data/NRC-VAD-Lexicon-v2.1.txt",
    compute_topics=False,
    num_topics=10,
):
    """Analyze input and create visualizations."""
    print(f"\n{'=' * 80}")
    print("Emotion Trajectory Analysis")
    print(f"{'=' * 80}")

    # Get trajectory and chunk scores
    try:
        trajectory, chunk_scores, title, author, book_topics = get_input_trajectory(
            spark,
            book_id=book_id,
            text_file=text_file,
            output_dir=output_dir,
            books_dir=books_dir,
            metadata_path=metadata_path,
            emotion_lexicon=emotion_lexicon,
            vad_lexicon=vad_lexicon,
            compute_topics=compute_topics,
            num_topics=num_topics,
        )
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return

    if trajectory is None or chunk_scores is None:
        return

    print(f"\n  Title: {title}")
    print(f"  Author: {author}")

    # Convert to pandas for plotting
    chunk_scores_pd = chunk_scores.orderBy("chunk_index").toPandas()

    # Create plot
    plot_id = (
        book_id
        if book_id
        else os.path.basename(text_file).replace(".txt", "")
        if text_file
        else "custom"
    )
    plot_id = str(plot_id).replace("/", "_").replace("\\", "_")
    os.makedirs("demo_output", exist_ok=True)

    output_path = f"demo_output/{plot_id}_trajectory.png"
    plot_emotion_trajectory(chunk_scores_pd, title, output_path)

    # Plot topic distribution if available
    if book_topics is not None:
        print("\n  Computing topic distribution...")
        book_topics_pd = book_topics.toPandas()
        topic_output_path = f"demo_output/{plot_id}_topics.png"
        if plot_topic_distribution(
            book_topics_pd, title, topic_output_path, num_topics
        ):
            topics = book_topics_pd.iloc[0]["book_topics"]
            if topics:
                # Get top 5 topics
                topic_scores = [(i, topics[i]) for i in range(len(topics))]
                topic_scores.sort(key=lambda x: x[1], reverse=True)

                print("\n  Top Topics:")
                for rank, (topic_idx, score) in enumerate(topic_scores[:5], 1):
                    print(f"    {rank}. Topic {topic_idx + 1}: {score:.4f}")

    # Print statistics
    print("\n  Emotion Statistics:")
    print(f"    Average Joy: {chunk_scores_pd['joy'].mean():.4f}")
    print(f"    Average Sadness: {chunk_scores_pd['sadness'].mean():.4f}")
    print(f"    Average Fear: {chunk_scores_pd['fear'].mean():.4f}")
    print(f"    Average Anger: {chunk_scores_pd['anger'].mean():.4f}")
    print(f"    Average Valence: {chunk_scores_pd['avg_valence'].mean():.4f}")
    print(f"    Average Arousal: {chunk_scores_pd['avg_arousal'].mean():.4f}")


def demo_recommendations(
    spark,
    book_id=None,
    text_file=None,
    output_dir="output",
    limit=None,
    books_dir="data/books",
    metadata_path="data/gutenberg_metadata.csv",
    emotion_lexicon="data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    vad_lexicon="data/NRC-VAD-Lexicon-v2.1.txt",
):
    """Get recommendations based on input."""
    print(f"\n{'=' * 80}")
    print("Recommendation System")
    print(f"{'=' * 80}")

    # Get trajectory for input
    try:
        liked_trajectory, _, title, author, _ = get_input_trajectory(
            spark,
            book_id=book_id,
            text_file=text_file,
            output_dir=output_dir,
            books_dir=books_dir,
            metadata_path=metadata_path,
            emotion_lexicon=emotion_lexicon,
            vad_lexicon=vad_lexicon,
            compute_topics=False,  # Don't need topics for recommendations
        )
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return

    if liked_trajectory is None:
        return

    print(f"  Input: {title} by {author}")

    # Load trajectories from main.py output for comparison (required)
    trajectories_path = f"{output_dir}/trajectories"

    if not os.path.exists(trajectories_path):
        print(f"  ❌ Error: No trajectories found in {output_dir}")
        print(
            "  Please run 'python main.py' first to generate trajectories for comparison."
        )
        print(f"  Example: python main.py --limit 100 --output {output_dir}")
        return

    try:
        print("  Loading trajectories from main.py output for comparison...")
        trajectories = load_trajectories(spark, output_dir)
        if trajectories is None:
            print(f"  ❌ Error: Could not load trajectories from {trajectories_path}")
            return

        # Log what features are available
        has_embeddings = "book_embedding" in trajectories.columns
        has_topics = "book_topics" in trajectories.columns
        has_emotion_traj = "emotion_trajectory" in trajectories.columns
        features = []
        if has_embeddings:
            features.append("embeddings")
        if has_topics:
            features.append("topics")
        if has_emotion_traj:
            features.append("emotion trajectories")
        if features:
            print(f"  ✓ Loaded trajectories with: {', '.join(features)}")

        # Filter to limit if specified
        total_count = trajectories.count()
        if limit and total_count > limit:
            print(f"  Limiting to {limit} books for comparison...")
            trajectories = trajectories.limit(limit)

        print(f"  ✓ Loaded {trajectories.count()} trajectories for comparison")
    except Exception as e:
        print(f"  ❌ Error loading trajectories from main.py output: {e}")
        print("  Please run 'python main.py' first to generate trajectories.")
        return

    # Get recommendations
    print("  Computing recommendations...")
    liked_id = liked_trajectory.select("book_id").first()["book_id"]

    # Combine liked trajectory with trajectories from main.py output
    # Use unionByName to handle schema differences (e.g., if CSV was created with old schema)
    # This allows union even if columns differ, filling missing columns with null
    try:
        all_trajectories = trajectories.unionByName(
            liked_trajectory, allowMissingColumns=True
        )
    except Exception:
        # Fallback: if unionByName fails, align schemas manually
        trajectories_cols = set(trajectories.columns)
        liked_cols = set(liked_trajectory.columns)
        common_cols = sorted(list(trajectories_cols & liked_cols))

        trajectories_aligned = trajectories.select(*common_cols)
        liked_trajectory_aligned = liked_trajectory.select(*common_cols)
        all_trajectories = trajectories_aligned.union(liked_trajectory_aligned)

    # Get recommendations (combines feature-based + trajectory similarity when available)
    recommendations = recommend(spark, all_trajectories, liked_id, top_n=10)

    # Display recommendations
    print("\n  Top 10 Recommendations:")
    print(f"  {'-' * 80}")
    recs_pd = recommendations.toPandas()
    for idx, row in recs_pd.iterrows():
        print(f"  {idx + 1}. {row['title']}")
        print(f"     Author: {row['author']}")
        print(f"     Similarity: {row['similarity']:.4f}")
        # Display key emotions (show all 8 Plutchik emotions)
        emotions_str = (
            f"Anger: {row.get('avg_anger', 0):.2f}, "
            f"Anticipation: {row.get('avg_anticipation', 0):.2f}, "
            f"Disgust: {row.get('avg_disgust', 0):.2f}, "
            f"Fear: {row.get('avg_fear', 0):.2f}, "
            f"Joy: {row.get('avg_joy', 0):.2f}, "
            f"Sadness: {row.get('avg_sadness', 0):.2f}, "
            f"Surprise: {row.get('avg_surprise', 0):.2f}, "
            f"Trust: {row.get('avg_trust', 0):.2f}"
        )
        print(f"     Emotions - {emotions_str}")
        print()

    # Save to CSV
    os.makedirs("demo_output", exist_ok=True)

    # Use generic name or filename
    rec_id = str(liked_id).replace("/", "_").replace("\\", "_")
    if rec_id == "text_file" and text_file:
        rec_id = os.path.basename(text_file).replace(".txt", "").replace("_", " ")

    recs_pd.to_csv(f"demo_output/recommendations_for_{rec_id}.csv", index=False)
    print(f"  ✓ Saved recommendations to demo_output/recommendations_for_{rec_id}.csv")


def main():
    """Main demo function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="EmoArc Demo - Analyzes text/book and provides recommendations",
        epilog="For recommendations, main.py should be run first to generate trajectories.",
    )
    parser.add_argument("--book-id", type=str, help="Gutenberg book ID to analyze")
    parser.add_argument("--text-file", type=str, help="Path to text file to analyze")
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze input and create visualizations"
    )
    parser.add_argument(
        "--recommend", action="store_true", help="Get recommendations based on input"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory with output from main.py (default: output)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of books to consider for recommendations (optional)",
    )
    parser.add_argument(
        "--books-dir",
        default="data/books",
        help="Directory containing Gutenberg book files",
    )
    parser.add_argument(
        "--metadata", default="data/gutenberg_metadata.csv", help="Path to metadata CSV"
    )
    parser.add_argument(
        "--emotion-lexicon",
        default="data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        help="Path to NRC Emotion Lexicon",
    )
    parser.add_argument(
        "--vad-lexicon",
        default="data/NRC-VAD-Lexicon-v2.1.txt",
        help="Path to NRC VAD Lexicon",
    )
    parser.add_argument(
        "--compute-topics",
        action="store_true",
        help="Compute topic modeling (slower but provides topic analysis)",
    )
    parser.add_argument(
        "--num-topics",
        type=int,
        default=10,
        help="Number of topics for LDA (default: 10)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.book_id and not args.text_file:
        print("❌ Error: Please specify either --book-id or --text-file")
        print("\nExample usage:")
        print("  python demo.py --book-id 11 --analyze")
        print("  python demo.py --text-file story.txt --analyze")
        print("  python demo.py --book-id 11 --recommend")
        print("  python demo.py --text-file story.txt --recommend")
        return

    if not args.analyze and not args.recommend:
        print("❌ Error: Please specify --analyze or --recommend (or both)")
        return

    spark = create_spark_session()

    try:
        if args.analyze:
            demo_analysis(
                spark,
                book_id=args.book_id,
                text_file=args.text_file,
                output_dir=args.output_dir,
                books_dir=args.books_dir,
                metadata_path=args.metadata,
                emotion_lexicon=args.emotion_lexicon,
                vad_lexicon=args.vad_lexicon,
                compute_topics=args.compute_topics,
                num_topics=args.num_topics,
            )

        if args.recommend:
            demo_recommendations(
                spark,
                book_id=args.book_id,
                text_file=args.text_file,
                output_dir=args.output_dir,
                limit=args.limit,
                books_dir=args.books_dir,
                metadata_path=args.metadata,
                emotion_lexicon=args.emotion_lexicon,
                vad_lexicon=args.vad_lexicon,
            )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
