"""
Streamlit app for EmoArc - Emotion Trajectory Analysis and Recommendation System
"""

import sys
import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from lexicon_loader import load_emotion_lexicon, load_vad_lexicon
from text_preprocessor import create_chunks_df
from emotion_scorer import (
    score_chunks_with_emotions,
    score_chunks_with_vad,
    combine_emotion_vad_scores,
)
from trajectory_analyzer import analyze_trajectory
from recommender import recommend

# Page config
st.set_page_config(
    page_title="EmoArc - Emotion Trajectory Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "spark" not in st.session_state:
    st.session_state.spark = None
if "emotion_lexicon" not in st.session_state:
    st.session_state.emotion_lexicon = None
if "vad_lexicon" not in st.session_state:
    st.session_state.vad_lexicon = None
if "metadata_df" not in st.session_state:
    st.session_state.metadata_df = None


@st.cache_resource
def get_spark_session():
    """Create and cache Spark session."""
    spark = (
        SparkSession.builder.appName("EmoArc Streamlit")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_metadata(metadata_path="data/gutenberg_metadata.csv"):
    """Load metadata (not cached due to Spark DataFrame serialization issues)."""
    spark = get_spark_session()
    metadata_df = spark.read.option("header", "true").csv(metadata_path)
    # Filter English books only
    metadata_df = metadata_df.filter(col("Language") == "en")
    return metadata_df


@st.cache_resource
def load_lexicons(
    emotion_lexicon="data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    vad_lexicon="data/NRC-VAD-Lexicon-v2.1.txt",
):
    """Load and cache lexicons."""
    spark = get_spark_session()
    emotion_df = load_emotion_lexicon(spark, emotion_lexicon)
    vad_df = load_vad_lexicon(spark, vad_lexicon)
    return emotion_df, vad_df


def search_books_by_title(title_query, metadata_df, limit=20):
    """Search books by title."""
    if not title_query:
        return None

    # Case-insensitive search
    results = (
        metadata_df.filter(col("Title").like(f"%{title_query}%"))
        .select(
            col("Etext Number").alias("book_id"),
            col("Title").alias("title"),
            col("Authors").alias("author"),
        )
        .limit(limit)
    )

    return results


def load_trajectories_with_types(spark, trajectories_path):
    """Load trajectories CSV and cast columns to proper types."""
    df = spark.read.option("header", "true").csv(trajectories_path)

    numeric_cols = [
        "max_anger",
        "max_anticipation",
        "max_disgust",
        "max_fear",
        "max_joy",
        "max_sadness",
        "max_surprise",
        "max_trust",
        "avg_anger",
        "avg_anticipation",
        "avg_disgust",
        "avg_fear",
        "avg_joy",
        "avg_sadness",
        "avg_surprise",
        "avg_trust",
        "avg_valence",
        "avg_arousal",
        "avg_dominance",
        "valence_std",
        "arousal_std",
        "num_chunks",
    ]

    for col_name in numeric_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast("double"))

    return df


def load_chunk_scores_with_types(spark, chunk_scores_path):
    """Load chunk scores CSV and cast columns to proper types."""
    df = spark.read.option("header", "true").csv(chunk_scores_path)

    if "chunk_index" in df.columns:
        df = df.withColumn("chunk_index", col("chunk_index").cast("int"))

    emotion_cols = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "negative",
        "positive",
        "sadness",
        "surprise",
        "trust",
    ]

    for col_name in emotion_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast("double"))

    vad_cols = ["avg_valence", "avg_arousal", "avg_dominance", "vad_word_count"]
    for col_name in vad_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast("double"))

    return df


def get_input_trajectory(
    spark,
    book_id=None,
    text_file=None,
    output_dir="output",
    books_dir="data/books",
    metadata_path="data/gutenberg_metadata.csv",
    emotion_lexicon="data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    vad_lexicon="data/NRC-VAD-Lexicon-v2.1.txt",
):
    """
    Get trajectory for input (book ID or text file).

    Returns:
        tuple: (trajectory_df, chunk_scores_df, title, author)
    """
    trajectory = None
    chunk_scores = None
    title = None
    author = None

    # Load lexicons
    emotion_df, vad_df = load_lexicons(emotion_lexicon, vad_lexicon)

    # Case 1: Text file input
    if text_file:
        try:
            with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            title = os.path.basename(text_file).replace(".txt", "").replace("_", " ")
            author = "File Input"

            from pyspark.sql.types import StructType, StructField, StringType

            schema = StructType(
                [
                    StructField("book_id", StringType(), True),
                    StructField("title", StringType(), True),
                    StructField("author", StringType(), True),
                    StructField("text", StringType(), True),
                ]
            )
            books_df = spark.createDataFrame(
                [("text_file", title, author, text)], schema
            )

            text_len = len(text)
            chunk_size = 10000
            if text_len < 20000:
                chunk_size = max(100, text_len // 10)

            chunks_df = create_chunks_df(spark, books_df, chunk_size=chunk_size)
            emotion_scores = score_chunks_with_emotions(spark, chunks_df, emotion_df)
            vad_scores = score_chunks_with_vad(spark, chunks_df, vad_df)
            chunk_scores = combine_emotion_vad_scores(emotion_scores, vad_scores)
            trajectory = analyze_trajectory(spark, chunk_scores)

        except Exception as e:
            st.error(f"Error processing text file: {e}")
            return None, None, None, None

    # Case 2: Book ID input
    elif book_id:
        # Try to load from main.py output first
        chunk_scores_path = f"{output_dir}/chunk_scores"
        trajectories_path = f"{output_dir}/trajectories"

        import glob

        chunk_csv_files = glob.glob(f"{chunk_scores_path}/*.csv")
        traj_csv_files = glob.glob(f"{trajectories_path}/*.csv")

        if (chunk_csv_files or os.path.exists(chunk_scores_path)) and (
            traj_csv_files or os.path.exists(trajectories_path)
        ):
            try:
                output_chunks = load_chunk_scores_with_types(spark, chunk_scores_path)
                output_chunks = output_chunks.filter(col("book_id") == book_id)

                output_trajectories = load_trajectories_with_types(
                    spark, trajectories_path
                )
                output_trajectories = output_trajectories.filter(
                    col("book_id") == book_id
                )

                if output_chunks.count() > 0 and output_trajectories.count() > 0:
                    chunk_scores = output_chunks
                    trajectory = output_trajectories
                    book_info = trajectory.select("title", "author").first()
                    title = book_info["title"]
                    author = book_info["author"]
            except Exception as e:
                st.warning(f"Could not load from main.py output: {e}")
                chunk_scores = None
                trajectory = None

        # Process from Gutenberg data if not in main.py output
        if chunk_scores is None or trajectory is None:
            metadata_df = load_metadata(metadata_path)
            # Trim whitespace and compare as strings to handle any type mismatches
            metadata_df = metadata_df.filter(
                (col("Language") == "en")
                & (trim(col("Etext Number")) == str(book_id).strip())
            )

            if metadata_df.count() == 0:
                # Try without language filter in case language column has issues
                metadata_df_retry = load_metadata(metadata_path)
                metadata_df_retry = metadata_df_retry.filter(
                    trim(col("Etext Number")) == str(book_id).strip()
                )
                if metadata_df_retry.count() == 0:
                    st.error(f"Book {book_id} not found in Gutenberg metadata!")
                    st.info(
                        "💡 Tip: Make sure the book ID exists and the book file is in the data/books/ directory."
                    )
                    return None, None, None, None
                else:
                    metadata_df = metadata_df_retry

            book_info = metadata_df.select(
                col("Etext Number").alias("book_id"),
                col("Title").alias("title"),
                col("Authors").alias("author"),
            ).first()

            title = book_info["title"]
            author = book_info["author"]

            from pyspark.sql.types import StructType, StructField, StringType
            import re

            def read_book_text(book_id: str) -> str:
                """Read book text from file."""
                try:
                    book_path = f"{books_dir}/{book_id}"
                    with open(book_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        text = re.sub(
                            r"\*\*\* START.*?\*\*\*", "", text, flags=re.DOTALL
                        )
                        text = re.sub(r"\*\*\* END.*?\*\*\*", "", text, flags=re.DOTALL)
                        return text
                except Exception as e:
                    st.warning(f"Could not read book file {book_id}: {e}")
                    return ""

            book_text = read_book_text(book_id)
            if not book_text:
                st.error(f"Could not read book file for {book_id}!")
                return None, None, None, None

            schema = StructType(
                [
                    StructField("book_id", StringType(), True),
                    StructField("title", StringType(), True),
                    StructField("author", StringType(), True),
                    StructField("text", StringType(), True),
                ]
            )
            books_df = spark.createDataFrame(
                [(book_id, title, author, book_text)], schema
            )

            text_len = len(book_text)
            chunk_size = 10000
            if text_len < 20000:
                chunk_size = max(100, text_len // 10)

            chunks_df = create_chunks_df(spark, books_df, chunk_size=chunk_size)
            emotion_scores = score_chunks_with_emotions(spark, chunks_df, emotion_df)
            vad_scores = score_chunks_with_vad(spark, chunks_df, vad_df)
            chunk_scores = combine_emotion_vad_scores(emotion_scores, vad_scores)
            trajectory = analyze_trajectory(spark, chunk_scores)

    else:
        st.error("No input specified!")
        return None, None, None, None

    return trajectory, chunk_scores, title, author


def plot_emotion_trajectory(chunk_scores_pd, book_title):
    """Plot emotion trajectory for a book using Plotly."""
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"Emotion Trajectory: {book_title}",
            "Valence-Arousal-Dominance Trajectory",
        ),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
    )

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
    colors = [
        "red",
        "orange",
        "brown",
        "black",
        "gold",
        "blue",
        "purple",
        "green",
    ]

    # Plot emotions
    has_data = False
    for emotion, color in zip(emotions, colors):
        if emotion in chunk_scores_pd.columns:
            if chunk_scores_pd[emotion].any():
                has_data = True
                fig.add_trace(
                    go.Scatter(
                        x=chunk_scores_pd["chunk_index"],
                        y=chunk_scores_pd[emotion],
                        mode="lines+markers",
                        name=emotion.capitalize(),
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        opacity=0.7,
                    ),
                    row=1,
                    col=1,
                )

    if not has_data:
        fig.add_annotation(
            text="No Emotion Data Available",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=1,
            col=1,
        )

    # Plot VAD scores
    vad_has_data = False
    if "avg_valence" in chunk_scores_pd.columns:
        if chunk_scores_pd["avg_valence"].any():
            vad_has_data = True
            fig.add_trace(
                go.Scatter(
                    x=chunk_scores_pd["chunk_index"],
                    y=chunk_scores_pd["avg_valence"],
                    mode="lines+markers",
                    name="Valence",
                    line=dict(color="purple", width=2),
                    marker=dict(size=4),
                ),
                row=2,
                col=1,
            )

    if "avg_arousal" in chunk_scores_pd.columns:
        if chunk_scores_pd["avg_arousal"].any():
            vad_has_data = True
            fig.add_trace(
                go.Scatter(
                    x=chunk_scores_pd["chunk_index"],
                    y=chunk_scores_pd["avg_arousal"],
                    mode="lines+markers",
                    name="Arousal",
                    line=dict(color="orange", width=2, dash="dash"),
                    marker=dict(size=4),
                ),
                row=2,
                col=1,
            )

    if "avg_dominance" in chunk_scores_pd.columns:
        if chunk_scores_pd["avg_dominance"].any():
            vad_has_data = True
            fig.add_trace(
                go.Scatter(
                    x=chunk_scores_pd["chunk_index"],
                    y=chunk_scores_pd["avg_dominance"],
                    mode="lines+markers",
                    name="Dominance",
                    line=dict(color="green", width=2, dash="dot"),
                    marker=dict(size=4),
                ),
                row=2,
                col=1,
            )

    if not vad_has_data:
        fig.add_annotation(
            text="No VAD Data Available",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=2,
            col=1,
        )

    # Update x-axis properties
    fig.update_xaxes(title_text="Chunk Index", row=1, col=1)
    fig.update_xaxes(title_text="Chunk Index", row=2, col=1)

    # Update y-axis properties
    fig.update_yaxes(title_text="Emotion Score", row=1, col=1)
    fig.update_yaxes(title_text="VAD Score", row=2, col=1)

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def main():
    """Main Streamlit app."""
    st.title("📚 EmoArc - Emotion Trajectory Analysis")
    st.markdown(
        "Analyze emotion trajectories in books and get recommendations based on emotional story arcs"
    )

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["Book Analysis & Recommendations", "Explore Books", "About"],
    )

    # Initialize Spark session
    if st.session_state.spark is None:
        with st.spinner("Initializing Spark session..."):
            st.session_state.spark = get_spark_session()

    # Page routing
    if page == "Book Analysis & Recommendations":
        show_book_analysis_and_recommendations()
    elif page == "Explore Books":
        show_explore_books()
    elif page == "About":
        show_about()


def show_book_analysis_and_recommendations():
    """Show combined book analysis and recommendations page."""
    st.header("Book Analysis & Recommendations")
    st.markdown("Analyze emotion trajectories and get book recommendations")

    # Output directory (default, not shown to user)
    output_dir = "output"
    trajectories_path = f"{output_dir}/trajectories"
    trajectories_available = os.path.exists(trajectories_path)

    if not trajectories_available:
        st.info(
            "💡 Tip: Run `python main.py` first to generate trajectories for recommendations."
        )
    else:
        # Show how many books are available for comparison
        try:
            spark = st.session_state.spark
            trajectories = load_trajectories_with_types(spark, trajectories_path)
            total_books = trajectories.count()
            st.info(
                f"📚 **{total_books}** books available in trajectory database for recommendations. "
                f"(Recommendations will compare against these {total_books} books)"
            )
        except Exception:
            # If we can't load trajectories, just show basic info
            st.info("💡 Trajectories found. Recommendations will compare against books in the trajectory database.")

    # Input method selection
    input_method = st.radio(
        "Select input method",
        ["Search by Title", "Enter Book ID", "Upload Text File"],
        horizontal=True,
    )

    book_id = None
    text_file = None

    if input_method == "Search by Title":
        title_query = st.text_input(
            "Enter book title (partial match supported)", key="title_search_input"
        )

        if title_query:
            with st.spinner("Searching books..."):
                metadata_df = load_metadata()
                results = search_books_by_title(title_query, metadata_df, limit=20)

                if results:
                    results_pd = results.toPandas()
                    st.success(f"Found {len(results_pd)} books")

                    # Display results in a selectbox
                    book_options = [
                        f"{row['title']} by {row['author']} (ID: {row['book_id']})"
                        for _, row in results_pd.iterrows()
                    ]
                    selected = st.selectbox(
                        "Select a book", book_options, key="book_selectbox"
                    )

                    # Store the selected book ID in session state (but don't use it until button is clicked)
                    if selected:
                        selected_idx = book_options.index(selected)
                        st.session_state.selected_book_id = results_pd.iloc[
                            selected_idx
                        ]["book_id"]
                else:
                    st.warning("No books found matching your query")
                    # Clear selected book if search fails
                    if "selected_book_id" in st.session_state:
                        del st.session_state.selected_book_id
        else:
            # Clear selected book when query is cleared
            if "selected_book_id" in st.session_state:
                del st.session_state.selected_book_id

    elif input_method == "Enter Book ID":
        book_id = st.text_input("Enter Gutenberg Book ID (e.g., 11)")

    elif input_method == "Upload Text File":
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            text_file = temp_path

    # Number of recommendations slider (shown if trajectories are available)
    top_n = 10
    if trajectories_available:
        top_n = st.slider("Number of recommendations", 5, 20, 10, key="rec_slider")

    # Single button that does both analysis and recommendations
    if st.button("Analyze Book & Get Recommendations", type="primary"):
        # Get book_id from selected option if using title search
        if input_method == "Search by Title" and "selected_book_id" in st.session_state:
            book_id = st.session_state.selected_book_id

        if book_id or text_file:
            with st.spinner("Processing book (this may take a minute)..."):
                spark = st.session_state.spark
                trajectory, chunk_scores, title, author = get_input_trajectory(
                    spark, book_id=book_id, text_file=text_file, output_dir=output_dir
                )

                if trajectory is not None and chunk_scores is not None:
                    # Store in session state for recommendations
                    st.session_state.current_trajectory = trajectory
                    st.session_state.current_chunk_scores = chunk_scores
                    st.session_state.current_title = title
                    st.session_state.current_author = author
                    st.session_state.current_book_id = (
                        book_id if book_id else "text_file"
                    )

                    st.success("Analysis complete!")

                    # Display book info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Book Information")
                        st.write(f"**Title:** {title}")
                        st.write(f"**Author:** {author}")
                        if book_id:
                            st.write(f"**Book ID:** {book_id}")

                    # Display trajectory plot
                    st.subheader("Emotion Trajectory")
                    chunk_scores_pd = chunk_scores.orderBy("chunk_index").toPandas()
                    fig = plot_emotion_trajectory(chunk_scores_pd, title)
                    st.plotly_chart(fig, width="stretch")

                    # Display statistics
                    st.subheader("Emotion Statistics")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Average Joy", f"{chunk_scores_pd['joy'].mean():.4f}")
                        st.metric(
                            "Average Sadness",
                            f"{chunk_scores_pd['sadness'].mean():.4f}",
                        )

                    with col2:
                        st.metric(
                            "Average Fear", f"{chunk_scores_pd['fear'].mean():.4f}"
                        )
                        st.metric(
                            "Average Anger", f"{chunk_scores_pd['anger'].mean():.4f}"
                        )

                    with col3:
                        st.metric(
                            "Average Valence",
                            f"{chunk_scores_pd['avg_valence'].mean():.4f}",
                        )
                        st.metric(
                            "Average Arousal",
                            f"{chunk_scores_pd['avg_arousal'].mean():.4f}",
                        )

                    with col4:
                        st.metric(
                            "Average Dominance",
                            f"{chunk_scores_pd['avg_dominance'].mean():.4f}",
                        )
                        st.metric("Number of Chunks", f"{len(chunk_scores_pd)}")

                    # Show trajectory statistics
                    trajectory_pd = trajectory.toPandas().iloc[0]
                    st.subheader("Trajectory Summary")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Peak Emotions:**")
                        peak_emotions = {
                            "Anger": trajectory_pd.get("max_anger", 0),
                            "Anticipation": trajectory_pd.get("max_anticipation", 0),
                            "Disgust": trajectory_pd.get("max_disgust", 0),
                            "Fear": trajectory_pd.get("max_fear", 0),
                            "Joy": trajectory_pd.get("max_joy", 0),
                            "Sadness": trajectory_pd.get("max_sadness", 0),
                            "Surprise": trajectory_pd.get("max_surprise", 0),
                            "Trust": trajectory_pd.get("max_trust", 0),
                        }
                        for emotion, value in peak_emotions.items():
                            st.write(f"- {emotion}: {value:.2f}")

                    with col2:
                        st.write("**Average Emotions:**")
                        avg_emotions = {
                            "Anger": trajectory_pd.get("avg_anger", 0),
                            "Anticipation": trajectory_pd.get("avg_anticipation", 0),
                            "Disgust": trajectory_pd.get("avg_disgust", 0),
                            "Fear": trajectory_pd.get("avg_fear", 0),
                            "Joy": trajectory_pd.get("avg_joy", 0),
                            "Sadness": trajectory_pd.get("avg_sadness", 0),
                            "Surprise": trajectory_pd.get("avg_surprise", 0),
                            "Trust": trajectory_pd.get("avg_trust", 0),
                        }
                        for emotion, value in avg_emotions.items():
                            st.write(f"- {emotion}: {value:.4f}")

                    # Automatically show recommendations if trajectories are available
                    if trajectories_available:
                        st.divider()
                        st.subheader("📚 Recommendations")

                        with st.spinner("Computing recommendations..."):
                            # Load trajectories for comparison
                            trajectories = load_trajectories_with_types(
                                spark, trajectories_path
                            )
                            
                            # Count total books available for comparison
                            total_books_count = trajectories.count()
                            
                            # Get liked book ID
                            liked_id = trajectory.select("book_id").first()["book_id"]
                            
                            # Check if the current book is already in the trajectories
                            current_book_in_trajectories = trajectories.filter(
                                col("book_id") == liked_id
                            ).count() > 0
                            
                            # Display comparison info
                            if current_book_in_trajectories:
                                st.info(
                                    f"📊 Comparing against **{total_books_count}** books from the trajectory database. "
                                    f"(Note: The current book is included in this database)"
                                )
                            else:
                                st.info(
                                    f"📊 Comparing against **{total_books_count}** books from the trajectory database."
                                )

                            # Combine trajectories
                            try:
                                all_trajectories = trajectories.unionByName(
                                    trajectory, allowMissingColumns=True
                                )
                            except Exception:
                                trajectories_cols = set(trajectories.columns)
                                liked_cols = set(trajectory.columns)
                                common_cols = sorted(
                                    list(trajectories_cols & liked_cols)
                                )

                                trajectories_aligned = trajectories.select(*common_cols)
                                liked_trajectory_aligned = trajectory.select(
                                    *common_cols
                                )
                                all_trajectories = trajectories_aligned.union(
                                    liked_trajectory_aligned
                                )

                            # Get recommendations
                            recommendations = recommend(
                                spark, all_trajectories, liked_id, top_n=top_n
                            )

                            # Display recommendations
                            st.success(
                                f"Top {top_n} recommendations for: **{title}** by {author}"
                            )

                            recs_pd = recommendations.toPandas()

                            for idx, row in recs_pd.iterrows():
                                with st.expander(
                                    f"{idx + 1}. {row['title']} by {row['author']} "
                                    f"(Similarity: {row['similarity']:.4f})"
                                ):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**Emotion Scores:**")
                                        st.write(f"- Joy: {row.get('avg_joy', 0):.4f}")
                                        st.write(
                                            f"- Sadness: {row.get('avg_sadness', 0):.4f}"
                                        )
                                        st.write(
                                            f"- Fear: {row.get('avg_fear', 0):.4f}"
                                        )
                                        st.write(
                                            f"- Anger: {row.get('avg_anger', 0):.4f}"
                                        )

                                    with col2:
                                        st.write("**VAD Scores:**")
                                        st.write(
                                            f"- Valence: {row.get('avg_valence', 0):.4f}"
                                        )
                                        st.write(
                                            f"- Arousal: {row.get('avg_arousal', 0):.4f}"
                                        )
                                        st.write(
                                            f"- Dominance: {row.get('avg_dominance', 0):.4f}"
                                        )

                            # Download button
                            csv = recs_pd.to_csv(index=False)
                            st.download_button(
                                label="Download Recommendations as CSV",
                                data=csv,
                                file_name=f"recommendations_{liked_id}.csv",
                                mime="text/csv",
                            )
                    else:
                        st.info(
                            "💡 Recommendations require trajectories. Run `python main.py` first."
                        )
                else:
                    st.error(
                        "Failed to analyze book. Please check if the book exists and try again."
                    )
        else:
            st.warning("Please provide a book ID or upload a text file")


def show_explore_books():
    """Show explore books page."""
    st.header("Explore Books")
    st.markdown("Discover books by emotion characteristics")

    # Output directory (default, not shown to user)
    output_dir = "output"
    trajectories_path = f"{output_dir}/trajectories"

    if not os.path.exists(trajectories_path):
        st.warning(
            f"⚠️ Trajectories not found in {trajectories_path}. "
            "Please run `python main.py` first to generate trajectories."
        )
        return

    emotion_type = st.selectbox(
        "Explore by emotion",
        [
            "Joy",
            "Sadness",
            "Fear",
            "Anger",
            "Anticipation",
            "Disgust",
            "Surprise",
            "Trust",
        ],
    )

    top_n = st.slider("Number of books to show", 10, 50, 20)

    if st.button("Show Top Books", type="primary"):
        with st.spinner("Loading trajectories..."):
            spark = st.session_state.spark
            trajectories = load_trajectories_with_types(spark, trajectories_path)

            emotion_col = f"avg_{emotion_type.lower()}"
            if emotion_col in trajectories.columns:
                top_books = (
                    trajectories.orderBy(col(emotion_col).desc())
                    .select(
                        "book_id",
                        "title",
                        "author",
                        emotion_col,
                        "avg_joy",
                        "avg_sadness",
                        "avg_fear",
                        "avg_anger",
                        "avg_valence",
                        "avg_arousal",
                    )
                    .limit(top_n)
                )

                top_books_pd = top_books.toPandas()

                st.success(f"Top {top_n} books by {emotion_type}")

                for idx, row in top_books_pd.iterrows():
                    st.write(f"**{idx + 1}. {row['title']}** by {row['author']}")
                    st.write(f"   {emotion_type}: {row[emotion_col]:.4f}")
                    st.write("---")


def show_about():
    """Show about page."""
    st.header("About EmoArc")
    st.markdown("""
    **EmoArc** is an emotion trajectory analysis and recommendation system for Project Gutenberg books.
    
    ### Features:
    - **Emotion Analysis**: Analyze emotion trajectories using NRC Emotion Lexicon (8 Plutchik emotions)
    - **VAD Analysis**: Analyze Valence-Arousal-Dominance scores
    - **Recommendations**: Get book recommendations based on similar emotion trajectories
    - **Visualizations**: Interactive plots showing emotion trajectories over time
    
    ### How it works:
    1. Books are segmented into fixed-length chunks (10,000 characters)
    2. Each chunk is scored using NRC Emotion and VAD lexicons
    3. Emotion trajectories are analyzed to identify patterns
    4. Similar books are found using feature-based similarity
    
    ### Technical Details:
    - Built with Apache Spark for big data processing
    - Uses NRC Emotion Lexicon and NRC VAD Lexicon
    - Processes 75,000+ books from Project Gutenberg
    
    ### Usage:
    1. Run `python main.py` to generate trajectories for all books
    2. Use this app to search, analyze, and get recommendations
    """)


if __name__ == "__main__":
    main()
