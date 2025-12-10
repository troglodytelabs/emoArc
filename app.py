"""
Streamlit app for EmoArc - Emotion Trajectory Analysis and Recommendation System
"""

import sys
import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from core import (
    create_spark_session,
    load_trajectories,
    load_metadata,
    get_input_trajectory,
    find_books_by_emotion_preferences,
)
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
    return create_spark_session("EmoArc Streamlit")


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


def plot_topic_distribution(book_topics_pd, book_title, num_topics=10):
    """Plot topic distribution for a book using Plotly."""
    if book_topics_pd is None or len(book_topics_pd) == 0:
        return None

    topics = book_topics_pd.iloc[0]["book_topics"]
    if topics is None:
        return None

    # Create bar chart
    fig = go.Figure()

    topic_labels = [f"Topic {i + 1}" for i in range(len(topics))]
    fig.add_trace(
        go.Bar(
            x=topic_labels,
            y=topics,
            marker=dict(color="steelblue", line=dict(color="navy", width=1)),
            text=[f"{t:.3f}" for t in topics],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=f"Topic Distribution: {book_title}",
        xaxis_title="Topic",
        yaxis_title="Probability",
        height=400,
        template="plotly_white",
        showlegend=False,
    )

    return fig


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
        [
            "Book Analysis & Recommendations",
            "Explore Books",
            "Find Books by Emotion Preferences",
            "About",
        ],
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
    elif page == "Find Books by Emotion Preferences":
        show_find_books_by_emotions()
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
            trajectories = load_trajectories(spark, output_dir)
            if trajectories is not None:
                total_books = trajectories.count()
                has_embeddings = "book_embedding" in trajectories.columns
                has_topics = "book_topics" in trajectories.columns
                features_info = []
                if has_embeddings:
                    features_info.append("embeddings")
                if has_topics:
                    features_info.append("topics")
                features_str = (
                    f" (includes {', '.join(features_info)})" if features_info else ""
                )
                st.info(
                    f"📚 **{total_books}** books available in trajectory database for recommendations{features_str}. "
                    f"(Recommendations will compare against these {total_books} books)"
                )
        except Exception:
            # If we can't load trajectories, just show basic info
            st.info(
                "💡 Trajectories found. Recommendations will compare against books in the trajectory database."
            )

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
                spark = st.session_state.spark
                metadata_df = load_metadata(spark)
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

    # Option to compute topics
    compute_topics = st.checkbox(
        "Compute topic modeling (slower but provides topic analysis)",
        value=False,
        key="compute_topics",
    )
    num_topics = 10
    if compute_topics:
        num_topics = st.slider("Number of topics", 5, 20, 10, key="num_topics_slider")

    # Single button that does both analysis and recommendations
    if st.button("Analyze Book & Get Recommendations", type="primary"):
        # Get book_id from selected option if using title search
        if input_method == "Search by Title" and "selected_book_id" in st.session_state:
            book_id = st.session_state.selected_book_id

        if book_id or text_file:
            with st.spinner("Processing book (this may take a minute)..."):
                spark = st.session_state.spark
                try:
                    trajectory, chunk_scores, title, author, book_topics = (
                        get_input_trajectory(
                            spark,
                            book_id=book_id,
                            text_file=text_file,
                            output_dir=output_dir,
                            compute_topics=compute_topics,
                            num_topics=num_topics,
                        )
                    )
                except Exception as e:
                    st.error(f"Error processing book: {e}")
                    trajectory = None
                    chunk_scores = None
                    title = None
                    author = None
                    book_topics = None

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

                    # Display topic modeling if available
                    if book_topics is not None:
                        st.divider()
                        st.subheader("📊 Topic Modeling")
                        book_topics_pd = book_topics.toPandas()
                        topic_fig = plot_topic_distribution(
                            book_topics_pd, title, num_topics
                        )
                        if topic_fig:
                            st.plotly_chart(topic_fig, width="stretch")

                            # Show top topics
                            topics = book_topics_pd.iloc[0]["book_topics"]
                            if topics:
                                # Get top 5 topics
                                topic_scores = [
                                    (i, topics[i]) for i in range(len(topics))
                                ]
                                topic_scores.sort(key=lambda x: x[1], reverse=True)

                                st.write("**Top Topics:**")
                                for rank, (topic_idx, score) in enumerate(
                                    topic_scores[:5], 1
                                ):
                                    st.write(
                                        f"{rank}. Topic {topic_idx + 1}: {score:.4f}"
                                    )
                        else:
                            st.info("Topic distribution not available")
                    elif compute_topics:
                        st.info("💡 Topic modeling is being computed...")

                    # Automatically show recommendations if trajectories are available
                    if trajectories_available:
                        st.divider()
                        st.subheader("📚 Recommendations")

                        with st.spinner("Computing recommendations..."):
                            # Load trajectories for comparison
                            trajectories = load_trajectories(spark, output_dir)
                            if trajectories is None:
                                st.error("Could not load trajectories")
                                return

                            # Count total books available for comparison
                            total_books_count = trajectories.count()

                            # Get liked book ID
                            liked_id = trajectory.select("book_id").first()["book_id"]

                            # Check if the current book is already in the trajectories
                            current_book_in_trajectories = (
                                trajectories.filter(col("book_id") == liked_id).count()
                                > 0
                            )

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
            trajectories = load_trajectories(spark, output_dir)
            if trajectories is None:
                st.error("Could not load trajectories")
                return

            emotion_col = f"avg_{emotion_type.lower()}"
            if emotion_col in trajectories.columns:
                # Select columns, avoiding duplicates
                select_cols = ["book_id", "title", "author", emotion_col]
                additional_cols = [
                    "avg_joy",
                    "avg_sadness",
                    "avg_fear",
                    "avg_anger",
                    "avg_valence",
                    "avg_arousal",
                ]
                # Only add additional cols if they're not already selected
                for col_name in additional_cols:
                    if col_name not in select_cols:
                        select_cols.append(col_name)

                top_books = (
                    trajectories.orderBy(col(emotion_col).desc())
                    .select(*select_cols)
                    .limit(top_n)
                )

                top_books_pd = top_books.toPandas()

                st.success(f"Top {top_n} books by {emotion_type}")

                for idx, row in top_books_pd.iterrows():
                    # Get emotion value safely
                    emotion_val = row[emotion_col]
                    # Handle case where it might be a Series (if duplicate columns exist)
                    if hasattr(emotion_val, "iloc"):
                        emotion_val = (
                            emotion_val.iloc[0] if len(emotion_val) > 0 else 0.0
                        )
                    emotion_val = float(emotion_val) if emotion_val is not None else 0.0

                    st.write(f"**{idx + 1}. {row['title']}** by {row['author']}")
                    st.write(f"   {emotion_type}: {emotion_val:.4f}")
                    st.write("---")


def show_find_books_by_emotions():
    """Show page for finding books by emotion preferences."""
    st.header("Find Books by Emotion Preferences")
    st.markdown("Pick a vibe in one click, optionally fine-tune, then get matches.")

    # Output directory
    output_dir = "output"
    trajectories_path = f"{output_dir}/trajectories"

    if not os.path.exists(trajectories_path):
        st.warning(
            f"⚠️ Trajectories not found in {trajectories_path}. "
            "Please run `python main.py` first to generate trajectories."
        )
        return

    # Simple presets to reduce friction; sliders are now optional fine-tuning
    st.subheader("Choose a vibe")
    presets = {
        "Balanced": {
            "joy": 0.5,
            "anticipation": 0.5,
            "surprise": 0.3,
            "trust": 0.5,
            "sadness": 0.0,
            "fear": 0.0,
            "anger": 0.0,
            "disgust": 0.0,
        },
        "Cozy & Uplifting": {
            "joy": 0.7,
            "anticipation": 0.6,
            "surprise": 0.2,
            "trust": 0.6,
            "sadness": 0.0,
            "fear": 0.0,
            "anger": 0.0,
            "disgust": 0.0,
        },
        "Adventurous": {
            "joy": 0.5,
            "anticipation": 0.7,
            "surprise": 0.5,
            "trust": 0.4,
            "sadness": 0.1,
            "fear": 0.2,
            "anger": 0.0,
            "disgust": 0.0,
        },
        "Dark & Tense": {
            "joy": 0.1,
            "anticipation": 0.4,
            "surprise": 0.3,
            "trust": 0.2,
            "sadness": 0.5,
            "fear": 0.5,
            "anger": 0.2,
            "disgust": 0.2,
        },
        "Surprising": {
            "joy": 0.4,
            "anticipation": 0.6,
            "surprise": 0.7,
            "trust": 0.3,
            "sadness": 0.0,
            "fear": 0.1,
            "anger": 0.0,
            "disgust": 0.0,
        },
    }

    preset_choice = st.radio(
        "Quick presets",
        list(presets.keys()),
        horizontal=True,
        key="emotion_pref_preset",
    )
    emotion_preferences = presets.get(preset_choice, presets["Balanced"]).copy()
    st.caption("Adjust nothing else to use the preset as-is.")

    with st.expander("Fine-tune (optional)", expanded=False):
        st.markdown("Pick a level instead of precise numbers.")
        level_opts = ["None", "Low", "Medium", "High"]
        level_map = {"None": 0.0, "Low": 0.25, "Medium": 0.5, "High": 0.75}
        level_map_high = {"None": 0.0, "Low": 0.33, "Medium": 0.66, "High": 1.0}

        def level_from_value(val, use_high=False):
            candidates = level_map_high if use_high else level_map
            return min(candidates.keys(), key=lambda k: abs(candidates[k] - val))

        col1, col2 = st.columns(2)

        with col1:
            joy_level = st.select_slider(
                "Joy",
                options=level_opts,
                value=level_from_value(emotion_preferences["joy"]),
                key="pref_joy",
            )
            emotion_preferences["joy"] = level_map[joy_level]

            sadness_level = st.select_slider(
                "Sadness",
                options=level_opts,
                value=level_from_value(emotion_preferences["sadness"]),
                key="pref_sadness",
            )
            emotion_preferences["sadness"] = level_map[sadness_level]

            fear_level = st.select_slider(
                "Fear",
                options=level_opts,
                value=level_from_value(emotion_preferences["fear"]),
                key="pref_fear",
            )
            emotion_preferences["fear"] = level_map[fear_level]

            anger_level = st.select_slider(
                "Anger",
                options=level_opts,
                value=level_from_value(emotion_preferences["anger"]),
                key="pref_anger",
            )
            emotion_preferences["anger"] = level_map[anger_level]

        with col2:
            anticipation_level = st.select_slider(
                "Anticipation",
                options=level_opts,
                value=level_from_value(
                    emotion_preferences["anticipation"], use_high=True
                ),
                key="pref_anticipation",
            )
            emotion_preferences["anticipation"] = level_map_high[anticipation_level]

            surprise_level = st.select_slider(
                "Surprise",
                options=level_opts,
                value=level_from_value(emotion_preferences["surprise"], use_high=True),
                key="pref_surprise",
            )
            emotion_preferences["surprise"] = level_map_high[surprise_level]

            trust_level = st.select_slider(
                "Trust",
                options=level_opts,
                value=level_from_value(emotion_preferences["trust"]),
                key="pref_trust",
            )
            emotion_preferences["trust"] = level_map[trust_level]

            disgust_level = st.select_slider(
                "Disgust",
                options=level_opts,
                value=level_from_value(emotion_preferences["disgust"]),
                key="pref_disgust",
            )
            emotion_preferences["disgust"] = level_map[disgust_level]

    # Number of results
    top_n = st.slider("Number of books to show", 10, 50, 20, key="emotion_pref_top_n")

    if st.button("Find Matching Books", type="primary"):
        with st.spinner("Finding books that match your preferences..."):
            spark = st.session_state.spark
            trajectories = load_trajectories(spark, output_dir)
            if trajectories is None:
                st.error("Could not load trajectories")
                return

            # Find matching books
            matching_books = find_books_by_emotion_preferences(
                spark, trajectories, emotion_preferences, top_n=top_n
            )

            if matching_books.count() == 0:
                st.warning(
                    "No books found. Try adjusting your emotion preferences or ensure trajectories are available."
                )
            else:
                matching_books_pd = matching_books.toPandas()

                st.success(
                    f"Found {len(matching_books_pd)} books matching your preferences!"
                )

                # Display results
                for idx, row in matching_books_pd.iterrows():
                    with st.expander(
                        f"{idx + 1}. {row['title']} by {row['author']} "
                        f"(Match: {row['match_score']:.3f})"
                    ):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Emotion Scores:**")
                            st.write(f"- Joy: {row.get('avg_joy', 0):.4f}")
                            st.write(f"- Sadness: {row.get('avg_sadness', 0):.4f}")
                            st.write(f"- Fear: {row.get('avg_fear', 0):.4f}")
                            st.write(f"- Anger: {row.get('avg_anger', 0):.4f}")

                        with col2:
                            st.write("**More Emotions:**")
                            st.write(
                                f"- Anticipation: {row.get('avg_anticipation', 0):.4f}"
                            )
                            st.write(f"- Surprise: {row.get('avg_surprise', 0):.4f}")
                            st.write(f"- Trust: {row.get('avg_trust', 0):.4f}")
                            st.write(f"- Disgust: {row.get('avg_disgust', 0):.4f}")

                        st.write("**VAD Scores:**")
                        st.write(f"- Valence: {row.get('avg_valence', 0):.4f}")
                        st.write(f"- Arousal: {row.get('avg_arousal', 0):.4f}")

                # Download button
                csv = matching_books_pd.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="emotion_preference_matches.csv",
                    mime="text/csv",
                )


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
    1. Books are segmented into percentage-based chunks (default: 50 chunks per book)
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
