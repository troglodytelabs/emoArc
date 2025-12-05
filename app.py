"""
streamlit app for emoarc emotion trajectory analysis and recommendation system
"""

import sys
import os
import re
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, explode
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# add src to path
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

# page config
st.set_page_config(
    page_title="EmoArc - Emotion Trajectory Analysis",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# initialize session state
if "spark" not in st.session_state:
    st.session_state.spark = None
if "emotion_lexicon" not in st.session_state:
    st.session_state.emotion_lexicon = None
if "vad_lexicon" not in st.session_state:
    st.session_state.vad_lexicon = None
if "metadata_df" not in st.session_state:
    st.session_state.metadata_df = None


def calculate_plutchik_dyads(emotion_scores):
    """
    calculate plutchik's emotion dyads from the 8 basic emotions.
    dyads are combinations of adjacent emotions on plutchik's wheel.

    primary dyads (adjacent emotions):
    - joy + trust = love
    - trust + fear = submission
    - fear + surprise = awe
    - surprise + sadness = disapproval
    - sadness + disgust = remorse
    - disgust + anger = contempt
    - anger + anticipation = aggressiveness
    - anticipation + joy = optimism
    """
    dyads = {}

    # calculate each dyad as the average of its component emotions
    dyads['love'] = (emotion_scores.get('joy', 0) + emotion_scores.get('trust', 0)) / 2
    dyads['submission'] = (emotion_scores.get('trust', 0) + emotion_scores.get('fear', 0)) / 2
    dyads['awe'] = (emotion_scores.get('fear', 0) + emotion_scores.get('surprise', 0)) / 2
    dyads['disapproval'] = (emotion_scores.get('surprise', 0) + emotion_scores.get('sadness', 0)) / 2
    dyads['remorse'] = (emotion_scores.get('sadness', 0) + emotion_scores.get('disgust', 0)) / 2
    dyads['contempt'] = (emotion_scores.get('disgust', 0) + emotion_scores.get('anger', 0)) / 2
    dyads['aggressiveness'] = (emotion_scores.get('anger', 0) + emotion_scores.get('anticipation', 0)) / 2
    dyads['optimism'] = (emotion_scores.get('anticipation', 0) + emotion_scores.get('joy', 0)) / 2

    return dyads


@st.cache_resource
def get_spark_session():
    """create and cache spark session"""
    spark = (
        SparkSession.builder.appName("EmoArc Streamlit")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_metadata(metadata_path="data/gutenberg_metadata.csv"):
    """load metadata from csv file"""
    spark = get_spark_session()
    metadata_df = spark.read.option("header", "true").csv(metadata_path)
    # filter english books only
    metadata_df = metadata_df.filter(col("Language") == "en")
    return metadata_df


@st.cache_resource
def load_lexicons(
    emotion_lexicon="data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    vad_lexicon="data/NRC-VAD-Lexicon-v2.1.txt",
):
    """load and cache emotion and vad lexicons"""
    spark = get_spark_session()
    emotion_df = load_emotion_lexicon(spark, emotion_lexicon)
    vad_df = load_vad_lexicon(spark, vad_lexicon)
    return emotion_df, vad_df


def search_books_by_title(title_query, metadata_df, limit=20):
    """search for books by title with partial matching"""
    if not title_query:
        return None

    # case-insensitive search
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


def generate_wordcloud_from_text(text, title=""):
    """generate wordcloud visualization from book text"""
    # clean and prepare text
    text_clean = text.lower()
    text_clean = re.sub(r'[^a-z\s]', ' ', text_clean)

    # create wordcloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text_clean)

    # create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    if title:
        ax.set_title(f'word cloud: {title}', fontsize=14, pad=10)
    plt.tight_layout()

    return fig


def load_trajectories_with_types(spark, trajectories_path):
    """load trajectories from csv and cast columns to numeric types"""
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
    """load chunk scores from csv and cast columns to numeric types"""
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
    get emotion trajectory for a book or text file
    returns: (trajectory_df, chunk_scores_df, title, author, full_text)
    """
    trajectory = None
    chunk_scores = None
    title = None
    author = None
    full_text = None

    # load lexicons
    emotion_df, vad_df = load_lexicons(emotion_lexicon, vad_lexicon)

    # case 1: text file input
    if text_file:
        try:
            with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            full_text = text
            title = os.path.basename(text_file).replace(".txt", "").replace("_", " ")
            author = "uploaded file"

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

            # use percentage-based chunking for comparable trajectories
            chunks_df = create_chunks_df(spark, books_df, num_chunks=20)
            emotion_scores = score_chunks_with_emotions(spark, chunks_df, emotion_df)
            vad_scores = score_chunks_with_vad(spark, chunks_df, vad_df)
            chunk_scores = combine_emotion_vad_scores(emotion_scores, vad_scores)
            trajectory = analyze_trajectory(spark, chunk_scores)

        except Exception as e:
            st.error(f"error processing text file: {e}")
            return None, None, None, None, None

    # case 2: book id input
    elif book_id:
        # try to load from precomputed output first
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
                st.warning(f"could not load from precomputed output: {e}")
                chunk_scores = None
                trajectory = None

        # process from gutenberg data if not in precomputed output
        if chunk_scores is None or trajectory is None:
            metadata_df = load_metadata(metadata_path)
            # trim whitespace and compare as strings
            metadata_df = metadata_df.filter(
                (col("Language") == "en")
                & (trim(col("Etext Number")) == str(book_id).strip())
            )

            if metadata_df.count() == 0:
                # try without language filter
                metadata_df_retry = load_metadata(metadata_path)
                metadata_df_retry = metadata_df_retry.filter(
                    trim(col("Etext Number")) == str(book_id).strip()
                )
                if metadata_df_retry.count() == 0:
                    st.error(f"book {book_id} not found in metadata")
                    st.info("tip: make sure the book id exists and the book file is in data/books/")
                    return None, None, None, None, None
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
                """read book text from file and remove gutenberg headers"""
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
                    st.warning(f"could not read book file {book_id}: {e}")
                    return ""

            book_text = read_book_text(book_id)
            if not book_text:
                st.error(f"could not read book file for {book_id}")
                return None, None, None, None, None

            full_text = book_text

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

            # use percentage-based chunking for comparable trajectories
            chunks_df = create_chunks_df(spark, books_df, num_chunks=20)
            emotion_scores = score_chunks_with_emotions(spark, chunks_df, emotion_df)
            vad_scores = score_chunks_with_vad(spark, chunks_df, vad_df)
            chunk_scores = combine_emotion_vad_scores(emotion_scores, vad_scores)
            trajectory = analyze_trajectory(spark, chunk_scores)

    else:
        st.error("no input specified")
        return None, None, None, None, None

    return trajectory, chunk_scores, title, author, full_text


def plot_emotion_trajectory(chunk_scores_pd, book_title):
    """plot emotion trajectory for a book using plotly"""
    # create subplots for emotions and vad scores
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"emotion trajectory: {book_title}",
            "valence-arousal-dominance trajectory",
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

    # plot each emotion as a line
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
            text="no emotion data available",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=1,
            col=1,
        )

    # plot vad scores
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
            text="no vad data available",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            row=2,
            col=1,
        )

    # update axes labels
    fig.update_xaxes(title_text="story progression (chunk index)", row=1, col=1)
    fig.update_xaxes(title_text="story progression (chunk index)", row=2, col=1)
    fig.update_yaxes(title_text="emotion score", row=1, col=1)
    fig.update_yaxes(title_text="vad score", row=2, col=1)

    # update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def main():
    """main streamlit application"""
    st.title("EmoArc - Emotion Trajectory Analysis")
    st.markdown(
        "analyze emotion trajectories in books and get recommendations based on emotional story arcs"
    )

    # sidebar navigation
    st.sidebar.title("navigation")
    page = st.sidebar.radio(
        "choose a page",
        ["book analysis & recommendations", "explore books", "about"],
    )

    # initialize spark session
    if st.session_state.spark is None:
        with st.spinner("initializing spark session..."):
            st.session_state.spark = get_spark_session()

    # page routing
    if page == "book analysis & recommendations":
        show_book_analysis_and_recommendations()
    elif page == "explore books":
        show_explore_books()
    elif page == "about":
        show_about()


def show_book_analysis_and_recommendations():
    """show book analysis page with plutchik dyads and wordclouds"""
    st.header("book analysis & recommendations")
    st.markdown("analyze emotion trajectories and get book recommendations")

    # check for precomputed trajectories
    output_dir = "output"
    trajectories_path = f"{output_dir}/trajectories"
    trajectories_available = os.path.exists(trajectories_path)

    if not trajectories_available:
        st.info(
            "tip: run `python main.py` first to generate trajectories for recommendations"
        )
    else:
        # show how many books are available for comparison
        try:
            spark = st.session_state.spark
            trajectories = load_trajectories_with_types(spark, trajectories_path)
            total_books = trajectories.count()
            st.info(
                f"{total_books} books available in trajectory database for recommendations "
                f"(recommendations will compare against these {total_books} books)"
            )
        except Exception:
            # if we can't load trajectories, show basic info
            st.info("trajectories found - recommendations will compare against books in the database")

    # input method selection
    input_method = st.radio(
        "select input method",
        ["search by title", "enter book id", "upload text file"],
        horizontal=True,
    )

    book_id = None
    text_file = None

    if input_method == "search by title":
        title_query = st.text_input(
            "enter book title (partial match supported)", key="title_search_input"
        )

        if title_query:
            with st.spinner("searching books..."):
                metadata_df = load_metadata()
                results = search_books_by_title(title_query, metadata_df, limit=20)

                if results:
                    results_pd = results.toPandas()
                    st.success(f"found {len(results_pd)} books")

                    # display results in selectbox
                    book_options = [
                        f"{row['title']} by {row['author']} (id: {row['book_id']})"
                        for _, row in results_pd.iterrows()
                    ]
                    selected = st.selectbox(
                        "select a book", book_options, key="book_selectbox"
                    )

                    # store the selected book id
                    if selected:
                        selected_idx = book_options.index(selected)
                        st.session_state.selected_book_id = results_pd.iloc[
                            selected_idx
                        ]["book_id"]
                else:
                    st.warning("no books found matching your query")
                    if "selected_book_id" in st.session_state:
                        del st.session_state.selected_book_id
        else:
            if "selected_book_id" in st.session_state:
                del st.session_state.selected_book_id

    elif input_method == "enter book id":
        book_id = st.text_input("enter gutenberg book id (e.g., 11)")

    elif input_method == "upload text file":
        uploaded_file = st.file_uploader("upload a text file", type=["txt"])
        if uploaded_file:
            # save uploaded file temporarily
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            text_file = temp_path

    # number of recommendations slider
    top_n = 10
    if trajectories_available:
        top_n = st.slider("number of recommendations", 5, 20, 10, key="rec_slider")

    # analyze book button
    if st.button("analyze book & get recommendations", type="primary"):
        # get book_id from selected option if using title search
        if input_method == "search by title" and "selected_book_id" in st.session_state:
            book_id = st.session_state.selected_book_id

        if book_id or text_file:
            with st.spinner("processing book (this may take a minute)..."):
                spark = st.session_state.spark
                trajectory, chunk_scores, title, author, full_text = get_input_trajectory(
                    spark, book_id=book_id, text_file=text_file, output_dir=output_dir
                )

                if trajectory is not None and chunk_scores is not None:
                    # store in session state
                    st.session_state.current_trajectory = trajectory
                    st.session_state.current_chunk_scores = chunk_scores
                    st.session_state.current_title = title
                    st.session_state.current_author = author
                    st.session_state.current_full_text = full_text
                    st.session_state.current_book_id = (
                        book_id if book_id else "text_file"
                    )

                    st.success("analysis complete")

                    # book information
                    st.subheader("book information")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write(f"title: {title}")
                        st.write(f"author: {author}")
                        if book_id:
                            st.write(f"book id: {book_id}")
                    with col2:
                        chunk_scores_pd = chunk_scores.orderBy("chunk_index").toPandas()
                        st.write(f"chunks analyzed: {len(chunk_scores_pd)} (percentage-based)")
                        if len(chunk_scores_pd) > 0:
                            est_text_len = len(chunk_scores_pd) * (len(full_text) // len(chunk_scores_pd)) if full_text else 0
                            st.write(f"estimated text length: ~{est_text_len:,} characters")

                    # wordcloud visualization
                    if full_text:
                        st.divider()
                        st.subheader("word cloud")
                        try:
                            wordcloud_fig = generate_wordcloud_from_text(full_text, title)
                            st.pyplot(wordcloud_fig)
                        except Exception as e:
                            st.warning(f"could not generate word cloud: {e}")

                    # emotion trajectory plot
                    st.divider()
                    st.subheader("emotion trajectory")
                    fig = plot_emotion_trajectory(chunk_scores_pd, title)
                    st.plotly_chart(fig, use_container_width=True)

                    # plutchik's emotion dyads
                    st.divider()
                    st.subheader("plutchik emotion dyads")
                    st.caption("complex emotions derived from combinations of basic emotions")

                    trajectory_pd = trajectory.toPandas().iloc[0]
                    avg_emotions = {
                        'joy': trajectory_pd.get('avg_joy', 0),
                        'trust': trajectory_pd.get('avg_trust', 0),
                        'fear': trajectory_pd.get('avg_fear', 0),
                        'surprise': trajectory_pd.get('avg_surprise', 0),
                        'sadness': trajectory_pd.get('avg_sadness', 0),
                        'disgust': trajectory_pd.get('avg_disgust', 0),
                        'anger': trajectory_pd.get('avg_anger', 0),
                        'anticipation': trajectory_pd.get('avg_anticipation', 0),
                    }

                    dyads = calculate_plutchik_dyads(avg_emotions)

                    # display dyads in columns
                    col1, col2, col3, col4 = st.columns(4)
                    dyad_items = list(dyads.items())

                    with col1:
                        st.metric(dyad_items[0][0], f"{dyad_items[0][1]:.4f}")
                        st.metric(dyad_items[1][0], f"{dyad_items[1][1]:.4f}")
                    with col2:
                        st.metric(dyad_items[2][0], f"{dyad_items[2][1]:.4f}")
                        st.metric(dyad_items[3][0], f"{dyad_items[3][1]:.4f}")
                    with col3:
                        st.metric(dyad_items[4][0], f"{dyad_items[4][1]:.4f}")
                        st.metric(dyad_items[5][0], f"{dyad_items[5][1]:.4f}")
                    with col4:
                        st.metric(dyad_items[6][0], f"{dyad_items[6][1]:.4f}")
                        st.metric(dyad_items[7][0], f"{dyad_items[7][1]:.4f}")

                    # basic emotion statistics
                    st.divider()
                    st.subheader("emotion statistics")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("average emotion scores:")
                        for emotion, value in avg_emotions.items():
                            st.write(f"{emotion}: {value:.4f}")

                    with col2:
                        st.write("vad scores:")
                        st.write(f"valence: {trajectory_pd.get('avg_valence', 0):.4f}")
                        st.write(f"arousal: {trajectory_pd.get('avg_arousal', 0):.4f}")
                        st.write(f"dominance: {trajectory_pd.get('avg_dominance', 0):.4f}")

                    # show recommendations if trajectories available
                    if trajectories_available:
                        st.divider()
                        st.subheader("recommendations")

                        with st.spinner("computing recommendations..."):
                            # load trajectories for comparison
                            trajectories = load_trajectories_with_types(
                                spark, trajectories_path
                            )

                            # count total books available
                            total_books_count = trajectories.count()

                            # get current book id
                            liked_id = trajectory.select("book_id").first()["book_id"]

                            # check if current book is in database
                            current_book_in_trajectories = trajectories.filter(
                                col("book_id") == liked_id
                            ).count() > 0

                            # display comparison info
                            if current_book_in_trajectories:
                                st.info(
                                    f"comparing against {total_books_count} books from database "
                                    f"(current book is included)"
                                )
                            else:
                                st.info(
                                    f"comparing against {total_books_count} books from database"
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

                            # get recommendations
                            recommendations = recommend(
                                spark, all_trajectories, liked_id, top_n=top_n
                            )

                            # display recommendations
                            st.success(
                                f"top {top_n} recommendations for {title} by {author}"
                            )

                            recs_pd = recommendations.toPandas()

                            for idx, row in recs_pd.iterrows():
                                with st.expander(
                                    f"{idx + 1}. {row['title']} by {row['author']} "
                                    f"(similarity: {row['similarity']:.4f})"
                                ):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("emotion scores:")
                                        st.write(f"joy: {row.get('avg_joy', 0):.4f}")
                                        st.write(f"sadness: {row.get('avg_sadness', 0):.4f}")
                                        st.write(f"fear: {row.get('avg_fear', 0):.4f}")
                                        st.write(f"anger: {row.get('avg_anger', 0):.4f}")

                                    with col2:
                                        st.write("vad scores:")
                                        st.write(f"valence: {row.get('avg_valence', 0):.4f}")
                                        st.write(f"arousal: {row.get('avg_arousal', 0):.4f}")
                                        st.write(f"dominance: {row.get('avg_dominance', 0):.4f}")

                            # download button
                            csv = recs_pd.to_csv(index=False)
                            st.download_button(
                                label="download recommendations as csv",
                                data=csv,
                                file_name=f"recommendations_{liked_id}.csv",
                                mime="text/csv",
                            )
                    else:
                        st.info(
                            "tip: recommendations require trajectories - run `python main.py` first"
                        )
                else:
                    st.error(
                        "failed to analyze book - check if the book exists and try again"
                    )
        else:
            st.warning("please provide a book id or upload a text file")


def show_explore_books():
    """show explore books page"""
    st.header("explore books")
    st.markdown("discover books by emotion characteristics")

    # check for trajectories
    output_dir = "output"
    trajectories_path = f"{output_dir}/trajectories"

    if not os.path.exists(trajectories_path):
        st.warning(
            f"trajectories not found in {trajectories_path} - "
            "run `python main.py` first to generate trajectories"
        )
        return

    emotion_type = st.selectbox(
        "explore by emotion",
        [
            "joy",
            "sadness",
            "fear",
            "anger",
            "anticipation",
            "disgust",
            "surprise",
            "trust",
        ],
    )

    top_n = st.slider("number of books to show", 10, 50, 20)

    if st.button("show top books", type="primary"):
        with st.spinner("loading trajectories..."):
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

                st.success(f"top {top_n} books by {emotion_type}")

                for idx, row in top_books_pd.iterrows():
                    st.write(f"{idx + 1}. {row['title']} by {row['author']}")
                    st.write(f"{emotion_type}: {row[emotion_col]:.4f}")
                    st.divider()


def show_about():
    """show about page"""
    st.header("about emoarc")
    st.markdown("""
    emoarc is an emotion trajectory analysis and recommendation system for project gutenberg books.

    features:
    - analyze emotion trajectories using nrc emotion lexicon (8 plutchik emotions)
    - calculate plutchik's emotion dyads for complex emotional understanding
    - generate word clouds for visual text analysis
    - analyze valence-arousal-dominance scores
    - get book recommendations based on similar emotion trajectories
    - interactive plots showing emotion trajectories over time

    how it works:
    1. books are segmented into 20 equal chunks (percentage-based chunking for trajectory comparability)
    2. each chunk is scored using nrc emotion and vad lexicons
    3. emotion trajectories are analyzed to identify patterns across the story arc
    4. plutchik's dyads combine basic emotions to reveal complex emotional tones
    5. lda topic modeling discovers thematic content from word distributions
    6. similar books are found using multi-signal similarity (emotions, topics, embeddings, trajectories)

    technical details:
    - built with apache spark for big data processing
    - uses nrc emotion lexicon and nrc vad lexicon
    - processes 75,000+ books from project gutenberg
    - implements plutchik's wheel of emotions theory

    usage:
    1. run `python main.py` to generate trajectories for all books
    2. use this app to search, analyze, and get recommendations
    3. explore word clouds and emotion dyads for deeper insights
    """)


if __name__ == "__main__":
    main()
