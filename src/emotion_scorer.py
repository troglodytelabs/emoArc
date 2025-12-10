"""
Score chunks with NRC Emotion and VAD lexicons.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as spark_sum, avg, count


def score_chunks_with_emotions(spark: SparkSession, chunks_df, emotion_df):
    """
    Score each chunk with NRC Emotion lexicon using normalized density scoring.

    Args:
        spark: SparkSession
        chunks_df: DataFrame with columns: book_id, title, author, chunk_index, chunk_word_count, word
        emotion_df: DataFrame with columns: word, emotion, value

    Returns:
        DataFrame with normalized emotion scores per chunk (per 1000 words)
    """
    from pyspark.sql.functions import first, when

    # Join chunks with emotion lexicon
    emotion_scores = chunks_df.join(
        emotion_df, chunks_df.word == emotion_df.word, "left"
    )

    # Group by book_id, chunk_index and emotion, count occurrences
    # Also preserve chunk_word_count using first() aggregation
    chunk_emotions = emotion_scores.groupBy(
        "book_id", "title", "author", "bookshelves", "chunk_index", "emotion"
    ).agg(
        count("emotion").alias("emotion_count"),
        first("chunk_word_count").alias("chunk_word_count")
    )

    # Pivot to get one row per chunk with all emotions
    # Use only Plutchik's 8 basic emotions (exclude "negative" and "positive" which are sentiment labels)
    chunk_emotions_pivot = (
        chunk_emotions.groupBy("book_id", "title", "author", "bookshelves", "chunk_index")
        .pivot(
            "emotion",
            values=[
                "anger",
                "anticipation",
                "disgust",
                "fear",
                "joy",
                "sadness",
                "surprise",
                "trust",
            ],
        )
        .agg(spark_sum("emotion_count").alias("count"))
        .fillna(0)
    )

    # Get chunk_word_count back (it's lost in pivot)
    word_counts = chunk_emotions.select(
        "book_id", "bookshelves", "chunk_index", "chunk_word_count"
    ).distinct()

    chunk_emotions_pivot = chunk_emotions_pivot.join(
        word_counts, on=["book_id", "bookshelves", "chunk_index"], how="left"
    )

    # NORMALIZE: Convert raw counts to density (per 1000 words)
    # Formula: (emotion_count / chunk_word_count) * 1000
    # This makes Shakespeare's complete works comparable to individual novels
    emotion_cols = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

    for emotion_col in emotion_cols:
        chunk_emotions_pivot = chunk_emotions_pivot.withColumn(
            emotion_col,
            when(col("chunk_word_count") > 0,
                 (col(emotion_col) / col("chunk_word_count")) * 1000.0)
            .otherwise(0.0)
        )

    return chunk_emotions_pivot


def score_chunks_with_vad(spark: SparkSession, chunks_df, vad_df):
    """
    Score each chunk with NRC VAD lexicon.

    Args:
        spark: SparkSession
        chunks_df: DataFrame with columns: book_id, title, author, chunk_index, word
        vad_df: DataFrame with columns: term, valence, arousal, dominance

    Returns:
        DataFrame with VAD scores per chunk
    """
    # Join chunks with VAD lexicon
    vad_scores = chunks_df.join(vad_df, chunks_df.word == vad_df.term, "left")

    # Group by book_id, chunk_index and compute averages
    chunk_vad = vad_scores.groupBy("book_id", "title", "author", "bookshelves", "chunk_index").agg(
        avg("valence").alias("avg_valence"),
        avg("arousal").alias("avg_arousal"),
        avg("dominance").alias("avg_dominance"),
        count("valence").alias("vad_word_count"),
    )

    # Fill NaN values with 0
    chunk_vad = chunk_vad.fillna(
        0, subset=["avg_valence", "avg_arousal", "avg_dominance"]
    )

    return chunk_vad


def combine_emotion_vad_scores(emotion_scores_df, vad_scores_df):
    """
    Combine emotion and VAD scores into a single DataFrame.

    Args:
        emotion_scores_df: DataFrame with emotion scores per chunk
        vad_scores_df: DataFrame with VAD scores per chunk

    Returns:
        Combined DataFrame with all scores
    """
    combined = emotion_scores_df.join(
        vad_scores_df, on=["book_id", "title", "author", "bookshelves", "chunk_index"], how="outer"
    )

    # Fill missing values with 0 (only Plutchik's 8 basic emotions)
    emotion_cols = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
    ]
    for col_name in emotion_cols:
        if col_name in combined.columns:
            combined = combined.fillna(0, subset=[col_name])

    return combined
