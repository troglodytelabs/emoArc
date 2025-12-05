"""
Topic modeling using Spark MLlib LDA.
Extracts thematic topics from books to enhance recommendation quality.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, IDF, IDFModel
from pyspark.ml.clustering import LDA, LDAModel
import numpy as np


def prepare_topic_features(
    spark: SparkSession, chunks_df, vocab_size: int = 5000, min_df: int = 2
):
    """
    Prepare features for LDA: convert word sequences to term frequency vectors.

    Args:
        spark: SparkSession
        chunks_df: DataFrame with columns: book_id, chunk_index, word
        vocab_size: Maximum vocabulary size (default: 5000)
        min_df: Minimum document frequency (default: 2)

    Returns:
        Tuple of (feature_df, count_vectorizer_model)
    """
    # Group words by chunk
    word_sequences = chunks_df.groupBy("book_id", "chunk_index").agg(
        collect_list("word").alias("words")
    )

    # Convert to term frequency vectors
    count_vectorizer = CountVectorizer(
        inputCol="words", outputCol="raw_features", vocabSize=vocab_size, minDF=min_df
    )

    cv_model = count_vectorizer.fit(word_sequences)
    feature_df = cv_model.transform(word_sequences)

    return feature_df, cv_model


def train_lda(
    spark: SparkSession, feature_df, num_topics: int = 10, max_iter: int = 50
):
    """
    Train LDA topic model.

    Args:
        spark: SparkSession
        feature_df: DataFrame with raw_features column (from prepare_topic_features)
        num_topics: Number of topics to extract (default: 10)
        max_iter: Maximum iterations (default: 50)

    Returns:
        Trained LDAModel
    """
    # Optionally apply IDF for better topic quality
    # For now, we'll use raw term frequencies as LDA works well with them

    # Train LDA
    lda = LDA(
        k=num_topics,
        maxIter=max_iter,
        featuresCol="raw_features",
        topicDistributionCol="topic_distribution",
    )

    model = lda.fit(feature_df)

    return model


def get_chunk_topics(spark: SparkSession, feature_df, lda_model: LDAModel):
    """
    Get topic distributions for each chunk.

    Args:
        spark: SparkSession
        feature_df: DataFrame with raw_features column
        lda_model: Trained LDAModel

    Returns:
        DataFrame with topic distributions per chunk
    """
    # Transform to get topic distributions
    topic_df = lda_model.transform(feature_df)

    # Extract topic distribution as array
    def extract_topics(distribution):
        """Extract topic distribution vector."""
        if distribution is None:
            return None
        if hasattr(distribution, "toArray"):
            return distribution.toArray().tolist()
        return None

    topic_array_udf = udf(extract_topics, ArrayType(DoubleType()))

    chunk_topics = topic_df.withColumn(
        "topics", topic_array_udf(col("topic_distribution"))
    ).select("book_id", "chunk_index", "topics")

    return chunk_topics


def compute_book_topics(spark: SparkSession, chunk_topics_df):
    """
    Compute average topic distribution for each book.

    Args:
        spark: SparkSession
        chunk_topics_df: DataFrame with columns: book_id, chunk_index, topics

    Returns:
        DataFrame with book-level topic distributions
    """

    def average_topics(topic_vectors):
        """Compute average of multiple topic distributions."""
        import numpy as np  # Import inside UDF for worker nodes

        if not topic_vectors or len(topic_vectors) == 0:
            return None

        # Filter out None values
        valid_topics = [t for t in topic_vectors if t is not None]
        if len(valid_topics) == 0:
            return None

        # # For very large books, sample to avoid memory issues
        # if len(valid_topics) > 200:
        #     import random

        #     valid_topics = random.sample(valid_topics, 200)

        # Convert to numpy arrays and compute mean
        np_topics = [np.array(t) for t in valid_topics]
        avg_topics = np.mean(np_topics, axis=0)
        return avg_topics.tolist()

    average_udf = udf(average_topics, ArrayType(DoubleType()))

    book_topics = (
        chunk_topics_df.groupBy("book_id")
        .agg(collect_list("topics").alias("chunk_topics"))
        .withColumn("book_topics", average_udf(col("chunk_topics")))
        .select("book_id", "book_topics")
    )

    return book_topics


def compute_topic_similarity(topics1, topics2):
    """
    Compute cosine similarity between two topic distributions.

    Args:
        topics1: First topic distribution (list)
        topics2: Second topic distribution (list)

    Returns:
        Cosine similarity score (0-1)
    """
    import numpy as np  # Import inside function for UDF compatibility

    if not topics1 or not topics2:
        return 0.0

    try:
        vec1 = np.array(topics1)
        vec2 = np.array(topics2)

        # Ensure same length
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        # Topic distributions are probabilities, so similarity is already 0-1
        return float(similarity)
    except Exception:
        return 0.0
