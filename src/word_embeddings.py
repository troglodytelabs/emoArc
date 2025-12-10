"""
Word embeddings using Spark MLlib Word2Vec.
Generates semantic representations of text chunks for better similarity computation.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.feature import Word2Vec, Word2VecModel
import numpy as np


def train_word2vec(
    spark: SparkSession, chunks_df, vector_size: int = 100, min_count: int = 5
):
    """
    Train Word2Vec model on the corpus.

    Args:
        spark: SparkSession
        chunks_df: DataFrame with columns: book_id, chunk_index, words (array of strings)
        vector_size: Dimension of word vectors (default: 100)
        min_count: Minimum word count to be included (default: 5)

    Returns:
        Trained Word2VecModel
    """
    # chunks_df already has words as array, just rename for Word2Vec
    word_sequences = chunks_df.select("book_id", "chunk_index", col("words"))

    # Train Word2Vec model
    word2vec = Word2Vec(
        vectorSize=vector_size,
        minCount=min_count,
        inputCol="words",
        outputCol="word_vectors",
    )

    model = word2vec.fit(word_sequences)

    return model


def compute_chunk_embeddings(
    spark: SparkSession, chunks_df, word2vec_model: Word2VecModel
):
    """
    Compute average word embeddings for each chunk.

    Args:
        spark: SparkSession
        chunks_df: DataFrame with columns: book_id, chunk_index, words (array of strings)
        word2vec_model: Trained Word2VecModel

    Returns:
        DataFrame with chunk embeddings (average of word vectors)
    """
    # chunks_df already has words as array
    word_sequences = chunks_df.select("book_id", "chunk_index", col("words"))

    # Transform to get word vectors
    chunk_vectors = word2vec_model.transform(word_sequences)

    # Extract vector values and compute average
    def average_vector(word_vectors):
        """Compute average of word vectors in a chunk."""
        if not word_vectors or len(word_vectors) == 0:
            return None

        # word_vectors is a DenseVector, convert to numpy array
        if hasattr(word_vectors, "toArray"):
            return word_vectors.toArray().tolist()
        return None

    # Extract vector as array
    vector_to_array_udf = udf(
        lambda v: v.toArray().tolist() if v is not None else None,
        ArrayType(DoubleType()),
    )

    chunk_embeddings = chunk_vectors.withColumn(
        "embedding_vector", vector_to_array_udf(col("word_vectors"))
    ).select("book_id", "chunk_index", "embedding_vector")

    return chunk_embeddings


def compute_book_embedding(spark: SparkSession, chunk_embeddings_df):
    """
    Compute average embedding for each book (average of all chunk embeddings).

    Args:
        spark: SparkSession
        chunk_embeddings_df: DataFrame with columns: book_id, chunk_index, embedding_vector

    Returns:
        DataFrame with book-level embeddings
    """
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        ArrayType,
        DoubleType,
    )

    # Get vector size from first non-null embedding
    vector_size = None
    sample = (
        chunk_embeddings_df.filter(col("embedding_vector").isNotNull())
        .limit(1)
        .collect()
    )
    if sample:
        vector_size = len(sample[0]["embedding_vector"])

    if vector_size is None:
        # No embeddings, return empty
        schema = StructType(
            [
                StructField("book_id", StringType(), True),
                StructField("book_embedding", ArrayType(DoubleType()), True),
            ]
        )
        return spark.createDataFrame([], schema)

    def average_embeddings(vectors):
        """Compute average of multiple embedding vectors."""
        import numpy as np

        if not vectors or len(vectors) == 0:
            return None

        # Filter out None values
        valid_vectors = [v for v in vectors if v is not None]
        if len(valid_vectors) == 0:
            return None

        # # Convert to numpy arrays and compute mean
        # # Limit to first 100 vectors to avoid memory issues
        # if len(valid_vectors) > 100:
        #     # Sample for very large books
        #     import random

        #     valid_vectors = random.sample(valid_vectors, 100)

        np_vectors = [np.array(v) for v in valid_vectors]
        avg_vector = np.mean(np_vectors, axis=0)
        return avg_vector.tolist()

    average_udf = udf(average_embeddings, ArrayType(DoubleType()))

    book_embeddings = (
        chunk_embeddings_df.groupBy("book_id")
        .agg(collect_list("embedding_vector").alias("chunk_embeddings"))
        .withColumn("book_embedding", average_udf(col("chunk_embeddings")))
        .select("book_id", "book_embedding")
    )

    return book_embeddings


def compute_embedding_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector (list)
        embedding2: Second embedding vector (list)

    Returns:
        Cosine similarity score (0-1)
    """
    import numpy as np  # Import inside function for UDF compatibility

    if not embedding1 or not embedding2:
        return 0.0

    try:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        # Normalize to 0-1 range (cosine similarity is -1 to 1, but embeddings are usually positive)
        return float((similarity + 1) / 2.0)
    except Exception:
        return 0.0
