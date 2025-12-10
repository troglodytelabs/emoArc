"""
Text preprocessing: chunking, stopwords removal, tokenization.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode, regexp_extract
from pyspark.sql.types import (
    ArrayType,
    StringType,
    IntegerType,
    StructType,
    StructField,
)
import re
import nltk

# Download required NLTK data (stopwords are downloaded inside UDF if needed)
# Module-level download ensures availability before UDF execution
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Download wordnet for lemmatization
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)


def load_books(
    spark: SparkSession,
    books_dir: str,
    metadata_path: str,
    language: str = "en",
    limit: int = None,
):
    """
    Load books from directory and join with metadata.

    Args:
        spark: SparkSession
        books_dir: Directory containing book files
        metadata_path: Path to metadata CSV
        language: Filter by language (default: "en")
        limit: Limit number of books to process (for testing)

    Returns:
        DataFrame with columns: book_id, title, author, text
    """
    # Load metadata
    metadata_df = spark.read.option("header", "true").csv(metadata_path)

    # Filter by language
    if language:
        metadata_df = metadata_df.filter(col("Language") == language)

    # Select relevant columns
    metadata_df = metadata_df.select(
        col("Etext Number").alias("book_id"),
        col("Title").alias("title"),
        col("Authors").alias("author"),
    )

    # Filter out null/empty book_ids from metadata
    metadata_df = metadata_df.filter(
        col("book_id").isNotNull() & (col("book_id") != "")
    )

    if limit:
        metadata_df = metadata_df.limit(limit)

    # Collect target book IDs for filtering
    # This is efficient for small limits
    target_book_ids = None
    if limit:
        target_book_ids = [
            row.book_id for row in metadata_df.select("book_id").collect()
        ]

    # Use binaryFile format to read files lazily
    # This allows us to filter by path BEFORE reading the content
    # path, modificationTime, length, content
    books_df = spark.read.format("binaryFile").load(books_dir)

    # Extract book_id from path
    # We use the 'path' column provided by binaryFile source
    books_df = books_df.withColumn(
        "book_id_extracted", regexp_extract(col("path"), r"/([^/]+)$", 1)
    )

    # Filter books if we have a limit
    if target_book_ids:
        books_df = books_df.filter(col("book_id_extracted").isin(target_book_ids))

    # Convert binary content to string
    books_df = books_df.withColumn("text", col("content").cast("string"))

    # Clean text UDF
    def clean_text(text):
        if text is None:
            return ""
        # Remove Project Gutenberg headers/footers
        text = re.sub(r"\*\*\* START.*?\*\*\*", "", text, flags=re.DOTALL)
        text = re.sub(r"\*\*\* END.*?\*\*\*", "", text, flags=re.DOTALL)
        return text

    clean_text_udf = udf(clean_text, StringType())
    books_df = books_df.withColumn("text", clean_text_udf(col("text")))

    # Select only needed columns and alias book_id
    books_df = books_df.select(col("book_id_extracted").alias("book_id"), col("text"))

    # Join with metadata
    result_df = metadata_df.join(books_df, on="book_id", how="inner")

    return result_df


def create_chunks_df(spark: SparkSession, books_df, num_chunks: int = 20):
    """
    Create chunks from books DataFrame using fixed percentage-based chunking.

    Each book is divided into exactly num_chunks chunks (default 20 = 5% each).
    Returns chunks with preprocessed words as arrays (NOT exploded), which is much more
    memory efficient than the exploded form.

    Args:
        spark: SparkSession
        books_df: DataFrame with columns: book_id, title, author, text
        num_chunks: Fixed number of chunks per book (default: 20 = 5% each)

    Returns:
        DataFrame with columns: book_id, title, author, chunk_index, words (array of strings)
    """

    # Define chunk creation function with num_chunks captured in closure
    def make_chunk_udf(n_chunks):
        """Factory function to create chunk UDF with captured parameters."""

        def _create_chunks_wrapper(text):
            """Create exactly n_chunks percentage-based chunks from text."""
            try:
                if not text:
                    return []

                text_len = len(text)
                if text_len == 0:
                    return []

                # Always create exactly n_chunks (5% each by default)
                chunk_size = max(1, text_len // n_chunks)

                chunks = []
                for i in range(n_chunks):
                    start_idx = i * chunk_size
                    # Last chunk gets all remaining text
                    if i == n_chunks - 1:
                        end_idx = text_len
                    else:
                        end_idx = start_idx + chunk_size

                    chunk_text = text[start_idx:end_idx]
                    if chunk_text:  # Only add non-empty chunks
                        chunks.append({"chunk_index": i, "chunk_text": chunk_text})

                return chunks if chunks else [{"chunk_index": 0, "chunk_text": text}]
            except Exception:
                # Fallback: return entire text as single chunk if anything goes wrong
                return [{"chunk_index": 0, "chunk_text": str(text) if text else ""}]

        return udf(
            _create_chunks_wrapper,
            ArrayType(
                StructType(
                    [
                        StructField("chunk_index", IntegerType(), True),
                        StructField("chunk_text", StringType(), True),
                    ]
                )
            ),
        )

    # Create UDF with num_chunks
    chunk_udf = make_chunk_udf(num_chunks)

    # Define preprocessing function - completely self-contained, defined inside this function
    def _preprocess_wrapper(text):
        """Wrapper function for UDF that handles preprocessing."""
        if not text:
            return []

        # Import inside function to ensure availability on workers
        import re
        import nltk

        # Get stopwords - download if needed
        try:
            from nltk.corpus import stopwords

            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", quiet=True)
            stop_words = set(stopwords.words("english"))
        except Exception:
            # Fallback if nltk not available
            stop_words = set()

        # Convert to lowercase
        text_lower = text.lower()

        # Remove special characters but keep apostrophes for contractions
        text_clean = re.sub(r"[^a-z0-9\s']", " ", text_lower)

        # Handle contractions to avoid matching partial words
        text_clean = re.sub(r"'s\b", "", text_clean)  # Remove 's
        text_clean = re.sub(r"n't\b", " not", text_clean)  # isn't -> is not
        text_clean = re.sub(r"'re\b", " are", text_clean)
        text_clean = re.sub(r"'ve\b", " have", text_clean)
        text_clean = re.sub(r"'ll\b", " will", text_clean)
        text_clean = re.sub(r"'d\b", " would", text_clean)
        text_clean = re.sub(r"'", "", text_clean)  # Remove remaining apostrophes

        # Tokenize
        words = text_clean.split()

        # Remove stopwords and very short words
        words = [w for w in words if w not in stop_words and len(w) > 2]

        return words

    # Create preprocessing UDF - defined inside function to avoid module serialization
    preprocess_udf = udf(_preprocess_wrapper, ArrayType(StringType()))

    # Create chunks
    chunks_df = books_df.select(
        col("book_id"),
        col("title"),
        col("author"),
        explode(chunk_udf(col("text"))).alias("chunk"),
    )

    # Extract chunk_index and preprocess text to words in one step
    # We skip chunk_text since it's not used anywhere downstream
    chunks_df = chunks_df.select(
        col("book_id"),
        col("title"),
        col("author"),
        col("chunk.chunk_index").alias("chunk_index"),
        preprocess_udf(col("chunk.chunk_text")).alias("words"),
    )

    # Return with words as array - NOT exploded
    # This is much more memory efficient: ~2000 rows vs ~10 million rows
    return chunks_df
