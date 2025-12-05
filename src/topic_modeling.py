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
        
        # Convert to numpy arrays and compute mean
        np_topics = [np.array(t) for t in valid_topics]
        avg_topics = np.mean(np_topics, axis=0)
        return avg_topics.tolist()
    
    average_udf = udf(
        average_topics,
        ArrayType(DoubleType())
    )
    
    book_topics = chunk_topics_df.groupBy("book_id").agg(
        collect_list("topics").alias("chunk_topics")
    ).withColumn(
        "book_topics",
        average_udf(col("chunk_topics"))
    ).select(
        "book_id",
        "book_topics"
    )
    
    return book_topics


def compute_topic_similarity(topics1, topics2):
    """
    compute cosine similarity between two topic distributions

    args:
        topics1: first topic distribution (list)
        topics2: second topic distribution (list)

    returns:
        cosine similarity score (0-1)
    """
    import numpy as np  # import inside function for udf compatibility

    if not topics1 or not topics2:
        return 0.0

    try:
        vec1 = np.array(topics1)
        vec2 = np.array(topics2)

        # ensure same length
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]

        # compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        # topic distributions are probabilities, so similarity is already 0-1
        return float(similarity)
    except Exception:
        return 0.0


def interpret_book_topics(book_topics, lda_model, cv_model, top_words_per_topic=5):
    """
    interpret book's topic distribution with human-readable topic summaries

    args:
        book_topics: list of topic probabilities for a book
        lda_model: trained lda model
        cv_model: count vectorizer model (for vocabulary)
        top_words_per_topic: number of top words to extract per topic

    returns:
        dict with topic interpretations and dominant themes
    """
    if not book_topics or len(book_topics) == 0:
        return {"dominant_themes": [], "all_topics": []}

    # get vocabulary from count vectorizer
    vocab = cv_model.vocabulary

    # get topic-word distributions from lda model
    topics_matrix = lda_model.topicsMatrix()

    # interpret each topic
    all_topics = []
    for topic_idx in range(len(book_topics)):
        topic_prob = book_topics[topic_idx]

        # get top words for this topic
        topic_words_weights = []
        for word_idx in range(len(vocab)):
            weight = topics_matrix[word_idx][topic_idx]
            word = vocab[word_idx]
            topic_words_weights.append((word, weight))

        # sort by weight and get top words
        topic_words_weights.sort(key=lambda x: x[1], reverse=True)
        top_words = [word for word, _ in topic_words_weights[:top_words_per_topic]]

        all_topics.append({
            "topic_id": topic_idx,
            "probability": float(topic_prob),
            "top_words": top_words,
            "theme": infer_theme_from_words(top_words)
        })

    # sort topics by probability
    all_topics.sort(key=lambda x: x["probability"], reverse=True)

    # get dominant themes (>10% probability)
    dominant_themes = [
        t for t in all_topics if t["probability"] > 0.10
    ]

    return {
        "dominant_themes": dominant_themes,
        "all_topics": all_topics
    }


def infer_theme_from_words(words):
    """
    infer thematic label from top words using keyword matching

    args:
        words: list of top words for a topic

    returns:
        thematic label string
    """
    words_str = " ".join(words).lower()

    # define theme patterns
    themes = {
        "romance/love": ["love", "heart", "kiss", "romance", "marry", "wedding", "passion", "desire"],
        "war/military": ["war", "battle", "soldier", "army", "fight", "weapon", "enemy", "fought"],
        "mystery/crime": ["murder", "detective", "clue", "suspect", "crime", "investigate", "mystery"],
        "adventure/travel": ["journey", "travel", "adventure", "explore", "voyage", "quest", "road"],
        "maritime/nautical": ["ship", "sea", "captain", "voyage", "sail", "ocean", "boat", "naval"],
        "fantasy/magic": ["magic", "wizard", "spell", "dragon", "enchant", "mystical", "sorcerer"],
        "family/domestic": ["family", "home", "mother", "father", "child", "house", "domestic"],
        "nature/rural": ["nature", "forest", "tree", "rural", "farm", "country", "land", "field"],
        "urban/city": ["city", "street", "town", "urban", "building", "crowd", "busy"],
        "religion/spiritual": ["god", "church", "pray", "faith", "divine", "spirit", "holy", "soul"],
        "wealth/society": ["money", "rich", "poor", "society", "class", "wealth", "fortune"],
        "education/knowledge": ["learn", "study", "school", "teach", "knowledge", "book", "read"],
        "emotion/psychology": ["feel", "emotion", "thought", "mind", "heart", "soul", "sense"],
        "power/politics": ["king", "queen", "power", "rule", "govern", "empire", "throne", "royal"]
    }

    # count matches for each theme
    theme_scores = {}
    for theme, keywords in themes.items():
        score = sum(1 for keyword in keywords if keyword in words_str)
        if score > 0:
            theme_scores[theme] = score

    # return best matching theme or generic label
    if theme_scores:
        best_theme = max(theme_scores.items(), key=lambda x: x[1])[0]
        return best_theme
    else:
        # create generic label from top 3 words
        return " / ".join(words[:3])


def generate_book_summary(book_topics_interpretation):
    """
    generate human-readable summary from book's topic interpretation

    args:
        book_topics_interpretation: dict from interpret_book_topics()

    returns:
        string summary of book's themes
    """
    dominant = book_topics_interpretation.get("dominant_themes", [])

    if not dominant:
        return "themes unclear - insufficient topic data"

    if len(dominant) == 1:
        theme = dominant[0]
        return f"primarily {theme['theme']} ({theme['probability']*100:.0f}%)"

    elif len(dominant) == 2:
        t1, t2 = dominant[0], dominant[1]
        return f"{t1['theme']} ({t1['probability']*100:.0f}%) and {t2['theme']} ({t2['probability']*100:.0f}%)"

    else:
        # 3+ dominant themes
        themes_str = ", ".join([
            f"{t['theme']} ({t['probability']*100:.0f}%)"
            for t in dominant[:3]
        ])
        return f"mixed themes: {themes_str}"

