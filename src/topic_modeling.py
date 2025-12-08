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
    # comprehensive english stop words list for better topic quality
    stop_words = [
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each",
        "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",
        "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
        "its", "itself", "just", "me", "might", "more", "most", "must", "my", "myself", "no",
        "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
        "ours", "ourselves", "out", "over", "own", "said", "same", "she", "should", "so", "some",
        "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
        "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
        "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
        # common narrative words that don't add semantic value
        "one", "two", "may", "upon", "also", "well", "much", "many", "make", "made", "get", "got",
        "go", "went", "come", "came", "take", "took", "see", "saw", "know", "knew", "think",
        "thought", "tell", "told", "ask", "asked", "give", "gave", "find", "found", "seem",
        "seemed", "look", "looked", "put", "without", "within", "toward", "however", "therefore",
        "thus", "hence", "indeed", "moreover", "nevertheless", "otherwise", "yet", "still", "even",
        "ever", "never", "always", "often", "sometimes", "already", "quite", "rather", "almost",
        "perhaps", "maybe", "certainly", "surely"
    ]

    # Group words by chunk
    word_sequences = chunks_df.groupBy("book_id", "chunk_index").agg(
        collect_list("word").alias("words")
    )

    # Convert to term frequency vectors with stop word filtering
    # maxDF filters words that appear in >50% of documents (too common)
    count_vectorizer = CountVectorizer(
        inputCol="words",
        outputCol="raw_features",
        vocabSize=vocab_size,
        minDF=min_df,
        maxDF=0.5  # remove words appearing in >50% of chunks
    )

    cv_model = count_vectorizer.fit(word_sequences)

    # filter stop words from vocabulary after fitting
    # we need to refit with filtered vocabulary
    vocab = cv_model.vocabulary
    filtered_vocab = [w for w in vocab if w.lower() not in stop_words and len(w) > 2]

    # create new word sequences with filtered words only
    def filter_words(words):
        if not words:
            return []
        return [w for w in words if w.lower() not in stop_words and len(w) > 2]

    from pyspark.sql.functions import udf
    from pyspark.sql.types import ArrayType, StringType
    filter_udf = udf(filter_words, ArrayType(StringType()))

    word_sequences = word_sequences.withColumn("words", filter_udf(col("words")))

    # refit with filtered words
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

    # get top words for each topic using describeTopics
    topics_described = lda_model.describeTopics(maxTermsPerTopic=top_words_per_topic)
    topics_list = topics_described.collect()

    # interpret each topic
    all_topics = []
    for topic_idx in range(len(book_topics)):
        topic_prob = book_topics[topic_idx]

        # get top words for this topic from describeTopics
        if topic_idx < len(topics_list):
            topic_row = topics_list[topic_idx]
            term_indices = topic_row['termIndices']

            # convert term indices to words
            top_words = []
            for term_idx in term_indices:
                if term_idx < len(vocab):
                    top_words.append(vocab[term_idx])
        else:
            top_words = []

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



def train_per_book_lda(
    spark: SparkSession,
    chunks_df,
    num_topics: int = 5,
    vocab_size: int = 1000,
    min_df: int = 1,
    max_iter: int = 20
):
    """
    Train separate LDA model for each book (per-book topics).
    
    Args:
        spark: SparkSession
        chunks_df: DataFrame with columns: book_id, chunk_index, word
        num_topics: Number of topics per book (default: 5, fewer than corpus-wide)
        vocab_size: Max vocabulary per book (default: 1000)
        min_df: Minimum document frequency (default: 1 for per-book)
        max_iter: LDA iterations (default: 20, fewer for speed)
    
    Returns:
        DataFrame with columns: book_id, topics (array of topic word arrays)
    """
    from pyspark.sql.functions import udf, collect_list
    from pyspark.sql.types import ArrayType, StringType, StructType, StructField, FloatType
    
    # comprehensive english stop words list
    stop_words = set([
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each",
        "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",
        "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
        "its", "itself", "just", "me", "might", "more", "most", "must", "my", "myself", "no",
        "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
        "ours", "ourselves", "out", "over", "own", "said", "same", "she", "should", "so", "some",
        "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
        "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
        "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
        "one", "two", "may", "upon", "also", "well", "much", "many", "make", "made", "get", "got",
        "go", "went", "come", "came", "take", "took", "see", "saw", "know", "knew", "think",
        "thought", "tell", "told", "ask", "asked", "give", "gave", "find", "found", "seem",
        "seemed", "look", "looked", "put", "without", "within", "toward", "however", "therefore",
        "thus", "hence", "indeed", "moreover", "nevertheless", "otherwise", "yet", "still", "even",
        "ever", "never", "always", "often", "sometimes", "already", "quite", "rather", "almost",
        "perhaps", "maybe", "certainly", "surely"
    ])
    
    # get unique book ids
    book_ids = chunks_df.select("book_id").distinct().rdd.flatMap(lambda x: x).collect()
    
    results = []
    
    for book_id in book_ids:
        try:
            # filter chunks for this book
            book_chunks = chunks_df.filter(col("book_id") == book_id)
            
            # group words by chunk, filter stop words
            word_sequences = book_chunks.groupBy("book_id", "chunk_index").agg(
                collect_list("word").alias("words")
            )
            
            # filter stop words and short words
            def filter_words(words):
                if not words:
                    return []
                return [w for w in words if w.lower() not in stop_words and len(w) > 2]
            
            filter_udf = udf(filter_words, ArrayType(StringType()))
            word_sequences = word_sequences.withColumn("words", filter_udf(col("words")))
            
            # check if we have enough data
            word_count = word_sequences.count()
            if word_count < 5:  # need at least 5 chunks
                # return empty topics
                results.append((book_id, []))
                continue
            
            # create count vectorizer for this book
            cv = CountVectorizer(
                inputCol="words",
                outputCol="features",
                vocabSize=vocab_size,
                minDF=min_df
            )
            
            cv_model = cv.fit(word_sequences)
            vectorized = cv_model.transform(word_sequences)
            
            # check vocabulary size
            vocab = cv_model.vocabulary
            if len(vocab) < num_topics:  # can't create more topics than vocabulary
                # return single topic with all words
                top_words = vocab[:min(5, len(vocab))]
                results.append((book_id, [top_words]))
                continue
            
            # train LDA model for this book
            lda = LDA(
                k=min(num_topics, len(vocab)),  # ensure k <= vocab size
                maxIter=max_iter,
                featuresCol="features",
                seed=42
            )
            
            lda_model = lda.fit(vectorized)
            
            # extract top words for each topic
            topics_desc = lda_model.describeTopics(maxTermsPerTopic=5).collect()
            
            book_topics = []
            for topic_row in topics_desc:
                term_indices = topic_row['termIndices']
                top_words = [vocab[idx] for idx in term_indices if idx < len(vocab)]
                book_topics.append(top_words[:5])  # top 5 words per topic
            
            results.append((book_id, book_topics))
            
        except Exception as e:
            # if LDA fails for this book, store empty topics
            print(f"Warning: LDA failed for book {book_id}: {e}")
            results.append((book_id, []))
    
    # create dataframe from results
    schema = StructType([
        StructField("book_id", StringType(), True),
        StructField("book_topics", ArrayType(ArrayType(StringType())), True)
    ])
    
    results_df = spark.createDataFrame(results, schema)
    
    return results_df
