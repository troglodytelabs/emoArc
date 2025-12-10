"""
Model persistence utilities for saving and loading trained ML models.
"""

import os


def save_models(models, output_dir):
    """
    Save trained models to disk for resume capability.

    Args:
        models: Dictionary containing word2vec, lda, and cv_model
        output_dir: Base output directory
    """
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    if models.get("word2vec") is not None:
        word2vec_path = os.path.join(models_dir, "word2vec")
        models["word2vec"].save(word2vec_path)
        print(f"  ✓ Saved Word2Vec model to {word2vec_path}")

    if models.get("lda") is not None:
        lda_path = os.path.join(models_dir, "lda")
        models["lda"].save(lda_path)
        print(f"  ✓ Saved LDA model to {lda_path}")

    if models.get("cv_model") is not None:
        cv_path = os.path.join(models_dir, "count_vectorizer")
        models["cv_model"].save(cv_path)
        print(f"  ✓ Saved CountVectorizer model to {cv_path}")


def load_models(output_dir):
    """
    Load previously trained models from disk.

    Args:
        output_dir: Base output directory

    Returns:
        Dictionary with loaded models, or empty dict if not found
    """
    from pyspark.ml.feature import Word2VecModel, CountVectorizerModel
    from pyspark.ml.clustering import LDAModel

    models = {}
    models_dir = os.path.join(output_dir, "models")

    if not os.path.exists(models_dir):
        return models

    word2vec_path = os.path.join(models_dir, "word2vec")
    if os.path.exists(word2vec_path):
        try:
            models["word2vec"] = Word2VecModel.load(word2vec_path)
            print(f"  ✓ Loaded Word2Vec model from {word2vec_path}")
        except Exception as e:
            print(f"  ⚠ Could not load Word2Vec model: {e}")

    lda_path = os.path.join(models_dir, "lda")
    if os.path.exists(lda_path):
        try:
            models["lda"] = LDAModel.load(lda_path)
            print(f"  ✓ Loaded LDA model from {lda_path}")
        except Exception as e:
            print(f"  ⚠ Could not load LDA model: {e}")

    cv_path = os.path.join(models_dir, "count_vectorizer")
    if os.path.exists(cv_path):
        try:
            models["cv_model"] = CountVectorizerModel.load(cv_path)
            print(f"  ✓ Loaded CountVectorizer model from {cv_path}")
        except Exception as e:
            print(f"  ⚠ Could not load CountVectorizer model: {e}")

    return models
