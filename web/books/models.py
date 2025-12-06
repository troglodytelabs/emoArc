"""
Django models for book emotion analysis.
"""
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator


class Book(models.Model):
    """Model representing a book with emotional analysis data."""

    # Basic metadata
    book_id = models.CharField(max_length=100, unique=True, db_index=True)
    title = models.CharField(max_length=500)
    author = models.CharField(max_length=300)

    # Emotion scores (Plutchik's 8 basic emotions)
    avg_anger = models.FloatField(default=0.0)
    avg_anticipation = models.FloatField(default=0.0)
    avg_disgust = models.FloatField(default=0.0)
    avg_fear = models.FloatField(default=0.0)
    avg_joy = models.FloatField(default=0.0)
    avg_sadness = models.FloatField(default=0.0)
    avg_surprise = models.FloatField(default=0.0)
    avg_trust = models.FloatField(default=0.0)

    # VAD scores
    avg_valence = models.FloatField(default=0.0)
    avg_arousal = models.FloatField(default=0.0)
    avg_dominance = models.FloatField(default=0.0)

    # Dyad scores (Plutchik combinations)
    avg_love = models.FloatField(default=0.0)  # joy + trust
    avg_submission = models.FloatField(default=0.0)  # trust + fear
    avg_alarm = models.FloatField(default=0.0)  # fear + surprise
    avg_disappointment = models.FloatField(default=0.0)  # surprise + sadness
    avg_remorse = models.FloatField(default=0.0)  # sadness + disgust
    avg_contempt = models.FloatField(default=0.0)  # disgust + anger
    avg_aggressiveness = models.FloatField(default=0.0)  # anger + anticipation
    avg_optimism = models.FloatField(default=0.0)  # anticipation + joy

    # Secondary dyads
    avg_guilt = models.FloatField(default=0.0)  # joy + fear
    avg_curiosity = models.FloatField(default=0.0)  # trust + surprise
    avg_despair = models.FloatField(default=0.0)  # fear + sadness
    avg_unbelief = models.FloatField(default=0.0)  # surprise + disgust
    avg_envy = models.FloatField(default=0.0)  # sadness + anger
    avg_cynicism = models.FloatField(default=0.0)  # disgust + anticipation
    avg_pride = models.FloatField(default=0.0)  # anger + joy
    avg_hope = models.FloatField(default=0.0)  # anticipation + trust
    avg_anxiety = models.FloatField(default=0.0)  # anticipation + fear
    avg_outrage = models.FloatField(default=0.0)  # surprise + anger

    # Narrative arc metrics
    num_chunks = models.IntegerField(default=20)
    rising_action_pct = models.FloatField(default=0.0)
    climax_position = models.FloatField(default=0.0)
    resolution_pct = models.FloatField(default=0.0)
    emotional_volatility = models.FloatField(default=0.0)

    # Genre classification (top 3 matches stored as JSON)
    primary_genre = models.CharField(max_length=100, blank=True)
    genre_scores = models.JSONField(default=dict)  # {"genre": score, ...}

    # Topic modeling results
    topic_summary = models.TextField(blank=True)
    dominant_themes = models.JSONField(default=list)  # [{"theme": "...", "probability": 0.3}, ...]

    # Emotion trajectory (stored as JSON array)
    emotion_trajectory = models.JSONField(default=list)  # [{chunk_index: 0, joy: 0.5, ...}, ...]

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['title']),
            models.Index(fields=['author']),
            models.Index(fields=['primary_genre']),
        ]

    def __str__(self):
        return f"{self.title} by {self.author}"

    @property
    def emotion_scores(self):
        """Get all basic emotion scores as a dict."""
        return {
            'anger': self.avg_anger,
            'anticipation': self.avg_anticipation,
            'disgust': self.avg_disgust,
            'fear': self.avg_fear,
            'joy': self.avg_joy,
            'sadness': self.avg_sadness,
            'surprise': self.avg_surprise,
            'trust': self.avg_trust,
        }

    @property
    def dyad_scores(self):
        """Get all dyad scores as a dict."""
        return {
            'love': self.avg_love,
            'submission': self.avg_submission,
            'alarm': self.avg_alarm,
            'disappointment': self.avg_disappointment,
            'remorse': self.avg_remorse,
            'contempt': self.avg_contempt,
            'aggressiveness': self.avg_aggressiveness,
            'optimism': self.avg_optimism,
            'guilt': self.avg_guilt,
            'curiosity': self.avg_curiosity,
            'despair': self.avg_despair,
            'unbelief': self.avg_unbelief,
            'envy': self.avg_envy,
            'cynicism': self.avg_cynicism,
            'pride': self.avg_pride,
            'hope': self.avg_hope,
            'anxiety': self.avg_anxiety,
            'outrage': self.avg_outrage,
        }

    @property
    def vad_scores(self):
        """Get VAD scores as a dict."""
        return {
            'valence': self.avg_valence,
            'arousal': self.avg_arousal,
            'dominance': self.avg_dominance,
        }

    def get_top_emotions(self, n=3):
        """Get top N emotions for this book."""
        emotions = self.emotion_scores
        return sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:n]

    def get_emotional_distance(self, other_book):
        """Calculate emotional distance to another book (Euclidean distance)."""
        if not isinstance(other_book, Book):
            return None

        # Calculate distance across all emotions
        distance = 0
        for emotion in ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']:
            self_val = getattr(self, f'avg_{emotion}')
            other_val = getattr(other_book, f'avg_{emotion}')
            distance += (self_val - other_val) ** 2

        return distance ** 0.5


class UploadedBook(models.Model):
    """Model for user-uploaded books for on-demand analysis."""

    title = models.CharField(max_length=500)
    author = models.CharField(max_length=300, blank=True)
    file = models.FileField(upload_to='uploads/')

    # Analysis results (links to a Book instance after processing)
    analyzed_book = models.OneToOneField(
        Book,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='upload_source'
    )

    # Processing status
    STATUS_CHOICES = [
        ('pending', 'Pending Analysis'),
        ('processing', 'Processing'),
        ('completed', 'Analysis Complete'),
        ('failed', 'Analysis Failed'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField(blank=True)

    # Metadata
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.title} ({self.status})"
