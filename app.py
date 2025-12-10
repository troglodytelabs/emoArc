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

from core import (
    create_spark_session,
    load_trajectories_with_types,
    load_chunk_scores_with_types,
    load_metadata,
    get_input_trajectory,
    find_books_by_emotion_preferences,
)
from recommender import recommend
from topic_modeling import (
    prepare_topic_features,
    train_lda,
    get_chunk_topics,
    compute_book_topics,
    interpret_book_topics,
    generate_book_summary,
)

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


def get_dyad_explanation(dyad_name, score):
    """get human-readable explanation for a dyad score"""
    explanations = {
        'love': {
            'components': 'joy + trust',
            'meaning': 'warmth, affection, connection',
            'high': 'strong themes of love, friendship, or loyalty',
            'low': 'absence of romantic or affectionate elements'
        },
        'submission': {
            'components': 'trust + fear',
            'meaning': 'respect for authority, compliance',
            'high': 'themes of hierarchy, obedience, or reverence',
            'low': 'independence or defiance of authority'
        },
        'awe': {
            'components': 'fear + surprise',
            'meaning': 'wonder mixed with apprehension',
            'high': 'mysterious, supernatural, or overwhelming experiences',
            'low': 'mundane or familiar situations'
        },
        'disapproval': {
            'components': 'surprise + sadness',
            'meaning': 'disappointment, letdown',
            'high': 'unexpected negative outcomes or betrayals',
            'low': 'predictable or satisfying events'
        },
        'remorse': {
            'components': 'sadness + disgust',
            'meaning': 'guilt, regret, self-reproach',
            'high': 'moral conflict or consequences of actions',
            'low': 'characters without regrets or self-doubt'
        },
        'contempt': {
            'components': 'disgust + anger',
            'meaning': 'scorn, disdain, hatred',
            'high': 'intense conflict, villains, or moral judgment',
            'low': 'neutral or positive interpersonal dynamics'
        },
        'aggressiveness': {
            'components': 'anger + anticipation',
            'meaning': 'hostility, confrontation',
            'high': 'action, combat, or interpersonal conflict',
            'low': 'passive or peaceful narrative'
        },
        'optimism': {
            'components': 'anticipation + joy',
            'meaning': 'hope, excitement for the future',
            'high': 'adventure, aspirations, positive outlook',
            'low': 'pessimistic or dark themes'
        }
    }

    info = explanations.get(dyad_name, {})
    if score > 0.15:
        intensity = 'high'
        desc = info.get('high', '')
    elif score < 0.05:
        intensity = 'low'
        desc = info.get('low', '')
    else:
        intensity = 'moderate'
        desc = info.get('meaning', '')

    return {
        'components': info.get('components', ''),
        'meaning': info.get('meaning', ''),
        'intensity': intensity,
        'description': desc
    }


def detect_narrative_arc(chunk_scores_pd):
    """detect narrative arc pattern from emotion trajectory"""
    if len(chunk_scores_pd) < 10:
        return "insufficient data for arc detection"

    # split into beginning, middle, end
    third = len(chunk_scores_pd) // 3
    beginning = chunk_scores_pd.iloc[:third]
    middle = chunk_scores_pd.iloc[third:2*third]
    end = chunk_scores_pd.iloc[2*third:]

    # analyze valence trajectory
    begin_valence = beginning['avg_valence'].mean()
    middle_valence = middle['avg_valence'].mean()
    end_valence = end['avg_valence'].mean()

    # analyze arousal (tension)
    begin_arousal = beginning['avg_arousal'].mean()
    middle_arousal = middle['avg_arousal'].mean()
    end_arousal = end['avg_arousal'].mean()

    # detect patterns
    patterns = []

    # classic narrative arcs
    if middle_valence < begin_valence and end_valence > middle_valence:
        patterns.append("tragedy-to-triumph (classic hero's journey)")
    elif begin_valence > middle_valence and end_valence > begin_valence:
        patterns.append("triumph-after-trial (overcoming adversity)")
    elif begin_valence > end_valence:
        patterns.append("descent (tragic arc)")
    elif end_valence > begin_valence:
        patterns.append("ascent (comedic arc)")
    else:
        patterns.append("steady emotional tone")

    # tension patterns
    if middle_arousal > begin_arousal and middle_arousal > end_arousal:
        patterns.append("rising-falling tension (climax in middle)")
    elif end_arousal > begin_arousal and end_arousal > middle_arousal:
        patterns.append("building tension (climax at end)")

    return " | ".join(patterns) if patterns else "complex narrative structure"


def analyze_emotional_journey(chunk_scores_pd):
    """analyze the emotional journey throughout the book"""
    insights = []

    # dominant emotion
    emotion_cols = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    avg_emotions = {col: chunk_scores_pd[col].mean() for col in emotion_cols if col in chunk_scores_pd.columns}

    if avg_emotions:
        dominant = max(avg_emotions.items(), key=lambda x: x[1])
        insights.append(f"dominant emotion: {dominant[0]} ({dominant[1]:.3f})")

    # emotional range
    if 'avg_valence' in chunk_scores_pd.columns:
        val_range = chunk_scores_pd['avg_valence'].max() - chunk_scores_pd['avg_valence'].min()
        if val_range > 0.3:
            insights.append(f"high emotional variability (range: {val_range:.3f})")
        else:
            insights.append(f"stable emotional tone (range: {val_range:.3f})")

    # trajectory volatility
    if 'joy' in chunk_scores_pd.columns:
        joy_std = chunk_scores_pd['joy'].std()
        if joy_std > 0.1:
            insights.append("volatile joy trajectory (dramatic emotional shifts)")
        else:
            insights.append("consistent joy levels")

    return insights


def get_genre_profiles():
    """
    get emotion profiles for different genres.
    each genre has expected ranges for emotions, dyads, vad, and narrative patterns.
    """
    return {
        'romance': {
            'emotions': {'joy': (0.15, 0.35), 'trust': (0.15, 0.30), 'anticipation': (0.12, 0.25), 'sadness': (0.05, 0.15)},
            'dyads': {'love': (0.15, 0.35), 'optimism': (0.12, 0.25)},
            'vad': {'valence': (0.55, 0.75), 'arousal': (0.45, 0.65)},
            'narrative': ['ascent (comedic arc)', 'triumph-after-trial'],
            'description': 'emotional stories focused on relationships and affection'
        },
        'thriller': {
            'emotions': {'fear': (0.15, 0.35), 'anticipation': (0.15, 0.30), 'surprise': (0.12, 0.25)},
            'dyads': {'awe': (0.15, 0.30), 'aggressiveness': (0.10, 0.25)},
            'vad': {'valence': (0.35, 0.55), 'arousal': (0.60, 0.85)},
            'narrative': ['building tension', 'rising-falling tension'],
            'description': 'suspenseful stories with high tension and unexpected twists'
        },
        'horror': {
            'emotions': {'fear': (0.20, 0.40), 'disgust': (0.15, 0.30), 'sadness': (0.12, 0.25)},
            'dyads': {'awe': (0.15, 0.35), 'remorse': (0.10, 0.25)},
            'vad': {'valence': (0.25, 0.45), 'arousal': (0.65, 0.90), 'dominance': (0.30, 0.50)},
            'narrative': ['descent (tragic arc)'],
            'description': 'dark stories designed to evoke fear and dread'
        },
        'war/military': {
            'emotions': {'anger': (0.15, 0.30), 'fear': (0.12, 0.25), 'sadness': (0.10, 0.22)},
            'dyads': {'aggressiveness': (0.15, 0.35), 'contempt': (0.10, 0.25), 'submission': (0.08, 0.20)},
            'vad': {'valence': (0.35, 0.55), 'arousal': (0.65, 0.85), 'dominance': (0.55, 0.75)},
            'narrative': ['tragedy-to-triumph', 'descent'],
            'description': 'stories of conflict, combat, and military operations'
        },
        'adventure': {
            'emotions': {'anticipation': (0.15, 0.35), 'joy': (0.12, 0.28), 'surprise': (0.10, 0.22)},
            'dyads': {'optimism': (0.15, 0.35), 'aggressiveness': (0.08, 0.20)},
            'vad': {'valence': (0.50, 0.70), 'arousal': (0.60, 0.80), 'dominance': (0.55, 0.75)},
            'narrative': ['tragedy-to-triumph', 'triumph-after-trial'],
            'description': 'exciting journeys with exploration and challenges'
        },
        'mystery': {
            'emotions': {'anticipation': (0.15, 0.30), 'surprise': (0.12, 0.25), 'fear': (0.08, 0.20)},
            'dyads': {'disapproval': (0.10, 0.25), 'awe': (0.08, 0.20)},
            'vad': {'valence': (0.40, 0.60), 'arousal': (0.55, 0.75)},
            'narrative': ['building tension', 'rising-falling tension'],
            'description': 'puzzling stories focused on solving crimes or secrets'
        },
        'comedy': {
            'emotions': {'joy': (0.20, 0.40), 'surprise': (0.12, 0.25), 'trust': (0.10, 0.22)},
            'dyads': {'optimism': (0.15, 0.35), 'love': (0.10, 0.25)},
            'vad': {'valence': (0.60, 0.80), 'arousal': (0.45, 0.65)},
            'narrative': ['ascent (comedic arc)', 'triumph-after-trial'],
            'description': 'humorous stories designed to entertain and amuse'
        },
        'tragedy': {
            'emotions': {'sadness': (0.18, 0.38), 'fear': (0.12, 0.25), 'anger': (0.10, 0.22)},
            'dyads': {'remorse': (0.15, 0.35), 'disapproval': (0.10, 0.25)},
            'vad': {'valence': (0.25, 0.45), 'arousal': (0.50, 0.70)},
            'narrative': ['descent (tragic arc)'],
            'description': 'sorrowful stories of downfall and loss'
        },
        'noir/dark': {
            'emotions': {'sadness': (0.15, 0.30), 'disgust': (0.12, 0.25), 'anger': (0.10, 0.22)},
            'dyads': {'contempt': (0.15, 0.30), 'remorse': (0.12, 0.25)},
            'vad': {'valence': (0.25, 0.45), 'arousal': (0.50, 0.70), 'dominance': (0.40, 0.60)},
            'narrative': ['descent', 'complex narrative structure'],
            'description': 'dark, cynical stories with morally complex characters'
        },
        'philosophical': {
            'emotions': {'trust': (0.12, 0.25), 'anticipation': (0.10, 0.22), 'sadness': (0.08, 0.18)},
            'dyads': {'submission': (0.10, 0.22), 'disapproval': (0.08, 0.18)},
            'vad': {'valence': (0.45, 0.65), 'arousal': (0.35, 0.55)},
            'narrative': ['steady emotional tone', 'complex narrative structure'],
            'description': 'contemplative stories exploring ideas and meaning',
            'topics': ['religion/spiritual', 'education/knowledge', 'emotion/psychology']
        },
        'history': {
            'emotions': {'anticipation': (0.10, 0.25), 'trust': (0.08, 0.20), 'anger': (0.08, 0.18)},
            'dyads': {'aggressiveness': (0.08, 0.20), 'submission': (0.08, 0.18)},
            'vad': {'valence': (0.40, 0.60), 'arousal': (0.45, 0.65)},
            'narrative': ['tragedy-to-triumph', 'descent', 'complex narrative structure'],
            'description': 'historical accounts and events from the past',
            'topics': ['war/military', 'power/politics', 'wealth/society', 'religion/spiritual']
        },
        'psychology': {
            'emotions': {'trust': (0.10, 0.25), 'sadness': (0.08, 0.20), 'fear': (0.08, 0.18)},
            'dyads': {'submission': (0.08, 0.20), 'remorse': (0.08, 0.18)},
            'vad': {'valence': (0.40, 0.65), 'arousal': (0.35, 0.60)},
            'narrative': ['steady emotional tone', 'complex narrative structure'],
            'description': 'exploration of human mind, behavior, and emotions',
            'topics': ['emotion/psychology', 'education/knowledge', 'family/domestic']
        },
        'biography': {
            'emotions': {'anticipation': (0.10, 0.25), 'joy': (0.08, 0.22), 'sadness': (0.08, 0.20)},
            'dyads': {'optimism': (0.10, 0.22), 'disapproval': (0.08, 0.18)},
            'vad': {'valence': (0.40, 0.65), 'arousal': (0.45, 0.70)},
            'narrative': ['tragedy-to-triumph', 'triumph-after-trial', 'descent'],
            'description': 'life stories and personal narratives',
            'topics': ['power/politics', 'wealth/society', 'family/domestic', 'education/knowledge']
        },
        'science': {
            'emotions': {'anticipation': (0.12, 0.28), 'surprise': (0.10, 0.22), 'trust': (0.08, 0.20)},
            'dyads': {'optimism': (0.10, 0.25), 'awe': (0.08, 0.20)},
            'vad': {'valence': (0.45, 0.70), 'arousal': (0.40, 0.65)},
            'narrative': ['steady emotional tone', 'ascent (comedic arc)'],
            'description': 'scientific exploration and discovery',
            'topics': ['education/knowledge', 'nature/rural', 'power/politics']
        },
        'travel/exploration': {
            'emotions': {'anticipation': (0.15, 0.30), 'joy': (0.10, 0.25), 'surprise': (0.10, 0.22)},
            'dyads': {'optimism': (0.12, 0.28), 'awe': (0.10, 0.22)},
            'vad': {'valence': (0.50, 0.70), 'arousal': (0.50, 0.75)},
            'narrative': ['ascent (comedic arc)', 'steady emotional tone'],
            'description': 'journeys, expeditions, and cultural exploration',
            'topics': ['adventure/travel', 'nature/rural', 'urban/city', 'maritime/nautical']
        },
        'social commentary': {
            'emotions': {'anger': (0.10, 0.25), 'disgust': (0.08, 0.20), 'sadness': (0.08, 0.20)},
            'dyads': {'contempt': (0.10, 0.25), 'disapproval': (0.10, 0.22)},
            'vad': {'valence': (0.30, 0.55), 'arousal': (0.45, 0.70)},
            'narrative': ['descent', 'complex narrative structure'],
            'description': 'critique of society, culture, and social issues',
            'topics': ['wealth/society', 'power/politics', 'urban/city', 'family/domestic']
        }
    }


def classify_genre_from_emotions(emotion_scores, dyad_scores, vad_scores, narrative_arc, lda_topics=None):
    """
    classify book genre based on emotion profile, dyads, vad scores, narrative arc, and optional lda topics.
    hybrid approach: uses plutchik emotions + lda topic themes for comprehensive classification.

    args:
        emotion_scores: dict of plutchik emotion scores
        dyad_scores: dict of plutchik dyad scores
        vad_scores: dict of valence/arousal/dominance scores
        narrative_arc: detected narrative arc pattern
        lda_topics: optional list of dominant lda topic themes (from topic modeling)

    returns list of matching genres with confidence scores.
    """
    genre_profiles = get_genre_profiles()
    genre_matches = []

    for genre_name, profile in genre_profiles.items():
        match_score = 0.0
        max_possible_score = 0.0

        # if lda topics provided, adjust weights to incorporate content themes
        # otherwise use emotion-only weighting
        if lda_topics:
            emotion_weight = 0.30  # reduced from 0.40
            dyad_weight = 0.20     # reduced from 0.25
            vad_weight = 0.20      # reduced from 0.25
            narrative_weight = 0.10  # same
            topic_weight = 0.20    # new weight for lda topics
        else:
            emotion_weight = 0.40
            dyad_weight = 0.25
            vad_weight = 0.25
            narrative_weight = 0.10
            topic_weight = 0.0

        # check emotion ranges
        for emotion, (min_val, max_val) in profile.get('emotions', {}).items():
            max_possible_score += emotion_weight / len(profile.get('emotions', {1: 1}))
            emotion_val = emotion_scores.get(emotion, 0)
            if min_val <= emotion_val <= max_val:
                match_score += emotion_weight / len(profile['emotions'])
            elif emotion_val > max_val:
                # partial credit if close to range
                overshoot = emotion_val - max_val
                if overshoot < 0.1:
                    match_score += (emotion_weight / len(profile['emotions'])) * 0.5
            elif emotion_val < min_val:
                # partial credit if close to range
                undershoot = min_val - emotion_val
                if undershoot < 0.1:
                    match_score += (emotion_weight / len(profile['emotions'])) * 0.5

        # check dyad ranges (25% of total weight)
        dyad_weight = 0.25
        for dyad, (min_val, max_val) in profile.get('dyads', {}).items():
            max_possible_score += dyad_weight / len(profile.get('dyads', {1: 1}))
            dyad_val = dyad_scores.get(dyad, 0)
            if min_val <= dyad_val <= max_val:
                match_score += dyad_weight / len(profile['dyads'])
            elif min_val - 0.05 <= dyad_val <= max_val + 0.05:
                # partial credit if very close
                match_score += (dyad_weight / len(profile['dyads'])) * 0.5

        # check vad ranges (25% of total weight)
        vad_weight = 0.25
        for vad_dim, (min_val, max_val) in profile.get('vad', {}).items():
            max_possible_score += vad_weight / len(profile.get('vad', {1: 1}))
            vad_val = vad_scores.get(vad_dim, 0.5)
            if min_val <= vad_val <= max_val:
                match_score += vad_weight / len(profile['vad'])
            elif min_val - 0.1 <= vad_val <= max_val + 0.1:
                # partial credit if close
                match_score += (vad_weight / len(profile['vad'])) * 0.5

        # check narrative arc
        max_possible_score += narrative_weight
        for pattern in profile.get('narrative', []):
            if pattern in narrative_arc:
                match_score += narrative_weight
                break

        # check lda topic themes (if provided)
        if lda_topics and topic_weight > 0:
            genre_topics = profile.get('topics', [])
            if genre_topics:
                max_possible_score += topic_weight
                # count how many lda topics match genre's expected topics
                matching_topics = sum(1 for lda_topic in lda_topics if lda_topic in genre_topics)
                topic_match_ratio = matching_topics / len(genre_topics) if genre_topics else 0
                match_score += topic_weight * topic_match_ratio

        # normalize to 0-100 scale
        if max_possible_score > 0:
            confidence = (match_score / max_possible_score) * 100
        else:
            confidence = 0.0

        genre_matches.append({
            'genre': genre_name,
            'confidence': confidence,
            'description': profile.get('description', '')
        })

    # sort by confidence
    genre_matches.sort(key=lambda x: x['confidence'], reverse=True)

    return genre_matches


def create_genre_radar_chart(emotion_scores, dyad_scores, genre_profiles_to_compare=None):
    """
    create radar chart showing emotion profile compared to genre expectations.
    """
    import plotly.graph_objects as go

    # combine emotions and top dyads for visualization
    categories = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']
    values = [emotion_scores.get(cat, 0) for cat in categories]

    # create figure
    fig = go.Figure()

    # add genre comparisons FIRST (so they appear behind the book's profile)
    if genre_profiles_to_compare:
        colors = [
            ('rgb(239, 85, 59)', 'rgba(239, 85, 59, 0.15)'),
            ('rgb(0, 204, 150)', 'rgba(0, 204, 150, 0.15)'),
            ('rgb(171, 99, 250)', 'rgba(171, 99, 250, 0.15)')
        ]
        genre_profiles = get_genre_profiles()

        for idx, genre_name in enumerate(genre_profiles_to_compare[:3]):
            if genre_name in genre_profiles:
                profile = genre_profiles[genre_name]
                # use midpoint of ranges for visualization
                genre_values = []
                for cat in categories:
                    if cat in profile.get('emotions', {}):
                        min_val, max_val = profile['emotions'][cat]
                        genre_values.append((min_val + max_val) / 2)
                    else:
                        genre_values.append(0.05)  # default low value

                line_color, fill_color = colors[idx % len(colors)]
                fig.add_trace(go.Scatterpolar(
                    r=genre_values,
                    theta=categories,
                    fill='toself',
                    name=f'typical {genre_name}',
                    line=dict(color=line_color, width=2, dash='dot'),
                    fillcolor=fill_color,
                    opacity=0.7
                ))

    # add book's emotion profile ON TOP (more visible)
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='this book',
        line=dict(color='rgb(99, 110, 250)', width=3),
        fillcolor='rgba(99, 110, 250, 0.4)',
        marker=dict(size=8)
    ))

    # calculate max value for better scaling
    all_values = values.copy()
    if genre_profiles_to_compare:
        for genre_name in genre_profiles_to_compare[:3]:
            if genre_name in get_genre_profiles():
                profile = get_genre_profiles()[genre_name]
                for cat in categories:
                    if cat in profile.get('emotions', {}):
                        min_val, max_val = profile['emotions'][cat]
                        all_values.append(max_val)

    max_val = max(all_values) if all_values else 0.4

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, min(0.5, max_val * 1.3)],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        title="emotion profile radar chart",
        height=550
    )

    return fig


def get_emotion_star_rating(score, typical_range=None):
    """
    convert emotion score to star rating (1-5 stars).
    if typical_range provided, compare to that; otherwise use absolute scale.
    """
    if typical_range:
        min_val, max_val = typical_range
        midpoint = (min_val + max_val) / 2
        # compare to midpoint of typical range
        if score >= max_val * 1.2:
            return "★★★★★", "very high"
        elif score >= midpoint:
            return "★★★★☆", "high"
        elif score >= min_val:
            return "★★★☆☆", "typical"
        elif score >= min_val * 0.5:
            return "★★☆☆☆", "low"
        else:
            return "★☆☆☆☆", "very low"
    else:
        # absolute scale
        if score >= 0.25:
            return "★★★★★", "very high"
        elif score >= 0.18:
            return "★★★★☆", "high"
        elif score >= 0.12:
            return "★★★☆☆", "moderate"
        elif score >= 0.06:
            return "★★☆☆☆", "low"
        else:
            return "★☆☆☆☆", "very low"


def extract_high_emotion_passages(chunk_scores_pd, full_text, num_chunks=20, passages_per_emotion=2):
    """
    extract example passages from chunks with highest emotion scores.
    returns dict mapping emotions to list of (chunk_index, score, passage) tuples.
    """
    if not full_text:
        return {}

    emotions = ['joy', 'fear', 'sadness', 'anger', 'surprise']
    passages = {}

    # calculate chunk boundaries
    text_len = len(full_text)

    for emotion in emotions:
        if emotion not in chunk_scores_pd.columns:
            continue

        # find top chunks for this emotion
        top_chunks = chunk_scores_pd.nlargest(passages_per_emotion, emotion)

        emotion_passages = []
        for _, row in top_chunks.iterrows():
            chunk_idx = int(row['chunk_index'])
            score = float(row[emotion])

            # extract passage from full text
            start = (chunk_idx * text_len) // num_chunks
            end = ((chunk_idx + 1) * text_len) // num_chunks

            passage = full_text[start:end]

            # find a good excerpt (around 200-300 chars)
            excerpt_start = max(0, len(passage) // 2 - 150)
            excerpt_end = min(len(passage), excerpt_start + 300)
            excerpt = passage[excerpt_start:excerpt_end]

            # clean up excerpt (remove partial words at edges)
            if excerpt_start > 0:
                # find first space
                first_space = excerpt.find(' ')
                if first_space > 0:
                    excerpt = excerpt[first_space + 1:]

            if excerpt_end < len(passage):
                # find last space
                last_space = excerpt.rfind(' ')
                if last_space > 0:
                    excerpt = excerpt[:last_space]

            # add ellipsis
            if excerpt_start > 0:
                excerpt = "..." + excerpt
            if excerpt_end < len(passage):
                excerpt = excerpt + "..."

            emotion_passages.append({
                'chunk_index': chunk_idx + 1,  # 1-indexed for display
                'score': score,
                'passage': excerpt.strip()
            })

        passages[emotion] = emotion_passages

    return passages


def filter_books_by_genre(spark, trajectories_df, target_genre, confidence_threshold=40):
    """
    filter books by genre classification.
    returns dataframe of books matching the target genre.
    """
    # convert to pandas for easier processing
    trajectories_pd = trajectories_df.toPandas()

    # classify genre for each book
    matching_books = []
    for idx, row in trajectories_pd.iterrows():
        # prepare emotion scores
        emotion_scores = {
            'joy': row.get('avg_joy', 0),
            'trust': row.get('avg_trust', 0),
            'fear': row.get('avg_fear', 0),
            'surprise': row.get('avg_surprise', 0),
            'sadness': row.get('avg_sadness', 0),
            'disgust': row.get('avg_disgust', 0),
            'anger': row.get('avg_anger', 0),
            'anticipation': row.get('avg_anticipation', 0),
        }

        # calculate dyads
        dyad_scores = calculate_plutchik_dyads(emotion_scores)

        # vad scores
        vad_scores = {
            'valence': row.get('avg_valence', 0.5),
            'arousal': row.get('avg_arousal', 0.5),
            'dominance': row.get('avg_dominance', 0.5)
        }

        # simple narrative arc (we don't have chunk data here, so use valence as proxy)
        narrative_arc = "unknown"

        # classify genre (without lda topics for bulk filtering - too expensive)
        genre_matches = classify_genre_from_emotions(
            emotion_scores, dyad_scores, vad_scores, narrative_arc, lda_topics=None
        )

        # check if target genre matches with sufficient confidence
        for match in genre_matches:
            if match['genre'] == target_genre and match['confidence'] >= confidence_threshold:
                matching_books.append(row)
                break

    # convert back to spark dataframe
    if matching_books:
        import pandas as pd
        filtered_pd = pd.DataFrame(matching_books)
        filtered_df = spark.createDataFrame(filtered_pd)
        return filtered_df
    else:
        return None


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
    returns: (trajectory_df, chunk_scores_df, title, author, full_text, topic_summary)
    """
    trajectory = None
    chunk_scores = None
    title = None
    author = None
    full_text = None
    topic_summary = None

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

            # compute topic modeling summary
            try:
                with st.spinner("analyzing themes and topics..."):
                    feature_df, cv_model = prepare_topic_features(spark, chunks_df, vocab_size=5000, min_df=2)
                    lda_model = train_lda(spark, feature_df, num_topics=10, max_iter=50)
                    chunk_topics = get_chunk_topics(spark, feature_df, lda_model)
                    book_topics_df = compute_book_topics(spark, chunk_topics)
                    book_topics_list = book_topics_df.select("book_topics").first()["book_topics"]

                    if book_topics_list:
                        topic_interpretation = interpret_book_topics(book_topics_list, lda_model, cv_model)
                        topic_summary = {
                            "summary": generate_book_summary(topic_interpretation),
                            "dominant_themes": topic_interpretation.get("dominant_themes", []),
                            "all_topics": topic_interpretation.get("all_topics", [])
                        }
            except Exception as topic_error:
                st.warning(f"could not generate topic summary: {topic_error}")
                topic_summary = None

        except Exception as e:
            st.error(f"error processing text file: {e}")
            return None, None, None, None, None, None

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

            # compute topic modeling summary
            try:
                with st.spinner("analyzing themes and topics..."):
                    feature_df, cv_model = prepare_topic_features(spark, chunks_df, vocab_size=5000, min_df=2)
                    lda_model = train_lda(spark, feature_df, num_topics=10, max_iter=50)
                    chunk_topics = get_chunk_topics(spark, feature_df, lda_model)
                    book_topics_df = compute_book_topics(spark, chunk_topics)
                    book_topics_list = book_topics_df.select("book_topics").first()["book_topics"]

                    if book_topics_list:
                        topic_interpretation = interpret_book_topics(book_topics_list, lda_model, cv_model)
                        topic_summary = {
                            "summary": generate_book_summary(topic_interpretation),
                            "dominant_themes": topic_interpretation.get("dominant_themes", []),
                            "all_topics": topic_interpretation.get("all_topics", [])
                        }
            except Exception as topic_error:
                st.warning(f"could not generate topic summary: {topic_error}")
                topic_summary = None

    else:
        st.error("no input specified")
        return None, None, None, None, None, None

    return trajectory, chunk_scores, title, author, full_text, topic_summary


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
                trajectory, chunk_scores, title, author, full_text, topic_summary = get_input_trajectory(
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

                    # topic modeling summary
                    if topic_summary:
                        st.divider()
                        st.subheader("thematic content analysis (lda topics)")
                        st.caption("discovered themes from text content using topic modeling")

                        # display summary
                        st.info(f"**summary:** {topic_summary['summary']}")

                        # display dominant themes
                        if topic_summary.get('dominant_themes'):
                            st.write("**dominant themes (>10% probability):**")
                            for theme in topic_summary['dominant_themes']:
                                st.markdown(
                                    f"- **{theme['theme']}** ({theme['probability']*100:.1f}%)"
                                )
                                st.caption(f"  top words: {', '.join(theme['top_words'][:8])}")

                        # expandable section for all topics
                        with st.expander("view all discovered topics"):
                            st.write("all topics detected in this book:")
                            for topic in topic_summary.get('all_topics', [])[:10]:
                                st.markdown(
                                    f"**topic {topic['topic_id']}: {topic['theme']}** "
                                    f"({topic['probability']*100:.1f}%)"
                                )
                                st.caption(f"keywords: {', '.join(topic['top_words'])}")
                                st.write("")  # spacing

                    # narrative arc detection
                    st.divider()
                    st.subheader("narrative arc analysis")
                    narrative_arc = detect_narrative_arc(chunk_scores_pd)
                    st.info(f"detected pattern: {narrative_arc}")

                    # emotional journey insights
                    journey_insights = analyze_emotional_journey(chunk_scores_pd)
                    if journey_insights:
                        st.write("emotional journey insights:")
                        for insight in journey_insights:
                            st.write(f"• {insight}")

                    # emotion trajectory plot
                    st.divider()
                    st.subheader("emotion trajectory")
                    fig = plot_emotion_trajectory(chunk_scores_pd, title)
                    st.plotly_chart(fig, use_container_width=True)

                    # prepare emotion data for genre classification and dyads
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
                    vad_scores = {
                        'valence': trajectory_pd.get('avg_valence', 0.5),
                        'arousal': trajectory_pd.get('avg_arousal', 0.5),
                        'dominance': trajectory_pd.get('avg_dominance', 0.5)
                    }

                    # genre classification
                    st.divider()
                    st.subheader("genre classification")
                    st.caption("hybrid classification using plutchik emotions + lda topic themes")

                    # extract lda topic themes if available
                    lda_topic_themes = None
                    if topic_summary and topic_summary.get('dominant_themes'):
                        lda_topic_themes = [theme['theme'] for theme in topic_summary['dominant_themes']]

                    # classify genres (with optional lda topics for non-fiction genres)
                    genre_matches = classify_genre_from_emotions(
                        avg_emotions, dyads, vad_scores, narrative_arc, lda_topics=lda_topic_themes
                    )

                    # display top 5 genre matches
                    st.write("top genre matches:")
                    for i, match in enumerate(genre_matches[:5]):
                        st.markdown(
                            f"{i+1}. **{match['genre'].title()}** - {match['confidence']:.1f}% confidence"
                        )
                        st.caption(f"   {match['description']}")

                    # store top genre for recommendations
                    st.session_state.top_genre = genre_matches[0]['genre'] if genre_matches else None

                    # genre radar chart (emotion fingerprint)
                    st.divider()
                    st.subheader("emotion profile fingerprint")
                    st.caption("radar chart comparing this book's emotion profile to typical genre patterns")

                    # get top 3 genres for comparison
                    top_genres = [match['genre'] for match in genre_matches[:3]]
                    radar_fig = create_genre_radar_chart(avg_emotions, dyads, genre_profiles_to_compare=top_genres)
                    st.plotly_chart(radar_fig, use_container_width=True)

                    # genre comparison details
                    with st.expander("detailed genre comparison"):
                        st.write("how this book compares to typical genre characteristics:")
                        genre_profiles = get_genre_profiles()

                        for match in genre_matches[:3]:
                            genre_name = match['genre']
                            profile = genre_profiles[genre_name]

                            st.markdown(f"### {genre_name.title()} ({match['confidence']:.1f}% match)")

                            # emotion comparison
                            st.write("**emotion alignment:**")
                            for emotion, (min_val, max_val) in profile.get('emotions', {}).items():
                                book_val = avg_emotions.get(emotion, 0)
                                if min_val <= book_val <= max_val:
                                    st.write(f"✓ {emotion}: {book_val:.3f} (expected: {min_val:.2f}-{max_val:.2f})")
                                elif book_val < min_val:
                                    st.write(f"↓ {emotion}: {book_val:.3f} (below typical {min_val:.2f}-{max_val:.2f})")
                                else:
                                    st.write(f"↑ {emotion}: {book_val:.3f} (above typical {min_val:.2f}-{max_val:.2f})")

                            # dyad comparison
                            st.write("**dyad alignment:**")
                            for dyad, (min_val, max_val) in profile.get('dyads', {}).items():
                                book_dyad = dyads.get(dyad, 0)
                                if min_val <= book_dyad <= max_val:
                                    st.write(f"✓ {dyad}: {book_dyad:.3f} (expected: {min_val:.2f}-{max_val:.2f})")
                                elif book_dyad < min_val:
                                    st.write(f"↓ {dyad}: {book_dyad:.3f} (below typical {min_val:.2f}-{max_val:.2f})")
                                else:
                                    st.write(f"↑ {dyad}: {book_dyad:.3f} (above typical {min_val:.2f}-{max_val:.2f})")

                            st.write("")  # spacing

                    # find top dyads for highlighting
                    sorted_dyads = sorted(dyads.items(), key=lambda x: x[1], reverse=True)
                    top_3_dyads = [name for name, _ in sorted_dyads[:3]]

                    # display dyads with explanations
                    col1, col2 = st.columns(2)

                    dyad_items = list(dyads.items())
                    for i, (dyad_name, dyad_score) in enumerate(dyad_items):
                        explanation = get_dyad_explanation(dyad_name, dyad_score)

                        # alternate columns
                        target_col = col1 if i % 2 == 0 else col2

                        with target_col:
                            # highlight top dyads
                            if dyad_name in top_3_dyads:
                                st.metric(
                                    f"⭐ {dyad_name}",
                                    f"{dyad_score:.4f}",
                                    delta=f"{explanation['intensity']} intensity"
                                )
                            else:
                                st.metric(dyad_name, f"{dyad_score:.4f}")

                            # show explanation
                            st.caption(f"{explanation['components']} → {explanation['description']}")
                            st.write("")  # spacing

                    # emotion statistics with comparative context
                    st.divider()
                    st.subheader("emotion statistics")
                    st.caption("scores compared to typical genre ranges")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**emotion scores with ratings:**")
                        # get top predicted genre for comparison
                        top_genre_profile = None
                        if genre_matches and genre_matches[0]['genre'] in get_genre_profiles():
                            top_genre_profile = get_genre_profiles()[genre_matches[0]['genre']]

                        for emotion, value in avg_emotions.items():
                            # get typical range for this emotion in top genre
                            typical_range = None
                            if top_genre_profile and emotion in top_genre_profile.get('emotions', {}):
                                typical_range = top_genre_profile['emotions'][emotion]

                            stars, label = get_emotion_star_rating(value, typical_range)

                            # format with stars and typical range
                            if typical_range:
                                min_val, max_val = typical_range
                                st.write(f"{emotion}: {value:.3f} {stars} ({label})")
                                st.caption(f"  typical {genre_matches[0]['genre']}: {min_val:.2f}-{max_val:.2f}")
                            else:
                                st.write(f"{emotion}: {value:.3f} {stars} ({label})")

                    with col2:
                        st.write("**vad scores:**")
                        valence = trajectory_pd.get('avg_valence', 0)
                        arousal = trajectory_pd.get('avg_arousal', 0)
                        dominance = trajectory_pd.get('avg_dominance', 0)

                        v_stars, v_label = get_emotion_star_rating(valence)
                        a_stars, a_label = get_emotion_star_rating(arousal)
                        d_stars, d_label = get_emotion_star_rating(dominance)

                        st.write(f"valence: {valence:.3f} {v_stars}")
                        st.caption(f"  {v_label} (emotional positivity)")
                        st.write(f"arousal: {arousal:.3f} {a_stars}")
                        st.caption(f"  {a_label} (emotional intensity)")
                        st.write(f"dominance: {dominance:.3f} {d_stars}")
                        st.caption(f"  {d_label} (sense of control)")

                    # example passages with high emotions
                    if full_text:
                        st.divider()
                        st.subheader("example emotional passages")
                        st.caption("excerpts from chunks with highest emotion scores")

                        emotion_passages = extract_high_emotion_passages(
                            chunk_scores_pd, full_text, num_chunks=20, passages_per_emotion=1
                        )

                        # display passages in tabs
                        if emotion_passages:
                            tabs = st.tabs([f"{e.title()}" for e in ['joy', 'fear', 'sadness', 'anger', 'surprise']])

                            for idx, emotion in enumerate(['joy', 'fear', 'sadness', 'anger', 'surprise']):
                                with tabs[idx]:
                                    if emotion in emotion_passages and emotion_passages[emotion]:
                                        for passage_info in emotion_passages[emotion]:
                                            st.markdown(
                                                f"**Peak {emotion.title()} (chunk {passage_info['chunk_index']}, "
                                                f"score: {passage_info['score']:.3f})**"
                                            )
                                            st.info(f"_{passage_info['passage']}_")
                                    else:
                                        st.write(f"no strong {emotion} passages detected")

                    # show recommendations if trajectories available
                    if trajectories_available:
                        st.divider()
                        st.subheader("recommendations")

                        # genre filter option
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            enable_genre_filter = st.checkbox(
                                "filter recommendations by genre",
                                value=False,
                                help="only show books matching the top predicted genre"
                            )
                        with col2:
                            if enable_genre_filter and st.session_state.get('top_genre'):
                                st.info(f"filtering by: {st.session_state.top_genre}")

                        with st.spinner("computing recommendations..."):
                            # load trajectories for comparison
                            trajectories = load_trajectories_with_types(
                                spark, trajectories_path
                            )

                            # apply genre filter if enabled
                            if enable_genre_filter and st.session_state.get('top_genre'):
                                st.write(f"filtering books by genre: **{st.session_state.top_genre}**")
                                with st.spinner(f"finding {st.session_state.top_genre} books..."):
                                    genre_filtered = filter_books_by_genre(
                                        spark,
                                        trajectories,
                                        st.session_state.top_genre,
                                        confidence_threshold=40
                                    )

                                    if genre_filtered and genre_filtered.count() > 0:
                                        trajectories = genre_filtered
                                        st.success(f"found {trajectories.count()} {st.session_state.top_genre} books")
                                    else:
                                        st.warning(
                                            f"no books found matching {st.session_state.top_genre} genre. "
                                            "showing all recommendations instead."
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

                st.success(f"top {top_n} books by {emotion_type}")

                for idx, row in top_books_pd.iterrows():
                    st.write(f"{idx + 1}. {row['title']} by {row['author']}")
                    st.write(f"{emotion_type}: {row[emotion_col]:.4f}")
                    st.divider()


def show_find_books_by_emotions():
    """Show page for finding books by emotion preferences."""
    st.header("Find Books by Emotion Preferences")
    st.markdown(
        "Pick a vibe in one click, optionally fine-tune, then get matches."
    )

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
                value=level_from_value(emotion_preferences["anticipation"], use_high=True),
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
            trajectories = load_trajectories_with_types(spark, trajectories_path)

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
