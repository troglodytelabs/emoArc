"""
Enhanced views for book emotion analysis with meaningful, interpretable insights.
"""
from django.shortcuts import render, get_object_or_404
from django.db.models import Q, Avg, Max, Min, Count, StdDev
from django.core.paginator import Paginator
from django.http import JsonResponse
from .models import Book
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np


# ============================================================================
# HELPER FUNCTIONS FOR INTERPRETABLE METRICS
# ============================================================================

def calculate_percentile_rank(value, all_values):
    """Calculate what percentile this value falls into (0-100)."""
    if not all_values or value is None:
        return 50
    sorted_vals = sorted([v for v in all_values if v is not None])
    if not sorted_vals:
        return 50
    rank = sum(1 for v in sorted_vals if v <= value)
    return (rank / len(sorted_vals)) * 100


def get_emotion_interpretation(emotion, percentile):
    """Get human-readable interpretation of emotion percentile."""
    interpretations = {
        'joy': {
            90: "exceptionally uplifting and cheerful",
            75: "notably optimistic and positive",
            50: "moderately joyful",
            25: "subdued joy",
            0: "minimal joy or positivity"
        },
        'sadness': {
            90: "profoundly melancholic and sorrowful",
            75: "notably sad and tragic",
            50: "moderate sadness",
            25: "light melancholy",
            0: "minimal sadness"
        },
        'fear': {
            90: "intensely terrifying and suspenseful",
            75: "notably fearful and tense",
            50: "moderate tension",
            25: "light apprehension",
            0: "minimal fear"
        },
        'anger': {
            90: "extremely intense and wrathful",
            75: "notably angry and confrontational",
            50: "moderate conflict",
            25: "light irritation",
            0: "minimal anger"
        },
        'surprise': {
            90: "full of shocking twists and revelations",
            75: "notably surprising and unexpected",
            50: "moderate unpredictability",
            25: "light surprises",
            0: "highly predictable"
        },
        'trust': {
            90: "deeply faithful and secure",
            75: "notably trusting and reliable",
            50: "moderate trust",
            25: "light wariness",
            0: "deeply suspicious"
        },
    }

    emotion_desc = interpretations.get(emotion, {})
    for threshold in [90, 75, 50, 25, 0]:
        if percentile >= threshold:
            return emotion_desc.get(threshold, "moderate")
    return "moderate"


def get_vad_interpretation(valence, arousal, dominance):
    """Interpret VAD scores into emotional tone description."""
    tone_parts = []

    # Valence interpretation
    if valence > 0.6:
        tone_parts.append("predominantly positive and upbeat")
    elif valence > 0.5:
        tone_parts.append("slightly positive")
    elif valence > 0.4:
        tone_parts.append("slightly negative")
    else:
        tone_parts.append("predominantly negative and dark")

    # Arousal interpretation
    if arousal > 0.6:
        tone_parts.append("highly intense and emotionally charged")
    elif arousal > 0.5:
        tone_parts.append("moderately intense")
    elif arousal > 0.4:
        tone_parts.append("calm and measured")
    else:
        tone_parts.append("very subdued and quiet")

    # Dominance interpretation
    if dominance > 0.6:
        tone_parts.append("characters exert strong control")
    elif dominance > 0.5:
        tone_parts.append("balanced power dynamics")
    elif dominance > 0.4:
        tone_parts.append("characters face external pressures")
    else:
        tone_parts.append("characters feel powerless")

    return ", ".join(tone_parts)


def detect_narrative_structure(trajectory):
    """Analyze emotion trajectory to detect narrative structure."""
    if not trajectory or len(trajectory) < 5:
        return None

    # Calculate overall emotional intensity (avg of all emotions)
    intensities = []
    for chunk in trajectory:
        emotions = [chunk.get(e, 0) for e in ['joy', 'fear', 'sadness', 'anger', 'surprise']]
        intensities.append(sum(emotions))

    if not intensities:
        return None

    # Find climax (point of highest intensity)
    climax_idx = intensities.index(max(intensities))
    climax_pct = (climax_idx / len(intensities)) * 100

    # Calculate rising action (increase before climax)
    if climax_idx > 0:
        rising_slope = (intensities[climax_idx] - intensities[0]) / climax_idx if climax_idx > 0 else 0
    else:
        rising_slope = 0

    # Calculate falling action (decrease after climax)
    if climax_idx < len(intensities) - 1:
        falling_slope = (intensities[-1] - intensities[climax_idx]) / (len(intensities) - climax_idx - 1)
    else:
        falling_slope = 0

    # Determine narrative arc type
    if climax_pct < 30:
        arc_type = "In Medias Res (starts at peak tension)"
    elif climax_pct > 70:
        arc_type = "Slow Burn (builds to climactic ending)"
    else:
        arc_type = "Classic Arc (rising action → climax → resolution)"

    return {
        'climax_position': climax_pct,
        'climax_intensity': max(intensities),
        'arc_type': arc_type,
        'rising_slope': rising_slope,
        'falling_slope': falling_slope,
        'intensities': intensities,
    }


# ============================================================================
# VIEWS
# ============================================================================

def home(request):
    """Home page with book search and filtering."""
    search_query = request.GET.get('q', '')
    genre_filter = request.GET.get('genre', '')
    emotion_filter = request.GET.get('emotion', '')
    sort_by = request.GET.get('sort', '-created_at')

    books = Book.objects.all()

    if search_query:
        books = books.filter(
            Q(title__icontains=search_query) |
            Q(author__icontains=search_query) |
            Q(book_id__icontains=search_query)
        )

    if genre_filter:
        books = books.filter(primary_genre__icontains=genre_filter)

    if emotion_filter and emotion_filter in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'trust', 'disgust', 'anticipation']:
        sort_by = f'-avg_{emotion_filter}'

    books = books.order_by(sort_by)
    paginator = Paginator(books, 24)
    page_obj = paginator.get_page(request.GET.get('page'))

    unique_genres = Book.objects.exclude(primary_genre='').values_list('primary_genre', flat=True).distinct()

    # Enhanced statistics
    total_books = Book.objects.count()
    all_books = Book.objects.all()

    # Calculate statistics for interpretation
    stats = {
        'total_books': total_books,
        'avg_valence': Book.objects.aggregate(Avg('avg_valence'))['avg_valence__avg'] or 0.5,
        'avg_arousal': Book.objects.aggregate(Avg('avg_arousal'))['avg_arousal__avg'] or 0.5,
        'most_joyful': all_books.order_by('-avg_joy').first(),
        'most_fearful': all_books.order_by('-avg_fear').first(),
        'most_sad': all_books.order_by('-avg_sadness').first(),
    }

    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'genre_filter': genre_filter,
        'emotion_filter': emotion_filter,
        'sort_by': sort_by,
        'unique_genres': unique_genres,
        'stats': stats,
    }

    return render(request, 'books/home.html', context)


def book_detail(request, book_id):
    """Detailed analysis page with interpretable insights."""
    book = get_object_or_404(Book, book_id=book_id)
    all_books = list(Book.objects.all())

    # Calculate percentile rankings for each emotion
    emotion_percentiles = {}
    emotion_interpretations = {}

    for emotion in ['joy', 'sadness', 'fear', 'anger', 'surprise', 'trust', 'disgust', 'anticipation']:
        value = getattr(book, f'avg_{emotion}')
        all_values = [getattr(b, f'avg_{emotion}') for b in all_books]
        percentile = calculate_percentile_rank(value, all_values)
        emotion_percentiles[emotion] = percentile
        emotion_interpretations[emotion] = get_emotion_interpretation(emotion, percentile)

    # VAD interpretation
    vad_interpretation = get_vad_interpretation(book.avg_valence, book.avg_arousal, book.avg_dominance)

    # Narrative structure analysis
    narrative_analysis = detect_narrative_structure(book.emotion_trajectory) if book.emotion_trajectory else None

    # Find similar books
    similar_books = []
    for other_book in all_books[:100]:
        if other_book.id != book.id:
            distance = book.get_emotional_distance(other_book)
            if distance is not None:
                similar_books.append((other_book, distance))
    similar_books.sort(key=lambda x: x[1])
    similar_books = [b[0] for b in similar_books[:5]]

    # Create enhanced visualizations
    radar_chart = create_enhanced_radar_chart(book, emotion_percentiles)
    trajectory_chart = create_narrative_arc_chart(book, narrative_analysis) if book.emotion_trajectory else None
    heatmap_chart = create_emotion_heatmap(book) if book.emotion_trajectory else None
    vad_chart = create_vad_chart(book)

    # Get dominant emotions and dyads
    top_emotions = book.get_top_emotions(5)
    top_dyads = sorted(book.dyad_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    context = {
        'book': book,
        'emotion_percentiles': emotion_percentiles,
        'emotion_interpretations': emotion_interpretations,
        'vad_interpretation': vad_interpretation,
        'narrative_analysis': narrative_analysis,
        'similar_books': similar_books,
        'radar_chart': radar_chart,
        'trajectory_chart': trajectory_chart,
        'heatmap_chart': heatmap_chart,
        'vad_chart': vad_chart,
        'top_emotions': top_emotions,
        'top_dyads': top_dyads,
    }

    return render(request, 'books/book_detail.html', context)


def methodology(request):
    """Explain how emotion scores are calculated and what they mean."""
    return render(request, 'books/methodology.html')


def compare_books(request):
    """Compare multiple books side-by-side."""
    book_ids = request.GET.getlist('books')

    if not book_ids:
        books = Book.objects.all().order_by('title')[:100]
        return render(request, 'books/compare_select.html', {'books': books})

    books = Book.objects.filter(book_id__in=book_ids)

    if len(books) < 2:
        return render(request, 'books/compare_select.html', {
            'error': 'Please select at least 2 books to compare',
            'books': Book.objects.all().order_by('title')[:100]
        })

    comparison_chart = create_comparison_radar_chart(books)

    comparison_data = []
    for book in books:
        comparison_data.append({
            'book': book,
            'top_emotions': book.get_top_emotions(3),
            'genre': book.primary_genre or 'Unknown',
        })

    context = {
        'books': books,
        'comparison_chart': comparison_chart,
        'comparison_data': comparison_data,
    }

    return render(request, 'books/compare.html', context)


def genre_explorer(request):
    """Explore books by genre with aggregate statistics."""
    from django.db.models import Count

    genre_stats = {}
    genres = Book.objects.exclude(primary_genre='').values_list('primary_genre', flat=True).distinct()

    for genre in genres:
        genre_books = Book.objects.filter(primary_genre=genre)
        stats = genre_books.aggregate(
            count=Count('id'),
            avg_joy=Avg('avg_joy'),
            avg_sadness=Avg('avg_sadness'),
            avg_fear=Avg('avg_fear'),
            avg_anger=Avg('avg_anger'),
            avg_valence=Avg('avg_valence'),
            avg_arousal=Avg('avg_arousal'),
        )
        genre_stats[genre] = stats

    selected_genre = request.GET.get('genre', '')
    genre_books = None
    if selected_genre:
        genre_books = Book.objects.filter(primary_genre=selected_genre).order_by('-avg_joy')[:20]

    context = {
        'genre_stats': genre_stats,
        'selected_genre': selected_genre,
        'genre_books': genre_books,
    }

    return render(request, 'books/genre_explorer.html', context)


def recommendations(request, book_id):
    """Get book recommendations based on emotional similarity."""
    book = get_object_or_404(Book, book_id=book_id)
    all_books = Book.objects.exclude(id=book.id)
    recommendations = []

    for other_book in all_books:
        distance = book.get_emotional_distance(other_book)
        if distance is not None:
            recommendations.append({
                'book': other_book,
                'distance': distance,
                'similarity': max(0, 100 - (distance * 100)),  # Convert to 0-100 similarity
            })

    recommendations.sort(key=lambda x: x['distance'])
    recommendations = recommendations[:10]

    context = {
        'source_book': book,
        'recommendations': recommendations,
    }

    return render(request, 'books/recommendations.html', context)


# ============================================================================
# ENHANCED CHART CREATION FUNCTIONS
# ============================================================================

def create_enhanced_radar_chart(book, percentiles):
    """Create radar chart with percentile context."""
    emotions = list(book.emotion_scores.keys())
    values = list(book.emotion_scores.values())
    percentile_values = [percentiles.get(e, 50) for e in emotions]

    emotions_display = [e.capitalize() for e in emotions]

    fig = go.Figure()

    # Add percentile ring (shows where this book ranks)
    fig.add_trace(go.Scatterpolar(
        r=percentile_values,
        theta=emotions_display,
        fill='toself',
        name='Percentile Rank',
        line=dict(color='rgba(150, 150, 150, 0.3)', width=1, dash='dot'),
        fillcolor='rgba(150, 150, 150, 0.1)',
    ))

    # Add actual values
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=emotions_display,
        fill='toself',
        name='This Book',
        line=dict(color='rgb(99, 110, 250)', width=3),
        fillcolor='rgba(99, 110, 250, 0.3)',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(values) * 1.2, 100) if values else 100]
            )
        ),
        showlegend=True,
        height=450,
        margin=dict(l=80, r=80, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_narrative_arc_chart(book, narrative_analysis):
    """Create comprehensive narrative arc visualization."""
    if not book.emotion_trajectory or not narrative_analysis:
        return None

    chunk_indices = list(range(len(book.emotion_trajectory)))
    intensities = narrative_analysis['intensities']
    climax_idx = int((narrative_analysis['climax_position'] / 100) * len(chunk_indices))

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=('Emotional Intensity Throughout Narrative', 'Individual Emotions'),
        vertical_spacing=0.12
    )

    # Top plot: Overall intensity with climax marker
    fig.add_trace(
        go.Scatter(
            x=chunk_indices,
            y=intensities,
            mode='lines+markers',
            name='Total Intensity',
            line=dict(color='rgb(99, 110, 250)', width=3),
            fill='toze ro',
            fillcolor='rgba(99, 110, 250, 0.2)',
        ),
        row=1, col=1
    )

    # Mark climax
    fig.add_trace(
        go.Scatter(
            x=[climax_idx],
            y=[intensities[climax_idx]],
            mode='markers+text',
            name='Climax',
            marker=dict(color='red', size=15, symbol='star'),
            text=['CLIMAX'],
            textposition='top center',
            textfont=dict(size=12, color='red'),
        ),
        row=1, col=1
    )

    # Bottom plot: Individual emotions
    emotions_to_plot = ['joy', 'sadness', 'fear', 'anger']
    colors = ['#10b981', '#3b82f6', '#ef4444', '#f59e0b']

    for emotion, color in zip(emotions_to_plot, colors):
        values = [chunk.get(emotion, 0) for chunk in book.emotion_trajectory]
        fig.add_trace(
            go.Scatter(
                x=chunk_indices,
                y=values,
                mode='lines',
                name=emotion.capitalize(),
                line=dict(color=color, width=2),
            ),
            row=2, col=1
        )

    fig.update_xaxes(title_text="Narrative Progress (%)", row=2, col=1)
    fig.update_yaxes(title_text="Intensity", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)

    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_emotion_heatmap(book):
    """Create heatmap showing emotion intensity across the narrative."""
    if not book.emotion_trajectory:
        return None

    emotions = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']

    # Build matrix: rows = emotions, cols = chunks
    z_data = []
    for emotion in emotions:
        values = [chunk.get(emotion, 0) for chunk in book.emotion_trajectory]
        z_data.append(values)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=list(range(len(book.emotion_trajectory))),
        y=[e.capitalize() for e in emotions],
        colorscale='RdYlGn',
        showscale=True,
        hoverongaps=False,
        colorbar=dict(title="Intensity"),
    ))

    fig.update_layout(
        title='Emotion Intensity Heatmap',
        xaxis_title='Narrative Progress',
        yaxis_title='Emotion',
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_vad_chart(book):
    """Create enhanced VAD visualization with interpretation."""
    vad_scores = book.vad_scores

    fig = go.Figure()

    # Create bars with gradient colors
    colors = ['#10b981', '#f59e0b', '#8b5cf6']

    for i, (dimension, value) in enumerate(vad_scores.items()):
        fig.add_trace(go.Bar(
            x=[dimension.capitalize()],
            y=[value],
            name=dimension.capitalize(),
            marker=dict(color=colors[i]),
            text=[f"{value:.3f}"],
            textposition='auto',
            showlegend=False,
        ))

    # Add reference line at 0.5
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="Neutral (0.5)", annotation_position="right")

    fig.update_layout(
        title='Emotional Dimensions (VAD Model)',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        bargap=0.3,
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_comparison_radar_chart(books):
    """Create comparison radar chart for multiple books."""
    fig = go.Figure()

    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    emotions_display = [e.capitalize() for e in emotions]

    colors = ['rgb(99, 110, 250)', 'rgb(239, 85, 59)', 'rgb(0, 204, 150)', 'rgb(171, 99, 250)', 'rgb(255, 159, 64)']

    for i, book in enumerate(books):
        values = [getattr(book, f'avg_{e}') for e in emotions]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=emotions_display,
            fill='toself',
            name=book.title[:30],
            line=dict(color=colors[i % len(colors)], width=2),
            fillcolor=colors[i % len(colors)].replace('rgb', 'rgba').replace(')', ', 0.2)'),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max([max([getattr(b, f'avg_{e}') for e in emotions]) for b in books]) * 1.2]
            )
        ),
        showlegend=True,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
