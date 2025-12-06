"""
Views for the book emotion analysis web app.
"""
from django.shortcuts import render, get_object_or_404
from django.db.models import Q, Avg
from django.core.paginator import Paginator
from django.http import JsonResponse
from .models import Book
import plotly.graph_objects as go
import plotly.express as px
import json


def home(request):
    """Home page with book search and filtering."""
    # Get search and filter parameters
    search_query = request.GET.get('q', '')
    genre_filter = request.GET.get('genre', '')
    emotion_filter = request.GET.get('emotion', '')
    sort_by = request.GET.get('sort', '-created_at')

    # Start with all books
    books = Book.objects.all()

    # Apply search filter
    if search_query:
        books = books.filter(
            Q(title__icontains=search_query) |
            Q(author__icontains=search_query) |
            Q(book_id__icontains=search_query)
        )

    # Apply genre filter
    if genre_filter:
        books = books.filter(primary_genre__icontains=genre_filter)

    # Apply emotion-based sorting if specified
    if emotion_filter and emotion_filter in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'trust', 'disgust', 'anticipation']:
        sort_by = f'-avg_{emotion_filter}'

    # Apply sorting
    books = books.order_by(sort_by)

    # Pagination
    paginator = Paginator(books, 24)  # 24 books per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Get unique genres for filter dropdown
    unique_genres = Book.objects.exclude(primary_genre='').values_list('primary_genre', flat=True).distinct()

    # Statistics for dashboard
    total_books = Book.objects.count()
    avg_stats = Book.objects.aggregate(
        avg_joy=Avg('avg_joy'),
        avg_sadness=Avg('avg_sadness'),
        avg_fear=Avg('avg_fear'),
        avg_anger=Avg('avg_anger'),
    )

    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'genre_filter': genre_filter,
        'emotion_filter': emotion_filter,
        'sort_by': sort_by,
        'unique_genres': unique_genres,
        'total_books': total_books,
        'avg_stats': avg_stats,
    }

    return render(request, 'books/home.html', context)


def book_detail(request, book_id):
    """Detailed analysis page for a single book."""
    book = get_object_or_404(Book, book_id=book_id)

    # Get similar books (based on emotional distance)
    all_books = Book.objects.exclude(id=book.id)[:100]  # Limit for performance
    similar_books = []
    for other_book in all_books:
        distance = book.get_emotional_distance(other_book)
        if distance is not None:
            similar_books.append((other_book, distance))

    # Sort by distance and get top 5
    similar_books.sort(key=lambda x: x[1])
    similar_books = [b[0] for b in similar_books[:5]]

    # Create emotion radar chart
    radar_chart = create_emotion_radar_chart(book)

    # Create emotion trajectory chart
    trajectory_chart = create_trajectory_chart(book) if book.emotion_trajectory else None

    # Create VAD chart
    vad_chart = create_vad_chart(book)

    # Get top emotions and dyads
    top_emotions = book.get_top_emotions(3)
    top_dyads = sorted(book.dyad_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    context = {
        'book': book,
        'similar_books': similar_books,
        'radar_chart': radar_chart,
        'trajectory_chart': trajectory_chart,
        'vad_chart': vad_chart,
        'top_emotions': top_emotions,
        'top_dyads': top_dyads,
    }

    return render(request, 'books/book_detail.html', context)


def compare_books(request):
    """Compare multiple books side-by-side."""
    book_ids = request.GET.getlist('books')

    if not book_ids:
        # Show book selection page
        books = Book.objects.all().order_by('title')[:100]
        return render(request, 'books/compare_select.html', {'books': books})

    # Get selected books
    books = Book.objects.filter(book_id__in=book_ids)

    if len(books) < 2:
        return render(request, 'books/compare_select.html', {
            'error': 'Please select at least 2 books to compare',
            'books': Book.objects.all().order_by('title')[:100]
        })

    # Create comparison radar chart
    comparison_chart = create_comparison_radar_chart(books)

    # Create comparison table data
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
    # Get genre statistics
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

    # Get selected genre details
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

    # Calculate emotional distance to all other books
    all_books = Book.objects.exclude(id=book.id)
    recommendations = []

    for other_book in all_books:
        distance = book.get_emotional_distance(other_book)
        if distance is not None:
            recommendations.append({
                'book': other_book,
                'distance': distance,
                'similarity': 1 / (1 + distance),  # Convert distance to similarity score
            })

    # Sort by distance (most similar first) and get top 10
    recommendations.sort(key=lambda x: x['distance'])
    recommendations = recommendations[:10]

    context = {
        'source_book': book,
        'recommendations': recommendations,
    }

    return render(request, 'books/recommendations.html', context)


# Chart creation functions
def create_emotion_radar_chart(book):
    """Create a radar chart for book's emotion scores."""
    emotions = list(book.emotion_scores.keys())
    values = list(book.emotion_scores.values())

    # Capitalize emotion names
    emotions_display = [e.capitalize() for e in emotions]

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=emotions_display,
        fill='toself',
        name=book.title[:30],
        line=dict(color='rgb(99, 110, 250)', width=2),
        fillcolor='rgba(99, 110, 250, 0.3)',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2] if values else [0, 1]
            )
        ),
        showlegend=False,
        height=400,
        margin=dict(l=80, r=80, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_trajectory_chart(book):
    """Create emotion trajectory line chart."""
    if not book.emotion_trajectory:
        return None

    # Extract data from trajectory
    chunk_indices = [item.get('chunk_index', i) for i, item in enumerate(book.emotion_trajectory)]
    emotions_to_plot = ['joy', 'sadness', 'fear', 'anger']

    fig = go.Figure()

    for emotion in emotions_to_plot:
        values = [item.get(emotion, 0) for item in book.emotion_trajectory]
        fig.add_trace(go.Scatter(
            x=chunk_indices,
            y=values,
            mode='lines+markers',
            name=emotion.capitalize(),
            line=dict(width=2),
        ))

    fig.update_layout(
        title='Emotional Journey',
        xaxis_title='Chapter Progress',
        yaxis_title='Emotion Intensity',
        height=400,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_vad_chart(book):
    """Create VAD (Valence-Arousal-Dominance) bar chart."""
    vad_scores = book.vad_scores

    fig = go.Figure(data=[
        go.Bar(
            x=list(vad_scores.keys()),
            y=list(vad_scores.values()),
            marker=dict(color=['#4ade80', '#f59e0b', '#8b5cf6']),
            text=[f"{v:.3f}" for v in vad_scores.values()],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='VAD Scores',
        xaxis_title='Dimension',
        yaxis_title='Score',
        height=300,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_comparison_radar_chart(books):
    """Create a comparison radar chart for multiple books."""
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
                range=[0, 0.5]
            )
        ),
        showlegend=True,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# Import Count for genre_explorer
from django.db.models import Count
