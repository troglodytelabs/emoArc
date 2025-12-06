"""
URL configuration for books app.
"""
from django.urls import path
from . import views

app_name = 'books'

urlpatterns = [
    path('', views.home, name='home'),
    path('book/<str:book_id>/', views.book_detail, name='book_detail'),
    path('compare/', views.compare_books, name='compare'),
    path('genres/', views.genre_explorer, name='genre_explorer'),
    path('recommendations/<str:book_id>/', views.recommendations, name='recommendations'),
]
