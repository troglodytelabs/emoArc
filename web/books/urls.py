"""
URL configuration for books app.
"""
from django.urls import path
from . import views

app_name = 'books'

urlpatterns = [
    path('', views.data_browser, name='home'),
    path('methodology/', views.methodology, name='methodology'),
    path('book/<str:book_id>/', views.book_detail, name='book_detail'),
    path('recommendations/<str:book_id>/', views.recommendations, name='recommendations'),
]
