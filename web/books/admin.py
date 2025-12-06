from django.contrib import admin
from .models import Book, UploadedBook


@admin.register(Book)
class BookAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'primary_genre', 'avg_joy', 'avg_sadness', 'avg_valence')
    list_filter = ('primary_genre',)
    search_fields = ('title', 'author', 'book_id')
    readonly_fields = ('created_at', 'updated_at')


@admin.register(UploadedBook)
class UploadedBookAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'status', 'uploaded_at', 'processed_at')
    list_filter = ('status',)
    search_fields = ('title', 'author')
    readonly_fields = ('uploaded_at', 'processed_at')
