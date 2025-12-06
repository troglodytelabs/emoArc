"""Custom template filters for book templates."""
from django import template

register = template.Library()


@register.filter
def get_item(dictionary, key):
    """Get item from dictionary by key in templates."""
    if dictionary and key:
        return dictionary.get(key, '')
    return ''


@register.filter
def multiply(value, arg):
    """Multiply two numbers."""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0
