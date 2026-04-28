# experiment/templatetags/custom_filters.py
from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """辞書からキーを指定して値を取り出す魔法のフィルター"""
    if dictionary:
        return dictionary.get(key)
    return ""