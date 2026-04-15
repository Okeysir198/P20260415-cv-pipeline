"""Reusable components for building custom demo tabs.

Provides class filtering, metrics display, model loading, and a generic tab
builder that can be used to create new detection tabs without duplicating code.
"""

from app_demo.src.class_filter import ClassFilterComponent
from app_demo.src.metrics_display import MetricsDisplay
from app_demo.src.model_loader import ModelLoader
from app_demo.src.generic_tab import build_tab_generic

__all__ = [
    "ClassFilterComponent",
    "MetricsDisplay",
    "ModelLoader",
    "build_tab_generic",
]
