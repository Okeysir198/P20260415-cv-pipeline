"""Class filtering component for demo tabs.

Provides a reusable UI component and detection filtering logic for showing/hiding
specific classes in detection results.
"""

import sys
from pathlib import Path
from typing import Dict, List

import gradio as gr
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class ClassFilterComponent:
    """Gradio CheckboxGroup component for class filtering.

    Creates a selectable list of class names that can be used to filter
    detection results. Default behavior selects all classes.
    """

    def __init__(self, class_names: Dict[int, str], default_select_all: bool = True) -> None:
        """Initialize the class filter component.

        Args:
            class_names: Mapping from class_id to display name.
            default_select_all: If True, all classes are selected by default.
                If False, no classes are selected initially.
        """
        self._class_names = class_names
        self._name_to_id = {name: cid for cid, name in class_names.items()}
        self._default_select_all = default_select_all

        if default_select_all:
            self._default_value = list(class_names.values())
        else:
            self._default_value = []

    def build_ui(self, label: str = "Filter Classes") -> gr.CheckboxGroup:
        """Build the Gradio CheckboxGroup UI component.

        Args:
            label: Label text for the checkbox group.

        Returns:
            gr.CheckboxGroup component with all class names as choices.
        """
        return gr.CheckboxGroup(
            choices=list(self._class_names.values()),
            value=self._default_value,
            label=label,
            interactive=True,
        )

    @staticmethod
    def filter_detections(
        detections: sv.Detections,
        selected_names: List[str],
        name_to_id: Dict[str, int],
    ) -> sv.Detections:
        """Filter detections to only include selected classes.

        Args:
            detections: supervision Detections object to filter.
            selected_names: List of class names to keep (empty list = keep all).
            name_to_id: Mapping from class name to class_id.

        Returns:
            Filtered sv.Detections object containing only selected classes.
            If selected_names is empty, returns all detections unchanged.
        """
        if not selected_names:
            return detections

        selected_ids = np.array([name_to_id[name] for name in selected_names])
        mask = np.isin(detections.class_id, selected_ids)
        return detections[mask]
