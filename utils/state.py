"""Centralized application state for the MedGemma dashboard."""

from __future__ import annotations

import traitlets


class AppState(traitlets.HasTraits):
    """Single source of truth shared across all dashboard components."""

    # Currently selected single DICOM (viewer + payload source)
    current_ds = traitlets.Any(default_value=None)
    current_png_bytes = traitlets.Bytes(default_value=b"")
    current_file_name = traitlets.Unicode(default_value="")
    current_file_path = traitlets.Unicode(default_value="")

    # DICOM series state
    series_datasets = traitlets.List(default_value=[])
    series_png_cache = traitlets.List(default_value=[])
    series_index = traitlets.Int(default_value=0)
    series_dir_name = traitlets.Unicode(default_value="")
    series_dir_path = traitlets.Unicode(default_value="")
