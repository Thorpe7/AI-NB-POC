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

    # Model names of currently in-flight inference jobs. Multiple concurrent
    # cross-model runs are supported (e.g. ["brainseg", "duneai-nsclc"]); the
    # status row derives "N running" from len(...). Same-model concurrency is
    # out of scope — callers don't enforce uniqueness here.
    inflight_models = traitlets.List(default_value=[])

    # Diagnostics tabs currently open in the pipeline panel: mask_path → diag_path.
    # seg_viewer adds/removes entries as the user toggles a mask's Show button;
    # diagnostics_panel observes the dict and rebuilds tabs to match. Closing
    # a tab via its × removes the entry, which seg_viewer reacts to by turning
    # the matching mask's Show toggle off — keeping the overlay and the tab in
    # lockstep.
    active_diagnostics = traitlets.Dict(default_value={})
