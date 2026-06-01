"""Image display, report display, and metadata panel."""

from __future__ import annotations

import ipywidgets as widgets
from ipyevents import Event

from utils.dicom_utils import extract_metadata

_PLACEHOLDER_HTML = (
    "<div style='display:flex;align-items:center;justify-content:center;"
    "width:100%;min-height:350px;background:#f8f9fa;border-radius:8px;"
    "border:2px dashed #dee2e6;color:#6c757d;font-size:14px;"
    "flex-direction:column;gap:8px;'>"
    "<span style='font-size:36px;opacity:0.5;'>&#128203;</span>"
    "<span>Select a file from the browser</span></div>"
)


def _metadata_table(rows):
    trs = "".join(
        f"<tr><td style='padding:4px 12px;font-weight:600;white-space:nowrap;"
        f"color:#495057;font-size:12px;border-bottom:1px solid #f0f0f0;'>{k}</td>"
        f"<td style='padding:4px 12px;font-size:12px;color:#6c757d;"
        f"border-bottom:1px solid #f0f0f0;'>{v}</td></tr>"
        for k, v in rows
    )
    return (
        f"<table style='font-size:12px;border-collapse:collapse;width:100%;'>"
        f"{trs}</table>"
    )


def build_viewer(state):
    """Build image display and metadata widgets.

    Returns a dict with keys:
        viewer_panel      - VBox for the main viewer area
        info_panel        - VBox for DICOM metadata (hidden until image loaded)
        image_widget      - Image widget (for cross-component updates)
        image_placeholder - HTML placeholder (hidden when content loaded)
        image_label       - HTML label showing filename
        metadata_html     - HTML widget for metadata table
        metadata_table    - formatter function for metadata rows
    """

    image_placeholder = widgets.HTML(value=_PLACEHOLDER_HTML)
    image_widget = widgets.Image(
        format="png",
        layout=widgets.Layout(
            max_width="100%", max_height="500px", display="none",
            object_fit="contain",
        ),
    )
    # Wrapping the Image in a Box gives wheel/key events a reliable DOM target.
    # Events attached to <img> are inconsistent across JupyterLab versions.
    image_container = widgets.Box(
        [image_widget],
        layout=widgets.Layout(
            display="flex",
            justify_content="center",
            align_items="center",
            width="100%",
            min_height="400px",
            max_height="500px",
            overflow="hidden",
        ),
    )
    image_label = widgets.HTML(value="")
    metadata_html = widgets.HTML(value="")

    series_info_label = widgets.HTML(
        value="",
        layout=widgets.Layout(display="none"),
    )

    prev_btn = widgets.Button(
        description="", icon="arrow-left",
        layout=widgets.Layout(width="40px", height="30px"),
    )
    next_btn = widgets.Button(
        description="", icon="arrow-right",
        layout=widgets.Layout(width="40px", height="30px"),
    )
    slice_slider = widgets.IntSlider(
        value=0, min=0, max=0, step=1,
        description="", readout=False, continuous_update=True,
        layout=widgets.Layout(flex="1", margin="0 12px"),
    )
    _syncing_slider = [False]
    series_nav = widgets.HBox(
        [prev_btn, slice_slider, series_info_label, next_btn],
        layout=widgets.Layout(
            display="none", align_items="center",
            justify_content="center", width="100%", padding="4px 0",
        ),
    )

    def _go_to_slice(idx):
        """Navigate to a specific slice index, updating all state and UI."""
        total = len(state.series_datasets)
        if total == 0 or idx < 0 or idx >= total:
            return
        state.series_index = idx
        ds, png = state.series_datasets[idx], state.series_png_cache[idx]
        state.current_ds = ds
        state.current_png_bytes = png
        state.current_file_name = f"{state.series_dir_name} [{idx + 1}/{total}]"

        image_widget.value = png
        series_info_label.value = (
            f"<div style='font-size:12px;color:#6c757d;padding:4px 0;"
            f"text-align:center;min-width:100px;'>"
            f"Slice {idx + 1} / {total}</div>"
        )

        if slice_slider.value != idx:
            _syncing_slider[0] = True
            try:
                slice_slider.value = idx
            finally:
                _syncing_slider[0] = False

        meta_rows = extract_metadata(ds)
        if meta_rows:
            metadata_html.value = _metadata_table(meta_rows)

    def _on_prev(_btn):
        _go_to_slice(state.series_index - 1)

    def _on_next(_btn):
        _go_to_slice(state.series_index + 1)

    def _on_slider(change):
        if _syncing_slider[0]:
            return
        _go_to_slice(int(change["new"]))

    def _on_series_datasets_change(_change):
        total = len(state.series_datasets)
        _syncing_slider[0] = True
        try:
            slice_slider.max = max(0, total - 1)
            slice_slider.value = min(state.series_index, max(0, total - 1))
        finally:
            _syncing_slider[0] = False

    prev_btn.on_click(_on_prev)
    next_btn.on_click(_on_next)
    slice_slider.observe(_on_slider, names="value")
    state.observe(_on_series_datasets_change, names="series_datasets")

    # Scroll-wheel + arrow-key nav via ipyevents on the image container box.
    # Attaching events to a Box (<div>) is more reliable than attaching to the
    # Image widget itself (<img>), which swallows wheel events in some browser
    # / JupyterLab combinations.
    _wheel_event = Event(
        source=image_container,
        watched_events=["wheel"],
        prevent_default_action=True,
    )
    _key_event = Event(
        source=image_container,
        watched_events=["keydown"],
    )

    def _on_wheel(event):
        dy = event.get("deltaY", 0)
        if dy > 0:
            _go_to_slice(state.series_index + 1)
        elif dy < 0:
            _go_to_slice(state.series_index - 1)

    def _on_key(event):
        key = event.get("key", "")
        if key in ("ArrowDown", "ArrowRight", "PageDown", "j"):
            _go_to_slice(state.series_index + 1)
        elif key in ("ArrowUp", "ArrowLeft", "PageUp", "k"):
            _go_to_slice(state.series_index - 1)

    _wheel_event.on_dom_event(_on_wheel)
    _key_event.on_dom_event(_on_key)

    viewer_panel = widgets.VBox(
        [
            image_label,
            image_placeholder,
            image_container,
            series_nav,
        ],
        layout=widgets.Layout(
            flex="1.2", padding="0 16px 0 0",
            min_height="400px",
        ),
    )

    info_panel = widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:13px;font-weight:700;color:#495057;"
                "padding:8px 0 4px;border-top:1px solid #e9ecef;margin-top:12px;'>"
                "DICOM Metadata</div>"
            ),
            metadata_html,
        ],
        layout=widgets.Layout(padding="0", width="100%", display="none"),
    )

    return {
        "viewer_panel": viewer_panel,
        "info_panel": info_panel,
        "image_widget": image_widget,
        "image_container": image_container,
        "image_placeholder": image_placeholder,
        "image_label": image_label,
        "metadata_html": metadata_html,
        "metadata_table": _metadata_table,
        "series_nav": series_nav,
        "series_info_label": series_info_label,
        "go_to_slice": _go_to_slice,
        "_wheel_event": _wheel_event,
        "_key_event": _key_event,
    }
