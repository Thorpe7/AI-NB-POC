"""Image display, report display, and metadata panel."""

from __future__ import annotations

import ipywidgets as widgets
from ipyevents import Event

from utils.dicom_utils import extract_metadata

_PLACEHOLDER_HTML = (
    "<div style='display:flex;align-items:center;justify-content:center;"
    "width:100%;min-height:350px;background:transparent;"
    "color:var(--text-muted);font-size:13px;"
    "flex-direction:column;gap:8px;'>"
    "<span style='font-size:36px;opacity:0.4;'>&#128203;</span>"
    "<span>Select a series from the left sidebar</span></div>"
)


def _metadata_table(rows):
    trs = "".join(
        f"<tr><td style='padding:4px 12px;font-weight:600;white-space:nowrap;"
        f"color:var(--text);font-size:11.5px;"
        f"border-bottom:1px solid var(--border);'>{k}</td>"
        f"<td style='padding:4px 12px;font-size:11.5px;color:var(--text-muted);"
        f"border-bottom:1px solid var(--border);'>{v}</td></tr>"
        for k, v in rows
    )
    return (
        f"<table style='font-size:11.5px;border-collapse:collapse;width:100%;'>"
        f"{trs}</table>"
    )


def _fmt(value, suffix: str = "") -> str:
    """Format a DICOM scalar/sequence for an overlay (best-effort)."""
    if value is None or value == "":
        return "&mdash;"
    # DICOM multi-value (DSfloat, IS, etc.) is iterable; show first element.
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            value = next(iter(value))
        except StopIteration:
            return "&mdash;"
    try:
        if isinstance(value, float) or (
            isinstance(value, str) and "." in value
        ):
            return f"{float(value):.2f}{suffix}"
    except (TypeError, ValueError):
        pass
    return f"{value}{suffix}"


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
            max_width="100%", max_height="80vh", display="none",
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
            min_height="500px",
            max_height="80vh",
            overflow="hidden",
        ),
    )
    # image_label and series_info_label are kept as no-op sinks for the
    # file_browser writes that still call into them. The corner overlays are
    # the user-visible labels now.
    image_label = widgets.HTML(value="", layout=widgets.Layout(display="none"))
    metadata_html = widgets.HTML(value="")

    series_info_label = widgets.HTML(
        value="",
        layout=widgets.Layout(display="none"),
    )

    # --- Corner overlays (top-left subject, top-right series, bottom-left
    # slice + W/L, bottom-right zoom + matrix). Absolutely positioned over the
    # image via the .nbpoc-viewer-canvas wrapper.
    overlay_tl = widgets.HTML(value="")
    overlay_tr = widgets.HTML(value="")
    overlay_bl = widgets.HTML(value="")
    overlay_br = widgets.HTML(value="")
    for ov, cls in (
        (overlay_tl, "tl"), (overlay_tr, "tr"),
        (overlay_bl, "bl"), (overlay_br, "br"),
    ):
        ov.add_class("nbpoc-viewer-overlay")
        ov.add_class(cls)

    def _update_overlays():
        ds = state.current_ds
        if ds is None:
            overlay_tl.value = overlay_tr.value = ""
            overlay_bl.value = overlay_br.value = ""
            return
        # Top-left: subject + study
        patient = str(
            getattr(ds, "PatientID", "") or getattr(ds, "PatientName", "") or ""
        )
        study_date = str(getattr(ds, "StudyDate", "") or "")
        study_desc = str(getattr(ds, "StudyDescription", "") or "")
        study_line = " ".join(s for s in (study_date, study_desc) if s)
        overlay_tl.value = (
            f"<div class='title'>{patient or '&mdash;'}</div>"
            f"<div class='muted'>{study_line or ''}</div>"
        )
        # Top-right: series description + number
        desc = str(getattr(ds, "SeriesDescription", "") or "")
        modality = str(getattr(ds, "Modality", "") or "")
        series_num = getattr(ds, "SeriesNumber", "")
        head_line = " ".join(s for s in (desc, modality) if s) or "&mdash;"
        overlay_tr.value = (
            f"<div class='title'>{head_line}</div>"
            f"<div class='muted'>Series: {_fmt(series_num)}</div>"
        )
        # Bottom-left: image i/n, location, thickness, W/L
        total = len(state.series_datasets)
        idx = state.series_index + 1 if total else 1
        loc = getattr(ds, "SliceLocation", None)
        thick = getattr(ds, "SliceThickness", None)
        ww = getattr(ds, "WindowWidth", None)
        wc = getattr(ds, "WindowCenter", None)
        overlay_bl.value = (
            f"<div>Image: {idx}{f' / {total}' if total else ''}</div>"
            f"<div>Loc: {_fmt(loc, ' mm')}</div>"
            f"<div>Thick: {_fmt(thick, ' mm')}</div>"
            f"<div>W: {_fmt(ww)} &nbsp; L: {_fmt(wc)}</div>"
        )
        # Bottom-right: zoom (stub) + matrix size
        rows = getattr(ds, "Rows", None)
        cols = getattr(ds, "Columns", None)
        overlay_br.value = (
            f"<div>Zoom: 100%</div>"
            f"<div>{_fmt(rows)} &times; {_fmt(cols)}</div>"
        )

    prev_btn = widgets.Button(
        description="", icon="arrow-up",
        tooltip="Previous slice",
        layout=widgets.Layout(width="32px", height="28px"),
    )
    next_btn = widgets.Button(
        description="", icon="arrow-down",
        tooltip="Next slice",
        layout=widgets.Layout(width="32px", height="28px"),
    )
    slice_slider = widgets.IntSlider(
        value=0, min=0, max=0, step=1,
        description="", readout=False, continuous_update=True,
        orientation="vertical",
        layout=widgets.Layout(height="240px", margin="0"),
    )
    _syncing_slider = [False]
    # Vertical scrub stack — prev / slider / next, absolutely positioned on
    # the right edge of the canvas via the .nbpoc-viewer-slider CSS class.
    series_nav = widgets.VBox(
        [prev_btn, slice_slider, next_btn],
        layout=widgets.Layout(display="none", align_items="center"),
    )
    series_nav.add_class("nbpoc-viewer-slider")

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
        _update_overlays()

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
    state.observe(lambda _c: _update_overlays(), names="current_ds")

    # Scroll-wheel + arrow-key nav via ipyevents. Attach to BOTH the Box and
    # the Image — the Box catches bubbled events in JupyterLab, but Voila and
    # some browsers route the wheel directly to the <img> and the Box never
    # sees it. Two sources, same handler.
    _wheel_box = Event(
        source=image_container,
        watched_events=["wheel"],
        prevent_default_action=True,
    )
    _wheel_img = Event(
        source=image_widget,
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

    _wheel_box.on_dom_event(_on_wheel)
    _wheel_img.on_dom_event(_on_wheel)
    _key_event.on_dom_event(_on_key)

    canvas = widgets.Box(
        [
            image_placeholder,
            image_container,
            overlay_tl,
            overlay_tr,
            overlay_bl,
            overlay_br,
            series_nav,
        ],
        layout=widgets.Layout(width="100%", flex="1"),
    )
    canvas.add_class("nbpoc-viewer-canvas")

    viewer_panel = widgets.VBox(
        [image_label, canvas],
        layout=widgets.Layout(width="100%", height="100%"),
    )
    viewer_panel.add_class("nbpoc-viewer")

    info_panel = widgets.VBox(
        [
            widgets.HTML(
                "<div class='nbpoc-section-label' "
                "style='padding:8px 0 4px;border-top:1px solid var(--border);"
                "margin-top:12px;'>DICOM METADATA</div>"
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
        "_wheel_box": _wheel_box,
        "_wheel_img": _wheel_img,
        "_key_event": _key_event,
    }
