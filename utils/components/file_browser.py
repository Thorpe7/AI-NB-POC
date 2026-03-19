"""Dual file browsers: one for DICOM images, one for text reports."""

from __future__ import annotations

import html as html_mod
from pathlib import Path

import ipywidgets as widgets

from utils.dicom_utils import (
    dicom_to_pil,
    dicom_to_png_bytes,
    extract_metadata,
    is_dicom_candidate,
    load_series,
    read_dicom,
)

_TEXT_EXTENSIONS = {
    ".txt", ".csv", ".json", ".xml", ".html", ".log",
    ".md", ".yaml", ".yml", ".cfg", ".ini", ".conf",
}


def _is_text_file(p):
    """Check if a path is a readable text file."""
    return p.suffix.lower() in _TEXT_EXTENSIONS


def _error_card(msg):
    return (
        f"<div style='color:#d32f2f;background:#fff3f3;padding:10px 12px;"
        f"border-left:3px solid #d32f2f;border-radius:4px;font-size:12px;"
        f"margin-top:4px;'>{msg}</div>"
    )


def _build_browser(title, default_path, file_filter, file_icon, on_file_click,
                   on_clear=None, rows=15, extra_controls=None,
                   on_dir_change=None):
    """Build a generic file browser panel.

    Args:
        title: Section heading (may contain HTML entities).
        default_path: Initial directory to browse.
        file_filter: callable(Path) -> bool, which non-directory files to show.
        file_icon: Emoji string used for matching files.
        on_file_click: callable(Path) -> str|None.  Called when a matching
            file is selected.  Return an error message string to display,
            or *None* on success.
        on_clear: optional callable() invoked when the user clicks "Clear".
        rows: Number of visible rows in the Select widget.
        extra_controls: optional list of widgets appended to the controls HBox.
        on_dir_change: optional callback(Path) fired when the browsed directory
            changes.

    Returns:
        VBox widget for the browser panel.
    """

    _current_dir = [None]
    _refreshing = [False]

    root_text = widgets.Text(
        value=default_path,
        placeholder="/path/to/data",
        layout=widgets.Layout(width="100%"),
    )
    go_btn = widgets.Button(
        description="Go", icon="folder-open", button_style="info",
        layout=widgets.Layout(width="55px", height="30px"),
    )
    up_btn = widgets.Button(
        description="", icon="arrow-up",
        layout=widgets.Layout(width="34px", height="30px"),
    )
    clear_btn = widgets.Button(
        description="", icon="times",
        tooltip="Clear selection",
        layout=widgets.Layout(width="34px", height="30px"),
    )
    breadcrumb = widgets.HTML(
        value=(
            "<div style='font-size:10px;color:#6c757d;padding:2px 0;'>"
            "No directory selected</div>"
        ),
    )
    file_list = widgets.Select(
        options=[], rows=rows,
        layout=widgets.Layout(width="100%"),
    )
    status = widgets.HTML(value="")

    # ---- internal helpers ------------------------------------------------

    def _list_directory(directory):
        _refreshing[0] = True
        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except PermissionError:
            status.value = _error_card("Permission denied.")
            _refreshing[0] = False
            return

        options = []
        if directory.parent != directory:
            options.append("\u2B06 ..")
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                options.append(f"\U0001F4C1 {entry.name}")
            elif file_filter(entry):
                options.append(f"{file_icon} {entry.name}")

        file_list.options = options if options else ["(empty)"]
        file_list.value = None
        _refreshing[0] = False

        dir_str = str(directory)
        if len(dir_str) > 30:
            dir_str = "\u2026" + dir_str[-27:]
        breadcrumb.value = (
            f"<div style='font-size:10px;color:#6c757d;padding:2px 0;"
            f"border-bottom:1px solid #e9ecef;margin-bottom:2px;'>"
            f"&#128194; {dir_str}</div>"
        )

    def _on_go(_btn=None):
        p = Path(root_text.value.strip())
        if not p.exists() or not p.is_dir():
            status.value = _error_card(f"Not found: {p}")
            return
        status.value = ""
        _current_dir[0] = p
        _list_directory(p)
        if on_dir_change:
            on_dir_change(p)

    def _on_up(_btn=None):
        if _current_dir[0] is None:
            return
        parent = _current_dir[0].parent
        if parent != _current_dir[0]:
            _current_dir[0] = parent
            root_text.value = str(parent)
            _list_directory(parent)
            if on_dir_change:
                on_dir_change(parent)

    def _on_select(change):
        if _refreshing[0]:
            return
        val = change["new"]
        if not val or val == "(empty)":
            return

        name = val.split(" ", 1)[1] if " " in val else val

        if name == "..":
            _on_up()
            return

        if _current_dir[0] is None:
            return
        target = _current_dir[0] / name

        if target.is_dir():
            _current_dir[0] = target
            root_text.value = str(target)
            _list_directory(target)
            if on_dir_change:
                on_dir_change(target)
        elif file_filter(target):
            error = on_file_click(target)
            if error:
                status.value = _error_card(error)
            else:
                status.value = ""

    def _on_clear_click(_btn):
        if on_clear:
            on_clear()
        _refreshing[0] = True
        file_list.value = None
        _refreshing[0] = False
        status.value = ""

    # ---- wire events -----------------------------------------------------

    go_btn.on_click(_on_go)
    up_btn.on_click(_on_up)
    clear_btn.on_click(_on_clear_click)
    file_list.observe(_on_select, names="value")

    # Auto-browse on startup
    _on_go()

    bottom_controls = [go_btn, clear_btn]
    if extra_controls:
        bottom_controls.extend(extra_controls)

    panel = widgets.VBox(
        [
            widgets.HTML(
                f"<div style='font-size:12px;font-weight:700;color:#495057;"
                f"padding:0 0 4px;'>{title}</div>"
            ),
            widgets.HBox(
                [root_text, up_btn],
                layout=widgets.Layout(width="100%"),
            ),
            breadcrumb,
            file_list,
            widgets.HBox(
                bottom_controls,
                layout=widgets.Layout(width="100%", padding="4px 0 0 0"),
            ),
            status,
        ],
        layout=widgets.Layout(
            width="250px", min_width="250px",
            padding="8px",
        ),
    )
    panel.add_class("medgemma-sidebar")

    return panel


# =========================================================================
# Public builders
# =========================================================================

def build_image_browser(state, viewer):
    """Build DICOM image file browser (left sidebar)."""

    image_widget = viewer["image_widget"]
    image_placeholder = viewer["image_placeholder"]
    image_label = viewer["image_label"]
    metadata_html = viewer["metadata_html"]
    metadata_table = viewer["metadata_table"]
    info_panel = viewer["info_panel"]
    series_nav = viewer["series_nav"]
    series_info_label = viewer["series_info_label"]

    def _clear_series_state():
        """Reset all series-related state and hide series UI."""
        state.series_datasets = []
        state.series_png_cache = []
        state.series_index = 0
        state.series_dir_name = ""
        series_nav.layout.display = "none"

    def _on_dicom_selected(file_path):
        _clear_series_state()

        ds = read_dicom(file_path)
        if ds is None:
            return "Could not read DICOM file."

        try:
            _ = ds.pixel_array
        except Exception:
            return "No pixel data or missing transfer syntax handler."

        try:
            dicom_to_pil(ds)
        except Exception as exc:
            return f"Render error: {exc}"

        state.current_ds = ds
        state.current_png_bytes = dicom_to_png_bytes(ds)
        state.current_file_name = file_path.name

        # Show image, hide placeholder
        image_widget.value = state.current_png_bytes
        image_widget.layout.display = ""
        image_placeholder.layout.display = "none"
        image_label.value = (
            f"<div style='font-size:13px;color:#495057;padding:0 0 8px;'>"
            f"&#x1F52C; <b>{file_path.name}</b></div>"
        )

        # Show metadata
        meta_rows = extract_metadata(ds)
        if meta_rows:
            metadata_html.value = metadata_table(meta_rows)
            info_panel.layout.display = ""
        else:
            info_panel.layout.display = "none"

        return None

    def _on_image_clear():
        _clear_series_state()
        state.current_ds = None
        state.current_png_bytes = None
        state.current_file_name = ""
        image_widget.layout.display = "none"
        image_placeholder.layout.display = ""
        image_label.value = ""
        metadata_html.value = ""
        info_panel.layout.display = "none"

    # -- Open Series button and wiring --

    open_series_btn = widgets.Button(
        description="Series",
        icon="layer-group",
        button_style="warning",
        disabled=True,
        tooltip="Load all DICOMs in this directory as a series",
        layout=widgets.Layout(width="80px", height="30px"),
    )

    _series_dir = [None]

    def _on_dir_change(directory):
        _series_dir[0] = directory
        try:
            count = sum(1 for p in directory.iterdir()
                        if is_dicom_candidate(p))
        except Exception:
            count = 0
        open_series_btn.disabled = count < 2

    def _on_open_series(_btn):
        if _series_dir[0] is None:
            return

        # Loading state
        open_series_btn.description = "Loading…"
        open_series_btn.disabled = True

        pairs = load_series(_series_dir[0])

        if not pairs:
            open_series_btn.description = "Series"
            open_series_btn.disabled = False
            return

        datasets = [ds for ds, _png in pairs]
        png_cache = [png for _ds, png in pairs]

        state.series_datasets = datasets
        state.series_png_cache = png_cache
        state.series_index = 0
        state.series_dir_name = _series_dir[0].name

        # Set initial current slice
        state.current_ds = datasets[0]
        state.current_png_bytes = png_cache[0]
        state.current_file_name = f"{_series_dir[0].name} [1/{len(datasets)}]"

        # Show image, hide placeholder
        image_widget.value = png_cache[0]
        image_widget.layout.display = ""
        image_placeholder.layout.display = "none"
        image_label.value = (
            f"<div style='font-size:13px;color:#495057;padding:0 0 8px;'>"
            f"&#x1F52C; <b>{_series_dir[0].name}</b>"
            f" &mdash; {len(datasets)} slices</div>"
        )

        # Show series navigation
        series_nav.layout.display = ""
        series_info_label.layout.display = ""
        series_info_label.value = (
            f"<div style='font-size:12px;color:#6c757d;padding:4px 0;"
            f"text-align:center;min-width:100px;'>"
            f"Slice 1 / {len(datasets)}</div>"
        )

        # Metadata from first slice
        meta_rows = extract_metadata(datasets[0])
        if meta_rows:
            metadata_html.value = metadata_table(meta_rows)
            info_panel.layout.display = ""

        # Restore button
        open_series_btn.description = "Series"
        open_series_btn.disabled = False

    open_series_btn.on_click(_on_open_series)

    return _build_browser(
        title="&#x1F52C; Image Files",
        default_path="/data",
        file_filter=is_dicom_candidate,
        file_icon="\U0001F52C",
        on_file_click=_on_dicom_selected,
        on_clear=_on_image_clear,
        extra_controls=[open_series_btn],
        on_dir_change=_on_dir_change,
    )


def build_report_browser(state, viewer):
    """Build text report file browser (second sidebar)."""

    report_display = viewer["report_display"]

    def _on_text_selected(file_path):
        try:
            content = file_path.read_text(errors="replace")[:50_000]
        except Exception as exc:
            return f"Could not read: {exc}"

        safe = html_mod.escape(content)
        state.report_text = content
        state.report_file_name = file_path.name

        report_display.value = (
            f"<div style='border:1px solid #e9ecef;border-radius:6px;"
            f"margin-top:8px;overflow:hidden;'>"
            f"<div style='background:#e8f4fd;padding:6px 12px;font-size:12px;"
            f"font-weight:600;color:#1565c0;border-bottom:1px solid #e9ecef;'>"
            f"&#x1F4C4; {file_path.name}</div>"
            f"<pre style='font-size:12px;line-height:1.5;margin:0;padding:10px 12px;"
            f"background:#ffffff;max-height:400px;overflow:auto;white-space:pre-wrap;"
            f"word-wrap:break-word;"
            f"font-family:SF Mono,Monaco,Consolas,monospace;'>{safe}</pre></div>"
        )
        return None

    def _on_report_clear():
        state.report_text = ""
        state.report_file_name = ""
        report_display.value = ""

    return _build_browser(
        title="&#x1F4C4; Text Reports",
        default_path="/data",
        file_filter=_is_text_file,
        file_icon="\U0001F4C4",
        on_file_click=_on_text_selected,
        on_clear=_on_report_clear,
    )
