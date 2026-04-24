"""DICOM SEG overlay panel: composite segmentation masks onto the series viewer."""

from __future__ import annotations

import io
from pathlib import Path

import ipywidgets as widgets

from utils.components.file_browser import _build_browser
from utils.config import LOCAL_DATA_ROOT
from utils.dicom_utils import composite_overlay, dicom_to_pil, load_dicom_seg

_SEG_OUTPUT_ROOT = str(Path(LOCAL_DATA_ROOT) / "resources" / "output" / "totalseg")


def _pil_to_png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _error_card(msg):
    return (
        f"<div style='color:#d32f2f;background:#fff3f3;padding:8px 10px;"
        f"border-left:3px solid #d32f2f;border-radius:4px;font-size:12px;'>{msg}</div>"
    )


def _success_card(msg):
    return (
        f"<div style='color:#2e7d32;background:#f1f8f4;padding:8px 10px;"
        f"border-left:3px solid #2e7d32;border-radius:4px;font-size:12px;'>{msg}</div>"
    )


def _legend_html(segments, matched_counts):
    if not segments:
        return ""
    items = []
    for num in sorted(segments.keys()):
        info = segments[num]
        r, g, b = info["color"]
        count = matched_counts.get(num, 0)
        dim = "" if count else "opacity:0.45;"
        items.append(
            f"<li style='display:flex;align-items:center;gap:8px;"
            f"font-size:12px;padding:3px 0;{dim}'>"
            f"<span style='width:12px;height:12px;background:rgb({r},{g},{b});"
            f"border-radius:2px;display:inline-block;flex-shrink:0;"
            f"border:1px solid rgba(0,0,0,0.1);'></span>"
            f"<span style='flex:1;'>{info['label']}</span>"
            f"<span style='color:#6c757d;font-size:11px;'>{count} slice{'s' if count != 1 else ''}</span>"
            f"</li>"
        )
    return (
        "<div style='font-size:12px;font-weight:600;color:#495057;"
        "padding:6px 0 4px;'>Segments</div>"
        "<ul style='list-style:none;padding:0;margin:0;max-height:220px;"
        "overflow:auto;border:1px solid #e9ecef;border-radius:4px;"
        "padding:6px 10px;background:#fafbfc;'>"
        + "".join(items) + "</ul>"
    )


def _find_latest_seg():
    root = Path(_SEG_OUTPUT_ROOT)
    if not root.is_dir():
        return None
    candidates = [p for p in root.glob("*/segmentations.dcm") if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def build_seg_viewer(state, viewer):
    """Build the SEG overlay panel.

    Observes `state.seg_file_path` (set by segmentation_tab's "Load overlay
    in viewer" button) and also exposes a file browser + "Use latest"
    shortcut so prior segmentations can be re-attached without re-running
    TotalSegmentator.
    """

    image_widget = viewer["image_widget"]

    _originals = [None]
    _overlay_pngs: dict[int, bytes] = {}
    _seg_cache = [None]
    _current_seg_path: list[str | None] = [None]

    latest_btn = widgets.Button(
        description="Use latest", icon="clock",
        tooltip=f"Pick the most recent segmentations.dcm under {_SEG_OUTPUT_ROOT}",
        layout=widgets.Layout(width="110px", height="30px"),
    )
    clear_btn = widgets.Button(
        description="Clear", icon="times",
        layout=widgets.Layout(width="80px", height="30px"),
    )
    overlay_toggle = widgets.Checkbox(
        value=True, description="Show overlay", indent=False,
    )
    alpha_slider = widgets.FloatSlider(
        value=0.4, min=0.1, max=0.9, step=0.05,
        description="Opacity:", continuous_update=False,
        style={"description_width": "60px"},
        layout=widgets.Layout(width="100%"),
    )
    status_html = widgets.HTML(value="")
    legend_box = widgets.HTML(value="")

    def _refresh_image():
        if state.series_datasets and state.series_png_cache:
            idx = state.series_index
            if 0 <= idx < len(state.series_png_cache):
                image_widget.value = state.series_png_cache[idx]

    def _discard_overlay_state():
        _originals[0] = None
        _overlay_pngs.clear()
        _seg_cache[0] = None
        legend_box.value = ""

    def _restore_originals():
        if _originals[0] is None or not state.series_datasets:
            return
        originals = _originals[0]
        cache = state.series_png_cache
        for idx in range(min(len(originals), len(cache))):
            cache[idx] = originals[idx]
        _refresh_image()

    def _apply_overlay_to_cache():
        if _originals[0] is None or not _overlay_pngs:
            return
        cache = state.series_png_cache
        for idx, png in _overlay_pngs.items():
            if 0 <= idx < len(cache):
                cache[idx] = png
        _refresh_image()

    def _apply_seg(path_str):
        path_str = (path_str or "").strip()
        if not path_str:
            status_html.value = _error_card(
                "Select a SEG file from the browser (or click \"Use latest\")."
            )
            return
        p = Path(path_str).expanduser()
        if not p.is_file():
            status_html.value = _error_card(f"File not found: {p}")
            return

        if not state.series_datasets:
            status_html.value = _error_card(
                "No series loaded. Open a DICOM series first, then load the SEG."
            )
            return

        try:
            seg = load_dicom_seg(p)
        except Exception as e:
            status_html.value = _error_card(f"Could not parse SEG: {e}")
            return

        by_sop = seg["by_source_sop"]
        segments = seg["segments"]
        alpha = float(alpha_slider.value)

        try:
            snapshot = list(state.series_png_cache)
            overlays: dict[int, bytes] = {}
            matched_counts: dict[int, int] = {}

            for idx, ds in enumerate(state.series_datasets):
                sop = getattr(ds, "SOPInstanceUID", None)
                if not sop:
                    continue
                masks = by_sop.get(str(sop))
                if not masks:
                    continue
                base = dicom_to_pil(ds)
                composited = composite_overlay(base, masks, segments, alpha=alpha)
                overlays[idx] = _pil_to_png_bytes(composited)
                for seg_num in masks:
                    matched_counts[seg_num] = matched_counts.get(seg_num, 0) + 1
        except Exception as e:
            status_html.value = _error_card(f"Compositing failed: {e}")
            return

        if not overlays:
            status_html.value = _error_card(
                "SEG loaded, but no slices matched SOPInstanceUIDs in the "
                "current series. Make sure the loaded series is the one the "
                "SEG was produced from."
            )
            return

        _originals[0] = snapshot
        _overlay_pngs.clear()
        _overlay_pngs.update(overlays)
        _seg_cache[0] = {"segments": segments, "matched": matched_counts}
        _current_seg_path[0] = path_str

        if overlay_toggle.value:
            _apply_overlay_to_cache()

        n_matched = len(overlays)
        n_total = len(state.series_datasets)
        status_html.value = _success_card(
            f"Overlay loaded: {n_matched} of {n_total} slices "
            f"({len(matched_counts)} segment(s))."
        )
        legend_box.value = _legend_html(segments, matched_counts)

    def _load_new_seg(path_str):
        _restore_originals()
        _discard_overlay_state()
        status_html.value = ""
        _apply_seg(path_str)
        if path_str and path_str != state.seg_file_path:
            state.seg_file_path = path_str

    def _on_seg_file_clicked(file_path):
        _load_new_seg(str(file_path))
        return None

    def _on_latest(_btn):
        latest = _find_latest_seg()
        if latest is None:
            status_html.value = _error_card(
                f"No segmentations found under {_SEG_OUTPUT_ROOT}."
            )
            return
        _load_new_seg(str(latest))

    def _on_clear(_btn):
        status_html.value = ""
        _restore_originals()
        _discard_overlay_state()
        _current_seg_path[0] = None
        if state.seg_file_path:
            state.seg_file_path = ""

    def _on_toggle(change):
        if _seg_cache[0] is None or _originals[0] is None:
            return
        if change["new"]:
            _apply_overlay_to_cache()
        else:
            state.series_png_cache = list(_originals[0])
            _refresh_image()

    def _on_alpha_change(_change):
        if _seg_cache[0] is None or not state.series_datasets:
            return
        current_path = _current_seg_path[0]
        if not current_path:
            return
        _restore_originals()
        _discard_overlay_state()
        _apply_seg(current_path)

    def _on_seg_path_state_change(change):
        new_path = (change["new"] or "").strip()
        if not new_path:
            return
        if new_path == _current_seg_path[0]:
            return
        _restore_originals()
        _discard_overlay_state()
        status_html.value = ""
        _apply_seg(new_path)

    def _on_series_change(_change):
        # New series PNGs are already in state.series_png_cache; drop our
        # stale originals without writing them back.
        if _originals[0] is None and _seg_cache[0] is None:
            return
        _discard_overlay_state()
        status_html.value = ""
        _current_seg_path[0] = None
        if state.seg_file_path:
            state.seg_file_path = ""

    latest_btn.on_click(_on_latest)
    clear_btn.on_click(_on_clear)
    overlay_toggle.observe(_on_toggle, names="value")
    alpha_slider.observe(_on_alpha_change, names="value")
    state.observe(_on_seg_path_state_change, names="seg_file_path")
    state.observe(_on_series_change, names="series_dir_path")

    seg_browser = _build_browser(
        title="&#x1F9E9; SEG Files",
        default_path=_SEG_OUTPUT_ROOT,
        file_filter=lambda p: p.suffix.lower() == ".dcm",
        file_icon="\U0001F9E9",
        on_file_click=_on_seg_file_clicked,
        rows=8,
    )
    seg_browser.layout.width = "100%"
    seg_browser.layout.min_width = "0"

    controls_row = widgets.HBox(
        [latest_btn, clear_btn],
        layout=widgets.Layout(gap="8px", padding="4px 0"),
    )
    toggle_row = widgets.HBox([overlay_toggle])
    toggle_row.add_class("medgemma-switch")

    return widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:13px;font-weight:700;color:#495057;"
                "padding:0 0 8px;'>Segmentation Overlay</div>"
            ),
            seg_browser,
            controls_row,
            toggle_row,
            alpha_slider,
            status_html,
            legend_box,
        ],
        layout=widgets.Layout(
            flex="1", padding="0 0 0 16px",
            border_left="1px solid #e9ecef",
        ),
    )
