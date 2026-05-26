"""DICOM SEG overlay panel: composite segmentation masks onto the series viewer."""

from __future__ import annotations

import io
from pathlib import Path

import ipywidgets as widgets
import numpy as np
from PIL import Image

from utils.components.file_browser import _build_browser
from utils.config import LOCAL_DATA_ROOT
from utils.dicom_utils import load_dicom_seg

_SEG_OUTPUT_ROOT = str(Path(LOCAL_DATA_ROOT) / "resources" / "output" / "totalseg")


def _pil_to_png_bytes(img):
    buf = io.BytesIO()
    # compress_level=1 is ~3-5x faster than the default (6) and keeps the
    # output small enough for in-memory caching.
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def _composite_cached(base_rgb, masks, segments, alpha):
    """Composite masks onto a cached float32 RGB array; return PIL Image.

    `base_rgb` is never mutated — we copy before blending so the cache stays
    pristine for subsequent rotate/flip/alpha changes.
    """
    out = base_rgb.copy()
    h, w = out.shape[:2]
    for seg_num, mask in masks.items():
        if mask.shape != (h, w):
            mask_pil = Image.fromarray(
                (mask.astype(np.uint8) * 255), mode="L"
            ).resize((w, h), Image.NEAREST)
            mask = np.array(mask_pil) > 127
        color = np.array(segments[seg_num]["color"], dtype=np.float32)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


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
    _selected_seg_path: list[str | None] = [None]
    # Orientation controls, applied to every mask before compositing so all
    # slices share the same transform.
    _rotation = [0]  # number of 90° CCW turns
    _flip_h = [False]
    _flip_v = [False]
    # Caches that survive rotate/flip/alpha re-applies. Cleared on series
    # change (base arrays) or new SEG file / clear (parsed seg).
    _parsed_seg_cache: list[dict | None] = [None]
    _base_arrays_cache: list[list | None] = [None]

    use_mask_btn = widgets.Button(
        description="Use mask", icon="check", button_style="primary",
        disabled=True,
        tooltip="Load the selected segmentation as an overlay",
        layout=widgets.Layout(width="110px", height="30px"),
    )
    latest_btn = widgets.Button(
        description="Use latest", icon="clock",
        tooltip=f"Pick the most recent segmentations.dcm under {_SEG_OUTPUT_ROOT}",
        layout=widgets.Layout(width="110px", height="30px"),
    )
    clear_btn = widgets.Button(
        description="Clear", icon="times",
        layout=widgets.Layout(width="80px", height="30px"),
    )
    rotate_btn = widgets.Button(
        description="Rotate 90°", icon="redo",
        tooltip="Rotate overlays 90° CCW (cycles 0/90/180/270)",
        layout=widgets.Layout(width="110px", height="30px"),
    )
    flip_h_btn = widgets.Button(
        description="Flip H", icon="arrows-h",
        tooltip="Mirror overlays horizontally",
        layout=widgets.Layout(width="80px", height="30px"),
    )
    flip_v_btn = widgets.Button(
        description="Flip V", icon="arrows-v",
        tooltip="Mirror overlays vertically",
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

    def _transform_masks(masks):
        if not _flip_h[0] and not _flip_v[0] and _rotation[0] == 0:
            return masks
        out = {}
        for k, m in masks.items():
            mm = m
            if _flip_h[0]:
                mm = np.fliplr(mm)
            if _flip_v[0]:
                mm = np.flipud(mm)
            if _rotation[0]:
                mm = np.rot90(mm, _rotation[0])
            out[k] = mm
        return out

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

        cached_parsed = _parsed_seg_cache[0]
        if cached_parsed and cached_parsed["path"] == path_str:
            seg = cached_parsed["seg"]
        else:
            try:
                seg = load_dicom_seg(p)
            except Exception as e:
                status_html.value = _error_card(f"Could not parse SEG: {e}")
                return
            _parsed_seg_cache[0] = {"path": path_str, "seg": seg}

        by_sop = seg["by_source_sop"]
        segments = seg["segments"]
        alpha = float(alpha_slider.value)

        try:
            snapshot = list(state.series_png_cache)
            # Cache float32 RGB arrays of the pristine base PNGs so repeated
            # rotate/flip/alpha re-applies skip PNG-decode on every slice.
            base_arrays = _base_arrays_cache[0]
            if base_arrays is None or len(base_arrays) != len(snapshot):
                base_arrays = [
                    np.array(
                        Image.open(io.BytesIO(png)).convert("RGB"),
                        dtype=np.float32,
                    )
                    for png in snapshot
                ]
                _base_arrays_cache[0] = base_arrays

            overlays: dict[int, bytes] = {}
            matched_counts: dict[int, int] = {}

            for idx, ds in enumerate(state.series_datasets):
                sop = getattr(ds, "SOPInstanceUID", None)
                if not sop:
                    continue
                masks = by_sop.get(str(sop))
                if not masks:
                    continue
                masks = _transform_masks(masks)
                composited = _composite_cached(
                    base_arrays[idx], masks, segments, alpha
                )
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
        _selected_seg_path[0] = str(file_path)
        use_mask_btn.disabled = False
        status_html.value = (
            f"<div style='color:#1565c0;background:#e8f4fd;padding:8px 10px;"
            f"border-left:3px solid #1976d2;border-radius:4px;font-size:12px;'>"
            f"Selected: <b>{Path(file_path).name}</b>. "
            f"Click \"Use mask\" to load.</div>"
        )
        return None

    def _on_use_mask(_btn):
        path = _selected_seg_path[0]
        if not path:
            status_html.value = _error_card(
                "Select a SEG file from the browser first."
            )
            return
        _load_new_seg(path)

    def _on_latest(_btn):
        latest = _find_latest_seg()
        if latest is None:
            status_html.value = _error_card(
                f"No segmentations found under {_SEG_OUTPUT_ROOT}."
            )
            return
        _selected_seg_path[0] = str(latest)
        use_mask_btn.disabled = False
        _load_new_seg(str(latest))

    def _on_clear(_btn):
        status_html.value = ""
        _restore_originals()
        _discard_overlay_state()
        _parsed_seg_cache[0] = None
        _current_seg_path[0] = None
        _selected_seg_path[0] = None
        use_mask_btn.disabled = True
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

    def _reapply_current():
        current_path = _current_seg_path[0]
        if not current_path:
            return
        _restore_originals()
        _discard_overlay_state()
        _apply_seg(current_path)

    def _on_rotate(_btn):
        _rotation[0] = (_rotation[0] + 1) % 4
        rotate_btn.button_style = "info" if _rotation[0] else ""
        _reapply_current()

    def _on_flip_h(_btn):
        _flip_h[0] = not _flip_h[0]
        flip_h_btn.button_style = "info" if _flip_h[0] else ""
        _reapply_current()

    def _on_flip_v(_btn):
        _flip_v[0] = not _flip_v[0]
        flip_v_btn.button_style = "info" if _flip_v[0] else ""
        _reapply_current()

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
        # stale originals without writing them back, and invalidate caches
        # that were tied to the old series.
        _base_arrays_cache[0] = None
        _parsed_seg_cache[0] = None
        if _originals[0] is None and _seg_cache[0] is None:
            return
        _discard_overlay_state()
        status_html.value = ""
        _current_seg_path[0] = None
        _selected_seg_path[0] = None
        use_mask_btn.disabled = True
        if state.seg_file_path:
            state.seg_file_path = ""

    use_mask_btn.on_click(_on_use_mask)
    latest_btn.on_click(_on_latest)
    clear_btn.on_click(_on_clear)
    rotate_btn.on_click(_on_rotate)
    flip_h_btn.on_click(_on_flip_h)
    flip_v_btn.on_click(_on_flip_v)
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
        [use_mask_btn, latest_btn, clear_btn],
        layout=widgets.Layout(gap="8px", padding="4px 0"),
    )
    orient_row = widgets.HBox(
        [rotate_btn, flip_h_btn, flip_v_btn],
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
            orient_row,
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
