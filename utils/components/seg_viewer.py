"""Auto-discovering masks panel for the loaded DICOM series.

On series load, scans ``{state.series_dir_path}/../masks/*.dcm`` for DICOM SEG
files and exposes one checkbox per file. Multiple masks can be enabled at
once; they composite in checkbox order (later masks layer on top).

SEGs with more than one labeled segment expand to a per-segment checkbox list
so individual structures can be hidden. The opacity slider and orientation
buttons (rotate, flip H, flip V) apply to every active overlay.
"""

from __future__ import annotations

import io
from pathlib import Path

import ipywidgets as widgets
import numpy as np
from PIL import Image

from utils.dicom_utils import load_dicom_seg

from utils.components.diagnostics_panel import build_diagnostics_panel

_MASKS_SUBDIR = "masks"


def _pil_to_png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def _error_card(msg):
    return (
        f"<div class='nbpoc-card severe' style='font-size:11.5px;"
        f"color:var(--severity-severe-fg);'>{msg}</div>"
    )


def _muted_card(msg):
    return (
        f"<div style='color:var(--text-muted);background:var(--bg-panel-alt);"
        f"padding:10px 12px;border-left:3px solid var(--border-strong);"
        f"border-radius:4px;font-size:11.5px;line-height:1.4;'>{msg}</div>"
    )


def _color_swatch(rgb):
    r, g, b = rgb
    return (
        f"<span style='display:inline-block;width:11px;height:11px;"
        f"background:rgb({r},{g},{b});border-radius:2px;"
        f"border:1px solid rgba(255,255,255,0.15);vertical-align:middle;'></span>"
    )


def _severity_for_seg(num_segments: int) -> str:
    """Bucket: 0 → normal, 1-2 → moderate, 3+ → severe. Tweakable."""
    if num_segments <= 0:
        return "normal"
    if num_segments <= 2:
        return "moderate"
    return "severe"


def build_seg_viewer(state, viewer):
    """Build the auto-discovering masks panel.

    Rebuilds the mask list whenever ``state.series_dir_path`` changes. Each
    discovered ``.dcm`` becomes a checkbox; toggling drives a recomposite of
    the series PNG cache so the viewer reflects the active overlay set.
    """

    image_widget = viewer["image_widget"]

    # Pristine snapshots of the per-slice PNGs at series-load time. We never
    # mutate this; recomposites always start from base_arrays (decoded once)
    # and write into state.series_png_cache.
    _originals: list[bytes] | None = None
    _base_arrays: list[np.ndarray] | None = None

    # Per-mask state. List ordering is the composite z-order (top of list
    # paints first, later entries layer on top).
    # Entry shape:
    #   {
    #     "path": str,
    #     "name": str,
    #     "parsed": dict | None,        # cached load_dicom_seg result
    #     "parse_error": str | None,
    #     "row": HBox,                  # the checkbox/expand row
    #     "checkbox": Checkbox,         # master enable
    #     "expand_btn": Button,
    #     "segments_box": VBox,         # per-segment checkbox container
    #     "segment_checkboxes": {segment_number: Checkbox},
    #     "expanded": bool,
    #   }
    _masks: list[dict] = []

    # Orientation, applied to every mask before compositing so all overlays
    # share the same axis convention. Identical semantics to the prior
    # rotate/flip controls.
    _rotation = [0]  # number of 90° CCW turns
    _flip_h = [False]
    _flip_v = [False]

    # --- widgets ------------------------------------------------------------

    header_label = widgets.HTML(
        "<div class='nbpoc-section-label'>RESULTS (0)</div>",
        layout=widgets.Layout(flex="1"),
    )
    refresh_btn = widgets.Button(
        icon="refresh",
        tooltip="Rescan the masks folder",
        layout=widgets.Layout(width="28px", height="24px"),
    )
    header = widgets.HBox(
        [header_label, refresh_btn],
        layout=widgets.Layout(align_items="center", padding="12px 0 4px"),
    )
    status_html = widgets.HTML(value="")
    mask_list_box = widgets.VBox([], layout=widgets.Layout(padding="4px 0"))

    alpha_slider = widgets.FloatSlider(
        value=0.4, min=0.1, max=0.9, step=0.05,
        description="Opacity:", continuous_update=False,
        style={"description_width": "60px"},
        layout=widgets.Layout(width="100%"),
    )
    rotate_btn = widgets.Button(
        description="Rotate 90°", icon="redo",
        tooltip="Rotate overlays 90° CCW (cycles 0/90/180/270)",
        layout=widgets.Layout(width="100px", height="28px"),
    )
    flip_h_btn = widgets.Button(
        description="Flip H", icon="arrows-h",
        tooltip="Mirror overlays horizontally",
        layout=widgets.Layout(width="74px", height="28px"),
    )
    flip_v_btn = widgets.Button(
        description="Flip V", icon="arrows-v",
        tooltip="Mirror overlays vertically",
        layout=widgets.Layout(width="74px", height="28px"),
    )
    orient_row = widgets.HBox(
        [rotate_btn, flip_h_btn, flip_v_btn],
        layout=widgets.Layout(gap="6px", padding="4px 0"),
    )
    display_controls_label = widgets.HTML(
        "<div class='nbpoc-section-label' style='padding-top:12px;'>DISPLAY</div>"
    )

    # --- helpers ------------------------------------------------------------

    def _refresh_viewer_image():
        if state.series_datasets and state.series_png_cache:
            idx = state.series_index
            if 0 <= idx < len(state.series_png_cache):
                image_widget.value = state.series_png_cache[idx]

    def _ensure_base_arrays():
        nonlocal _originals, _base_arrays
        if _originals is None and state.series_png_cache:
            _originals = list(state.series_png_cache)
        if _originals is None:
            return False
        if _base_arrays is None or len(_base_arrays) != len(_originals):
            _base_arrays = [
                np.array(
                    Image.open(io.BytesIO(png)).convert("RGB"),
                    dtype=np.float32,
                )
                for png in _originals
            ]
        return True

    def _reset_caches():
        nonlocal _originals, _base_arrays
        _originals = None
        _base_arrays = None

    def _transform_mask_array(arr):
        if _flip_h[0]:
            arr = np.fliplr(arr)
        if _flip_v[0]:
            arr = np.flipud(arr)
        if _rotation[0]:
            arr = np.rot90(arr, _rotation[0])
        return arr

    def _active_layers():
        """Return list of (by_sop, segments, segments_enabled) for enabled masks."""
        layers = []
        for entry in _masks:
            if not entry["checkbox"].value:
                continue
            parsed = entry["parsed"]
            if parsed is None:
                continue
            enabled_segs = {
                num for num, cb in entry["segment_checkboxes"].items() if cb.value
            }
            if not enabled_segs:
                continue
            layers.append((parsed["by_source_sop"], parsed["segments"], enabled_segs))
        return layers

    def _recomposite():
        """Rebuild state.series_png_cache from base arrays + active layers."""
        if not state.series_datasets:
            return
        if not _ensure_base_arrays():
            return
        layers = _active_layers()
        alpha = float(alpha_slider.value)

        if not layers:
            state.series_png_cache = list(_originals)
            _refresh_viewer_image()
            return

        new_cache = list(_originals)
        for idx, ds in enumerate(state.series_datasets):
            sop = getattr(ds, "SOPInstanceUID", None)
            if not sop:
                continue
            sop_key = str(sop)

            base = _base_arrays[idx]
            composite = None
            for by_sop, segments, enabled_segs in layers:
                slice_masks = by_sop.get(sop_key)
                if not slice_masks:
                    continue
                filtered = {
                    n: slice_masks[n]
                    for n in enabled_segs
                    if n in slice_masks
                }
                if not filtered:
                    continue
                if composite is None:
                    composite = base.copy()
                h, w = composite.shape[:2]
                for seg_num, mask in filtered.items():
                    mask = _transform_mask_array(mask)
                    if mask.shape != (h, w):
                        mask_pil = Image.fromarray(
                            (mask.astype(np.uint8) * 255), mode="L"
                        ).resize((w, h), Image.NEAREST)
                        mask = np.array(mask_pil) > 127
                    color = np.array(segments[seg_num]["color"], dtype=np.float32)
                    composite[mask] = (
                        (1.0 - alpha) * composite[mask] + alpha * color
                    )
            if composite is not None:
                img = Image.fromarray(
                    np.clip(composite, 0, 255).astype(np.uint8), mode="RGB"
                )
                new_cache[idx] = _pil_to_png_bytes(img)

        state.series_png_cache = new_cache
        _refresh_viewer_image()

    def _restore_originals():
        if _originals is None:
            return
        state.series_png_cache = list(_originals)
        _refresh_viewer_image()

    # --- mask discovery + UI build -----------------------------------------

    def _masks_dir_for_series():
        if not state.series_dir_path:
            return None
        series_dir = Path(state.series_dir_path)
        return series_dir.parent / _MASKS_SUBDIR

    def _build_segment_row(entry, num, label, color, slice_count):
        cb = widgets.Checkbox(
            value=True,
            description="",
            indent=False,
            layout=widgets.Layout(width="22px", margin="0"),
        )
        cb.observe(lambda _c: _recomposite(), names="value")
        entry["segment_checkboxes"][num] = cb
        meta = widgets.HTML(
            f"<div style='font-size:12px;color:#495057;line-height:1.4;'>"
            f"{_color_swatch(color)} &nbsp;{label}"
            f"<span style='color:#6c757d;font-size:11px;'> &nbsp;&middot;&nbsp; "
            f"{slice_count} slice{'s' if slice_count != 1 else ''}</span></div>"
        )
        return widgets.HBox(
            [cb, meta],
            layout=widgets.Layout(
                align_items="center",
                padding="2px 0 2px 18px",
            ),
        )

    def _populate_segments(entry):
        parsed = entry["parsed"]
        seg_rows = []
        segments = parsed["segments"]
        matched_counts: dict[int, int] = {}
        for sop_masks in parsed["by_source_sop"].values():
            for num in sop_masks:
                matched_counts[num] = matched_counts.get(num, 0) + 1
        for num in sorted(segments.keys()):
            info = segments[num]
            seg_rows.append(
                _build_segment_row(
                    entry,
                    num=num,
                    label=info["label"],
                    color=info["color"],
                    slice_count=matched_counts.get(num, 0),
                )
            )
        entry["segments_box"].children = seg_rows

    def _on_expand_clicked(entry):
        def _handler(_btn):
            entry["expanded"] = not entry["expanded"]
            if entry["expanded"]:
                if entry["parsed"] is None and not entry["parse_error"]:
                    _ensure_parsed(entry)
                if entry["parsed"] is not None and not entry["segment_checkboxes"]:
                    _populate_segments(entry)
                if entry["parsed"] is not None:
                    _refresh_card_severity(entry)
            target = entry.get("drawer") or entry["segments_box"]
            target.layout.display = "" if entry["expanded"] else "none"
        return _handler

    def _refresh_card_severity(entry):
        """Recompute card severity from parsed segment count + restyle."""
        parsed = entry.get("parsed")
        title_html = entry.get("title_html")
        card = entry.get("card")
        if parsed is None or title_html is None or card is None:
            return
        seg_count = len(parsed.get("segments") or {})
        severity = _severity_for_seg(seg_count)
        for cls in ("normal", "moderate", "severe"):
            card.remove_class(cls)
        card.add_class(severity)
        title_html.value = (
            f"<div class='nbpoc-card-head'>"
            f"<div class='nbpoc-card-title'>{Path(entry['path']).stem}</div>"
            f"<span class='nbpoc-card-severity {severity}'>{severity}</span>"
            f"</div>"
            f"<div class='nbpoc-card-meta'>"
            f"<span>See report</span>"
            f"<span class='sep'>&middot;</span>"
            f"<span>{seg_count} segment{'s' if seg_count != 1 else ''}</span>"
            f"<span class='sep'>&middot;</span>"
            f"<span>50%</span>"
            f"</div>"
        )

    def _ensure_parsed(entry):
        if entry["parsed"] is not None or entry["parse_error"]:
            return
        try:
            entry["parsed"] = load_dicom_seg(Path(entry["path"]))
        except Exception as e:
            entry["parse_error"] = str(e)
            entry["checkbox"].disabled = True
            entry["checkbox"].description = f"{entry['name']}  (not a SEG: {e})"

    def _on_mask_toggle(entry):
        def _handler(change):
            if change["new"]:
                _ensure_parsed(entry)
                if entry["parse_error"]:
                    return
                if not entry["segment_checkboxes"]:
                    _populate_segments(entry)
            _recomposite()
        return _handler

    def _build_mask_entry(path: Path) -> dict:
        entry: dict = {
            "path": str(path),
            "name": path.name,
            "parsed": None,
            "parse_error": None,
            "segment_checkboxes": {},
            "expanded": False,
        }
        # Master enable for the overlay — lives alongside Accept/Edit/Reject
        # so the user can toggle visibility without opening the Edit drawer.
        checkbox = widgets.Checkbox(
            value=False,
            description="Show",
            indent=False,
            tooltip="Toggle this mask's overlay on the viewer",
            layout=widgets.Layout(width="auto"),
        )
        # Per-segment toggles, populated lazily once the SEG is parsed.
        segments_box = widgets.VBox([], layout=widgets.Layout(display="none"))

        # Action row: Accept / Edit / Reject. Accept is a no-op flag, Edit
        # toggles the Edit drawer (enable checkbox + segments box), Reject
        # hides the card and disables any active overlay for that mask.
        accept_btn = widgets.Button(
            description="Accept", icon="check",
            layout=widgets.Layout(width="auto", height="26px"),
        )
        edit_btn = widgets.Button(
            description="Edit", icon="pencil",
            layout=widgets.Layout(width="auto", height="26px"),
        )
        reject_btn = widgets.Button(
            description="Reject", icon="times",
            layout=widgets.Layout(width="auto", height="26px"),
        )
        for b in (accept_btn, edit_btn, reject_btn):
            b.add_class("nbpoc-card-action")
        accept_btn.add_class("accept")
        edit_btn.add_class("edit")
        reject_btn.add_class("reject")
        action_row = widgets.HBox(
            [checkbox, accept_btn, edit_btn, reject_btn],
            layout=widgets.Layout(gap="6px", padding="0"),
        )
        action_row.add_class("nbpoc-card-actions")

        # The Edit drawer holds per-segment toggles; hidden by default,
        # revealed by the Edit button. The master Show toggle moved up into
        # the action row.
        drawer = widgets.VBox(
            [segments_box],
            layout=widgets.Layout(display="none"),
        )
        drawer.add_class("nbpoc-card-drawer")

        # Card title + severity tag. Severity stays 'normal' until the SEG is
        # parsed and we can count segments; updated lazily inside _ensure_parsed.
        title_html = widgets.HTML(
            value=(
                f"<div class='nbpoc-card-head'>"
                f"<div class='nbpoc-card-title'>{path.stem}</div>"
                f"<span class='nbpoc-card-severity normal'>normal</span>"
                f"</div>"
                f"<div class='nbpoc-card-meta'>"
                f"<span>See report</span>"
                f"<span class='sep'>&middot;</span>"
                f"<span>{path.name}</span>"
                f"<span class='sep'>&middot;</span>"
                f"<span>50%</span>"
                f"</div>"
            )
        )

        card = widgets.VBox(
            [title_html, action_row, drawer],
            layout=widgets.Layout(margin="0 0 10px 0"),
        )
        card.add_class("nbpoc-card")
        card.add_class("normal")

        entry["checkbox"] = checkbox
        entry["expand_btn"] = edit_btn
        entry["segments_box"] = segments_box
        entry["accept_btn"] = accept_btn
        entry["reject_btn"] = reject_btn
        entry["title_html"] = title_html
        entry["card"] = card
        entry["drawer"] = drawer

        checkbox.observe(_on_mask_toggle(entry), names="value")
        edit_btn.on_click(_on_expand_clicked(entry))

        def _on_reject(_btn):
            if checkbox.value:
                checkbox.value = False
            card.layout.display = "none"

        def _on_accept(_btn):
            accept_btn.disabled = True

        reject_btn.on_click(_on_reject)
        accept_btn.on_click(_on_accept)

        entry["row"] = card
        return entry

    def _discover_masks():
        nonlocal _masks
        _masks = []
        mask_list_box.children = []
        header_label.value = "<div class='nbpoc-section-label'>RESULTS (0)</div>"

        masks_dir = _masks_dir_for_series()
        if masks_dir is None:
            status_html.value = _muted_card("Load a series to see available masks.")
            return
        if not masks_dir.is_dir():
            status_html.value = _muted_card(
                f"No <code>{_MASKS_SUBDIR}/</code> folder next to this series "
                f"(<code>{masks_dir}</code>). Run a segmentation to generate one."
            )
            return

        dcm_files = sorted(p for p in masks_dir.glob("*.dcm") if p.is_file())
        if not dcm_files:
            status_html.value = _muted_card(
                f"No <code>.dcm</code> files in <code>{masks_dir}</code>."
            )
            return

        rows = []
        for p in dcm_files:
            entry = _build_mask_entry(p)
            _masks.append(entry)
            rows.append(entry["row"])

        mask_list_box.children = rows
        header_label.value = (
            f"<div class='nbpoc-section-label'>RESULTS ({len(dcm_files)})</div>"
        )
        status_html.value = ""

    # --- event handlers -----------------------------------------------------

    def _on_alpha_change(_change):
        _recomposite()

    def _on_rotate(_btn):
        _rotation[0] = (_rotation[0] + 1) % 4
        rotate_btn.button_style = "info" if _rotation[0] else ""
        _recomposite()

    def _on_flip_h(_btn):
        _flip_h[0] = not _flip_h[0]
        flip_h_btn.button_style = "info" if _flip_h[0] else ""
        _recomposite()

    def _on_flip_v(_btn):
        _flip_v[0] = not _flip_v[0]
        flip_v_btn.button_style = "info" if _flip_v[0] else ""
        _recomposite()

    def _on_series_dir_change(_change):
        _reset_caches()
        _discover_masks()
        _restore_originals()

    def _on_series_datasets_change(_change):
        # series_png_cache is repopulated by file_browser when a new series
        # loads; capture a fresh snapshot before any compositing happens.
        _reset_caches()

    def _on_refresh(_btn):
        # Preserve the active overlay set across a rescan by carrying enabled
        # filenames forward — running a fresh segmentation often adds a new
        # mask but should not silently disable ones the user already toggled.
        previously_enabled = {
            entry["name"]
            for entry in _masks
            if entry["checkbox"].value
        }
        _discover_masks()
        if not previously_enabled:
            return
        any_restored = False
        for entry in _masks:
            if entry["name"] in previously_enabled:
                entry["checkbox"].value = True
                any_restored = True
        if not any_restored:
            _recomposite()

    alpha_slider.observe(_on_alpha_change, names="value")
    rotate_btn.on_click(_on_rotate)
    flip_h_btn.on_click(_on_flip_h)
    flip_v_btn.on_click(_on_flip_v)
    refresh_btn.on_click(_on_refresh)
    state.observe(_on_series_dir_change, names="series_dir_path")
    state.observe(_on_series_datasets_change, names="series_datasets")

    _discover_masks()

    diagnostics_panel = build_diagnostics_panel(state)

    return widgets.VBox(
        [
            header,
            status_html,
            mask_list_box,
            display_controls_label,
            alpha_slider,
            orient_row,
            diagnostics_panel,
        ],
        layout=widgets.Layout(width="100%", padding="0"),
    )


