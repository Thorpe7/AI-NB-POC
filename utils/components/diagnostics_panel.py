"""Auto-discovering pipeline-diagnostics panel for the loaded DICOM series.

Mirrors ``seg_viewer``'s discovery: on series load, scans
``{state.series_dir_path}/../masks/*_diagnostics.json`` for sidecar timing
files emitted by transformer pods (BrainSeg today, DuneAI later). The
contract is generic across models — any JSON shaped as

    {model, version, schema_version, timings_ms, preprocess_breakdown?}

renders into a collapsible "Pipeline diagnostics" panel. Unknown
``schema_version`` values fall back to a raw-JSON dump.
"""

from __future__ import annotations

import json
from pathlib import Path

import ipywidgets as widgets

_MASKS_SUBDIR = "masks"
_DIAGNOSTICS_SUFFIX = "_diagnostics.json"
_KNOWN_SCHEMA_VERSIONS = {1}

_TOP_STAGE_ORDER = ("preprocess", "model_inference", "postprocess", "total")


def _muted_card(msg):
    return (
        f"<div style='color:#6c757d;background:#f8f9fa;padding:8px 10px;"
        f"border-left:3px solid #adb5bd;border-radius:4px;font-size:12px;'>{msg}</div>"
    )


def _error_card(msg):
    return (
        f"<div style='color:#d32f2f;background:#fff3f3;padding:8px 10px;"
        f"border-left:3px solid #d32f2f;border-radius:4px;font-size:12px;'>{msg}</div>"
    )


def _format_ms(value_ms):
    try:
        ms = float(value_ms)
    except (TypeError, ValueError):
        return str(value_ms)
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{ms:.0f} ms"


def _stage_bar(label, value_ms, max_ms, accent="#1976d2"):
    width_pct = 0
    if max_ms > 0:
        try:
            width_pct = max(0.0, min(100.0, float(value_ms) / float(max_ms) * 100.0))
        except (TypeError, ValueError):
            width_pct = 0
    return (
        f"<div style='display:flex;align-items:center;gap:10px;"
        f"padding:3px 0;font-size:12px;'>"
        f"<div style='flex:0 0 150px;color:#495057;'>{label}</div>"
        f"<div style='flex:1;background:#eef1f4;border-radius:3px;height:10px;"
        f"position:relative;overflow:hidden;'>"
        f"<div style='width:{width_pct:.1f}%;background:{accent};"
        f"height:100%;border-radius:3px;'></div></div>"
        f"<div style='flex:0 0 80px;text-align:right;color:#6c757d;"
        f"font-family:monospace;'>{_format_ms(value_ms)}</div></div>"
    )


def _render_known_schema(data):
    timings = data.get("timings_ms") or {}
    breakdown = data.get("preprocess_breakdown") or {}
    model = data.get("model", "?")
    version = data.get("version", "?")

    top_values = [
        (key, timings[key]) for key in _TOP_STAGE_ORDER if key in timings
    ]
    bar_basis = max(
        (v for k, v in top_values if k != "total" and isinstance(v, (int, float))),
        default=0,
    )

    header = (
        f"<div style='font-size:11px;color:#6c757d;padding:0 0 6px;'>"
        f"<b>{model}</b> &middot; v{version}</div>"
    )
    rows = []
    for key, val in top_values:
        accent = "#9e9e9e" if key == "total" else "#1976d2"
        basis = max(bar_basis, val) if key == "total" else bar_basis
        rows.append(_stage_bar(key, val, basis, accent=accent))
    top_html = widgets.HTML(value=header + "".join(rows))

    if not breakdown:
        return widgets.VBox([top_html])

    breakdown_basis = max(
        (v for v in breakdown.values() if isinstance(v, (int, float))),
        default=0,
    )
    breakdown_rows = "".join(
        _stage_bar(name, val, breakdown_basis, accent="#26a69a")
        for name, val in breakdown.items()
    )
    breakdown_html = widgets.HTML(
        value=(
            "<div style='font-size:11px;color:#6c757d;"
            "padding:8px 0 4px;border-top:1px solid #f0f0f0;margin-top:6px;'>"
            "Preprocess breakdown</div>"
            + breakdown_rows
        ),
        layout=widgets.Layout(display="none"),
    )

    toggle = widgets.Button(
        description="Show preprocess breakdown",
        icon="chevron-right",
        layout=widgets.Layout(width="240px", height="26px"),
    )

    state = {"expanded": False}

    def _on_toggle(_btn):
        state["expanded"] = not state["expanded"]
        breakdown_html.layout.display = "" if state["expanded"] else "none"
        toggle.icon = "chevron-down" if state["expanded"] else "chevron-right"
        toggle.description = (
            "Hide preprocess breakdown"
            if state["expanded"]
            else "Show preprocess breakdown"
        )

    toggle.on_click(_on_toggle)

    return widgets.VBox(
        [top_html, toggle, breakdown_html],
        layout=widgets.Layout(padding="4px 0 0 0"),
    )


def _render_unknown_schema(data, schema_version):
    raw = json.dumps(data, indent=2)
    warning = _muted_card(
        f"Unknown <code>schema_version={schema_version}</code>; showing raw JSON."
    )
    pre = widgets.HTML(
        value=(
            f"<pre style='font-size:11px;background:#f8f9fa;padding:8px 10px;"
            f"border-radius:4px;border:1px solid #e9ecef;overflow:auto;"
            f"max-height:240px;white-space:pre-wrap;'>{raw}</pre>"
        )
    )
    return widgets.VBox([widgets.HTML(value=warning), pre])


def _render_diagnostics_file(path: Path):
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return widgets.HTML(value=_error_card(f"Failed to read {path.name}: {exc}"))

    if not isinstance(data, dict):
        return widgets.HTML(
            value=_error_card(f"{path.name}: expected JSON object, got {type(data).__name__}.")
        )

    schema_version = data.get("schema_version")
    if schema_version in _KNOWN_SCHEMA_VERSIONS:
        body = _render_known_schema(data)
    else:
        body = _render_unknown_schema(data, schema_version)

    title = widgets.HTML(
        value=(
            f"<div style='font-size:12px;font-weight:600;color:#495057;"
            f"padding:6px 0 4px;'>{path.name}</div>"
        )
    )
    return widgets.VBox(
        [title, body],
        layout=widgets.Layout(
            padding="6px 10px",
            border="1px solid #e9ecef",
            border_radius="4px",
            margin="0 0 6px 0",
        ),
    )


def build_diagnostics_panel(state):
    """Build the collapsible Pipeline-diagnostics panel.

    Returns a VBox that auto-hides when no sibling diagnostics JSON exists
    for the currently loaded series.
    """

    header_label = widgets.HTML(
        "<div style='font-size:13px;font-weight:700;color:#495057;'>"
        "Pipeline diagnostics</div>",
        layout=widgets.Layout(flex="1"),
    )
    toggle_btn = widgets.Button(
        icon="chevron-down",
        tooltip="Collapse / expand",
        layout=widgets.Layout(width="30px", height="24px"),
    )
    refresh_btn = widgets.Button(
        icon="refresh",
        tooltip="Rescan diagnostics files",
        layout=widgets.Layout(width="30px", height="24px"),
    )
    header = widgets.HBox(
        [header_label, refresh_btn, toggle_btn],
        layout=widgets.Layout(align_items="center", padding="0 0 6px"),
    )
    body_box = widgets.VBox([], layout=widgets.Layout(padding="4px 0"))

    container = widgets.VBox(
        [header, body_box],
        layout=widgets.Layout(
            display="none",
            padding="12px 16px",
            border_top="1px solid #e9ecef",
            margin="12px 0 0 0",
        ),
    )

    state_flags = {"expanded": True}

    def _on_toggle(_btn):
        state_flags["expanded"] = not state_flags["expanded"]
        body_box.layout.display = "" if state_flags["expanded"] else "none"
        toggle_btn.icon = "chevron-down" if state_flags["expanded"] else "chevron-right"

    toggle_btn.on_click(_on_toggle)

    def _masks_dir_for_series():
        if not state.series_dir_path:
            return None
        return Path(state.series_dir_path).parent / _MASKS_SUBDIR

    def _discover():
        masks_dir = _masks_dir_for_series()
        if masks_dir is None or not masks_dir.is_dir():
            container.layout.display = "none"
            body_box.children = []
            return

        files = sorted(
            p for p in masks_dir.glob(f"*{_DIAGNOSTICS_SUFFIX}") if p.is_file()
        )
        if not files:
            container.layout.display = "none"
            body_box.children = []
            return

        body_box.children = [_render_diagnostics_file(p) for p in files]
        container.layout.display = ""

    refresh_btn.on_click(lambda _b: _discover())
    state.observe(lambda _c: _discover(), names="series_dir_path")

    _discover()
    return container
