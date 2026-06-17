"""Mask-driven pipeline-diagnostics panel.

Tabs are driven by ``state.active_diagnostics``: a dict
``{mask_path: diag_path}`` populated by ``seg_viewer`` as the user toggles
each mask's Show button. Per-mask diagnostics files are written by the
transformer pods with a stem that matches the SEG file (e.g.,
``lesion_mask_seg_v1.dcm`` → ``lesion_mask_seg_v1_diagnostics.json``).

Closing a tab via its × removes the entry from ``state.active_diagnostics``;
``seg_viewer`` observes that mutation and turns the matching mask's Show
toggle off — keeping the overlay and the diagnostics tab in lockstep.

JSON contract — any object shaped as

    {model, version, schema_version, timings_ms, preprocess_breakdown?}

renders into a structured per-stage view. Unknown ``schema_version`` values
fall back to a raw-JSON dump.
"""

from __future__ import annotations

import json
from pathlib import Path

import ipywidgets as widgets

_KNOWN_SCHEMA_VERSIONS = {1}

_TOP_STAGE_ORDER = ("preprocess", "model_inference", "postprocess", "total")


def _muted_card(msg):
    return (
        f"<div style='color:var(--text-muted);background:var(--bg-panel-alt);"
        f"padding:8px 10px;border-left:3px solid var(--border-strong);"
        f"border-radius:4px;font-size:11.5px;'>{msg}</div>"
    )


def _error_card(msg):
    return (
        f"<div style='color:var(--severity-severe-fg);"
        f"background:rgba(211,47,47,0.10);padding:8px 10px;"
        f"border-left:3px solid var(--severity-severe-fg);"
        f"border-radius:4px;font-size:11.5px;'>{msg}</div>"
    )


def _format_ms(value_ms):
    try:
        ms = float(value_ms)
    except (TypeError, ValueError):
        return str(value_ms)
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{ms:.0f} ms"


def _stage_bar(label, value_ms, max_ms, accent="var(--accent)"):
    width_pct = 0
    if max_ms > 0:
        try:
            width_pct = max(0.0, min(100.0, float(value_ms) / float(max_ms) * 100.0))
        except (TypeError, ValueError):
            width_pct = 0
    return (
        f"<div style='display:flex;align-items:center;gap:10px;"
        f"padding:3px 0;font-size:11.5px;'>"
        f"<div style='flex:0 0 130px;color:var(--text-muted);'>{label}</div>"
        f"<div style='flex:1;background:var(--bg-input);border-radius:3px;"
        f"height:8px;position:relative;overflow:hidden;'>"
        f"<div style='width:{width_pct:.1f}%;background:{accent};"
        f"height:100%;border-radius:3px;'></div></div>"
        f"<div style='flex:0 0 70px;text-align:right;color:var(--text-dim);"
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
        f"<div style='font-size:10.5px;color:var(--text-muted);padding:0 0 6px;'>"
        f"<b style='color:var(--text);'>{model}</b> &middot; v{version}</div>"
    )
    rows = []
    for key, val in top_values:
        accent = "var(--text-dim)" if key == "total" else "var(--accent)"
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
            "<div style='font-size:10.5px;color:var(--text-muted);"
            "padding:8px 0 4px;border-top:1px solid var(--border);margin-top:6px;'>"
            "Preprocess breakdown</div>"
            + breakdown_rows
        ),
        layout=widgets.Layout(display="none"),
    )

    toggle = widgets.Button(
        description="Show preprocess breakdown",
        icon="chevron-right",
        layout=widgets.Layout(width="220px", height="24px"),
    )

    inner_state = {"expanded": False}

    def _on_toggle(_btn):
        inner_state["expanded"] = not inner_state["expanded"]
        breakdown_html.layout.display = "" if inner_state["expanded"] else "none"
        toggle.icon = "chevron-down" if inner_state["expanded"] else "chevron-right"
        toggle.description = (
            "Hide preprocess breakdown"
            if inner_state["expanded"]
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
            f"<pre style='font-size:10.5px;background:var(--bg-input);"
            f"color:var(--text);padding:8px 10px;border-radius:4px;"
            f"border:1px solid var(--border);overflow:auto;max-height:240px;"
            f"white-space:pre-wrap;'>{raw}</pre>"
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
            f"<div style='font-size:11.5px;font-weight:600;color:var(--text);"
            f"padding:6px 0 4px;'>{path.name}</div>"
        )
    )
    return widgets.VBox(
        [title, body],
        layout=widgets.Layout(
            padding="6px 10px",
            border="1px solid var(--border)",
            border_radius="4px",
            margin="0 0 6px 0",
        ),
    )


def _tab_title_for(mask_path: str) -> str:
    p = Path(mask_path)
    stem = p.stem
    parts = p.parts
    try:
        scans_idx = parts.index("SCANS")
        scan_id = parts[scans_idx + 1]
        return f"{scan_id} / {stem}"
    except (ValueError, IndexError):
        return stem


def build_diagnostics_panel(state):
    """Build the mask-driven Pipeline-diagnostics panel.

    Returns a VBox that auto-hides when no mask has surfaced a paired
    diagnostics file via ``state.active_diagnostics``.
    """

    header_label = widgets.HTML(
        "<div class='nbpoc-section-label' style='padding-top:12px;'>"
        "PIPELINE DIAGNOSTICS</div>",
        layout=widgets.Layout(flex="1"),
    )
    toggle_btn = widgets.Button(
        icon="chevron-down",
        tooltip="Collapse / expand",
        layout=widgets.Layout(width="28px", height="22px"),
    )
    header = widgets.HBox(
        [header_label, toggle_btn],
        layout=widgets.Layout(align_items="center", padding="0 0 6px"),
    )
    tabs = widgets.Tab(layout=widgets.Layout(padding="4px 0"))
    tabs.add_class("nbpoc-diagnostics-tabs")

    container = widgets.VBox(
        [header, tabs],
        layout=widgets.Layout(
            display="none",
            padding="6px 0 0 0",
            margin="0",
        ),
    )

    state_flags = {"expanded": True}
    # mask_path -> tab content widget. Keyed by mask path so we can
    # update/remove based on state.active_diagnostics mutations without
    # re-rendering tabs the user already has open.
    open_tabs: dict[str, widgets.VBox] = {}

    def _on_toggle(_btn):
        state_flags["expanded"] = not state_flags["expanded"]
        tabs.layout.display = "" if state_flags["expanded"] else "none"
        toggle_btn.icon = "chevron-down" if state_flags["expanded"] else "chevron-right"

    toggle_btn.on_click(_on_toggle)

    def _remove_from_state(mask_path: str):
        """Drop ``mask_path`` from state.active_diagnostics — triggers seg_viewer
        to turn the matching mask's Show toggle off via its observer."""
        current = dict(state.active_diagnostics)
        if mask_path not in current:
            return
        del current[mask_path]
        state.active_diagnostics = current

    def _build_tab(mask_path: str, diag_path: str):
        close_btn = widgets.Button(
            description="× Close",
            tooltip="Close this tab (and hide the matching mask)",
            layout=widgets.Layout(width="80px", height="22px"),
        )
        body = _render_diagnostics_file(Path(diag_path))
        tab_content = widgets.VBox(
            [
                widgets.HBox(
                    [close_btn],
                    layout=widgets.Layout(justify_content="flex-end", padding="0"),
                ),
                body,
            ],
            layout=widgets.Layout(padding="4px 0"),
        )
        close_btn.on_click(lambda _b: _remove_from_state(mask_path))
        return tab_content

    def _sync_tabs(active: dict[str, str]):
        """Make the tab set match ``active`` (mask_path -> diag_path)."""
        # Remove tabs whose mask is no longer active.
        to_remove = [mp for mp in open_tabs if mp not in active]
        if to_remove:
            remaining_children = [
                w for mp, w in open_tabs.items() if mp not in to_remove
            ]
            remaining_titles = [
                _tab_title_for(mp) for mp in open_tabs if mp not in to_remove
            ]
            for mp in to_remove:
                del open_tabs[mp]
            tabs.children = tuple(remaining_children)
            for i, t in enumerate(remaining_titles):
                tabs.set_title(i, t)

        # Append tabs for newly-active masks.
        new_masks = [mp for mp in active if mp not in open_tabs]
        if new_masks:
            existing_children = list(tabs.children)
            existing_titles = [
                tabs.get_title(i) for i in range(len(existing_children))
            ]
            new_widgets = [_build_tab(mp, active[mp]) for mp in new_masks]
            new_titles = [_tab_title_for(mp) for mp in new_masks]
            for mp, w in zip(new_masks, new_widgets):
                open_tabs[mp] = w
            tabs.children = tuple(existing_children + new_widgets)
            for i, t in enumerate(existing_titles + new_titles):
                tabs.set_title(i, t)

        container.layout.display = "" if open_tabs else "none"

    def _on_active_diagnostics_change(change):
        _sync_tabs(change["new"] or {})

    state.observe(_on_active_diagnostics_change, names="active_diagnostics")

    _sync_tabs(state.active_diagnostics or {})
    return container
