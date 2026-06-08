"""Top toolbar — visual stub matching the clinical-workstation reference.

Every button is a visual no-op for the modern-gui refresh; the real controls
(task dropdown, run button, masks/metadata visibility) live in the inference
panel and elsewhere. This module just renders the chrome.
"""

from __future__ import annotations

import ipywidgets as widgets


def _btn(label: str, *, active: bool = False, dropdown: bool = False) -> str:
    cls = "nbpoc-toolbar-btn" + (" active" if active else "")
    arrow = " <span style='font-size:9px;opacity:0.7;'>&#x25BE;</span>" if dropdown else ""
    return f"<button class='{cls}' type='button'>{label}{arrow}</button>"


def _icon(glyph: str, title: str = "") -> str:
    return (
        f"<button class='nbpoc-toolbar-icon' type='button' title='{title}'>"
        f"{glyph}</button>"
    )


def build_top_toolbar() -> widgets.HTML:
    """Return the top toolbar as a single HTML widget."""

    left = (
        "<div class='nbpoc-toolbar-group'>"
        "<span class='nbpoc-toolbar-brand'>"
        "<span class='dot'></span>"
        "<span>cnda.wustl.edu</span>"
        "</span>"
        + _icon("&#x25A0;", "Single view")
        + _icon("&#x25AE;&#x25AE;", "1&times;2")
        + _icon("&#x25A6;", "2&times;2")
        + "<span class='nbpoc-toolbar-divider'></span>"
        + _btn("Protocol", dropdown=True)
        + "</div>"
    )

    center = (
        "<div class='nbpoc-toolbar-group'>"
        + _btn("MPR")
        + _btn("W/L", active=True)
        + _btn("Pan")
        + _btn("Zoom")
        + _btn("Measure", dropdown=True)
        + _btn("Segment", dropdown=True)
        + _btn("Presets", dropdown=True)
        + "</div>"
    )

    right = (
        "<div class='nbpoc-toolbar-group'>"
        + _icon("&#x21BA;", "Undo")
        + _icon("&#x21BB;", "Redo")
        + "<span class='nbpoc-toolbar-divider'></span>"
        + _icon("&#x25C1;", "Previous")
        + _icon("&#x25B7;", "Play")
        + _icon("&#x25B7;&#x25B7;", "Next")
        + "<span class='nbpoc-toolbar-fps'>15 fps</span>"
        + "<span class='nbpoc-toolbar-divider'></span>"
        + _icon("&#x22EE;", "More")
        + _icon("&#x263C;", "Theme")
        + "</div>"
    )

    html = (
        f"<div class='nbpoc-toolbar'>{left}{center}{right}</div>"
    )
    return widgets.HTML(value=html)
