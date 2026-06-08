"""Dashboard orchestrator — assembles the modern-gui AppLayout.

Five-region layout filling the viewport:

    +-----------------------------------------------------+
    |                  top toolbar                        |   header
    +--------+-----------------------------+--------------+
    |  scan  |     viewer + overlays       |  inference   |
    |  list  |                             |    panel     |
    +--------+-----------------------------+--------------+
    |                    footer                           |
    +-----------------------------------------------------+

All styling lives in `utils/styles/dark.css`, injected once at app boot via
a <style> HTML widget. No inline CSS in this module.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display

from utils.state import AppState
from utils.components.top_toolbar import build_top_toolbar
from utils.components.file_browser import build_image_browser
from utils.components.viewer_tab import build_viewer
from utils.components.inference_panel import build_inference_panel

warnings.filterwarnings("ignore", category=UserWarning)

_STYLES_PATH = Path(__file__).resolve().parent / "styles" / "dark.css"


def _load_styles() -> widgets.HTML:
    """Read dark.css and emit a single <style> HTML widget."""
    try:
        css = _STYLES_PATH.read_text()
    except OSError:
        css = ""
    return widgets.HTML(value=f"<style>{css}</style>")


def _footer() -> widgets.HTML:
    return widgets.HTML(
        value=(
            "<div class='nbpoc-footer'>"
            "<span>AI-NB-POC &middot; KServe inference &middot; modern-gui</span>"
            "</div>"
        )
    )


def build_and_display_app():
    """Build the modern-gui dashboard and display it."""

    state = AppState()

    css = _load_styles()
    toolbar = build_top_toolbar()
    viewer = build_viewer(state)
    image_browser = build_image_browser(state, viewer)
    inference_panel = build_inference_panel(state, viewer)
    footer = _footer()

    layout = widgets.AppLayout(
        header=toolbar,
        left_sidebar=image_browser,
        center=viewer["viewer_panel"],
        right_sidebar=inference_panel,
        footer=footer,
        pane_widths=["280px", 1, "360px"],
        pane_heights=["52px", 1, "28px"],
        merge=False,
    )
    layout.add_class("nbpoc-app")

    container = widgets.VBox(
        [css, layout],
        layout=widgets.Layout(width="100%"),
    )
    display(container)
