"""Dashboard orchestrator: assembles components and opens in JupyterLab side panel."""

from __future__ import annotations

import warnings

import ipywidgets as widgets
from IPython.display import display

try:
    from sidecar import Sidecar
except ImportError:
    Sidecar = None

from utils.state import AppState
from utils.components.app_bar import build_app_bar
from utils.components.file_browser import build_file_browser
from utils.components.viewer_tab import build_viewer
from utils.components.chat_tab import build_chat

warnings.filterwarnings("ignore", category=UserWarning)


def build_and_display_app():
    """Build the full dashboard and display it in a JupyterLab side panel.

    Falls back to cell output if sidecar is not installed.
    """

    state = AppState()
    viewer = build_viewer(state)
    chat = build_chat(state)

    app = widgets.VBox([
        build_app_bar(state),
        widgets.HBox(
            [viewer["image_panel"], chat],
            layout=widgets.Layout(min_height="420px"),
        ),
        build_file_browser(state, viewer),
        viewer["info_panel"],
    ])

    if Sidecar is not None:
        sc = Sidecar(title="XNAT AI Assistant", anchor="split-right")
        with sc:
            display(app)
    else:
        display(app)
