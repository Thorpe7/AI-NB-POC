"""Dashboard orchestrator: assembles components and displays in cell output."""

from __future__ import annotations

import warnings

import ipywidgets as widgets
from IPython.display import display

from utils.state import AppState
from utils.components.app_bar import build_app_bar
from utils.components.file_browser import build_file_browser
from utils.components.viewer_tab import build_viewer
from utils.components.chat_tab import build_chat

warnings.filterwarnings("ignore", category=UserWarning)


def build_and_display_app():
    """Build the full dashboard widget tree and display it."""

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

    display(app)
