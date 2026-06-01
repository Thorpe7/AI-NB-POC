"""Dashboard orchestrator: assembles segmentation-focused components."""

from __future__ import annotations

import warnings

import ipywidgets as widgets
from IPython.display import display

from utils.state import AppState
from utils.components.app_bar import build_app_bar
from utils.components.file_browser import build_image_browser
from utils.components.viewer_tab import build_viewer
from utils.components.segmentation_tab import build_segmentation
from utils.components.seg_viewer import build_seg_viewer

warnings.filterwarnings("ignore", category=UserWarning)

_APP_CSS = """<style>
.medgemma-app .widget-select select {
    font-size: 13px;
    font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
    cursor: pointer;
    border: 1px solid #dee2e6 !important;
    border-radius: 6px !important;
}
.medgemma-app .widget-text input,
.medgemma-app .widget-password input,
.medgemma-app .widget-textarea textarea {
    border-radius: 6px !important;
    border: 1px solid #dee2e6 !important;
}
.medgemma-app .widget-text input:focus,
.medgemma-app .widget-password input:focus,
.medgemma-app .widget-textarea textarea:focus {
    border-color: #1976d2 !important;
    box-shadow: 0 0 0 2px rgba(25,118,210,0.12) !important;
    outline: none !important;
}
.medgemma-app .widget-button button {
    border-radius: 6px !important;
    font-weight: 500 !important;
    transition: opacity 0.15s;
}
.medgemma-app .widget-button button:hover { opacity: 0.85; }
.medgemma-sidebar {
    background-color: #f8f9fa !important;
    border-right: 1px solid #dee2e6 !important;
}
.medgemma-switch .widget-checkbox input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 40px;
    height: 22px;
    background: #adb5bd;
    border-radius: 11px;
    position: relative;
    cursor: pointer;
    transition: background 0.2s;
    margin: 0;
    flex-shrink: 0;
}
.medgemma-switch .widget-checkbox input[type="checkbox"]::before {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 18px;
    height: 18px;
    background: #fff;
    border-radius: 50%;
    transition: transform 0.2s;
}
.medgemma-switch .widget-checkbox input[type="checkbox"]:checked {
    background: #1976d2;
}
.medgemma-switch .widget-checkbox input[type="checkbox"]:checked::before {
    transform: translateX(18px);
}
.medgemma-switch .widget-checkbox label {
    font-size: 13px;
    font-weight: 500;
    color: #495057;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>"""


def build_and_display_app():
    """Build the segmentation-focused dashboard and display it."""

    state = AppState()
    viewer = build_viewer(state)
    segmentation = build_segmentation(state)
    masks_panel = build_seg_viewer(state, viewer)
    header = build_app_bar()

    image_browser = build_image_browser(state, viewer)

    browser_toggle = widgets.Checkbox(
        value=True,
        description="Show Image Browser",
        indent=False,
    )

    def _on_browser_toggle(change):
        image_browser.layout.display = "" if change["new"] else "none"

    browser_toggle.observe(_on_browser_toggle, names="value")

    toggle_bar = widgets.HBox(
        [browser_toggle],
        layout=widgets.Layout(
            padding="8px 16px",
            gap="24px",
            border_bottom="1px solid #dee2e6",
        ),
    )
    toggle_bar.add_class("medgemma-switch")

    main_content = widgets.VBox(
        [
            widgets.HBox(
                [viewer["viewer_panel"], segmentation, masks_panel],
                layout=widgets.Layout(min_height="450px"),
            ),
            viewer["info_panel"],
        ],
        layout=widgets.Layout(flex="1", padding="16px"),
    )

    content = widgets.HBox(
        [image_browser, main_content],
        layout=widgets.Layout(width="100%"),
    )

    css = widgets.HTML(_APP_CSS)
    app = widgets.VBox(
        [css, header, toggle_bar, content],
        layout=widgets.Layout(width="100%"),
    )
    app.add_class("medgemma-app")

    display(app)
