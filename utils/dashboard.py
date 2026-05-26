"""Dashboard orchestrator: assembles components and displays in cell output."""

from __future__ import annotations

import warnings

import ipywidgets as widgets
from IPython.display import display

from utils.state import AppState
from utils.components.app_bar import build_app_bar
from utils.components.file_browser import build_image_browser, build_report_browser
from utils.components.viewer_tab import build_viewer
from utils.components.chat_tab import build_chat
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
.medgemma-cred {
    background-color: #f8f9fa !important;
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
    """Build the full dashboard widget tree and display it."""

    state = AppState()
    viewer = build_viewer(state)
    chat = build_chat(state)
    segmentation = build_segmentation(state)
    seg_overlay = build_seg_viewer(state, viewer)
    action_tabs = widgets.Tab(
        children=[chat, segmentation, seg_overlay],
        layout=widgets.Layout(flex="1"),
    )
    action_tabs.set_title(0, "Chat")
    action_tabs.set_title(1, "Segmentation")
    action_tabs.set_title(2, "Overlay")
    header = build_app_bar()

    image_browser = build_image_browser(state, viewer)
    report_browser = build_report_browser(state, viewer)

    # --- Toggle switches (both on by default) ---
    image_toggle = widgets.Checkbox(
        value=True,
        description="Image Selection",
        indent=False,
    )
    report_toggle = widgets.Checkbox(
        value=True,
        description="Text Report",
        indent=False,
    )

    def _on_image_toggle(change):
        image_browser.layout.display = "" if change["new"] else "none"

    def _on_report_toggle(change):
        report_browser.layout.display = "" if change["new"] else "none"
        if not change["new"]:
            # Clear report when toggled off
            state.report_text = ""
            state.report_file_name = ""
            viewer["report_display"].value = ""

    image_toggle.observe(_on_image_toggle, names="value")
    report_toggle.observe(_on_report_toggle, names="value")

    browser_toggle = widgets.Checkbox(
        value=True,
        description="Show Browsers",
        indent=False,
    )

    def _on_browser_toggle(change):
        if change["new"]:
            image_browser.layout.display = "" if image_toggle.value else "none"
            report_browser.layout.display = "" if report_toggle.value else "none"
        else:
            image_browser.layout.display = "none"
            report_browser.layout.display = "none"

    browser_toggle.observe(_on_browser_toggle, names="value")

    toggle_bar = widgets.HBox(
        [image_toggle, report_toggle, browser_toggle],
        layout=widgets.Layout(
            padding="8px 16px",
            gap="24px",
            border_bottom="1px solid #dee2e6",
        ),
    )
    toggle_bar.add_class("medgemma-switch")

    # Report content panel (between viewer and chat, hidden until a report is loaded)
    report_content_panel = widgets.VBox(
        [viewer["report_display"]],
        layout=widgets.Layout(
            flex="0.8",
            padding="0 16px",
            border_left="1px solid #e9ecef",
            border_right="1px solid #e9ecef",
            display="none",
        ),
    )

    def _on_report_display_change(change):
        report_content_panel.layout.display = "" if change["new"] else "none"

    viewer["report_display"].observe(_on_report_display_change, names="value")

    # Main content: viewer + report + action tabs (chat, segmentation) side by
    # side, metadata below
    main_content = widgets.VBox(
        [
            widgets.HBox(
                [viewer["viewer_panel"], report_content_panel, action_tabs],
                layout=widgets.Layout(min_height="450px"),
            ),
            viewer["info_panel"],
        ],
        layout=widgets.Layout(flex="1", padding="16px"),
    )

    # Content area: browsers + main
    content = widgets.HBox(
        [image_browser, report_browser, main_content],
        layout=widgets.Layout(width="100%"),
    )

    css = widgets.HTML(_APP_CSS)
    app = widgets.VBox(
        [css, header, toggle_bar, content],
        layout=widgets.Layout(width="100%"),
    )
    app.add_class("medgemma-app")

    display(app)
