"""Right-pane inference panel — disclaimer + status + Analyze + results.

Visually wraps the existing segmentation form (`segmentation_tab`) and the
mask-discovery + display (`seg_viewer`, which itself embeds the diagnostics
panel) inside the clinical-workstation chrome: header + amber disclaimer +
Ready/Stop status row. The big blue "Analyze Current Slice" button is the
existing segmentation run-button restyled at the segmentation_tab layer.

This panel is presentational only — all inference logic stays in the wrapped
modules. Naming choice: "Inference Panel" (not "AI Findings") because the
pipeline today is KServe segmentation, not an LLM/VLM. See memory note
`feedback_inference_vs_ai_naming`.
"""

from __future__ import annotations

import ipywidgets as widgets

from utils.components.segmentation_tab import build_segmentation
from utils.components.seg_viewer import build_seg_viewer


_HEADER_HTML = (
    "<div class='nbpoc-inference-header'>"
    "<span class='title'><span class='glyph'>&#x2726;</span> Inference Panel</span>"
    "<span style='font-size:16px;color:var(--text-dim);cursor:pointer;'>&#x2715;</span>"
    "</div>"
)

_DISCLAIMER_HTML = (
    "<div class='nbpoc-disclaimer'>"
    "Inference Output &mdash; Not a Clinical Diagnosis. "
    "All results must be reviewed by a qualified radiologist."
    "</div>"
)

_STATUS_HTML = (
    "<div class='nbpoc-status-row'>"
    "<span class='nbpoc-status'><span class='dot'></span> Ready</span>"
    "<button class='nbpoc-stop-btn' type='button'>Stop</button>"
    "</div>"
)


def build_inference_panel(state, viewer) -> widgets.VBox:
    """Build the right-side Inference Panel.

    Composes:
      * static chrome (header, disclaimer, status row)
      * existing `segmentation_tab` (task dropdown + form + Analyze button)
      * existing `seg_viewer` (mask list + transform controls + diagnostics)

    Returns:
        VBox with the `nbpoc-inference` CSS class. Caller embeds this in the
        AppLayout's `right_sidebar` region.
    """

    header = widgets.HTML(value=_HEADER_HTML)
    disclaimer = widgets.HTML(value=_DISCLAIMER_HTML)
    status_row = widgets.HTML(value=_STATUS_HTML)

    segmentation_form = build_segmentation(state)
    segmentation_form.add_class("nbpoc-inference-form")

    results_section, display_section = build_seg_viewer(state, viewer)
    results_section.add_class("nbpoc-inference-results")

    # Three-zone layout: a fixed top (header/disclaimer/status), a flexing
    # middle that contains the segmentation form + results (the only zone
    # allowed to overflow — its scrollbar is hidden via CSS), and a pinned
    # bottom holding the DISPLAY/Opacity/orient controls so they're never
    # clipped regardless of how many result cards exist.
    top_zone = widgets.VBox(
        [header, disclaimer, status_row],
        layout=widgets.Layout(width="100%"),
    )
    top_zone.add_class("nbpoc-inference-top")

    middle_zone = widgets.VBox(
        [segmentation_form, results_section],
        layout=widgets.Layout(width="100%"),
    )
    middle_zone.add_class("nbpoc-inference-middle")

    bottom_zone = display_section
    bottom_zone.add_class("nbpoc-inference-bottom")

    body = widgets.VBox(
        [top_zone, middle_zone, bottom_zone],
        layout=widgets.Layout(width="100%", height="100%"),
    )
    body.add_class("nbpoc-inference")
    return body
