"""Segmentation panel for the KServe TotalSegmentator endpoint."""

from __future__ import annotations

import time

import ipywidgets as widgets
import requests

from utils.http_client import predict, translate_path, translate_pod_path_to_local

TASKS: dict[str, str] = {"TotalSegmentator": "totalseg"}

_SPINNER_HTML = (
    '<div style="display:flex;align-items:center;gap:8px;padding:8px 0;">'
    '<div style="width:18px;height:18px;border:2px solid #e0e0e0;'
    'border-top-color:#1976d2;border-radius:50%;animation:spin 0.8s linear infinite;"></div>'
    '<span style="font-size:13px;color:#6c757d;">Running segmentation...</span></div>'
)

_PLACEHOLDER = (
    "<div style='color:#6c757d;padding:24px;text-align:center;"
    "font-size:13px;'>Select a DICOM series from the browser and run "
    "segmentation to see results here.</div>"
)


def _error_card(msg: str) -> str:
    return (
        f"<div style='color:#d32f2f;background:#fff3f3;padding:10px 12px;"
        f"border-left:3px solid #d32f2f;border-radius:4px;font-size:13px;'>{msg}</div>"
    )


def _response_card(result: dict, local_seg_path: str) -> str:
    rows = [
        f"<div><b>task</b> &nbsp; {result.get('task')}</div>",
        f"<div><b>fast</b> &nbsp; {result.get('fast')}</div>",
    ]
    roi = result.get("roi_subset")
    if roi:
        rows.append(f"<div><b>roi_subset</b> &nbsp; {', '.join(roi)}</div>")
    return (
        "<div style='background:#f0f4f8;border:1px solid #dee2e6;border-radius:6px;"
        "padding:12px 14px;font-size:12px;font-family:monospace;line-height:1.6;'>"
        "<div style='font-weight:700;color:#495057;margin-bottom:6px;'>"
        "Segmentation Result</div>"
        f"<div style='word-break:break-all;'><b>seg_path</b> &nbsp; "
        f"{local_seg_path}</div>"
        + "".join(rows) +
        "</div>"
    )


def build_segmentation(state):
    """Build the segmentation request panel (TotalSegmentator)."""

    task_labels = list(TASKS.keys())
    task_dropdown = widgets.Dropdown(
        options=task_labels,
        value=task_labels[0],
        description="Task:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "50px"},
    )

    series_label = widgets.HTML(value="")

    fast_checkbox = widgets.Checkbox(
        value=True, description="Fast (3 mm, ~20–30 s)", indent=False,
    )
    fast_checkbox_bar = widgets.HBox([fast_checkbox])
    fast_checkbox_bar.add_class("medgemma-switch")

    roi_input = widgets.Text(
        value="",
        placeholder="Optional: liver, heart, aorta",
        description="ROI subset:",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="100%"),
    )

    run_button = widgets.Button(
        description="Run Segmentation",
        icon="cog",
        button_style="primary",
        layout=widgets.Layout(width="100%", height="38px"),
    )

    spinner = widgets.HTML(value="")
    response_area = widgets.HTML(value=_PLACEHOLDER)

    load_overlay_button = widgets.Button(
        description="Load overlay in viewer",
        icon="eye",
        button_style="",
        layout=widgets.Layout(width="100%", height="34px", display="none"),
    )

    last_local_seg_path = {"path": None}

    def _refresh_series_label(*_):
        n = len(state.series_datasets)
        if state.series_dir_name and n > 0:
            series_label.value = (
                f"<div style='font-size:12px;color:#495057;padding:4px 0;'>"
                f"<b>Series:</b> {state.series_dir_name} "
                f"({n} slice{'s' if n != 1 else ''})</div>"
            )
            run_button.disabled = False
        else:
            series_label.value = (
                "<div style='font-size:12px;color:#b26a00;padding:4px 0;'>"
                "Load a DICOM series from the file browser to enable segmentation."
                "</div>"
            )
            run_button.disabled = True

    state.observe(
        _refresh_series_label, names=["series_dir_name", "series_datasets"]
    )
    _refresh_series_label()

    def _parse_roi(value: str) -> list[str] | None:
        items = [s.strip() for s in value.split(",") if s.strip()]
        return items or None

    def _build_payload() -> tuple[dict | None, str | None]:
        if not state.series_dir_path:
            return None, "No series loaded — pick one from the file browser first."
        try:
            dicom_dir = translate_path(state.series_dir_path)
        except ValueError as e:
            return None, str(e)
        payload: dict = {"dicom_dir": dicom_dir, "fast": fast_checkbox.value}
        roi = _parse_roi(roi_input.value)
        if roi:
            payload["roi_subset"] = roi
        return payload, None

    def _on_run(_btn):
        payload, err = _build_payload()
        if err:
            response_area.value = _error_card(err)
            return

        model_name = TASKS[task_dropdown.value]
        run_button.disabled = True
        load_overlay_button.layout.display = "none"
        last_local_seg_path["path"] = None
        spinner.value = _SPINNER_HTML
        t0 = time.time()

        try:
            result = predict(payload, model_name)
            elapsed = time.time() - t0
            try:
                local_seg = translate_pod_path_to_local(result["seg_path"])
            except ValueError as e:
                response_area.value = _error_card(
                    f"Segmentation succeeded but seg_path could not be "
                    f"translated: {e}"
                )
                return

            last_local_seg_path["path"] = local_seg
            predictor_elapsed = float(result.get("elapsed_s", 0.0))
            footer = (
                f"<div style='font-size:11px;color:#adb5bd;margin-top:8px;'>"
                f"Response time: {elapsed:.1f}s &nbsp;|&nbsp; "
                f"Predictor: {predictor_elapsed:.1f}s</div>"
            )
            response_area.value = _response_card(result, local_seg) + footer
            load_overlay_button.layout.display = ""

        except requests.Timeout:
            response_area.value = _error_card(
                "Request timed out. KServe may be cold-starting "
                "(can take several minutes on first request)."
            )
        except requests.HTTPError as e:
            body = ""
            try:
                body = e.response.text[:500]
            except Exception:
                pass
            response_area.value = _error_card(
                f"HTTP {e.response.status_code} from predictor: {body}"
            )
        except requests.RequestException as e:
            response_area.value = _error_card(f"Request failed: {e}")
        except Exception as e:
            response_area.value = _error_card(f"Unexpected error: {e}")
        finally:
            run_button.disabled = not state.series_dir_path
            spinner.value = ""

    def _on_load_overlay(_btn):
        if last_local_seg_path["path"] is None:
            return
        state.seg_file_path = last_local_seg_path["path"]

    run_button.on_click(_on_run)
    load_overlay_button.on_click(_on_load_overlay)

    return widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:13px;font-weight:700;color:#495057;"
                "padding:0 0 8px;'>Segmentation Panel</div>"
            ),
            task_dropdown,
            series_label,
            fast_checkbox_bar,
            roi_input,
            response_area,
            run_button,
            spinner,
            load_overlay_button,
        ],
        layout=widgets.Layout(
            flex="1", padding="0 0 0 16px",
            border_left="1px solid #e9ecef",
        ),
    )
