"""Single-shot inference panel for the KServe MedGemma endpoint."""

from __future__ import annotations

import json
import re
import time

import ipywidgets as widgets
import requests

from utils.config import MODELS, get_model_caps
from utils.http_client import predict, translate_path

_SPINNER_HTML = (
    '<div style="display:flex;align-items:center;gap:8px;padding:8px 0;">'
    '<div style="width:18px;height:18px;border:2px solid #e0e0e0;'
    'border-top-color:#1976d2;border-radius:50%;animation:spin 0.8s linear infinite;"></div>'
    '<span style="font-size:13px;color:#6c757d;">Analyzing...</span></div>'
)

_PLACEHOLDER = (
    "<div style='color:#6c757d;padding:24px;text-align:center;"
    "font-size:13px;'>Response will appear here after inference.</div>"
)

def _md_to_html(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text.replace("\n", "<br>")


def _error_card(msg: str) -> str:
    return (
        f"<div style='color:#d32f2f;background:#fff3f3;padding:10px 12px;"
        f"border-left:3px solid #d32f2f;border-radius:4px;font-size:13px;'>{msg}</div>"
    )


def _response_card(prompt: str, response: str, model_label: str) -> str:
    return (
        "<div style='display:flex;flex-direction:column;gap:10px;'>"
        "<div style='background:#e3f2fd;padding:10px 14px;border-radius:10px 10px 10px 2px;"
        "font-size:13px;line-height:1.5;'>"
        "<div style='font-size:11px;font-weight:600;color:#1565c0;margin-bottom:4px;'>You</div>"
        f"{prompt}</div>"
        "<div style='background:#f5f5f5;padding:10px 14px;border-radius:10px 10px 2px 10px;"
        "font-size:13px;line-height:1.6;'>"
        "<div style='font-size:11px;font-weight:600;color:#495057;margin-bottom:4px;'>"
        f"{model_label}</div>"
        f"{_md_to_html(response)}</div></div>"
    )


def _payload_summary_html(payload: dict, model_name: str) -> str:
    rows = [
        "<div style='background:#f0f4f8;border:1px solid #dee2e6;border-radius:6px;"
        "padding:10px 14px;margin:4px 0 8px;font-size:12px;font-family:monospace;'>",
        "<div style='font-weight:700;margin-bottom:6px;color:#495057;'>Payload Summary</div>",
        f"<div><b>model</b> &nbsp; {model_name}</div>",
    ]
    for key in ("prompt", "dicom_dir", "slice_range", "image_paths",
                "text_paths", "max_new_tokens"):
        if key not in payload:
            continue
        val = payload[key]
        if key == "prompt":
            display = f"{len(val):,} chars"
        elif isinstance(val, list):
            display = json.dumps(val) if len(val) <= 3 else f"{len(val)} items"
        else:
            display = str(val)
        rows.append(f"<div><b>{key}</b> &nbsp; {display}</div>")
    rows.append("</div>")
    return "".join(rows)


def build_chat(state):
    """Build the inference prompt and response panel."""

    model_labels = list(MODELS.keys())
    model_dropdown = widgets.Dropdown(
        options=model_labels,
        value=model_labels[0],
        description="Model:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "50px"},
    )

    prompt_area = widgets.Textarea(
        value="Describe this medical image. What findings are visible?",
        placeholder="Enter your question about the image...",
        rows=3,
        layout=widgets.Layout(width="100%"),
    )

    max_new_tokens_input = widgets.BoundedIntText(
        value=512, min=1, max=4096, step=64,
        description="Max new tokens:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="100%"),
    )

    slice_start_input = widgets.BoundedIntText(
        value=0, min=0, max=0,
        description="Slice start:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="48%"),
    )
    slice_end_input = widgets.BoundedIntText(
        value=0, min=0, max=0,
        description="Slice end:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="48%"),
    )
    slice_range_box = widgets.HBox(
        [slice_start_input, slice_end_input],
        layout=widgets.Layout(
            display="none", justify_content="space-between", width="100%",
        ),
    )
    slice_hint = widgets.HTML(
        value="",
        layout=widgets.Layout(display="none"),
    )

    send_button = widgets.Button(
        description="Run Inference",
        icon="paper-plane",
        button_style="primary",
        layout=widgets.Layout(width="100%", height="38px"),
    )

    payload_toggle = widgets.Checkbox(
        value=False, description="Show Payload", indent=False,
    )
    payload_toggle_bar = widgets.HBox([payload_toggle])
    payload_toggle_bar.add_class("medgemma-switch")

    spinner = widgets.HTML(value="")
    payload_display = widgets.HTML(
        value="", layout=widgets.Layout(display="none"),
    )
    response_area = widgets.HTML(value=_PLACEHOLDER)

    def _on_payload_toggle(change):
        payload_display.layout.display = "" if change["new"] else "none"

    payload_toggle.observe(_on_payload_toggle, names="value")

    def _refresh_slice_inputs(*_):
        n = len(state.series_datasets)
        caps = get_model_caps(MODELS[model_dropdown.value])

        if n > 1 and caps["supports_series"]:
            slice_range_box.layout.display = ""
            slice_hint.layout.display = ""
            slice_hint.value = (
                f"<div style='font-size:11px;color:#6c757d;padding:2px 0 6px;'>"
                f"Series loaded ({n} slices). Slice range is required, "
                f"0-indexed inclusive.</div>"
            )
            for w in (slice_start_input, slice_end_input):
                w.max = n - 1
                if w.value > n - 1:
                    w.value = n - 1
        elif n > 1 and not caps["supports_series"]:
            slice_range_box.layout.display = "none"
            slice_hint.layout.display = ""
            slice_hint.value = (
                f"<div style='font-size:11px;color:#b26a00;padding:2px 0 6px;'>"
                f"{model_dropdown.value} accepts a single image only. "
                f"Click an individual DICOM in the file browser to select "
                f"one slice from this series.</div>"
            )
        else:
            slice_range_box.layout.display = "none"
            slice_hint.layout.display = "none"

    state.observe(_refresh_slice_inputs, names=["series_datasets"])
    model_dropdown.observe(_refresh_slice_inputs, names="value")
    _refresh_slice_inputs()

    def _build_payload():
        prompt = prompt_area.value.strip()
        if not prompt:
            return None, "Please enter a prompt."

        model_name = MODELS[model_dropdown.value]
        caps = get_model_caps(model_name)

        payload = {
            "prompt": prompt,
            "max_new_tokens": int(max_new_tokens_input.value),
        }

        n_slices = len(state.series_datasets)
        if n_slices > 1 and caps["supports_series"]:
            if not state.series_dir_path:
                return None, "Series loaded but series directory path is missing."
            try:
                payload["dicom_dir"] = translate_path(state.series_dir_path)
            except ValueError as e:
                return None, str(e)
            start = int(slice_start_input.value)
            end = int(slice_end_input.value)
            if start > end:
                return None, f"slice_range start ({start}) must be <= end ({end})."
            count = end - start + 1
            if count > caps["max_images"]:
                return None, (
                    f"slice_range selects {count} slices but "
                    f"{model_dropdown.value} accepts at most {caps['max_images']}."
                )
            payload["slice_range"] = [start, end]
        elif n_slices > 1 and not caps["supports_series"]:
            return None, (
                f"{model_dropdown.value} accepts a single image only. "
                f"Click an individual DICOM in the file browser to pick one slice."
            )
        elif state.current_file_path:
            try:
                payload["image_paths"] = [translate_path(state.current_file_path)]
            except ValueError as e:
                return None, str(e)

        if state.report_file_path:
            try:
                payload["text_paths"] = [translate_path(state.report_file_path)]
            except ValueError as e:
                return None, str(e)

        return payload, None

    def _on_send(_btn):
        payload, err = _build_payload()
        if err:
            response_area.value = _error_card(err)
            return

        model_label = model_dropdown.value
        model_name = MODELS[model_label]

        payload_display.value = _payload_summary_html(payload, model_name)

        send_button.disabled = True
        spinner.value = _SPINNER_HTML
        t0 = time.time()

        try:
            result = predict(payload, model_name)
            elapsed = time.time() - t0
            generated = result.get("generated_text", "(No text in response)")
            slice_info = result.get("slice_info")

            footer_bits = [f"Response time: {elapsed:.1f}s"]
            if slice_info:
                footer_bits.append(
                    f"slices: {slice_info.get('used_slices')}/"
                    f"{slice_info.get('total_slices')} "
                    f"(indices {slice_info.get('used_indices')})"
                )
            footer = (
                "<div style='font-size:11px;color:#adb5bd;margin-top:8px;'>"
                + " &nbsp;|&nbsp; ".join(footer_bits)
                + "</div>"
            )
            response_area.value = (
                _response_card(payload["prompt"], generated, model_label) + footer
            )

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
            send_button.disabled = False
            spinner.value = ""

    send_button.on_click(_on_send)

    return widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:13px;font-weight:700;color:#495057;"
                "padding:0 0 8px;'>AI Inference Panel</div>"
            ),
            model_dropdown,
            payload_toggle_bar,
            response_area,
            prompt_area,
            max_new_tokens_input,
            slice_hint,
            slice_range_box,
            send_button,
            spinner,
            payload_display,
        ],
        layout=widgets.Layout(
            flex="1", padding="0 0 0 16px",
            border_left="1px solid #e9ecef",
        ),
    )
