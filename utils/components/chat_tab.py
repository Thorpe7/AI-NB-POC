"""MedGemma prompt, inference, and response display."""

from __future__ import annotations

import base64
import json
import re
import time

import botocore
import ipywidgets as widgets

from utils.config import CONTENT_TYPE, ENDPOINT_NAME

# Model registry: display name → model ID sent in the payload
_MODELS = {
    "MedGemma": "medgemma",
}

_SPINNER_HTML = (
    '<div style="display:flex;align-items:center;gap:8px;padding:8px 0;">'
    '<div style="width:18px;height:18px;border:2px solid #e0e0e0;'
    'border-top-color:#1976d2;border-radius:50%;animation:spin 0.8s linear infinite;"></div>'
    '<span style="font-size:13px;color:#6c757d;">Analyzing...</span></div>'
)


def _md_to_html(text):
    """Convert basic markdown (bold, italic, newlines) to HTML."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = text.replace("\n", "<br>")
    return text


def _chat_bubble(prompt, response, model_label="MedGemma"):
    response_html = _md_to_html(response)
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
        f"{response_html}</div></div>"
    )


def _error_card(msg):
    return (
        f"<div style='color:#d32f2f;background:#fff3f3;padding:10px 12px;"
        f"border-left:3px solid #d32f2f;border-radius:4px;font-size:13px;'>{msg}</div>"
    )


def _format_payload_summary(payload, has_report_context):
    """Return an HTML summary of the inference payload (no raw data)."""
    text_len = len(payload.get("text", ""))
    img_b64 = payload.get("image", "")
    img_kb = len(img_b64) * 3 / 4 / 1024  # approx decoded size
    model_id = payload.get("model_id", "unknown")
    report_flag = "Yes (prepended)" if has_report_context else "No"
    return (
        "<div style='background:#f0f4f8;border:1px solid #dee2e6;border-radius:6px;"
        "padding:10px 14px;margin:4px 0 8px;font-size:12px;font-family:monospace;'>"
        "<div style='font-weight:700;margin-bottom:6px;color:#495057;'>Payload Summary</div>"
        f"<div><b>model_id</b> &nbsp; {model_id}</div>"
        f"<div><b>text</b> &nbsp; String &mdash; {text_len:,} chars &nbsp;|&nbsp; "
        f"Report context: {report_flag}</div>"
        f"<div><b>image</b> &nbsp; Base64 PNG &mdash; ~{img_kb:,.1f} KB (encoded)</div>"
        "</div>"
    )


def build_chat(state):
    """Build the AI inference prompt and response panel."""

    model_dropdown = widgets.Dropdown(
        options=list(_MODELS.keys()),
        value=list(_MODELS.keys())[0],
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
    send_button = widgets.Button(
        description="Run Inference",
        icon="paper-plane",
        button_style="primary",
        layout=widgets.Layout(width="100%", height="38px"),
    )
    payload_toggle = widgets.Checkbox(
        value=False,
        description="Show Payload",
        indent=False,
    )
    payload_toggle_bar = widgets.HBox([payload_toggle])
    payload_toggle_bar.add_class("medgemma-switch")

    spinner = widgets.HTML(value="")
    payload_display = widgets.HTML(value="", layout=widgets.Layout(display="none"))

    def _on_payload_toggle(change):
        payload_display.layout.display = "" if change["new"] else "none"

    payload_toggle.observe(_on_payload_toggle, names="value")
    response_area = widgets.HTML(
        value=(
            "<div style='color:#6c757d;padding:24px;text-align:center;"
            "font-size:13px;'>Response will appear here after inference.</div>"
        ),
    )

    def _on_send(_btn):
        if not state.current_png_bytes:
            response_area.value = _error_card(
                "No image loaded. Select a DICOM file first."
            )
            return
        if state.sm_client is None:
            response_area.value = _error_card(
                "Not connected to AWS. Enter credentials above."
            )
            return

        prompt = prompt_area.value.strip()
        if not prompt:
            response_area.value = _error_card("Please enter a prompt.")
            return

        send_button.disabled = True
        spinner.value = _SPINNER_HTML

        # Prepend report context if attached
        full_prompt = prompt
        if state.report_text:
            full_prompt = (
                f"Clinical report ({state.report_file_name}):\n"
                f"{state.report_text}\n\n---\n\n{prompt}"
            )

        model_label = model_dropdown.value
        model_id = _MODELS[model_label]

        payload = {
            "text": full_prompt,
            "image": base64.b64encode(state.current_png_bytes).decode("utf-8"),
            "model_id": model_id,
        }
        payload_display.value = _format_payload_summary(
            payload, has_report_context=bool(state.report_text)
        )

        t0 = time.time()
        try:
            resp = state.sm_client.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType=CONTENT_TYPE,
                Body=json.dumps(payload),
            )
            elapsed = time.time() - t0
            result = json.loads(resp["Body"].read().decode("utf-8"))
            generated = result.get("generated_text", "(No text in response)")
            timing = (
                f"<div style='font-size:11px;color:#adb5bd;margin-top:8px;'>"
                f"Response time: {elapsed:.1f}s</div>"
            )
            report_note = ""
            if state.report_text:
                report_note = (
                    f"<div style='font-size:11px;color:#1565c0;margin-bottom:8px;'>"
                    f"&#x1F4CE; Report attached: {state.report_file_name}</div>"
                )
            response_area.value = (
                report_note + _chat_bubble(prompt, generated, model_label) + timing
            )

        except botocore.exceptions.ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "ModelError":
                response_area.value = _error_card(
                    "Model error &mdash; endpoint may still be loading."
                )
            else:
                response_area.value = _error_card(f"AWS error ({code}): {e}")
        except botocore.exceptions.ReadTimeoutError:
            response_area.value = _error_card(
                "Request timed out. Endpoint may be cold-starting (5-10 min)."
            )
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
            prompt_area,
            send_button,
            spinner,
            payload_display,
            response_area,
        ],
        layout=widgets.Layout(
            flex="1", padding="0 0 0 16px",
            border_left="1px solid #e9ecef",
        ),
    )
