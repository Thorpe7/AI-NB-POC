"""AI inference prompt, response display, and multi-model support."""

from __future__ import annotations

import base64
import json
import re
import time

import botocore
import ipywidgets as widgets

from utils.config import (
    BACKGROUND_POLL_INTERVAL,
    BACKGROUND_POLL_TIMEOUT,
    CONTENT_TYPE,
    ENDPOINT_NAME,
    VOLUME_S3_BUCKET,
    VOLUME_S3_PREFIX,
)
from utils.volume_utils import (
    dicom_series_to_nifti_bytes,
    series_fingerprint,
    upload_volume_to_s3,
    volume_exists_in_s3,
)

# Model registry: display name -> model ID sent in the payload
_MODELS = {
    "MedGemma": "medgemma",
    "Merlin 3D CT": "merlin",
}

# Merlin modes: display name -> mode ID sent in the payload
_MERLIN_MODES = {
    "Findings": "findings",
    "Phenotype": "phenotype",
    "Retrieval": "retrieval",
    "Prediction": "prediction",
    "Report": "report",
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


_PLACEHOLDER = (
    "<div style='color:#6c757d;padding:24px;text-align:center;"
    "font-size:13px;'>Response will appear here after inference.</div>"
)


def _error_card(msg):
    return (
        f"<div style='color:#d32f2f;background:#fff3f3;padding:10px 12px;"
        f"border-left:3px solid #d32f2f;border-radius:4px;font-size:13px;'>{msg}</div>"
    )


def _render_history(history, model_label):
    """Build full conversation HTML from chat history."""
    if not history:
        return _PLACEHOLDER
    bubbles = []
    for i in range(0, len(history), 2):
        user_msg = history[i]["content"]
        asst_msg = history[i + 1]["content"] if i + 1 < len(history) else "..."
        bubbles.append(_chat_bubble(user_msg, asst_msg, model_label))
    return (
        "<div id='chat-scroll' style='max-height:500px;overflow-y:auto;display:flex;"
        "flex-direction:column;gap:16px;'>"
        + "".join(bubbles)
        + "</div>"
        "<script>document.getElementById('chat-scroll').scrollTop = "
        "document.getElementById('chat-scroll').scrollHeight;</script>"
    )


def _format_payload_summary(payload, has_report_context, turn):
    """Return an HTML summary of the inference payload (no raw data)."""
    model_id = payload.get("model", "unknown")
    lines = [
        "<div style='background:#f0f4f8;border:1px solid #dee2e6;border-radius:6px;"
        "padding:10px 14px;margin:4px 0 8px;font-size:12px;font-family:monospace;'>",
        "<div style='font-weight:700;margin-bottom:6px;color:#495057;'>Payload Summary</div>",
        f"<div><b>model</b> &nbsp; {model_id}</div>",
    ]

    if model_id == "merlin":
        mode = payload.get("mode", "?")
        uri = payload.get("volume_s3_uri", "?")
        lines.append(f"<div><b>mode</b> &nbsp; {mode}</div>")
        lines.append(f"<div><b>volume</b> &nbsp; {uri}</div>")
        if "query_texts" in payload:
            lines.append(
                f"<div><b>queries</b> &nbsp; {len(payload['query_texts'])} text(s)</div>"
            )
    else:
        text_len = len(payload.get("text", ""))
        img_b64 = payload.get("image", "")
        img_kb = len(img_b64) * 3 / 4 / 1024
        report_flag = "Yes (prepended)" if has_report_context else "No"
        session_flag = payload.get("session_id", "New session")
        lines.append(
            f"<div><b>session</b> &nbsp; {session_flag} &nbsp;|&nbsp; Turn: {turn}</div>"
        )
        lines.append(
            f"<div><b>text</b> &nbsp; String &mdash; {text_len:,} chars &nbsp;|&nbsp; "
            f"Report context: {report_flag}</div>"
        )
        lines.append(
            f"<div><b>image</b> &nbsp; Base64 PNG &mdash; ~{img_kb:,.1f} KB (encoded)</div>"
        )

    lines.append("</div>")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Merlin response renderers
# ---------------------------------------------------------------------------

def _render_merlin_response(result):
    """Dispatch Merlin result to the appropriate renderer by mode."""
    mode = result.get("mode", "unknown")
    renderers = {
        "findings": _render_findings,
        "phenotype": _render_phenotype,
        "retrieval": _render_retrieval,
        "prediction": _render_prediction,
        "report": _render_report,
    }
    renderer = renderers.get(mode)
    if renderer is None:
        return _error_card(f"Unknown Merlin mode: {mode}")
    return renderer(result)


def _table_style():
    return (
        "border-collapse:collapse;width:100%;font-size:13px;"
        "border:1px solid #dee2e6;border-radius:6px;"
    )


def _th_style():
    return (
        "background:#e3f2fd;padding:8px 12px;text-align:left;"
        "font-weight:600;border-bottom:2px solid #1976d2;font-size:12px;"
    )


def _td_style():
    return "padding:8px 12px;border-bottom:1px solid #eee;"


def _render_findings(result):
    findings = result.get("findings", {})
    if isinstance(findings, dict):
        rows = "".join(
            f"<tr><td style='{_td_style()}'>{k}</td>"
            f"<td style='{_td_style()}'>{v}</td></tr>"
            for k, v in findings.items()
        )
    elif isinstance(findings, list):
        rows = "".join(
            f"<tr><td style='{_td_style()}'>{item}</td></tr>" for item in findings
        )
    else:
        rows = f"<tr><td style='{_td_style()}'>{findings}</td></tr>"
    header = (
        f"<tr><th style='{_th_style()}'>Finding</th>"
        f"<th style='{_th_style()}'>Value</th></tr>"
        if isinstance(findings, dict)
        else f"<tr><th style='{_th_style()}'>Finding</th></tr>"
    )
    return (
        "<div style='padding:8px 0;'>"
        "<div style='font-weight:600;color:#1565c0;margin-bottom:8px;'>Findings</div>"
        f"<table style='{_table_style()}'>{header}{rows}</table></div>"
    )


def _render_phenotype(result):
    top_k = result.get("top_k", [])
    rows = ""
    for item in top_k:
        name = item.get("name", "?")
        score = item.get("score", 0)
        pct = int(float(score) * 100)
        bar = (
            f"<div style='background:#e0e0e0;border-radius:4px;height:16px;width:100%;'>"
            f"<div style='background:#1976d2;border-radius:4px;height:16px;"
            f"width:{pct}%;'></div></div>"
        )
        rows += (
            f"<tr><td style='{_td_style()}'>{name}</td>"
            f"<td style='{_td_style()};width:50%;'>{bar}</td>"
            f"<td style='{_td_style()};text-align:right;'>{score:.3f}</td></tr>"
        )
    return (
        "<div style='padding:8px 0;'>"
        "<div style='font-weight:600;color:#1565c0;margin-bottom:8px;'>Phenotype</div>"
        f"<table style='{_table_style()}'>"
        f"<tr><th style='{_th_style()}'>Phenotype</th>"
        f"<th style='{_th_style()}'>Score</th>"
        f"<th style='{_th_style()}'>Value</th></tr>"
        f"{rows}</table></div>"
    )


def _render_retrieval(result):
    scores = result.get("similarity_scores", [])
    rows = "".join(
        f"<tr><td style='{_td_style()}'>{s.get('query', '?')}</td>"
        f"<td style='{_td_style()};text-align:right;'>{s.get('score', 0):.4f}</td></tr>"
        for s in scores
    )
    return (
        "<div style='padding:8px 0;'>"
        "<div style='font-weight:600;color:#1565c0;margin-bottom:8px;'>Retrieval</div>"
        f"<table style='{_table_style()}'>"
        f"<tr><th style='{_th_style()}'>Query</th>"
        f"<th style='{_th_style()}'>Similarity</th></tr>"
        f"{rows}</table></div>"
    )


def _render_prediction(result):
    risk = result.get("risk_scores", {})
    if isinstance(risk, dict):
        items = risk.items()
    elif isinstance(risk, list):
        items = [(f"Risk {i+1}", v) for i, v in enumerate(risk)]
    else:
        items = []
    rows = ""
    for name, score in items:
        score_f = float(score)
        pct = int(score_f * 100)
        bar = (
            f"<div style='background:#e0e0e0;border-radius:4px;height:16px;width:100%;'>"
            f"<div style='background:#ef6c00;border-radius:4px;height:16px;"
            f"width:{pct}%;'></div></div>"
        )
        rows += (
            f"<tr><td style='{_td_style()}'>{name}</td>"
            f"<td style='{_td_style()};width:50%;'>{bar}</td>"
            f"<td style='{_td_style()};text-align:right;'>{score_f:.4f}</td></tr>"
        )
    return (
        "<div style='padding:8px 0;'>"
        "<div style='font-weight:600;color:#1565c0;margin-bottom:8px;'>"
        "5-Year Disease Prediction</div>"
        f"<table style='{_table_style()}'>"
        f"<tr><th style='{_th_style()}'>Risk Category</th>"
        f"<th style='{_th_style()}'>Score</th>"
        f"<th style='{_th_style()}'>Value</th></tr>"
        f"{rows}</table></div>"
    )


def _render_report(result):
    text = result.get("report_text", "(No report text)")
    return (
        "<div style='padding:8px 0;'>"
        "<div style='font-weight:600;color:#1565c0;margin-bottom:8px;'>Report</div>"
        f"<div style='background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;"
        f"padding:12px 16px;font-size:13px;line-height:1.7;'>"
        f"{_md_to_html(text)}</div></div>"
    )


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_chat(state):
    """Build the AI inference prompt and response panel."""

    model_dropdown = widgets.Dropdown(
        options=list(_MODELS.keys()),
        value=list(_MODELS.keys())[0],
        description="Model:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "50px"},
    )

    mode_dropdown = widgets.Dropdown(
        options=list(_MERLIN_MODES.keys()),
        value=list(_MERLIN_MODES.keys())[0],
        description="Mode:",
        layout=widgets.Layout(width="100%", display="none"),
        style={"description_width": "50px"},
    )

    query_texts_area = widgets.Textarea(
        placeholder="Enter query texts (one per line) for retrieval mode...",
        rows=3,
        layout=widgets.Layout(width="100%", display="none"),
    )

    volume_status = widgets.HTML(
        value="",
        layout=widgets.Layout(display="none"),
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

    new_conversation_btn = widgets.Button(
        description="New Conversation",
        icon="refresh",
        button_style="warning",
        layout=widgets.Layout(width="100%", height="32px"),
    )

    spinner = widgets.HTML(value="")
    payload_display = widgets.HTML(value="", layout=widgets.Layout(display="none"))

    def _on_payload_toggle(change):
        payload_display.layout.display = "" if change["new"] else "none"

    payload_toggle.observe(_on_payload_toggle, names="value")
    response_area = widgets.HTML(value=_PLACEHOLDER)

    # --- Model / mode switching ---

    def _update_volume_status():
        n = len(state.series_datasets)
        if n > 0:
            volume_status.value = (
                f"<div style='font-size:12px;color:#2e7d32;padding:4px 0;'>"
                f"&#10003; {n} slices loaded</div>"
            )
        else:
            volume_status.value = (
                "<div style='font-size:12px;color:#d32f2f;padding:4px 0;'>"
                "&#10007; No series loaded &mdash; load a DICOM series first.</div>"
            )

    def _on_model_change(change):
        is_merlin = _MODELS.get(change["new"]) == "merlin"
        if is_merlin:
            prompt_area.layout.display = "none"
            new_conversation_btn.layout.display = "none"
            mode_dropdown.layout.display = ""
            volume_status.layout.display = ""
            _update_volume_status()
            _on_mode_change({"new": mode_dropdown.value})
        else:
            prompt_area.layout.display = ""
            new_conversation_btn.layout.display = ""
            mode_dropdown.layout.display = "none"
            volume_status.layout.display = "none"
            query_texts_area.layout.display = "none"

    model_dropdown.observe(_on_model_change, names="value")

    def _on_mode_change(change):
        mode_id = _MERLIN_MODES.get(change["new"], "")
        if mode_id == "retrieval":
            query_texts_area.layout.display = ""
        else:
            query_texts_area.layout.display = "none"

    mode_dropdown.observe(_on_mode_change, names="value")

    # Update volume status when series changes
    def _on_series_change(change):
        if _MODELS.get(model_dropdown.value) == "merlin":
            _update_volume_status()

    state.observe(_on_series_change, names=["series_datasets"])

    # --- Inference dispatchers ---

    def _send_medgemma():
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
            "model": model_id,
        }
        if state.session_id:
            payload["session_id"] = state.session_id
        payload_display.value = _format_payload_summary(
            payload, has_report_context=bool(state.report_text),
            turn=state.current_turn,
        )

        t0 = time.time()
        try:
            resp = state.sm_client.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType=CONTENT_TYPE,
                Body=json.dumps(payload),
            )
            result = json.loads(resp["Body"].read().decode("utf-8"))

            elapsed = time.time() - t0
            generated = result.get("generated_text", "(No text in response)")

            if result.get("session_id"):
                state.session_id = result["session_id"]
            if result.get("turn") is not None:
                state.current_turn = result["turn"]

            state.chat_history = state.chat_history + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": generated},
            ]

            resp_keys = ", ".join(sorted(result.keys()))
            sid_val = result.get("session_id", "<missing>")
            turn_val = result.get("turn", "<missing>")
            timing = (
                f"<div style='font-size:11px;color:#adb5bd;margin-top:8px;'>"
                f"Response time: {elapsed:.1f}s &nbsp;|&nbsp; "
                f"keys: [{resp_keys}] &nbsp;|&nbsp; "
                f"session_id: {sid_val} &nbsp;|&nbsp; turn: {turn_val}</div>"
            )
            response_area.value = (
                _render_history(state.chat_history, model_label) + timing
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

    def _send_merlin():
        if not state.series_datasets:
            response_area.value = _error_card(
                "No DICOM series loaded. Load a series directory first."
            )
            return
        if len(state.series_datasets) < 10:
            response_area.value = _error_card(
                f"Series too small ({len(state.series_datasets)} slices). "
                "Need at least 10 slices for volumetric analysis."
            )
            return
        if state.sm_client is None:
            response_area.value = _error_card(
                "Not connected to AWS. Enter credentials above."
            )
            return
        if state.s3_client is None:
            response_area.value = _error_card(
                "S3 client not available. Reconnect to AWS."
            )
            return

        mode_label = mode_dropdown.value
        mode_id = _MERLIN_MODES[mode_label]

        if mode_id == "retrieval":
            raw_queries = query_texts_area.value.strip()
            if not raw_queries:
                response_area.value = _error_card(
                    "Retrieval mode requires at least one query text."
                )
                return
            query_texts = [q.strip() for q in raw_queries.split("\n") if q.strip()]
        else:
            query_texts = None

        send_button.disabled = True
        spinner.value = _SPINNER_HTML

        t0 = time.time()
        try:
            # Check if this series volume already exists in S3 — the
            # fingerprint is derived purely from DICOM metadata so it
            # survives kernel restarts and endpoint redeployments.
            datasets = list(state.series_datasets)
            s3_key = f"{VOLUME_S3_PREFIX}test.nii.gz"
            volume_uri = f"s3://{VOLUME_S3_BUCKET}/{s3_key}"

            spinner.value = (
                '<div style="display:flex;align-items:center;gap:8px;padding:8px 0;">'
                '<div style="width:18px;height:18px;border:2px solid #e0e0e0;'
                'border-top-color:#1976d2;border-radius:50%;'
                'animation:spin 0.8s linear infinite;"></div>'
                '<span style="font-size:13px;color:#6c757d;">'
                'Checking for cached volume in S3...</span></div>'
            )

            if volume_exists_in_s3(state.s3_client, VOLUME_S3_BUCKET, s3_key):
                spinner.value = (
                    '<div style="display:flex;align-items:center;gap:8px;padding:8px 0;">'
                    '<span style="font-size:13px;color:#43a047;">'
                    'Volume found in S3 — skipping upload</span></div>'
                )
            else:
                spinner.value = (
                    '<div style="display:flex;align-items:center;gap:8px;padding:8px 0;">'
                    '<div style="width:18px;height:18px;border:2px solid #e0e0e0;'
                    'border-top-color:#1976d2;border-radius:50%;'
                    'animation:spin 0.8s linear infinite;"></div>'
                    '<span style="font-size:13px;color:#6c757d;">'
                    'Converting DICOM to NIfTI...</span></div>'
                )
                nifti_bytes = dicom_series_to_nifti_bytes(datasets)

                spinner.value = (
                    '<div style="display:flex;align-items:center;gap:8px;padding:8px 0;">'
                    '<div style="width:18px;height:18px;border:2px solid #e0e0e0;'
                    'border-top-color:#1976d2;border-radius:50%;'
                    'animation:spin 0.8s linear infinite;"></div>'
                    '<span style="font-size:13px;color:#6c757d;">'
                    'Uploading volume to S3...</span></div>'
                )
                volume_uri = upload_volume_to_s3(
                    state.s3_client, nifti_bytes, VOLUME_S3_BUCKET, VOLUME_S3_PREFIX,
                    fingerprint="test",
                )

            spinner.value = (
                '<div style="display:flex;align-items:center;gap:8px;padding:8px 0;">'
                '<div style="width:18px;height:18px;border:2px solid #e0e0e0;'
                'border-top-color:#1976d2;border-radius:50%;'
                'animation:spin 0.8s linear infinite;"></div>'
                '<span style="font-size:13px;color:#6c757d;">'
                'Submitting inference job...</span></div>'
            )

            payload = {
                "model": "merlin",
                "mode": mode_id,
                "volume_s3_uri": volume_uri,
                "background": True,
            }
            if query_texts:
                payload["query_texts"] = query_texts

            payload_display.value = _format_payload_summary(
                payload, has_report_context=False, turn=0,
            )

            # Submit background job
            resp = state.sm_client.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType=CONTENT_TYPE,
                Body=json.dumps(payload),
            )
            submit_result = json.loads(resp["Body"].read().decode("utf-8"))
            job_id = submit_result["job_id"]

            # Poll until completed
            elapsed_poll = 0
            result = None
            while elapsed_poll < BACKGROUND_POLL_TIMEOUT:
                time.sleep(BACKGROUND_POLL_INTERVAL)
                elapsed_poll += BACKGROUND_POLL_INTERVAL
                spinner.value = (
                    '<div style="display:flex;align-items:center;gap:8px;padding:8px 0;">'
                    '<div style="width:18px;height:18px;border:2px solid #e0e0e0;'
                    'border-top-color:#1976d2;border-radius:50%;'
                    'animation:spin 0.8s linear infinite;"></div>'
                    f'<span style="font-size:13px;color:#6c757d;">'
                    f'Analyzing... {elapsed_poll}s</span></div>'
                )

                poll_resp = state.sm_client.invoke_endpoint(
                    EndpointName=ENDPOINT_NAME,
                    ContentType=CONTENT_TYPE,
                    Body=json.dumps({"model": "merlin", "poll_job_id": job_id}),
                )
                poll_result = json.loads(poll_resp["Body"].read().decode("utf-8"))

                if poll_result["status"] == "completed":
                    result = poll_result["result"]
                    break
                elif poll_result["status"] == "failed":
                    raise RuntimeError(poll_result.get("error", "Inference failed"))
                elif poll_result["status"] == "not_found":
                    raise RuntimeError("Job lost — endpoint may have restarted")
            else:
                raise TimeoutError("Inference timed out after 5 minutes")

            elapsed = time.time() - t0

            timing = (
                f"<div style='font-size:11px;color:#adb5bd;margin-top:8px;'>"
                f"Response time: {elapsed:.1f}s &nbsp;|&nbsp; "
                f"Mode: {mode_label}</div>"
            )
            response_area.value = _render_merlin_response(result) + timing

        except botocore.exceptions.ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "ModelError":
                response_area.value = _error_card(
                    "Model error &mdash; Merlin may still be loading."
                )
            else:
                response_area.value = _error_card(f"AWS error ({code}): {e}")
        except TimeoutError as e:
            response_area.value = _error_card(str(e))
        except Exception as e:
            response_area.value = _error_card(f"Unexpected error: {e}")
        finally:
            send_button.disabled = False
            spinner.value = ""

    def _on_send(_btn):
        model_id = _MODELS.get(model_dropdown.value, "")
        if model_id == "merlin":
            _send_merlin()
        else:
            _send_medgemma()

    send_button.on_click(_on_send)

    def _on_new_conversation(_btn):
        if state.session_id and state.sm_client is not None:
            try:
                payload = {
                    "model": "medgemma",
                    "clear_session": True,
                    "session_id": state.session_id,
                }
                state.sm_client.invoke_endpoint(
                    EndpointName=ENDPOINT_NAME,
                    ContentType=CONTENT_TYPE,
                    Body=json.dumps(payload),
                )
            except Exception:
                pass  # best-effort server cleanup
        state.session_id = ""
        state.chat_history = []
        state.current_turn = 0
        response_area.value = _PLACEHOLDER

    new_conversation_btn.on_click(_on_new_conversation)

    def _on_context_change(change):
        state.session_id = ""
        state.chat_history = []
        state.current_turn = 0
        response_area.value = _PLACEHOLDER

    state.observe(_on_context_change, names=["current_png_bytes"])
    state.observe(_on_context_change, names=["report_text"])

    return widgets.VBox(
        [
            widgets.HTML(
                "<div style='font-size:13px;font-weight:700;color:#495057;"
                "padding:0 0 8px;'>AI Inference Panel</div>"
            ),
            model_dropdown,
            mode_dropdown,
            volume_status,
            payload_toggle_bar,
            new_conversation_btn,
            response_area,
            prompt_area,
            query_texts_area,
            send_button,
            spinner,
            payload_display,
        ],
        layout=widgets.Layout(
            flex="1", padding="0 0 0 16px",
            border_left="1px solid #e9ecef",
        ),
    )
