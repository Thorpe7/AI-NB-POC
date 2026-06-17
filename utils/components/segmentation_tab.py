"""Segmentation panel for the KServe TotalSegmentator endpoint."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import ipywidgets as widgets
import pydicom
import requests

from utils.http_client import predict, translate_path, translate_pod_path_to_local

# InferenceService names. Payload shape varies: TotalSegmentator uses
# fast/roi_subset, NSCLC uses output_format/threshold/mask_tag, BrainSeg
# uses four modality dirs (t1/t2/t1ce/flair).
NSCLC_MODEL = "duneai-nsclc"
BRAINSEG_MODEL = "brainseg"

# Dropdown label -> InferenceService name. The name is substituted into
# INFERENCE_URL_TEMPLATE by http_client.predict().
TASKS: dict[str, str] = {
    "nsclc_segmentation": NSCLC_MODEL,
    "brainseg_segmentation": BRAINSEG_MODEL,
    "TotalSegmentator": "totalseg",
}

_BRAINSEG_MODALITIES = ("t1", "t1ce", "t2", "flair")
_BRAINSEG_MODALITY_LABELS = {
    "t1": "T1",
    "t1ce": "T1ce (post)",
    "t2": "T2",
    "flair": "FLAIR",
}

_PLACEHOLDER = (
    "<div style='color:var(--text-muted);padding:16px 4px;text-align:center;"
    "font-size:12px;'>Select a DICOM series and click "
    "<b>Analyze Current Slice</b> to generate a result.</div>"
)


def _card_header(model_name: str, started_at: str, suffix: str = "") -> str:
    tail = f" &nbsp;&middot;&nbsp; {suffix}" if suffix else ""
    return (
        "<div style='font-weight:700;color:var(--text);margin-bottom:6px;"
        "font-family:inherit;display:flex;justify-content:space-between;"
        "align-items:center;gap:8px;'>"
        f"<span>{model_name}</span>"
        f"<span style='font-weight:500;color:var(--text-dim);font-size:10.5px;"
        f"font-family:inherit;'>{started_at}{tail}</span>"
        "</div>"
    )


def _spinner_card(model_name: str, started_at: str) -> str:
    return (
        "<div class='nbpoc-card normal' "
        "style='font-size:11px;font-family:monospace;line-height:1.55;'>"
        + _card_header(model_name, started_at, "running") +
        '<div style="display:flex;align-items:center;gap:8px;padding:4px 0;">'
        '<span class="nbpoc-spinner"></span>'
        '<span style="font-size:12px;color:var(--text-muted);'
        'font-family:inherit;">Running inference&hellip;</span></div>'
        "</div>"
    )


def _error_card(msg: str, model_name: str = "", started_at: str = "") -> str:
    if not model_name:
        return (
            f"<div class='nbpoc-card severe' style='font-size:12px;color:var(--severity-severe-fg);'>"
            f"{msg}</div>"
        )
    return (
        "<div class='nbpoc-card severe' "
        "style='font-size:12px;color:var(--severity-severe-fg);'>"
        + _card_header(model_name, started_at, "failed") +
        f"<div style='font-family:monospace;'>{msg}</div>"
        "</div>"
    )


def _response_card(
    result: dict, local_seg_path: str | None, model_name: str, started_at: str
) -> str:
    # Show whichever fields the predictor echoed back; TotalSegmentator and
    # NSCLC return different keys.
    rows = []
    if local_seg_path:
        rows.append(
            f"<div style='word-break:break-all;'><b>seg_path</b> &nbsp; "
            f"{local_seg_path}</div>"
        )
    for key in ("task", "fast", "output_format", "threshold"):
        val = result.get(key)
        if val is not None:
            rows.append(f"<div><b>{key}</b> &nbsp; {val}</div>")
    roi = result.get("roi_subset")
    if roi:
        rows.append(f"<div><b>roi_subset</b> &nbsp; {', '.join(roi)}</div>")
    return (
        "<div class='nbpoc-card normal' "
        "style='font-size:11px;font-family:monospace;line-height:1.55;'>"
        + _card_header(model_name, started_at, "done") +
        "".join(rows) +
        "</div>"
    )


def _guess_brainseg_modality(series_desc: str) -> str | None:
    """Best-effort SeriesDescription -> {t1, t1ce, t2, flair}. None if unsure."""
    s = (series_desc or "").lower()
    if "flair" in s:
        return "flair"
    if "t2" in s:
        return "t2"
    if "t1" in s:
        post_cues = ("post", "+c", "stealth-post", "contrast", " ce", "_ce")
        gd_cue = "gd"  # word-bounded check below to avoid matching e.g. "edge"
        if any(c in s for c in post_cues) or any(
            tok == gd_cue for tok in s.replace("+", " ").replace("_", " ").split()
        ):
            return "t1ce"
        return "t1"
    return None


def _discover_brainseg_scans(series_dir_path: str) -> dict[str, dict]:
    """Find sibling scans under the XNAT SCANS/ root of the selected series.

    Returns a dict keyed by scan ID (the dir name under SCANS/), each value
    {path, series_desc, guess}. ``path`` is the DICOM dir (prefers SCANS/<id>/
    DICOM/ if present, else SCANS/<id>/) — i.e. what we'll send to the
    transformer after translate_path().
    """
    if not series_dir_path:
        return {}
    p = Path(series_dir_path)
    scans_root: Path | None = None
    for ancestor in [p, *p.parents]:
        if ancestor.name == "SCANS":
            scans_root = ancestor
            break
    if scans_root is None or not scans_root.is_dir():
        return {}

    result: dict[str, dict] = {}
    for child in sorted(scans_root.iterdir(), key=lambda c: c.name):
        if not child.is_dir():
            continue
        dicom_dir = child / "DICOM" if (child / "DICOM").is_dir() else child
        series_desc = ""
        try:
            for f in dicom_dir.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    try:
                        ds = pydicom.dcmread(
                            str(f), stop_before_pixels=True, force=True
                        )
                        series_desc = str(getattr(ds, "SeriesDescription", "") or "")
                    except Exception:
                        series_desc = ""
                    break
        except OSError:
            continue
        result[child.name] = {
            "path": str(dicom_dir),
            "series_desc": series_desc,
            "guess": _guess_brainseg_modality(series_desc),
        }
    return result


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

    # NSCLC-only control: tumor probability cutoff sent as `threshold`.
    threshold_input = widgets.BoundedFloatText(
        value=0.99,
        min=0.0,
        max=1.0,
        step=0.01,
        description="Threshold:",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="100%"),
    )

    mask_tag_input = widgets.Text(
        value="",
        placeholder="Optional: high_thresh_v1",
        description="Mask tag:",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="100%"),
    )

    # BrainSeg requires four co-registered modalities — one dropdown per
    # modality, populated from sibling scan dirs under .../SCANS/ of whatever
    # series is currently loaded in the file browser.
    brainseg_dropdowns: dict[str, widgets.Dropdown] = {}
    for key in _BRAINSEG_MODALITIES:
        brainseg_dropdowns[key] = widgets.Dropdown(
            options=[("(load a series first)", None)],
            value=None,
            description=f"{_BRAINSEG_MODALITY_LABELS[key]}:",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "90px"},
        )
    brainseg_status = widgets.HTML(value="")
    brainseg_mask_tag_input = widgets.Text(
        value="",
        placeholder="Optional: post_contrast_v2",
        description="Mask tag:",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="100%"),
    )
    brainseg_block = widgets.VBox(
        [
            brainseg_status,
            *(brainseg_dropdowns[k] for k in _BRAINSEG_MODALITIES),
            brainseg_mask_tag_input,
        ]
    )
    _brainseg_scans_cache: dict[str, dict] = {}

    run_button = widgets.Button(
        description="Analyze Current Slice",
        icon="bolt",
        button_style="primary",
        layout=widgets.Layout(width="100%", height="38px"),
    )
    run_button.add_class("nbpoc-analyze-wrap")

    placeholder_card = widgets.HTML(value=_PLACEHOLDER)
    response_stack = widgets.VBox(children=[placeholder_card])
    # job_id -> card widget for in-flight jobs. Threads update card.value in
    # the finally block to swap the spinner for the result HTML in place.
    inflight_cards: dict[int, widgets.HTML] = {}
    next_job_id = [0]

    def _refresh_series_label(*_):
        n = len(state.series_datasets)
        if state.series_dir_name and n > 0:
            series_label.value = (
                f"<div style='font-size:11.5px;color:var(--text-muted);padding:4px 0;'>"
                f"<b style='color:var(--text);'>Series:</b> {state.series_dir_name} "
                f"({n} slice{'s' if n != 1 else ''})</div>"
            )
            run_button.description = "Analyze Series"
            run_button.tooltip = f"Run inference on the loaded series ({n} slices)"
            run_button.disabled = False
        elif state.current_ds is not None:
            # A single slice is loaded but no series — the current predictors
            # need a directory, so keep the button disabled and tell the user
            # what to do next.
            series_label.value = (
                "<div style='font-size:11.5px;color:var(--warn-fg);padding:4px 0;'>"
                "Click <b>Load Entire Series</b> to enable inference for this scan."
                "</div>"
            )
            run_button.description = "Analyze Current Slice"
            run_button.tooltip = "Load the full series first — inference runs on the directory"
            run_button.disabled = True
        else:
            series_label.value = (
                "<div style='font-size:11.5px;color:var(--warn-fg);padding:4px 0;'>"
                "Load a DICOM series to enable inference."
                "</div>"
            )
            run_button.description = "Analyze"
            run_button.tooltip = "Load a DICOM series first"
            run_button.disabled = True

    state.observe(
        _refresh_series_label,
        names=["series_dir_name", "series_datasets", "current_ds"],
    )
    _refresh_series_label()

    def _apply_task_visibility(*_):
        """Show only the controls relevant to the selected task."""
        model = TASKS[task_dropdown.value]
        is_nsclc = model == NSCLC_MODEL
        is_brainseg = model == BRAINSEG_MODEL
        is_totalseg = not (is_nsclc or is_brainseg)
        fast_checkbox_bar.layout.display = "" if is_totalseg else "none"
        roi_input.layout.display = "" if is_totalseg else "none"
        threshold_input.layout.display = "" if is_nsclc else "none"
        mask_tag_input.layout.display = "" if is_nsclc else "none"
        brainseg_block.layout.display = "" if is_brainseg else "none"

    task_dropdown.observe(_apply_task_visibility, names="value")
    _apply_task_visibility()

    def _refresh_brainseg_scans(*_):
        scans = _discover_brainseg_scans(state.series_dir_path)
        _brainseg_scans_cache.clear()
        _brainseg_scans_cache.update(scans)

        if scans:
            opts = [("(choose scan)", None)] + [
                (
                    f"{sid} — {info['series_desc'] or '(no description)'}",
                    sid,
                )
                for sid, info in scans.items()
            ]
        else:
            opts = [("(no SCANS/ sibling dir found)", None)]

        for key, dd in brainseg_dropdowns.items():
            dd.options = opts
            default = next(
                (sid for sid, info in scans.items() if info["guess"] == key), None
            )
            dd.value = default

        if not scans:
            brainseg_status.value = (
                "<div style='font-size:11.5px;color:var(--warn-fg);padding:4px 0;'>"
                "Load a series under <code>.../SCANS/&lt;id&gt;/DICOM/</code> "
                "to populate modality choices.</div>"
            )
        else:
            unmatched = [
                _BRAINSEG_MODALITY_LABELS[k]
                for k, dd in brainseg_dropdowns.items()
                if dd.value is None
            ]
            tail = (
                f" Could not auto-detect: {', '.join(unmatched)}."
                if unmatched
                else " Auto-detected — verify before running."
            )
            brainseg_status.value = (
                f"<div style='font-size:11.5px;color:var(--text-muted);padding:4px 0;'>"
                f"Found {len(scans)} sibling scan{'s' if len(scans) != 1 else ''}.{tail}"
                "</div>"
            )

    state.observe(_refresh_brainseg_scans, names="series_dir_path")
    _refresh_brainseg_scans()

    def _on_series_change(_change):
        # Wipe completed cards so they don't masquerade as the new series'
        # result. In-flight cards stay — their threads will land the result
        # in place — and series identification is the user's job once they
        # have multiple completed cards in the stack.
        inflight_widgets = set(id(c) for c in inflight_cards.values())
        keep = [c for c in response_stack.children if id(c) in inflight_widgets]
        response_stack.children = tuple(keep) if keep else (placeholder_card,)

    state.observe(_on_series_change, names="series_dir_path")

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

        if TASKS[task_dropdown.value] == NSCLC_MODEL:
            payload: dict = {
                "dicom_dir": dicom_dir,
                "output_format": "dicom",
                "threshold": float(threshold_input.value),
            }
            tag = mask_tag_input.value.strip()
            if tag:
                payload["mask_tag"] = tag
            return payload, None

        if TASKS[task_dropdown.value] == BRAINSEG_MODEL:
            if not _brainseg_scans_cache:
                return None, (
                    "No sibling scans found. Load a series under "
                    ".../SCANS/<id>/DICOM/ first."
                )
            selected = {k: brainseg_dropdowns[k].value for k in _BRAINSEG_MODALITIES}
            missing = [
                _BRAINSEG_MODALITY_LABELS[k] for k, v in selected.items() if not v
            ]
            if missing:
                return None, f"Pick a scan for: {', '.join(missing)}"
            if len(set(selected.values())) != len(selected):
                return None, "Each modality must map to a different scan."
            payload = {"output_format": "dicom"}
            for key, scan_id in selected.items():
                local_path = _brainseg_scans_cache[scan_id]["path"]
                try:
                    payload[f"{key}_dir"] = translate_path(local_path)
                except ValueError as e:
                    return None, str(e)
            return payload, None

        payload = {"dicom_dir": dicom_dir, "fast": fast_checkbox.value}
        roi = _parse_roi(roi_input.value)
        if roi:
            payload["roi_subset"] = roi
        return payload, None

    def _remove_inflight(job_id: int, model_name: str) -> None:
        inflight_cards.pop(job_id, None)
        new_inflight = list(state.inflight_models)
        if model_name in new_inflight:
            new_inflight.remove(model_name)
        state.inflight_models = new_inflight

    def _run_in_thread(payload, model_name, job_id, card, started_at):
        t0 = time.time()
        result_html = None
        try:
            result = predict(payload, model_name)
            elapsed = time.time() - t0

            local_seg = None
            seg_path = result.get("seg_path")
            if seg_path:
                try:
                    local_seg = translate_pod_path_to_local(seg_path)
                except ValueError as e:
                    result_html = _error_card(
                        f"Segmentation succeeded but seg_path could not be "
                        f"translated: {e}",
                        model_name=model_name,
                        started_at=started_at,
                    )

            if result_html is None:
                predictor_elapsed = float(result.get("elapsed_s", 0.0))
                footer = (
                    f"<div style='font-size:10.5px;color:var(--text-dim);margin-top:8px;'>"
                    f"Response: {elapsed:.1f}s &nbsp;&middot;&nbsp; "
                    f"Predictor: {predictor_elapsed:.1f}s &nbsp;&middot;&nbsp; "
                    f"Refresh the results section to load the new mask.</div>"
                )
                result_html = (
                    _response_card(result, local_seg, model_name, started_at)
                    + footer
                )
        except requests.Timeout:
            result_html = _error_card(
                "Request timed out. KServe may be cold-starting "
                "(can take several minutes on first request).",
                model_name=model_name,
                started_at=started_at,
            )
        except requests.HTTPError as e:
            body = ""
            try:
                body = e.response.text[:500]
            except Exception:
                pass
            result_html = _error_card(
                f"HTTP {e.response.status_code} from predictor: {body}",
                model_name=model_name,
                started_at=started_at,
            )
        except requests.RequestException as e:
            result_html = _error_card(
                f"Request failed: {e}", model_name=model_name, started_at=started_at
            )
        except Exception as e:
            result_html = _error_card(
                f"Unexpected error: {e}", model_name=model_name, started_at=started_at
            )
        finally:
            card.value = result_html or _error_card(
                "Unknown error.", model_name=model_name, started_at=started_at
            )
            _remove_inflight(job_id, model_name)

    def _on_run(_btn):
        payload, err = _build_payload()
        if err:
            # Surface validation errors above the existing cards without
            # consuming a job slot — render directly into a transient card.
            transient = widgets.HTML(value=_error_card(err))
            existing = [c for c in response_stack.children if c is not placeholder_card]
            response_stack.children = (transient, *existing)
            return

        model_name = TASKS[task_dropdown.value]
        next_job_id[0] += 1
        job_id = next_job_id[0]
        started_at = datetime.now().strftime("%H:%M:%S")

        card = widgets.HTML(value=_spinner_card(model_name, started_at))
        inflight_cards[job_id] = card
        # Newest card on top; drop the placeholder once we have real content.
        existing = [c for c in response_stack.children if c is not placeholder_card]
        response_stack.children = (card, *existing)

        state.inflight_models = state.inflight_models + [model_name]

        threading.Thread(
            target=_run_in_thread,
            args=(payload, model_name, job_id, card, started_at),
            daemon=True,
        ).start()

    run_button.on_click(_on_run)

    return widgets.VBox(
        [
            task_dropdown,
            series_label,
            fast_checkbox_bar,
            roi_input,
            threshold_input,
            mask_tag_input,
            brainseg_block,
            run_button,
            response_stack,
        ],
        layout=widgets.Layout(width="100%", padding="0"),
    )
