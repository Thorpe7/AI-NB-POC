"""Inference endpoint and path-translation configuration for the Medical AI POC."""

import os


def _parse_models(raw: str) -> dict[str, str]:
    """Parse MODELS env var as comma-separated 'Label=name' pairs (or bare names)."""
    out: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            label, name = entry.split("=", 1)
            out[label.strip()] = name.strip()
        else:
            out[entry] = entry
    return out


# Dropdown options: display label -> KServe InferenceService name.
MODELS = _parse_models(
    os.environ.get("MODELS", "MedGemma=medgemma,Llava-Med=llava-med")
)

# Per-model capabilities. Keys are InferenceService names (the values in MODELS).
# max_images: upper bound on images per request (must match --max_images in the
#   InferenceService manifest).
# supports_series: whether the model accepts a DICOM series via dicom_dir; when
#   False the UI must send a single file via image_paths instead.
MODEL_CAPS: dict[str, dict] = {
    "medgemma": {"max_images": 32, "supports_series": True},
    "llava-med": {"max_images": 1, "supports_series": False},
}


def get_model_caps(name: str) -> dict:
    """Return caps for the given InferenceService name, conservatively defaulting."""
    return MODEL_CAPS.get(name, {"max_images": 1, "supports_series": False})

# In-cluster URL template. {model} is substituted with the selected model name.
INFERENCE_URL_TEMPLATE = os.environ.get(
    "INFERENCE_URL_TEMPLATE",
    "http://{model}-predictor.xnat.svc.cluster.local/v1/models/{model}:predict",
)

# KServe scale-to-zero cold starts can take several minutes while the model
# weights mount and vLLM spins up.
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "900"))

# PVC path translation. The JupyterHub kernel and the inference pod mount the
# same archive under different prefixes, and the first segment under AI-POC/
# differs between the two views (notebook = "experiments", XNAT archive =
# "arc001"). SEGMENT_MAP rewrites that first segment; anything not in the map
# passes through unchanged.
LOCAL_DATA_ROOT = os.environ.get("LOCAL_DATA_ROOT", "/data/projects/AI-POC")
POD_DATA_ROOT = os.environ.get("POD_DATA_ROOT", "/data/xnat/archive/AI-POC")
SEGMENT_MAP = {"experiments": "arc001"}
