"""Inference endpoint and path-translation configuration for the Medical AI POC."""

import os

# Inference endpoint template. {model} is substituted with the selected model
# name. Default points at the in-cluster transformer Service so traffic from
# a kernel inside the xnat namespace stays in-cluster — the public ALB URL
# would hairpin through the AWS NAT Gateway, whose 350s idle-TCP timeout
# silently drops the connection while the transformer is mid-inference
# (>350s wall-clock on brainseg). Override via env var for out-of-cluster
# kernels, e.g. http://{model}-xnat.tap.dev.embarklabs.ai/v1/models/{model}:predict
INFERENCE_URL_TEMPLATE = os.environ.get(
    "INFERENCE_URL_TEMPLATE",
    "http://{model}-transformer.xnat/v1/models/{model}:predict",
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
