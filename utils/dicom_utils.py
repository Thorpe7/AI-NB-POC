"""DICOM reading, windowing, and conversion utilities."""

from __future__ import annotations

import io

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image

# Modality-specific default windows: (center, width)
_DEFAULT_WINDOWS = {
    "CT": (40.0, 400.0),      # soft-tissue
    "CR": (2048.0, 4096.0),   # 12-bit X-ray
    "DX": (2048.0, 4096.0),   # digital X-ray
}


def is_dicom_candidate(p):
    """Check if a path is likely a DICOM file."""
    if p.is_dir():
        return False
    return p.suffix.lower() in {".dcm", ".dicom"} or p.suffix == ""


def read_dicom(path):
    """Read a DICOM file, return Dataset or None on failure."""
    try:
        return pydicom.dcmread(str(path), force=True)
    except Exception:
        return None


def _dynamic_window(pixels):
    """Compute a window from pixel statistics (for MR / unknown)."""
    mean, std = float(np.mean(pixels)), float(np.std(pixels))
    return mean, max(std * 4.0, 1.0)


def _has_voi_lut_sequence(ds):
    """Return True if ds has a non-empty VOILUTSequence (non-linear LUT)."""
    seq = getattr(ds, "VOILUTSequence", None)
    return seq is not None and len(seq) > 0


def _percentile_normalize(arr, lo_pct=1.0, hi_pct=99.0):
    """Normalize to [0,1] using percentile clipping (outlier-robust)."""
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi - lo < 1e-6:
        lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def apply_windowing(ds):
    """Apply VOI LUT or manual windowing, return float64 array in [0, 1]."""
    pixels = ds.pixel_array.astype(np.float64)

    # Path A: Non-linear VOI LUT sequence (rare — sigmoid/table LUT)
    if _has_voi_lut_sequence(ds):
        try:
            windowed = apply_voi_lut(pixels, ds).astype(np.float64)
            return _percentile_normalize(windowed)
        except Exception:
            pass

    # Path B: Linear window/level (common case)
    modality = str(getattr(ds, "Modality", ""))
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    if wc is not None and ww is not None:
        if hasattr(wc, "__iter__") and not isinstance(wc, str):
            wc, ww = float(wc[0]), float(ww[0])
        else:
            wc, ww = float(wc), float(ww)
    elif modality in _DEFAULT_WINDOWS:
        wc, ww = _DEFAULT_WINDOWS[modality]
    else:
        wc, ww = _dynamic_window(pixels)

    lower = wc - ww / 2.0
    upper = wc + ww / 2.0
    clipped = np.clip(pixels, lower, upper)
    return (clipped - lower) / max(upper - lower, 1e-6)


def dicom_to_pil(ds):
    """Convert DICOM Dataset to an 8-bit grayscale PIL Image."""
    arr = apply_windowing(ds)
    # Invert MONOCHROME1 (white = low values)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = 1.0 - arr
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L")


def dicom_to_png_bytes(ds):
    """Convert DICOM to PNG bytes for the endpoint."""
    buf = io.BytesIO()
    dicom_to_pil(ds).save(buf, format="PNG")
    return buf.getvalue()


def load_series(directory_path):
    """Load all valid DICOM files in a directory as a sorted series.

    Returns list of (dataset, png_bytes) tuples sorted by InstanceNumber,
    SliceLocation, or filename as fallback.  Returns [] if <1 valid DICOM.
    """
    from pathlib import Path

    directory_path = Path(directory_path)
    candidates = [p for p in directory_path.iterdir() if is_dicom_candidate(p)]

    loaded = []
    for p in candidates:
        ds = read_dicom(p)
        if ds is None:
            continue
        try:
            _ = ds.pixel_array
            png = dicom_to_png_bytes(ds)
        except Exception:
            continue
        loaded.append((ds, png, p.name))

    def _sort_key(item):
        ds, _png, fname = item
        inst = getattr(ds, "InstanceNumber", None)
        if inst is not None:
            try:
                return (0, int(inst), fname)
            except (TypeError, ValueError):
                pass
        sloc = getattr(ds, "SliceLocation", None)
        if sloc is not None:
            try:
                return (1, float(sloc), fname)
            except (TypeError, ValueError):
                pass
        return (2, 0, fname)

    loaded.sort(key=_sort_key)
    return [(ds, png) for ds, png, _fname in loaded]


def extract_metadata(ds):
    """Pull a curated set of DICOM tags for display."""
    tags = [
        ("Patient", "PatientName"), ("Patient ID", "PatientID"),
        ("Study Date", "StudyDate"), ("Study", "StudyDescription"),
        ("Series", "SeriesDescription"), ("Modality", "Modality"),
        ("Body Part", "BodyPartExamined"), ("Institution", "InstitutionName"),
        ("Size", None), ("Bits", "BitsAllocated"),
        ("Photometric", "PhotometricInterpretation"),
        ("Window Center", "WindowCenter"), ("Window Width", "WindowWidth"),
    ]
    rows = []
    for label, attr in tags:
        if label == "Size":
            r = getattr(ds, "Rows", "?")
            c = getattr(ds, "Columns", "?")
            rows.append((label, f"{r} x {c}"))
        else:
            val = getattr(ds, attr, None)
            if val is not None:
                rows.append((label, str(val)))
    return rows
