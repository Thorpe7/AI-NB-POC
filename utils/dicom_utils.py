"""DICOM reading, windowing, and conversion utilities."""

from __future__ import annotations

import io

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from PIL import Image


_KNOWN_NON_DICOM = {
    ".txt", ".csv", ".json", ".xml", ".html", ".log", ".md",
    ".yaml", ".yml", ".cfg", ".ini", ".conf",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".svg",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".rs", ".go",
    ".zip", ".tar", ".gz", ".bz2", ".rar", ".7z", ".xz",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".mp4", ".avi", ".mov", ".mkv", ".mp3", ".wav", ".flac",
    ".nii", ".nrrd", ".mha", ".mhd", ".stl", ".obj", ".vtk",
    ".sh", ".bat", ".ps1",
}


def is_dicom_candidate(p):
    """Check if a path is likely a DICOM file.

    Uses an exclusion-based approach: any file that doesn't have a known
    non-DICOM extension is treated as a candidate.  This catches numeric
    extensions (.001, .1), Siemens .ima, UID-fragment names, and
    extension-less files common in PACS exports.
    """
    if p.is_dir():
        return False
    return p.suffix.lower() not in _KNOWN_NON_DICOM


def is_nifti_file(p):
    """Check if a path is a NIfTI file (.nii or .nii.gz)."""
    if p.is_dir():
        return False
    name = p.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def read_dicom(path):
    """Read a DICOM file, return Dataset or None on failure."""
    try:
        return pydicom.dcmread(str(path), force=True)
    except Exception:
        return None


def apply_windowing(ds):
    """Apply modality + VOI LUTs via pydicom, return float64 array in [0, 1]."""
    pixels = ds.pixel_array
    pixels = apply_modality_lut(pixels, ds)
    pixels = apply_voi_lut(pixels, ds)
    pixels = pixels.astype(np.float64)
    lo, hi = float(pixels.min()), float(pixels.max())
    if hi - lo < 1e-6:
        return np.zeros_like(pixels, dtype=np.float64)
    return (pixels - lo) / (hi - lo)


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
