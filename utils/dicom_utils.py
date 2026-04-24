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


_DEFAULT_SEG_PALETTE = [
    (239, 83, 80),   # red
    (66, 165, 245),  # blue
    (102, 187, 106), # green
    (255, 167, 38),  # orange
    (171, 71, 188),  # purple
    (38, 198, 218),  # cyan
    (255, 238, 88),  # yellow
    (141, 110, 99),  # brown
    (236, 64, 122),  # pink
    (156, 204, 101), # lime
]


def _cielab_dicom_to_rgb(lab):
    """DICOM RecommendedDisplayCIELabValue (uint16 x3) -> sRGB 0-255 tuple.

    DICOM encodes L in [0, 65535] -> [0, 100]; a,b in [0, 65535] -> [-128, 127].
    """
    L = float(lab[0]) / 65535.0 * 100.0
    a = float(lab[1]) / 65535.0 * 255.0 - 128.0
    b = float(lab[2]) / 65535.0 * 255.0 - 128.0

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def _f_inv(t):
        t3 = t ** 3
        return t3 if t3 > 0.008856 else (t - 16.0 / 116.0) / 7.787

    # D65 reference white
    X = 0.95047 * _f_inv(fx)
    Y = 1.00000 * _f_inv(fy)
    Z = 1.08883 * _f_inv(fz)

    r =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    bl = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def _gamma(u):
        u = max(0.0, u)
        return 1.055 * (u ** (1 / 2.4)) - 0.055 if u > 0.0031308 else 12.92 * u

    return tuple(
        max(0, min(255, int(round(_gamma(c) * 255))))
        for c in (r, g, bl)
    )


def _segment_color(seg, num):
    lab = getattr(seg, "RecommendedDisplayCIELabValue", None)
    if lab is not None and len(lab) == 3:
        try:
            return _cielab_dicom_to_rgb(lab)
        except Exception:
            pass
    return _DEFAULT_SEG_PALETTE[(num - 1) % len(_DEFAULT_SEG_PALETTE)]


def _source_sop_uid(fg):
    try:
        drv = fg.DerivationImageSequence[0]
        src = drv.SourceImageSequence[0]
        return str(src.ReferencedSOPInstanceUID)
    except (AttributeError, IndexError):
        return None


def _segment_number(fg):
    try:
        seg_id = fg.SegmentIdentificationSequence[0]
        return int(seg_id.ReferencedSegmentNumber)
    except (AttributeError, IndexError, ValueError):
        return None


def load_dicom_seg(path):
    """Load a DICOM SEG file into per-slice masks keyed by source SOP UID.

    Returns dict with:
        by_source_sop: {sop_instance_uid: {segment_number: bool ndarray (H, W)}}
        segments:      {segment_number: {"label": str, "color": (r, g, b)}}
        referenced_series_uid: SeriesInstanceUID of the source series (or None)
    """
    ds = pydicom.dcmread(str(path))
    if getattr(ds, "Modality", "") != "SEG":
        raise ValueError(
            f"Not a DICOM SEG file (Modality={getattr(ds, 'Modality', None)!r})"
        )

    segments = {}
    for seg in getattr(ds, "SegmentSequence", []):
        num = int(seg.SegmentNumber)
        segments[num] = {
            "label": getattr(seg, "SegmentLabel", f"Segment {num}"),
            "color": _segment_color(seg, num),
        }

    ref_series_uid = None
    rss = getattr(ds, "ReferencedSeriesSequence", None)
    if rss and len(rss) > 0:
        ref_series_uid = getattr(rss[0], "SeriesInstanceUID", None)

    frames = ds.pixel_array
    if frames.ndim == 2:
        frames = frames[None, ...]

    by_source_sop = {}
    for i, fg in enumerate(ds.PerFrameFunctionalGroupsSequence):
        sop_uid = _source_sop_uid(fg)
        seg_num = _segment_number(fg)
        if sop_uid is None or seg_num is None:
            continue
        # TotalSegmentator writes SEG via nibabel (RAS), which flips the
        # row axis relative to the source DICOM's pixel data. Undo that so
        # the mask lines up with the displayed image.
        mask = np.flipud(frames[i]).astype(bool)
        if not mask.any():
            continue
        by_source_sop.setdefault(sop_uid, {})[seg_num] = mask

    return {
        "by_source_sop": by_source_sop,
        "segments": segments,
        "referenced_series_uid": ref_series_uid,
    }


def composite_overlay(base_pil, segment_masks, segments, alpha=0.4):
    """Blend colored segmentation masks onto a base image.

    base_pil:       PIL.Image (any mode; converted to RGB internally)
    segment_masks:  {segment_number: bool ndarray (H, W)} for this slice; may be empty
    segments:       {segment_number: {"label": str, "color": (r, g, b)}}
    alpha:          overlay opacity, 0..1

    Masks whose shape differs from the base are resampled with
    nearest-neighbor so SEGs produced at a resampled resolution
    (e.g. TotalSegmentator --fast) still render correctly.

    Returns an RGB PIL.Image.
    """
    base_rgb = np.array(base_pil.convert("RGB"), dtype=np.float32)
    h, w = base_rgb.shape[:2]

    if not segment_masks:
        return Image.fromarray(base_rgb.astype(np.uint8), mode="RGB")

    for seg_num, mask in segment_masks.items():
        if mask.shape != (h, w):
            mask_pil = Image.fromarray(
                (mask.astype(np.uint8) * 255), mode="L"
            ).resize((w, h), Image.NEAREST)
            mask = np.array(mask_pil) > 127
        color = np.array(segments[seg_num]["color"], dtype=np.float32)
        base_rgb[mask] = (1.0 - alpha) * base_rgb[mask] + alpha * color

    return Image.fromarray(np.clip(base_rgb, 0, 255).astype(np.uint8), mode="RGB")


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
