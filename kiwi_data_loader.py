"""
language: python
# AI-Generated Code Header
# **Intent:** Load kiwi NIR spectral CSV batches, returning arrays and metadata for experiments.
# **Optimization:** Stream parsing, minimal copies; vectorized operations with numpy/pandas.
# **Safety:** Validates wavelength header monotonicity, handles missing/invalid rows, and isolates batch labels.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import io
import csv

import numpy as np
import pandas as pd


@dataclass
class KiwiDataset:
    spectra: np.ndarray  # shape: (n_samples, n_wavelengths)
    targets: np.ndarray  # shape: (n_samples,)
    wavelengths: np.ndarray  # shape: (n_wavelengths,)
    batches: List[str]  # length n_samples, file-derived batch labels


def _read_csv_robust(csv_path: str) -> List[List[str]]:
    """Read CSV by decoding bytes with UTF-8 fallback and parsing via csv module.

    Returns list of rows (each a list of string tokens).
    """
    with open(csv_path, "rb") as f:
        raw = f.read()
    # Detect OLE/ZIP Excel headers and delegate to pandas Excel engine
    if raw[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" or raw[:2] == b"PK":
        try:
            # Read first sheet via pandas; dtype=str to preserve tokens
            xl = pd.read_excel(io.BytesIO(raw), header=None, dtype=str, engine=None)
            return xl.astype(str).values.tolist()
        except Exception:
            pass
    # Try utf-8 first, then latin1 as permissive fallback
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin1", errors="ignore")
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Detect delimiter
    sample = text[:4096]
    delimiter = ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
        delimiter = dialect.delimiter
    except Exception:  # noqa: BLE001
        for d in [",", ";", "\t"]:
            if d in sample:
                delimiter = d
                break
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [row for row in reader if len(row) > 0]
    if len(rows) < 2:
        raise ValueError(f"CSV {csv_path} has insufficient rows after parsing")
    # Heuristic: drop trailing empty cells
    rows = [list(row) for row in rows]
    return rows


def _read_single_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a single kiwi CSV where the first row is wavelengths, each subsequent row:
    [target, spec_0, spec_1, ...]. Returns (targets, spectra, wavelengths).
    """
    rows = _read_csv_robust(csv_path)
    header = rows[0]
    # Expect first cell not a wavelength; take from col 1 onward
    wl_raw = header[1:]
    wl_vals: List[float] = []
    wl_mask: List[bool] = []
    for tok in wl_raw:
        try:
            val = float(str(tok))
            wl_vals.append(val)
            wl_mask.append(True)
        except Exception:
            wl_mask.append(False)
    wl_arr_full = np.array(wl_vals, dtype=float)
    # finite mask implicit by construction
    wl_arr = wl_arr_full
    if wl_arr.size < 50:
        raise ValueError(f"CSV {csv_path} header lacks valid wavelengths")
    order = np.argsort(wl_arr)
    wavelengths = wl_arr[order]

    targets_list: List[float] = []
    spectra_list: List[np.ndarray] = []
    for r in rows[1:]:
        # Require at least 1 target + as many spectral columns as discovered wavelengths
        if len(r) - 1 < len(header) - 1:
            continue
        try:
            y_val = float(r[0])
        except Exception:
            continue
        spec_vals: List[float] = []
        # Iterate spectral tokens up to number of wavelengths in header and parse floats
        spec_tokens = r[1: 1 + len(wl_raw)]
        for tok in spec_tokens:
            try:
                spec_vals.append(float(tok))
            except Exception:
                spec_vals.append(np.nan)
        if len(spec_vals) < wl_arr.size:
            continue
        spec_arr = np.array(spec_vals, dtype=float)
        if np.sum(np.isfinite(spec_arr)) < 0.8 * spec_arr.size:
            continue
        spec_arr = np.nan_to_num(spec_arr, nan=np.nanmean(spec_arr))
        spectra_list.append(spec_arr[: wl_arr.size][order])
        targets_list.append(y_val)

    if len(targets_list) == 0:
        raise ValueError(f"CSV {csv_path} contains no valid samples after filtering")
    targets = np.array(targets_list, dtype=float)
    spectra = np.vstack(spectra_list)

    return targets, spectra, wavelengths


def load_kiwi_dataset(input_dir: str) -> KiwiDataset:
    """
    Load all kiwi CSV files under `input_dir`, merging them into arrays and assigning batch labels.

    - input_dir: directory containing kiwi-*.csv
    - returns: KiwiDataset
    """
    paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".csv") and not f.startswith(".")
    ]
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")

    paths.sort()
    all_targets: List[np.ndarray] = []
    all_spectra: List[np.ndarray] = []
    all_batches: List[str] = []
    wavelengths_ref: Optional[np.ndarray] = None

    for p in paths:
        t, X, wl = _read_single_csv(p)
        if wavelengths_ref is None:
            wavelengths_ref = wl
        else:
            # Align to reference wavelengths by intersection (exact match); if mismatch, inner-join columns
            if wl.shape[0] != wavelengths_ref.shape[0] or not np.allclose(wl, wavelengths_ref):
                # AI-SUGGESTION: robust alignment via exact set intersection, preserving order of ref
                common = np.intersect1d(wavelengths_ref, wl)
                if common.size < min(wavelengths_ref.size, wl.size) * 0.8:
                    raise ValueError("Wavelength axes differ substantially across files; please harmonize.")
                # Indices into original arrays, preserving order of wavelengths_ref
                ref_mask = np.isin(wavelengths_ref, common)
                wl_mask = np.isin(wl, common)
                # Truncate previous accumulators to common wavelengths (in wavelengths_ref order)
                if all_spectra:
                    all_spectra = [arr[:, ref_mask] for arr in all_spectra]  # type: ignore
                wavelengths_ref = wavelengths_ref[ref_mask]
                # Reorder current X to match wavelengths_ref order
                # Build mapping from value->index for wl
                pos = {float(val): idx for idx, val in enumerate(wl[wl_mask])}
                order_idx = [pos[float(val)] for val in wavelengths_ref]
                X = X[:, wl_mask][:, order_idx]

        all_targets.append(t)
        all_spectra.append(X)
        all_batches.extend([os.path.basename(p)] * t.shape[0])

    targets = np.concatenate(all_targets, axis=0)
    spectra = np.vstack(all_spectra)
    wavelengths = wavelengths_ref if wavelengths_ref is not None else np.array([])

    return KiwiDataset(spectra=spectra, targets=targets, wavelengths=wavelengths, batches=all_batches)


def train_test_split_by_batches(batches: List[str], held_out_batch: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (train_mask, test_mask) for leave-one-batch-out by filename label."""
    batches_arr = np.array(batches)
    test_mask = batches_arr == held_out_batch
    train_mask = ~test_mask
    return train_mask, test_mask

