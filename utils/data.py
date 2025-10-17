import re
import csv
import argparse
from typing import List, Tuple
import pandas as pd
import numpy as np
import requests
REMOVE_ALWAYS = {"N", "Val", "B", "Cat", "Cod", "Chi", "A", "M", "K", "T"}
REMOVE_THIRD_ONLY = {"RMS", "Resid"}

def read_lines(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.readlines()

def find_header_line(lines: List[str]) -> Tuple[int, str]:
    # Find END_OF_HEADER then the '! Design' line
    try:
        end_idx = next(i for i, ln in enumerate(lines) if 'END_OF_HEADER' in ln)
    except StopIteration:
        raise ValueError("END_OF_HEADER not found")
    for i in range(end_idx + 1, len(lines)):
        if lines[i].lstrip().startswith('! Design'):
            raw = lines[i].rstrip("\n")
            clean = raw.lstrip()
            if clean.startswith('!'):
                clean = clean[1:].lstrip()
            return i, clean
    raise ValueError("Header template starting with '! Design' not found")

def compute_bounds(template: str) -> List[Tuple[int, int, str]]:
    """Return (start, end, label) per subcolumn using space->non-space transitions."""
    starts = [i for i, ch in enumerate(template) if ch != ' ' and (i == 0 or template[i-1] == ' ')]
    if not starts:
        raise ValueError("Could not determine column starts from template")
    bounds = [(starts[k], starts[k+1]) for k in range(len(starts)-1)] + [(starts[-1], len(template))]
    return [(s, e, template[s:e].strip()) for s, e in bounds]

def prune_cols(cols: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Apply pruning rules: always-remove set; remove only the 3rd 'RMS' and 'Resid'."""
    counts = {k: 0 for k in REMOVE_THIRD_ONLY}
    kept = []
    for s, e, lbl in cols:
        if lbl in REMOVE_ALWAYS:
            continue
        if lbl in REMOVE_THIRD_ONLY:
            counts[lbl] += 1
            if counts[lbl] == 3:
                # Drop ONLY the 3rd appearance
                continue
        kept.append((s, e, lbl))
    return kept

def header_with_suffixes(kept: List[Tuple[int, int, str]]) -> List[str]:
    name_counts = {}
    headers = []
    for _, _, lbl in kept:
        base = lbl if lbl else "col"
        c = name_counts.get(base, 0) + 1
        name_counts[base] = c
        headers.append(f"{base}_{c}" if c > 1 else base)
    return headers

def slice_row_aligned(line: str, kept_bounds: List[Tuple[int, int, str]]) -> List[str]:
    """Slice a single data row using per-row alignment and sign handoff rule."""
    # Row alignment: compute offset to the first non-space char
    first_nonspace = next((i for i, ch in enumerate(line) if ch != ' '), 0)
    raw_slices = []
    for s, e, _ in kept_bounds:
        s_off = s + first_nonspace
        e_off = e + first_nonspace
        if len(line) < e_off:
            frag = (line[s_off:] if s_off < len(line) else '').ljust(e_off - s_off)
        else:
            frag = line[s_off:e_off]
        raw_slices.append(frag)

    # Sign handoff at every boundary
    for i in range(len(raw_slices) - 1):
        left = raw_slices[i]
        right = raw_slices[i+1]
        if left.rstrip().endswith('-'):
            rtrim = right.lstrip()
            if (len(rtrim) > 0 and rtrim[0] in '0123456789.') and not rtrim.startswith('-'):
                # Remove trailing '-' from left
                raw_slices[i] = left.rstrip()[:-1].rstrip()
                # Prepend '-' to right at first non-space
                lead_spaces = len(right) - len(right.lstrip(' '))
                raw_slices[i+1] = (' ' * lead_spaces) + '-' + right.lstrip(' ')

    # Final trim
    return [cell.strip() for cell in raw_slices]

def rwo_to_csv(input_path: str, output_path: str) -> None:
    lines = read_lines(input_path)
    hdr_idx, template = find_header_line(lines)
    cols = compute_bounds(template)
    kept = prune_cols(cols)
    headers = header_with_suffixes(kept)

    # Gather data rows (skip blank and comment lines)
    data_lines = [ln.rstrip('\n') for ln in lines[hdr_idx+1:] if ln.strip() and not ln.lstrip().startswith('!')]

    # Slice rows
    out_rows = [slice_row_aligned(ln, kept) for ln in data_lines]

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(headers)
        writer.writerows(out_rows)

def csv_to_pandas(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path, skipinitialspace=True)

def get_tle_celestrak(norad_id: int) -> tuple[str, str, str]:
    """
    Return (name, line1, line2) for the satellite's latest TLE.
    """
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    lines = [ln.strip() for ln in r.text.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        raise RuntimeError("Response did not contain a 3-line TLE.")
    # TLE is 3 lines: name, L1, L2 (may be multiple blocks; we take the first)
    name, line1, line2 = lines[0], lines[1], lines[2]
    return name, line1, line2

