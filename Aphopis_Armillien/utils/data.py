import re
import csv
import argparse
from typing import List, Tuple
import pandas as pd
import numpy as np

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

def obs_extractN(NEA: pd.DataFrame, N_lower: int, N_upper: int) -> NDArray:
    """
    Extracts the observation data from the NEA DataFrame THROUGH OBSERVATION INSTANCES.
    :param NEA: DataFrame containing NEA observation data.
    :param N: The number of observation instances to extract.

    :return time: Time of observation (YYYY, Month, Day.dd)
    :return RA: Right Ascension of Observation (Hours, Minutes, Seconds)
    :return DEC: Declination of Observation (Degrees, Minutes, Hours)
    :return RA_sigma: Time of observation (YYYY, Month, Day.dd)
    :return DEC_sigma: Time of observation (YYYY, Month, Day.dd)
    """
    rows = NEA.iloc[N_lower:N_upper]
    obs_time = rows[["YYYY", "MM", "DD.dddddddddd"]]
    RA = rows[["HH", "MM_2", "SS.sss"]]
    DEC = rows[["sDD","MM_3","SS.ss"]]

    # Extract RA and DEC uncertainties
    RA_sigma = rows[["Accuracy_2"]]
    DEC_sigma = rows[["Accuracy_3"]]

    # Conver to numpy arrays
    obs_time = obs_time.to_numpy()
    RA = RA.to_numpy()
    DEC = DEC.to_numpy()
    RA_sigma = RA_sigma.to_numpy()
    DEC_sigma = DEC_sigma.to_numpy()

    return obs_time, RA, DEC, RA_sigma, DEC_sigma

def obs_extractDT(NEA: pd.DataFrame, start_index: int, DT: float, include_match: bool = False, tol: float = 1e-9, day_column: str = "DD.dddddddddd" ):
    """
        Extract Observation based on first observation and the time after first observation 'arc length'.
    """

    day = pd.to_numeric(NEA[day_column], errors="coerce")

    start_val = float(day.loc[start_index])
    target = start_val + DT

    subset = day.loc[start_index:]                 # search forward from N_0
    diffs = (subset - target).abs()
    match_index = diffs.idxmin()
    match_value = float(day.loc[match_index])

    if abs(match_value - target) <= tol:
        match_type = "exact"
        print(f"EXACT match at index {match_index}: {match_value:.12f}")
    else:
        match_type = "rounded"
        print(f"ROUNDED match at index {match_index}: {match_value:.12f} (Target Value: {target:.12f})")
    
    start_pos = NEA.index.get_loc(start_index)
    match_pos = NEA.index.get_loc(match_index)
    
    if include_match:
        lo, hi = sorted((start_pos, match_pos))
        result = NEA.loc[lo:hi+1]
    else:
        if match_pos <= start_pos: 
            result = NEA.iloc[start_pos:start_pos+1]    # Only return the first row
        else:
            result = NEA.iloc[start_pos:match_pos]      # up to match index but not including
    

    # Calculate forward differences: i+1 - i
    day_values = pd.to_numeric(result[day_column], errors="coerce")
    result["DD_diff"] = day_values.shift(-1) - day_values  # forward diff
    result["DD_diff"] = result["DD_diff"].fillna(0)        # last row gets 0
    result.iloc[0, result.columns.get_loc("DD_diff")] = 0  # ensure first row = 0

    # Obtain the RA, DEC and Tim measurements in Numpy form 
    obs_time = result[["YYYY", "MM", "DD.dddddddddd"]]
    RA = result[["HH", "MM_2", "SS.sss"]]
    DEC = result[["sDD","MM_3","SS.ss"]]

    # Extract RA and DEC uncertainties
    RA_sigma = result[["Accuracy_2"]]
    DEC_sigma = result[["Accuracy_3"]]

    # Time Difference
    dt = result[["DD_diff"]]
    # Conver to numpy arrays
    obs_time = obs_time.to_numpy()
    RA = RA.to_numpy()
    DEC = DEC.to_numpy()
    RA_sigma = RA_sigma.to_numpy()
    DEC_sigma = DEC_sigma.to_numpy()

    return obs_time, RA, DEC, RA_sigma, DEC_sigma, dt
    
