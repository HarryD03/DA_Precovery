#Integrated Observation Extract Package: Extract NEO CC observations, format to Guass IOD / DAIOD
#Try Extract .row Except if .csv of same name is in the same file directory as this script

import utils.data as data
import utils.time_reference as time_ref
import pandas as pd
import numpy as np
from typing import Callable, List, Union, overload, Tuple
import poliastro as pl
import astropy.time as at
import astropy.constants as ac
import astropy.coordinates as acoords
import astropy.units as u
from numpy.typing import NDArray
 


def load_observation_file(filepath: str) -> pd.DataFrame:
    """
    Extract Observations to Pandas, from .row (or .csv) file. FILE MUST BE IN MPC FORMAT
            See if csv file in 05_Thesis_code exisits
            if not get the .row file and conduct conversion
        
    
    Args:
        filepath (str): Path to the observation file (.rwo or .csv)
        
    Returns:
        pd.DataFrame: DataFrame containing the observation data
        
    Raises:
        ValueError: If file is neither .rwo nor .csv format
    """
    if filepath.endswith('.rwo'):
        csv_path = filepath.replace('.rwo', '.csv')
        try:
            NEA = pd.read_csv(csv_path)
        except FileNotFoundError:
            # Convert RWO to CSV if CSV doesn't exist
            data.rwo_to_csv(filepath, csv_path)
            NEA = pd.read_csv(csv_path)
    elif filepath.endswith('.csv'):
        NEA = pd.read_csv(filepath)
    else:
        raise ValueError("File must be either .rwo or .csv format")
    
    return NEA

def extract_obs(NEA, extraction_method: str = "instance", **kwargs) -> Tuple:
    """
        Extract Observations to Pandas, from .row (or .csv) file. FILE MUST BE IN MPC FORMAT
            See if csv file in 05_Thesis_code exisits
            if not get the .row file and conduct conversion
        
        Observation Extraction from pandas Dataframe based on what user wants to extract:
            Observation Instances
            Observation Arc length/Time Gap
            Observation Degree Seperation

        Args:
        filepath (str): Path to the observation file (.rwo or .csv)
        extraction_method (str): Method to extract observations:
            - "instance": Extract by observation instances (N_lower to N_upper)
            - "arc": Extract by time arc length from start index
            - "time": Extract by specific time

        **kwargs: Additional arguments based on extraction method:
            For "instance":
                N_lower (int): Lower bound of observation instances
                N_upper (int): Upper bound of observation instances
            For "arc_time":
                start_index (int): Starting observation index
                DT (float): Time arc length in days
                include_match (bool, optional): Include matching observation
            For "arc_seperation":
                TBD
    
    Returns:
        Tuple containing observation data based on extraction method
        
    """
    #Extract Observations and Observation dataes based on method
    if extraction_method == "instance":
        if 'N_lower' not in kwargs or 'N_upper' not in kwargs:
            raise ValueError("N_lower and N_upper required for instance extraction")
        return obs_extractN(NEA, kwargs['N_lower'], kwargs['N_upper'])
    
    elif extraction_method == 'arc_time':
        if 'start_index' not in kwargs or 'DT' not in kwargs:
            raise ValueError("start_index and DT required for arc extraction")
        include_match = kwargs.get('include_match', False)
        return obs_extractDT(NEA, kwargs['start_index'], kwargs['DT'], include_match)  

    elif extraction_method == 'seperation':
        raise ValueError("This extraction Method has not been Implimented yet")
    
    else:
        raise ValueError("Invalid extraction method. Use 'instance', 'arc_time','arc_sepration")

def convert_obs(obs_time: pd.DataFrame, RA: pd.DataFrame, DEC: pd.DataFrame, 
               RA_sigma: pd.DataFrame, DEC_sigma: pd.DataFrame) -> Tuple[np.ndarray, ...]:
    """
    Convert observation data from Pandas DataFrames to numpy arrays and convert units.
    
    Args:
        obs_time: DataFrame with columns ["YYYY", "MM", "DD.dddddddddd"]
        RA: DataFrame with columns ["HH", "MM_2", "SS.sss"]
        DEC: DataFrame with columns ["sDD", "MM_3", "SS.ss"]
        RA_sigma: DataFrame with column ["Accuracy_2"] in arcseconds precision
        DEC_sigma: DataFrame with column ["Accuracy_3"] in arcseconds precision
    
    Returns:
        Tuple containing:
        - obs_time_np: array of [day, hour, minute, second]
        - RA_deg: Right Ascension in degrees
        - DEC_deg: Declination in degrees
        - RA_sigma_deg: RA uncertainty in degrees (converted from seconds)
        - DEC_sigma_deg: DEC uncertainty in degrees (converted from arcseconds)
    """
    # 1. Convert observation time from decimal days to [day, hour, minute, second]
    
    obs_time_np = time_ref.timeday_to_hhmmss(obs_time["DD.dddddddddd"].to_numpy())

    # 2. Convert RA from HH:MM:SS to degrees
    RA_np = RA.to_numpy()

    RA_deg = time_ref.HH_MM_SS_to_Degrees(RA_np)

    # 3. Convert DEC from DD:MM:SS to degrees
    DEC_np = DEC.to_numpy()
    DEC_deg = time_ref.dms_to_degrees(DEC_np)

    # 4. Convert uncertainties to degrees
    # RA sigma: seconds of time to degrees (15° per hour, so 15/3600 degrees per second)
    RA_sigma_rad = RA_sigma["Accuracy_2"].to_numpy() * 4.8481e-6
    
    # DEC sigma: arcseconds to degrees (1/3600 degrees per arcsecond)
    DEC_sigma_rad = DEC_sigma["Accuracy_3"].to_numpy() * 4.8481e-6

    return obs_time_np, RA_deg, DEC_deg, RA_sigma_rad, DEC_sigma_rad

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

    return obs_time, RA, DEC, RA_sigma, DEC_sigma, dt
    
def extract_N0(NEA: pd.DataFrame, t, year_col: str = "YYYY", month_col: str = "MM", day_col:str = "DD.dddddddddd", tol = 1e-9):
    """
        Extract the Index of variable based on [YYYY, MM, DD.ddddd]
        If no extact match, it finds the closest one
        :params NEA: Near Earth Asteroid Data Set
        :params t: Time of interest in [YYYY, MM, DD.dddddd]
    """

    Y, M, D = t
    
    # Coerce to numeric (without mutating original dtypes)
    year = pd.to_numeric(NEA[year_col], errors="coerce")
    month = pd.to_numeric(NEA[month_col], errors="coerce")
    day = pd.to_numeric(NEA[day_col], errors="coerce")

    # 1) Try exact YYYY & MM subset
    mask_exact_ym = (year == Y) & (month == M)      #If exact store as true

    if mask_exact_ym.any():         #If Any Entry of Year or Month is exact
        sub = day[mask_exact_ym]    #Get the Subset where they're exact
        # Find closest day within same year & month
        diffs = (sub - D).abs()
        match_index = diffs.idxmin()
        # Exact if within tolerance
        match_type = "exact" if abs(day.loc[match_index] - D) <= tol else "closest"
        
        return match_index, match_type
    
    # 2) No exact YYYY/MM: pick closest by hierarchy: year → month → day
    # Compute hierarchical distance: large weight for year, then month, then day
    # (weights chosen to ensure hierarchy: year dominates, then month)

    dist_year = (year - Y).abs()
    # pick rows with minimal year distance
    cand = dist_year == dist_year.min()
    year_best_idx = year[cand].index

    # Get the year difference to determine direction
    year_diff = year.loc[year_best_idx].iloc[0] - Y

    if year_diff < 0:  # Previous year - get latest possible date
        # Find latest month in that year
        latest_month_idx = month.loc[year_best_idx].idxmax()
        latest_month = month.loc[latest_month_idx]
        
        # Find all entries with that year and latest month
        mask_latest = (year.loc[year_best_idx] == year.loc[latest_month_idx]) & \
                     (month.loc[year_best_idx] == latest_month)
        candidates = year_best_idx[mask_latest]
        
        # Get latest day in that month
        match_index = day.loc[candidates].idxmax()
        
    elif year_diff > 0:  # Next year - get earliest possible date
        # Find earliest month in that year
        earliest_month_idx = month.loc[year_best_idx].idxmin()
        earliest_month = month.loc[earliest_month_idx]
        
        # Find all entries with that year and earliest month
        mask_earliest = (year.loc[year_best_idx] == year.loc[earliest_month_idx]) & \
                       (month.loc[year_best_idx] == earliest_month)
        candidates = year_best_idx[mask_earliest]
        
        # Get earliest day in that month
        match_index = day.loc[candidates].idxmin()
        
    else:  # Same year (year_diff == 0)
        dist_month = (month.loc[year_best_idx] - M).abs()
        cand2 = dist_month == dist_month.min()
        ym_best_idx = dist_month.index[cand2]

        # Get the month difference to determine if we're looking forward or backward
        month_diff = month.loc[ym_best_idx].iloc[0] - M

        # If we're looking at a previous month, we want the latest day
        # If we're looking at a next month, we want the earliest day
        if month_diff < 0:  # Previous month
            match_index = day.loc[ym_best_idx].idxmax()
        elif month_diff > 0:  # Next month
            match_index = day.loc[ym_best_idx].idxmin()
        else:
            # Same month, find closest day
            dist_day = (day.loc[ym_best_idx] - D).abs()
            match_index = dist_day.idxmin()

    return match_index, "closest"

def filter_observations_to_three(obs_times, RA, DEC, RA_sigma, DEC_sigma, obs_J2000):
    """
    Filter observations to keep only first, last, and temporal midpoint when more than 3 observations exist.
    Midpoint is determined by finding the closest observation to the temporal center using J2000 day values.
    
    Args:
        obs_times: Astropy Time array of observation times
        RA, DEC: Right ascension and declination arrays
        RA_sigma, DEC_sigma: Uncertainty arrays
        obs_J2000: Observations from J2000
        
    Returns:
        Tuple of filtered arrays: (obs_times, RA, DEC, RA_sigma, DEC_sigma, time_sec)
    """
    import numpy as np
    import astropy.units as u
    import astropy.time as at
    
    n_obs = len(obs_times)
    
    if n_obs <= 3:
        print(f"Using all {n_obs} observations (≤3)")
        return obs_times, RA, DEC, RA_sigma, DEC_sigma, obs_J2000

    print(f"\nFiltering {n_obs} observations to 3 (first, temporal midpoint, last)...")
    
    
    # Calculate indices for first and last observations
    first_idx = 0
    last_idx = n_obs - 1
    
    # Calculate temporal midpoint using obs_J2000 values
    first_time_val = obs_J2000.value[first_idx]
    last_time_val = obs_J2000.value[last_idx]
    mid_time_val = first_time_val + (last_time_val - first_time_val) / 2
    
    # Find observation closest to temporal midpoint
    time_diffs = np.abs(obs_J2000.value - mid_time_val)
    mid_idx = np.argmin(time_diffs)
    
    # Create index array for filtering
    keep_indices = [first_idx, mid_idx, last_idx]
    
    # Remove duplicates and sort (in case mid_idx equals first_idx or last_idx)
    keep_indices = sorted(list(set(keep_indices)))
    
    # Logging
    print(f"Original observations: {n_obs}")
    print(f"Time span (days since J2000): {first_time_val:.6f} to {last_time_val:.6f}")
    print(f"Temporal midpoint (days since J2000): {mid_time_val:.6f}")
    print(f"Keeping observations at indices: {keep_indices}")
    print(f"Selected observation times:")
    for i, idx in enumerate(keep_indices):
        time_diff_days = abs(obs_J2000.value[idx] - mid_time_val) if idx == mid_idx else None
        time_diff_hours = time_diff_days * 24 if time_diff_days is not None else None
        label = "FIRST" if idx == first_idx else "LAST" if idx == last_idx else f"MID (Δt={time_diff_hours:.1f}h)"
        print(f"  {i+1}. Index {idx}: {obs_times[idx].iso} [{label}]")
        print(f"      J2000 + {obs_J2000.value[idx]:.6f} days")
    
    # Filter observation arrays only
    obs_times_filtered = obs_times[keep_indices]
    RA_filtered = RA[keep_indices]
    DEC_filtered = DEC[keep_indices]
    RA_sigma_filtered = RA_sigma[0,keep_indices]
    DEC_sigma_filtered = DEC_sigma[0,keep_indices]
    obs_J2000_filtered = obs_J2000[keep_indices]

    print(f"Filtered to {len(obs_times_filtered)} observations")
    print(f"Time intervals (days): {np.diff(obs_J2000_filtered.value)}")
    print(f"Time intervals (seconds): {np.diff(obs_J2000_filtered.to(u.s).value)}")

    return obs_times_filtered, RA_filtered, DEC_filtered, RA_sigma_filtered, DEC_sigma_filtered, obs_J2000_filtered
