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
        RA_sigma: DataFrame with column ["Accuracy_2"] in seconds precision
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
    RA_sigma_deg = RA_sigma["Accuracy_2"].to_numpy() * (15.0/3600.0)
    
    # DEC sigma: arcseconds to degrees (1/3600 degrees per arcsecond)
    DEC_sigma_deg = DEC_sigma["Accuracy_3"].to_numpy() * (1.0/3600.0)

    return obs_time_np, RA_deg, DEC_deg, RA_sigma_deg, DEC_sigma_deg

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


