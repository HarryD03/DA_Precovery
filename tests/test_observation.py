import pandas as pd
from utils.observation import extract_obs, obs_extractN, obs_extractDT, extract_N0, convert_obs

def test_extract_obs_instance():
    # Create test input
    sample_data = pd.DataFrame({
        'YYYY': [2023, 2023, 2023],
        'MM': [1, 1, 1],
        'DD.dddddddddd': [1.5, 2.5, 3.5],
        'HH': [12, 13, 14],
        'MM_2': [30, 45, 15],
        'SS.sss': [45.123, 30.456, 15.789],
        'sDD': [45, -30, 60],
        'MM_3': [30, 45, 15],
        'SS.ss': [45.12, 30.45, 15.78],
        'Accuracy_2': [0.5, 0.6, 0.7],
        'Accuracy_3': [0.4, 0.5, 0.6]
    })

    # Execute function
    obs_time, ra, dec, ra_sigma, dec_sigma = extract_obs(
        sample_data,
        extraction_method="instance",
        N_lower=0,
        N_upper=2
    )

    # Assert results
    assert len(obs_time) == 2
    assert list(obs_time.columns) == ['YYYY', 'MM', 'DD.dddddddddd']
    assert list(ra.columns) == ['HH', 'MM_2', 'SS.sss']
    assert list(dec.columns) == ['sDD', 'MM_3', 'SS.ss']

def test_obs_extractN():
    # Create test input
    sample_data = pd.DataFrame({
        'YYYY': [2023, 2023, 2023],
        'MM': [1, 1, 1],
        'DD.dddddddddd': [1.5, 2.5, 3.5],
        'HH': [12, 13, 14],
        'MM_2': [30, 45, 15],
        'SS.sss': [45.123, 30.456, 15.789],
        'sDD': [45, -30, 60],
        'MM_3': [30, 45, 15],
        'SS.ss': [45.12, 30.45, 15.78],
        'Accuracy_2': [0.5, 0.6, 0.7],
        'Accuracy_3': [0.4, 0.5, 0.6]
    })

    # Execute function
    obs_time, ra, dec, ra_sigma, dec_sigma = obs_extractN(sample_data, 0, 2)

    # Assert results
    assert len(obs_time) == 2
    assert obs_time.iloc[0]['YYYY'] == 2023
    assert obs_time.iloc[0]['DD.dddddddddd'] == 1.5

def test_obs_extractDT():
    # Create test input
    sample_data = pd.DataFrame({
        'YYYY': [2023, 2023, 2023],
        'MM': [1, 1, 1],
        'DD.dddddddddd': [1.5, 2.5, 3.5],
        'HH': [12, 13, 14],
        'MM_2': [30, 45, 15],
        'SS.sss': [45.123, 30.456, 15.789],
        'sDD': [45, -30, 60],
        'MM_3': [30, 45, 15],
        'SS.ss': [45.12, 30.45, 15.78],
        'Accuracy_2': [0.5, 0.6, 0.7],
        'Accuracy_3': [0.4, 0.5, 0.6]
    })

    # Execute function
    obs_time, ra, dec, ra_sigma, dec_sigma, dt = obs_extractDT(
        sample_data, 
        start_index=0,
        DT=1.0
    )

    # Assert results
    assert 'DD_diff' in dt.columns
    assert dt.iloc[0]['DD_diff'] == 0

def test_extract_N0():
    # Create test input
    sample_data = pd.DataFrame({
        'YYYY': [2023, 2023, 2023],
        'MM': [1, 1, 1],
        'DD.dddddddddd': [1.5, 2.5, 3.5],
        'HH': [12, 13, 14],
        'MM_2': [30, 45, 15],
        'SS.sss': [45.123, 30.456, 15.789],
        'sDD': [45, -30, 60],
        'MM_3': [30, 45, 15],
        'SS.ss': [45.12, 30.45, 15.78],
        'Accuracy_2': [0.5, 0.6, 0.7],
        'Accuracy_3': [0.4, 0.5, 0.6]
    })

    # Test exact match
    time = [2023, 1, 1.5]
    index, match_type = extract_N0(sample_data, time)

    # Assert results
    assert match_type == "exact"
    assert sample_data.iloc[index]['YYYY'] == 2023
    assert sample_data.iloc[index]['DD.dddddddddd'] == 1.5

def test_extract_N0_closest():
    # Create test input
    sample_data = pd.DataFrame({
        'YYYY': [2023, 2023, 2023],
        'MM': [1, 1, 1],
        'DD.dddddddddd': [1.5, 2.5, 3.5],
        'HH': [12, 13, 14],
        'MM_2': [30, 45, 15],
        'SS.sss': [45.123, 30.456, 15.789],
        'sDD': [45, -30, 60],
        'MM_3': [30, 45, 15],
        'SS.ss': [45.12, 30.45, 15.78],
        'Accuracy_2': [0.5, 0.6, 0.7],
        'Accuracy_3': [0.4, 0.5, 0.6]
    })

    # Test closest match
    time = [2023, 1, 1.7]
    index, match_type = extract_N0(sample_data, time)

    # Assert results
    assert match_type == "closest"
    assert sample_data.iloc[index]['DD.dddddddddd'] in [1.5, 2.5]

def test_extract_N0_closest_month():
    # Create test input
    sample_data = pd.DataFrame({
        'YYYY': [2023, 2023, 2023],
        'MM': [2, 2, 2],
        'DD.dddddddddd': [1.5, 2.5, 3.5],
        'HH': [12, 13, 14],
        'MM_2': [30, 45, 15],
        'SS.sss': [45.123, 30.456, 15.789],
        'sDD': [45, -30, 60],
        'MM_3': [30, 45, 15],
        'SS.ss': [45.12, 30.45, 15.78],
        'Accuracy_2': [0.5, 0.6, 0.7],
        'Accuracy_3': [0.4, 0.5, 0.6]
    })

    # Test closest match
    time = [2023, 3, 1.7]
    index, match_type = extract_N0(sample_data, time)

    # Assert results
    assert match_type == "closest"
    assert sample_data.iloc[index]['DD.dddddddddd'] == 3.5

    time = [2023, 1, 1.7]
    index, match_type = extract_N0(sample_data, time)

    # Assert results
    assert match_type == "closest"
    assert sample_data.iloc[index]['DD.dddddddddd'] == 1.5

def test_extract_N0_closest_year():
    # Create test input
    sample_data = pd.DataFrame({
        'YYYY': [2024, 2024, 2024],
        'MM': [2, 2, 2],
        'DD.dddddddddd': [1.5, 2.5, 3.5],
        'HH': [12, 13, 14],
        'MM_2': [30, 45, 15],
        'SS.sss': [45.123, 30.456, 15.789],
        'sDD': [45, -30, 60],
        'MM_3': [30, 45, 15],
        'SS.ss': [45.12, 30.45, 15.78],
        'Accuracy_2': [0.5, 0.6, 0.7],
        'Accuracy_3': [0.4, 0.5, 0.6]
    })

    # Test closest match
    time = [2023, 2, 1.7]
    index, match_type = extract_N0(sample_data, time)

    # Assert results
    assert match_type == "closest"
    assert sample_data.iloc[index]['DD.dddddddddd'] == 1.5

    time = [2025, 2, 1.7]
    index, match_type = extract_N0(sample_data, time)

    # Assert results
    assert match_type == "closest"
    assert sample_data.iloc[index]['DD.dddddddddd'] == 3.5

def test_convert_obs():
    # Create test input DataFrames
    obs_time = pd.DataFrame({
        'YYYY': [2023],
        'MM': [1],
        'DD.dddddddddd': [1.5]
    })
    
    ra = pd.DataFrame({
        'HH': [12],
        'MM_2': [30],
        'SS.sss': [45.123]
    })
    
    dec = pd.DataFrame({
        'sDD': [45],
        'MM_3': [30],
        'SS.ss': [45.12]
    })
    
    ra_sigma = pd.DataFrame({
        'Accuracy_2': [0.5]
    })
    
    dec_sigma = pd.DataFrame({
        'Accuracy_3': [0.4]
    })

    # Call the function
    obs_time_np, RA_deg, DEC_deg, RA_sigma_deg, DEC_sigma_deg = convert_obs(
        obs_time, ra, dec, ra_sigma, dec_sigma
    )

    # Assertions
    assert obs_time_np.shape == (1, 4)  # [day, hour, minute, second]
    
    # Check uncertainty conversions
    assert RA_sigma_deg.shape == (1,)
    assert abs(RA_sigma_deg[0] - (0.5 * 15.0/3600.0)) < 1e-6  # RA uncertainty in degrees

    assert DEC_sigma_deg.shape == (1,)
    assert abs(DEC_sigma_deg[0] - (0.4 * 1.0/3600.0)) < 1e-6  # DEC uncertainty in degrees

    # Since we're not mocking the time_reference functions, we'll just check that
    # the outputs have the right shape and are non-zero
    assert RA_deg.shape == (1,)
    assert DEC_deg.shape == (1,)
    assert all(abs(RA_deg) > 0)  # Check that conversion happened
    assert all(abs(DEC_deg) > 0)  # Check that conversion happened


