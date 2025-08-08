import Aphopis_Armillien.utils.data as data
import os


def test_convert_rwo_to_csv():
    # Define input and output file paths
    rwo_file = "C:\\Users\\chezh\\OneDrive - Cranfield University\\Documents\\IRP\\02_References\\99942 Apophis.rwo"
    csv_file = "C:\\Users\\chezh\\OneDrive\\Documents\\Masters_Thesis_misc\\apophis_data.csv"

    # Call the conversion function
    data.rwo_to_csv(rwo_file, csv_file)

    # Check if the CSV file was created
    assert os.path.exists(csv_file), "CSV file was not created."

    # Check if the CSV file is not empty
    assert os.path.getsize(csv_file) > 0, "CSV file is empty."

def test_csv_to_pandas():
    # Define the path to the CSV file
    input_csv_file = "C:\\Users\\chezh\\OneDrive - Cranfield University\\Documents\\IRP\\05_Thesis_code\\apophis_data.csv"

    # Call the conversion function
    df = data.csv_to_pandas(input_csv_file)

    # Check if the DataFrame is not empty
    assert not df.empty, "DataFrame is empty."

    # Check if the DataFrame has the expected columns
    expected_columns = ["Design", "YYYY", "MM","DD.dddddddddd", "Accuracy", "HH", "MM_2"]
    for col in expected_columns:
        assert col in df.columns, f"Expected column '{col}' not found in DataFrame."

def 
