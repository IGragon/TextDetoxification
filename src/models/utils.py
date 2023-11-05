# Import necessary modules and functions
from datasets import Dataset
import pandas as pd


# Define a function to load a dataset from a CSV file
def load_csv_dataset(csv_path: str):
    # Read data from the CSV file using pandas, assuming tab-separated values
    dataframe = pd.read_csv(csv_path, sep='\t')

    # Create a dataset from selected columns and shuffle it with a fixed seed
    dataset = Dataset.from_pandas(dataframe[["tox_high", "tox_low"]]).shuffle(seed=42)

    # Return the resulting dataset
    return dataset
