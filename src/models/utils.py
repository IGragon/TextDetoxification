from datasets import Dataset
import pandas as pd


def load_csv_dataset(csv_path: str):
    dataframe = pd.read_csv(csv_path, sep='\t')
    dataset = Dataset.from_pandas(dataframe[["tox_high", "tox_low"]]).shuffle(seed=42)
    return dataset
