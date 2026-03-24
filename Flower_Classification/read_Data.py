from pathlib import Path

import pandas as pd


def read_Data(input_file: Path) -> pd.DataFrame:
    # reading the CSV as a dataframe and reading the columns
    df: pd.DataFrame = pd.read_csv(input_file)
    # duplicating input file to work on.
    new_file = df.copy()
    new_file.columns = new_file.columns.str.strip()

    return new_file
