import pandas as pd


def clean_Data(input_file: pd.DataFrame) -> tuple:
    # making sure that from every row in every column, the values inside is an integer
    # checking each column for sum of empty values and then adding all the sums of each column together to check if greater than 0
    # dropping empty rows and dopping duplicate rows
    if input_file.isnull().sum().sum() > 0:
        input_file = input_file.dropna()

    input_file = input_file.drop_duplicates()

    # basically making an array
    # selecting everything but the last column which contains the species specification
    X = input_file.iloc[:, :-1]
    # selecting only the species specification
    y = input_file.iloc[:, -1]

    # print(X.shape)
    # print(y.shape)
    # print(input_file.name, input_file.stem, input_file.suffix, input_file.resolve())

    return X, y
