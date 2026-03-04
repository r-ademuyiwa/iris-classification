# importing pandas
import pandas as pd  # type: ignore[import-untyped]

# importing Path so that it can find the path of a file that im trying to run without the need to actually place the
# csv in the same directory as my class.code
from pathlib import Path  # type: ignore[import-untyped]

# importing typer so that I can call methods directly from the terminal
import typer  # type: ignore[import-untyped]
from typing import Annotated  # type: ignore[import-untyped]

# importing the method of splitting the test data.
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# importing the model for scaling the data so that everything is in the same scale
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

# importing numpy
import numpy as np  # type: ignore[import-untyped]

# the method from typer that i used to run methods from terminal ie main()
app = typer.Typer()


@app.command()

# main method
def main(
    # calling the method and files straight from terminal
    input_file: Annotated[Path, typer.Argument()],
) -> None:
    # reading the CSV as a dataframe and reading the columns
    df: pd.DataFrame = pd.read_csv(input_file)
    df.column = df.columns.str.strip()
    # making sure that from every row in every column, the values inside is an integer
    # basically making an array
    X: np.ndarray = df.iloc[:, :-1].values
    y: np.ndarray = df.iloc[:, -1].values

    print(X.shape)
    print(y.shape)
    # printing out the file name, what type of file, and the directory of the file
    print(input_file.name, input_file.stem, input_file.suffix, input_file.resolve())

    # spilliting the data into training and testing data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    # 0.25 x 0.8 = 0.2
    scaler = StandardScaler()
    X_train_scaled= scaler.fit_transform(X_train)
    X_test_scaled= scaler.transform(X_test)
    X_val_scaled= scaler.transform(X_val)

    # print(scaler.mean_)
    # print(scaler.transform(X_train))
    # print(scaler.transform([[2, 2]]))

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    y_val_pred = knn.predict(X_val_scaled)
    # prediction = knn.predict(df.values.reshape(1, -1))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print(confusion_matrix(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))


# @app.command()
# def run():
#     print("test")

if __name__ == "__main__":
    app()
