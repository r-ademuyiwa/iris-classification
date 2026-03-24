# importing the method of splitting the test data.
import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]


def split_Data(X: np.ndarray, y: np.ndarray) -> tuple:
    # splitting the data into training, testing and validation data.
    # stratify to make sure that the amount of each classification in each subset is the same. 33% of each subset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=1, stratify=y_train)
    # 0.25 x 0.8 = 0.2

    return X_train, X_val, X_test, y_train, y_val, y_test
