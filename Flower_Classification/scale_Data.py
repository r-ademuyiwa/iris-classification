# importing the model for scaling the data so that everything is in the same scale
import numpy as np
from sklearn.preprocessing import (  # type: ignore[import-untyped]
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


def standard_scale_Data(train_split: np.ndarray, validation_split: np.ndarray, test_split: np.ndarray) -> tuple:
    # scalling and fitting the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_split)
    X_val_scaled = scaler.transform(validation_split)
    X_test_scaled = scaler.transform(test_split)

    # print(scaler.mean_)
    # print(scaler.transform(X_train))
    # print(scaler.transform([[2, 2]]))

    return X_train_scaled, X_val_scaled, X_test_scaled


def minmax_scale_Data(train_split: np.ndarray, validation_split: np.ndarray, test_split: np.ndarray) -> tuple:
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(train_split)
    X_val_scaled = scaler.transform(validation_split)
    X_test_scaled = scaler.transform(test_split)

    return X_train_scaled, X_val_scaled, X_test_scaled


def robust_scale_Data(train_split: np.ndarray, validation_split: np.ndarray, test_split: np.ndarray) -> tuple:
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(train_split)
    X_val_scaled = scaler.transform(validation_split)
    X_test_scaled = scaler.transform(test_split)

    return X_train_scaled, X_val_scaled, X_test_scaled
