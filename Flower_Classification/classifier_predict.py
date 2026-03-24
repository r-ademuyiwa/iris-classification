import numpy as np
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.neighbors import KNeighborsClassifier  # type: ignore[import-untyped]
from sklearn.tree import DecisionTreeClassifier  # type: ignore[import-untyped]


def k_predict(
    X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, X_val_scaled: np.ndarray, y_train: np.ndarray
) -> tuple:
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    y_val_pred = knn.predict(X_val_scaled)
    print(y_pred)

    return y_pred, y_val_pred


def random_forest_predict(
    X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, X_val_scaled: np.ndarray, y_train: np.ndarray
) -> tuple:
    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    y_val_pred = rf.predict(X_val_scaled)

    return y_pred, y_val_pred


def d_tree_predict(
    X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, X_val_scaled: np.ndarray, y_train: np.ndarray
) -> tuple:
    dt = DecisionTreeClassifier(criterion="gini", random_state=1, max_leaf_nodes=5)
    dt.fit(X_train_scaled, y_train)
    y_pred = dt.predict(X_test_scaled)
    y_val_pred = dt.predict(X_val_scaled)

    return y_pred, y_val_pred
