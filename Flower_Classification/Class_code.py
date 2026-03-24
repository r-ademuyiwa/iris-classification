# importing Path so that it can find the path of a file that im trying to run without the need to actually place the
# csv in the same directory as my class.code
from pathlib import Path
from typing import Annotated

# importing typer so that I can call methods directly from the terminal
import typer
from sklearn.metrics import accuracy_score  # type: ignore[import-untyped]
from sklearn.metrics import classification_report, confusion_matrix

from Flower_Classification.classifier_predict import d_tree_predict, k_predict, random_forest_predict
from Flower_Classification.clean_Data import clean_Data
from Flower_Classification.read_Data import read_Data
from Flower_Classification.scale_Data import minmax_scale_Data, robust_scale_Data, standard_scale_Data
from Flower_Classification.split_Data import split_Data

# the method from typer that i used to run methods from terminal ie main()
app = typer.Typer()


@app.command()

# main method
def main(
    # calling the method and files straight from terminal
    input_file: Annotated[Path, typer.Argument()],
    classifier: Annotated[str, typer.Argument()],
) -> None:
    new_file = read_Data(input_file)
    x, y = clean_Data(new_file)
    x_train, x_val, x_test, y_train, y_val, y_test = split_Data(x, y)
    classifier = classifier.lower()
    selected = False

    while not selected:
        if classifier == "knn":
            scalercall: str = input("Please Select a type of scaler to use: standard, robust or minmax\n").lower()
            if scalercall == "standard":
                x_train_scaled, x_val_scaled, x_test_scaled = standard_scale_Data(x_train, x_val, x_test)
                y_pred, y_val_pred = k_predict(x_train_scaled, x_test_scaled, x_val_scaled, y_train)
                selected = True
            elif scalercall == "robust":
                x_train_scaled, x_val_scaled, x_test_scaled = robust_scale_Data(x_train, x_val, x_test)
                y_pred, y_val_pred = k_predict(x_train_scaled, x_test_scaled, x_val_scaled, y_train)
                selected = True
            elif scalercall == "minmax":
                x_train_scaled, x_val_scaled, x_test_scaled = minmax_scale_Data(x_train, x_val, x_test)
                y_pred, y_val_pred = k_predict(x_train_scaled, x_test_scaled, x_val_scaled, y_train)
                selected = True
            else:
                print("Please select a valid scaler")

        elif classifier == "randomforest":
            x_train_scaled, x_val_scaled, x_test_scaled = standard_scale_Data(x_train, x_val, x_test)
            y_pred, y_val_pred = random_forest_predict(x_train_scaled, x_test_scaled, x_val_scaled, y_train)
            selected = True
        elif classifier == "decisiontree":
            x_train_scaled, x_val_scaled, x_test_scaled = standard_scale_Data(x_train, x_val, x_test)
            y_pred, y_val_pred = d_tree_predict(x_train_scaled, x_test_scaled, x_val_scaled, y_train)
            selected = True
        else:
            print("Classifier not recognized")
            return

    # print(confusion_matrix(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_val, y_val_pred))
    # print(classification_report(y_val, y_val_pred))


# @app.command()
# def run():
#     print("test")

if __name__ == "__main__":
    app()
