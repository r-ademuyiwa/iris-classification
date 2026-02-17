import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df= pd.read_csv("sleep_deprevation.csv")
df.columns = df.columns.str.strip()

target_col = "PVT_Reaction_Time"

for col in df.columns:
    if col == target_col:
        continue
    if not pd.api.types.is_numeric_dtype(df[col]):
        print("Skipping non-numeric column:", col)
        continue

    temp = df[[col, target_col]].dropna()

    x = df[[col]]
    y = df[target_col]

    model = LinearRegression()
    model.fit(x,y)

    print("Feature:" , col)
    print("Slope:" , model.coef_[0])
    print("Intercept:" , model.intercept_)
    print()

def divide (x, y):
    z = x/ y
    return z


