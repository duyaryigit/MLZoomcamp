import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv")
df.head()

print(f"Q1 = {pd.__version__}")
print(f"Q2 = {df.shape[1]}")

for col in df.columns:
    if df[col].isnull().sum() > 0:
        print(f"Q3 = {col} : {df[col].isnull().sum()}")

print(f"Q4 = {df.ocean_proximity.nunique()}")

x = round(df[df.ocean_proximity == "NEAR BAY"]["median_house_value"].mean(), 0)
print(f"Q5 = {x}")

print(f"Q6 = {round(df.total_bedrooms.mean(), 4)}")
df["total_bedrooms"].fillna(df["total_bedrooms"].mean(), inplace=True)
print(f"Q6 = {round(df.total_bedrooms.mean(), 4)}")

island_options = df[df["ocean_proximity"] == "ISLAND"]
selected_columns = island_options[["housing_median_age", "total_rooms", "total_bedrooms"]]
X = selected_columns.values
XTX = X.T @ X
XTX_inverse = np.linalg.inv(XTX)
y = np.array([950, 1300, 800, 1000, 1300])
a = XTX_inverse @ X.T
w = a @ y
print(f"Q7 = {round(w[-1], 4)}")
