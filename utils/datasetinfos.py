import pandas as pd
import numpy as np

df = pd.read_csv("../datas/Reviews.csv")

print(df.isnull().sum())

print(f"Dimension du dataset : {df.shape}")