import pandas as pd
import numpy as np


df = pd.read_csv("Train.csv")
df.count
x_df = df.iloc[:, :-2]
x_df.columns

for column in df:
    print(column)
    col = df[column]
    m = col.min()
    
    print(df[column].min())
    print(m)
    break


