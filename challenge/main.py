import pandas as pd
import numpy as np

# exact match ratio
#MR = np.all(y_pred == y_true, axis=1).mean()

# Hamming Loss
# def Hamming_Loss(y_true, y_pred):
#   temp=0
#   for i in range(y_true.shape[0]):
#       temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
#   return temp/(y_true.shape[0] * y_true.shape[1])

df = pd.read_csv("Train.csv")
df.count
x_df = df.iloc[:, :-2]
x_df.columns

for column in x_df:
    col = df[column]
    print(column, col.min(), col.mean(), col.max())
    break

print(type(x_df[['F12', 'F13']]))

x_df[['F12', 'F13']].corr()


