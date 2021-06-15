import pandas as pd
import numpy as np
from collections import defaultdict

# exact match ratio
#MR = np.all(y_pred == y_true, axis=1).mean()

# Hamming Loss
# def Hamming_Loss(y_true, y_pred):
#   temp=0
#   for i in range(y_true.shape[0]):
#       temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
#   return temp/(y_true.shape[0] * y_true.shape[1])

df = pd.read_csv("Train.csv")
x_df = df.iloc[:, :-2].astype(np.float)
y_df = df.iloc[:, -2:].astype(np.int64)

cordf = x_df.corr().abs()
cordf = cordf.replace(1.0, 0.0)

threshhold = 0.94
todrop = defaultdict(list)
for column in cordf:
    s = cordf[column]
    if any(s > threshhold):
        for i, v in s.items():
            if v > threshhold:
                if i not in todrop:
                    todrop[column].append((i, v))
todroplist = list(set([item[0] for k,v in todrop.items() for item in v]))

x_df = x_df.drop(todroplist, axis=1)

y_df['Label'] = y_df.apply(lambda r: r['Label1'] * (-1 if r['Label2'] == 0 else 1), axis=1)

for column in y_df:
    col = y_df[column]
    print(column, pd.unique(col), col.min(), col.mean(), col.max())
