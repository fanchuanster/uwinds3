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

def remove_correlations(df, threshold = 0.94):
    cordf = df.corr().abs()
    todrop = defaultdict(list)
    for i in range(cordf.shape[0]):
        for j in range(cordf.shape[1]):
            v = cordf.iat[i, j]
            if v > threshold and i != j and j not in todrop:
                todrop[i].append((j, v))
    print(todrop)
    todroplist = list(set([item[0] for k,v in todrop.items() for item in v]))
    print(len(todroplist), todroplist)
    df.drop(df.columns[todroplist], axis=1, inplace=True)
    return df

df = pd.read_csv("Train.csv")
df = remove_correlations(df)
df.columns
x_df = df.iloc[:, :-2].astype(np.float)
y_df = df.iloc[:, -2:].astype(np.int64)

x_df.head(10)
y_df['Label'] = y_df.apply(lambda r: r['Label1'] * (-1 if r['Label2'] == 0 else 1), axis=1)

for column in x_df:
    col = x_df[column]
    if len(pd.unique(col)) < 100:
        print(column, type(col.value_counts()), '\n', col.value_counts())

for column in x_df:
    col = x_df[column]
    highest = col.value_counts().iat[0]
    if highest / len(x_df.index) > 0.9:
        print(col.value_counts().iat[0])
        #print(column, type(col.value_counts()), '\n', col.value_counts())


y_df = y_df['Label1']
y_df.head(-10)

x_df.columns

def main():
    pass

if __name__ == '__main__':
    main()
