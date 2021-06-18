import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow.keras as tfk

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

def remove_sparses(df, threshold = 0.89):
    sparse_columns = []
    for column in df:
        col = df[column]
        highest = col.value_counts().iat[0]
        if highest / len(df.index) > threshold:
            print(column, col.value_counts().index[0], col.value_counts().iat[0], highest / len(x_df.index))
            sparse_columns.append(column)
    df.drop(sparse_columns, axis=1, inplace=True)
    return df


df = pd.read_csv("Train.csv")
df = remove_correlations(df)
df = remove_sparses(df)

df_7 = df[df['Label1'] == 7]

x_df = df.iloc[:, :-2].astype('float32')
y_df = df.iloc[:, -2:].astype('int32')

sc = preprocessing.StandardScaler()
x = sc.fit_transform(x_df)

y_df['Label'] = y_df.apply(lambda r: r['Label1'] * (-1 if r['Label2'] == 0 else 1), axis=1).astype(np.int64)
y_df = y_df[['Label1']]
for column in y_df:
    col = y_df[column]
    print(column, '\n', col.value_counts())
    
 y = tfk.utils.to_categorical(y_df, dtype="int32")
 y_df.columns
 y[:10]
print(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(x, y_df.to_numpy(), test_size=0.2, random_state=0)
print(X_train.shape)
print(y_train.shape)

for column in x_df:
    col = x_df[column]
    if len(pd.unique(col)) < 1000:
        print(column, type(col.value_counts()), '\n', col.value_counts())
        

# create model
model = tfk.Sequential()
model.add(tfk.layers.Dense(50,input_shape=(26,), activation='relu')) #First Hidden Layer
model.add(tfk.layers.Dense(100, activation='relu')) #Second  Hidden Layer
model.add(tfk.layers.Dense(11, activation='softmax')) #Output Layer
#model Looks like:  784 input -> [50 units in layer1] ->[100 units in layer2] -> 1 output

# Compiling the model  
model.compile(optimizer='adam',                         
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
#Model's Summary
model.summary()


for i in [X_train, X_test, y_train, y_test]:
    print(type(i))

#Training the model 
training = model.fit(X_train, y_train, epochs = 20, batch_size = 5000, validation_data = (X_test, y_test))
        

if __name__ == '__main__':
    main()
