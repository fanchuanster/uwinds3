import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn import preprocessing, decomposition
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import tensorflow.keras as tfk
import tensorflow.keras.backend as kb
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.datasets import make_classification

# exact match ratio
#MR = np.all(y_pred == y_true, axis=1).mean()
# def exact_match_ratio_loss(y_actual, y_pred):
#     return y_atual != y_pred

# Hamming Loss
# def Hamming_Loss(y_true, y_pred):
#   temp=0
#   for i in range(y_true.shape[0]):
#       temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
#   return temp/(y_true.shape[0] * y_true.shape[1])

def remove_correlations(df, threshold = 0.94):
    cordf = df.corr().abs()
    print(cordf)
    todrop = defaultdict(list)
    for i in range(cordf.shape[0]):
        for j in range(cordf.shape[1]):
            v = cordf.iat[i, j]
            if v > threshold and i != j and j not in todrop:
                todrop[i].append((j, v))
    todroplist = list(set([item[0] for k,v in todrop.items() for item in v]))
    print(len(todroplist), todroplist)
    df.drop(df.columns[todroplist], axis=1, inplace=True)
    print("remove_correlations - {} features remaining".format(len(df.columns)))
    return df

def remove_sparses(df, threshold = 0.89):
    sparse_columns = []
    for column in df:
        col = df[column]
        highest = col.value_counts().iat[0]
        if highest / len(df.index) > threshold:
            print(column, col.value_counts().index[0], col.value_counts().iat[0], highest / len(df.index))
            sparse_columns.append(column)
    df.drop(sparse_columns, axis=1, inplace=True)
    print("remove_sparses - {} features remaining".format(len(df.columns)))
    return df

def remove_lowcor_with_label(df, labelname):
    cor = df.corr().abs()    
    relevant_cor = cor[(cor[labelname] > 0.1)]
    df = df[relevant_cor.index]
    print("remove_lowcor_with_label - {} features remaining".format(len(df.columns)))
    return df

def visualize(training):
    #Visulaizing the Training and Validation Sets Loss and Accuracy
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    #Plot training and validation accuracy values
    #axes[0].set_ylim(0,1) #if we want to limit axis in certain range
    axes[0].plot(training.history['sparse_categorical_accuracy'], label='Train')
    axes[0].plot(training.history['val_sparse_categorical_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    #Plot training and validation loss values
    #axes[1].set_ylim(0,1)
    axes[1].plot(training.history['loss'], label='Train')
    axes[1].plot(training.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    
# https://machinelearningmastery.com/chi-squared-test-for-machine-learning/
def select_kbest(x,y):
    fs = SelectKBest(score_func=f_classif, k=10)
    X_selected = fs.fit_transform(X, y)
    print(X_selected.shape)

df = pd.read_csv("Train.csv")
df = remove_correlations(df)
df = remove_sparses(df)
df = df.drop(['Label2'], axis=1)
df = remove_lowcor_with_label(df, 'Label1')

# remove linear features
for column in df:
    col = df[column]
    if len(pd.unique(col)) > 10000:
        print(column, type(col.value_counts()), '\t{}\n'.format(len(col.value_counts())), col.value_counts())
        df.drop([column], axis=1, inplace=True)
print(df.shape)
        
x_df = df.iloc[:, :-1].astype('float32')
y_df = df.iloc[:, -1:].astype('int32')
sc = preprocessing.StandardScaler()
x = sc.fit_transform(x_df)
# x = preprocessing.normalize(x_df, norm='l2', axis=1, copy=True, return_norm=False)
# y_df['Label'] = y_df.apply(lambda r: r['Label1'] * (-1 if r['Label2'] == 0 else 1), axis=1).astype(np.int64)
# y_df = y_df[['Label1']]

for column in y_df:
    col = y_df[column]
    print(column, '\n', col.value_counts())
    
# X_train, X_test, y_train, y_test = train_test_split(x, y_df-1, test_size=0.2, random_state=0)

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# fs = SelectKBest(score_func=f_classif, k=10)
# X_train_selected = fs.fit_transform(X_train, np.ravel(y_train))
# print(X_selected.shape)
# x_test_selected = fs.transform(X_test)
# print(x_test_selected.shape)
# print(y_train.shape)

# pca_components = 12
# pca = decomposition.PCA(pca_components, svd_solver='full')
# x_train_pca = pca.fit_transform(X_train)
# x_test_pca = pca.transform(X_test)
# pca_components = len(x_train_pca[0])
# print(x_train_pca[0])
# print(x_test_pca[0])
# print(pca.explained_variance_ratio_)
# print(pca_transform.shape)
# print(pca_components)
# # print(pca.n_features_)


# 1. try without reducing
# 2. apply pca  - dimensions reduction
# 3. switch from standardization to normalization
# 4. more layers, more nodes

# 5. try activation different
# 6. try more epochs

model = tfk.Sequential()
model.add(tfk.layers.Dense(27,input_shape=(x_df.shape[1],), activation='relu')) #First Hidden Layer
model.add(tfk.layers.Dense(108, activation='relu')) #Second  Hidden Layer
model.add(tfk.layers.Dense(10, activation='softmax')) #Output Layer
#model Looks like:  784 input -> [50 units in layer1] ->[100 units in layer2] -> 1 output

model.compile(optimizer='adam',                         
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
model.summary()

#Training the model 
sss = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
training = None
count= 0
for train_index, test_index in sss.split(x, (y_df-1).to_numpy()):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    result = model.fit(X_train, y_train, batch_size = 1280, validation_data = (X_test, y_test))
    if not training:
        training = result
    else:
        training.history['sparse_categorical_accuracy'].extend(result.history['sparse_categorical_accuracy'])
        training.history['val_sparse_categorical_accuracy'].extend(result.history['val_sparse_categorical_accuracy'])
        training.history['loss'].extend(result.history['loss'])
        training.history['val_loss'].extend(result.history['val_loss'])
    count += 1
    print("iteration", count)

visualize(training)

print(X_train.shape)
print(y_train.shape)
print((y_df-1).to_numpy())

if __name__ == '__main__':
    main()
