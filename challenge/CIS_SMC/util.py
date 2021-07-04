import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn import preprocessing, decomposition
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.backend as kb
import matplotlib.pyplot as plt
from collections import defaultdict
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

from sklearn.datasets import make_classification

# exact match ratio
#MR = np.all(y_pred == y_true, axis=1).mean()
# def exact_match_ratio_loss(y_actual, y_pred):
#     return y_atual != y_pred

def remove_correlations(df, threshold = 0.94):
    cordf = df.corr().abs()
    todrop = defaultdict(list)
    for i in range(cordf.shape[0]):
        for j in range(cordf.shape[1]):
            v = cordf.iat[i, j]
            if v > threshold and i != j and j not in todrop:
                todrop[i].append((j, v))
    todroplist = list(set([item[0] for k,v in todrop.items() for item in v]))
    df.drop(df.columns[todroplist], axis=1, inplace=True)
    print("remove_correlations - {} features remaining".format(len(df.columns)))
    return df

def remove_sparses(df, threshold = 0.89, col_number=1):
    sparse_columns = []
    for column in df:
        col = df[column]
        highest = 0
        for i in range(0,col_number):
            highest += col.value_counts().iat[i]
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

def remove_linears(df, reverse=False):
    # remove linear features
    for column in df:
        if 'Label' in column:
            continue
        col = df[column]
        if not reverse:
            if len(pd.unique(col)) > 1000:
                df.drop([column], axis=1, inplace=True)
        else:
            if len(pd.unique(col)) < 1000:
                df.drop([column], axis=1, inplace=True)
    print("remove_linears - {} features remaining".format(len(df.columns)))
    return df

def visualize(training):
    #Visulaizing the Training and Validation Sets Loss and Accuracy
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    #Plot training and validation accuracy values
    #axes[0].set_ylim(0,1) #if we want to limit axis in certain range
    axes[0].plot(training.history['accuracy'], label='Train')
    axes[0].plot(training.history['val_accuracy'], label='Validation')
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
def hamming_loss(y_true, y_pred):
    print("hamming_loss", y_true.shape, y_true)
    print("hamming_loss", y_pred.shape, y_pred)
    # kb.print_tensor(y_true)
    # loss = tfk.losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    y_pred_index = y_pred.argmax(axis=1)
    
    # scce = tfk.losses.SparseCategoricalCrossentropy()
    # # print(scce(y_true, y_pred).numpy())
    # return scce(y_true, y_pred)
    special_class = 6
    temp=0
    assert(len(y_true) == len(y_pred_index))
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            temp += 2
        elif y_true[i] == special_class and y_pred[i] != special_class:
            temp += 0
        else:
            temp += 1
    return temp/(len(y_true) * 2)

def Custom_Hamming_Loss(y_true, y_pred):
  return kb.mean(y_true*(1-y_pred)+(1-y_true)*y_pred)

def Custom_Hamming_Loss1(y_true, y_pred):
  tmp = kb.abs(y_true-y_pred)
  return kb.mean(K.cast(K.greater(tmp,0.5),dtype=float))

def hamming_loss_2(y_true, y_pred):
    print("y_pred", y_pred.shape, y_pred)
    print("y_true", y_true.shape, y_true)
    hamming_loss(y_true, y_pred)

def select_kbest(x,y, k=17):
    fs = SelectKBest(score_func=f_classif, k=k)
    X_selected = fs.fit_transform(x, y)
    print('select_kbest', X_selected.shape)
    return X_selected
    
def df_statistics(df):
    for column in df:
        col = df[column]
        print(column, '\n', col.value_counts())
def analyze_labels(y_df):
    y_df['Label'] = y_df.apply(lambda r: r['Label1'] * (-1 if r['Label2'] == 0 else 1), axis=1).astype(np.int64)    
    df_statistics(y_df)
    
def quantile(X_train):
    # NON-LINEAR Column-wise,  STANDARDIZATION in Column-wise
    # 'skewed/congested' or 'highly-spread' data to standard normal
    quantile_trans = preprocessing.QuantileTransformer(output_distribution='uniform', random_state=48)
    X_train_2 = quantile_trans.fit_transform(X_train)
    return X_train_2
	
#Training the model 
# sss = StratifiedShuffleSplit(n_splits=40, test_size=0.2, random_state=0)
# training = None
# count= 0
# for train_index, test_index in sss.split(x, (y_df-1).to_numpy()):
#     print("iteration", count)
#     # print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = x[train_index], x[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     result = model.fit(X_train, y_train, batch_size = 128, validation_data = (X_test, y_test))
#     if not training:
#         training = result
#     else:
#         training.history['sparse_categorical_accuracy'].extend(result.history['sparse_categorical_accuracy'])
#         training.history['val_sparse_categorical_accuracy'].extend(result.history['val_sparse_categorical_accuracy'])
#         training.history['loss'].extend(result.history['loss'])
#         training.history['val_loss'].extend(result.history['val_loss'])
#     count += 1



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