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
from sklearn.metrics import hamming_loss

from sklearn.datasets import make_classification

import sys
sys.path.append('C:/Users/donwen.CORPDOM\WS/wen/uwinds3/challenge/CIS_SMC')
from util import *

# exact match ratio
#MR = np.all(y_pred == y_true, axis=1).mean()
# def exact_match_ratio_loss(y_actual, y_pred):
#     return y_atual != y_pred

    
# https://machinelearningmastery.com/chi-squared-test-for-machine-learning/
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

df = pd.read_csv("Train.csv")
df = df.drop(['Label2'], axis=1)
df = remove_correlations(df)
# df = remove_sparses(df, col_number=2)
# df = remove_lowcor_with_label(df, 'Label1')
# df = remove_linears(df, reverse=True)
# df = df.drop(['F10', 'F12', 'F13', 'F20', 'F27'], axis=1)
        
x_df = df.iloc[:, :-1].astype('float32')
y_df = df.iloc[:, -1:].astype('int32')
y_df = y_df - 1

x = select_kbest(x, np.ravel(y_df), k='all')

# t = select_kbest(x_df, np.ravel(y_df))
sc = preprocessing.StandardScaler()
x = sc.fit_transform(x_df)
x = quantile(x)


enc = preprocessing.OneHotEncoder()
onehot_y = enc.fit_transform(y_df).toarray()



# print(type(onehot_y))
# np.shape(onehot_y)
# print(onehot_y)

# normaliztn = preprocessing.Normalizer(norm='l2')
# x = normaliztn.fit_transform(x)

# df_statistics(x_df)
# x = preprocessing.normalize(x_df, norm='l2', axis=1, copy=True, return_norm=False)

X_train, X_test, y_train, y_test = train_test_split(x, onehot_y, test_size=0.2, random_state=0)

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

def build_model(hp):
    model = tfk.Sequential()
    model.add(tfk.layers.Dense(
        X_train.shape[1]+1,
        input_shape=(X_train.shape[1],), 
        activation='relu')
        )
    model.add(tfk.layers.Dense(108, activation='relu'))
    model.add(tfk.layers.Dense(10, activation='softmax'))
    
    # optimizer=tfk.optimizers.Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
    model.compile(
                    optimizer='adam',
                    # loss='sparse_categorical_crossentropy',
                    # loss='categorical_crossentropy',
                    loss = hamming_loss_2,
                    # metrics=['sparse_categorical_accuracy'],
                    metrics=['categorical_accuracy'],
                    # run_eagerly = True
                  )
    model.summary()
    return model

model = build_model(None)
training = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), callbacks=[tfk.callbacks.EarlyStopping('val_loss', patience=3)])
# visualize(training)


                        # callbacks=[tfk.callbacks.EarlyStopping('val_loss', patience=2)]



# tuner = RandomSearch(
#                         tuner_build_model,
#                         objective='sparse_categorical_accuracy',
#                         max_trials = 1,
#                         executions_per_trial=1, # reduce variance.
#                         )

# tuner.search(X_train, y_train, epochs=2, batch_size=16, validation_data=(X_test, y_test),
#                         )
# best_model = tuner.get_best_models()[0]
# best_model.summary()


# 1. try without reducing - 0.8072, 0.8094, 0.8058
# 2. remove correlations - 0.7999, 0.8005, 0.7995
# 3. remove sparse - 0.8067, 0.8055, 0.8048 ; col_number=2 - 0.8045, 0.8065
# 4. remove linear - 0.7872, 0.7859, 0.7860; reverse - 
# 2. apply pca  - dimensions reduction
# 3. switch from standardization to normalization
# 4. more layers, more nodes

# selectkbest - 0.7395, 0.7395

# tanh activation - 0.7989, 0.8007; relu - 0.8028

# batch size - small size 0.8114, 0.8105(normalize rows)

# remvoe correlations, sparse, lowcor_with_label - 0.7912, 0.7907

# 5. try activation different
# 6. try more epochs


# visualize(training)
