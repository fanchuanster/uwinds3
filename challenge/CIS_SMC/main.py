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
sys.path.append('C:/Users/donwen.CORPDOM/WS/wen/t3/challenge/CIS_SMC')
from util import *

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

# x = select_kbest(x, np.ravel(y_df), k='all')

sc = preprocessing.StandardScaler()
x = sc.fit_transform(x_df)
x = quantile(x)
normaliztn = preprocessing.Normalizer(norm='l2')
x = normaliztn.fit_transform(x)

# enc = preprocessing.OneHotEncoder()
# y_df = enc.fit_transform(y_df).toarray()

# print(type(onehot_y))
# np.shape(onehot_y)
# print(onehot_y)



# df_statistics(x_df)
# x = preprocessing.normalize(x_df, norm='l2', axis=1, copy=True, return_norm=False)

X_train, X_test, y_train, y_test = train_test_split(x, y_df, test_size=0.2, random_state=0)

def build_model(hp):
    model = tfk.Sequential()
    model.add(tfk.layers.Dense(
        X_train.shape[1]+1,
        input_shape=(X_train.shape[1],), 
        activation='relu')
        )
    model.add(tfk.layers.Dense(108, activation='relu'))
    model.add(tfk.layers.Dense(108, activation='relu'))
    model.add(tfk.layers.Dense(10, activation='softmax'))
    
    # optimizer=tfk.optimizers.Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
    model.compile(
                    optimizer=tfk.optimizers.Adam(learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
                    loss='sparse_categorical_crossentropy',
                    # loss='categorical_crossentropy',
                    metrics=['sparse_categorical_accuracy'],
                    # metrics=['categorical_accuracy']
                  )
    model.summary()
    return model

# model = build_model(None)
# training = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), callbacks=[tfk.callbacks.EarlyStopping('val_loss', patience=2)])
# visualize(training)

tuner = RandomSearch(
                        build_model,
                        objective='sparse_categorical_accuracy',
                        max_trials = 10,
                        executions_per_trial=1, # reduce variance.
                        )

tuner.search(X_train, y_train, epochs=2, batch_size=64, validation_data=(X_test, y_test),
                        )
best_model = tuner.get_best_models()[0]
best_model.summary()


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
