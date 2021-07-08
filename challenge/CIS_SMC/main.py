import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn import preprocessing, decomposition
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.backend as kb
import matplotlib.pyplot as plt
from collections import defaultdict
from kerastuner import RandomSearch, Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
from sklearn.metrics import hamming_loss
from datetime import datetime
from sklearn.datasets import make_classification
from tensorflow.keras.callbacks import History

# ensure to set current working directory to 'CIS_SMC/'
from util import *

def select_features(train_X, train_y):
    reg = ExtraTreesRegressor(n_estimators=100, random_state=0).fit(train_X, np.ravel(train_y))
    columns_to_drop = []
    for i in range(train_X.shape[1]):
        if reg.feature_importances_[i] < 0.01:
            columns_to_drop.append(train_X.columns[i])
    train_X = train_X.drop(columns_to_drop, axis=1)
    print("select_features by ExtraTreesRegressor - {} featuers selected".format(len(train_X.columns)))
    # res = reg.score(train_X, train_y)
    # print(res)
    return train_X

def read_dataset(filename, withLabel2=False):
    df = pd.read_csv(filename)
    if not withLabel2:
        df = df.drop(['Label2'], axis=1)
        x_df = df.iloc[:, :-1].astype('float32')
        y_df = df.iloc[:, -1:].astype('int32')
    else:
        x_df = df.iloc[:, :-2].astype('float32')
        y_df = df.iloc[:, -2:].astype('int32')
    y_df['Label1'] = y_df['Label1'] - 1
    # df = remove_sparses(df, col_number=2)
    # df = remove_lowcor_with_label(df, 'Label1')
    # df = remove_linears(df, reverse=True)
    # df = df.drop(['F10', 'F12', 'F13', 'F20', 'F27'], axis=1)
            
    return x_df, y_df

def hamming_accuracy(y_true_df, y_pred):
    y_true = y_true_df.to_numpy()
    return np.count_nonzero(y_true==y_pred) / y_true.size
    
def build_model(hp):
    model = tfk.Sequential()
    model.add(tfk.layers.Dense(
        units=hp.Choice("first_layer_units", values=[X_train.shape[1]+1, 2*(X_train.shape[1]+1), 3*(X_train.shape[1]+1)]),
        input_shape=(X_train.shape[1],), 
        activation='relu')
        )
    model.add(tfk.layers.Dropout(rate=hp.Choice("dropout_rate", values=[0.0, 0.2, 0.4])))
    for i in range(hp.Int("num_layers", min_value=2, max_value=6, step=2)):
        model.add(tfk.layers.Dense(hp.Int(f"units_{i}", min_value=32, max_value=108, step=16), 
                                   activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh'])))
    model.add(tfk.layers.Dense(108, activation='relu'))
    model.add(tfk.layers.Dense(10, activation='softmax'))
    
    model.compile(
                    optimizer=tfk.optimizers.Adam(learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
                    # loss='sparse_categorical_crossentropy',
                    loss=tfk.losses.SparseCategoricalCrossentropy(from_logits=hp.Choice("from_logits", values=[True, False])),
                    # loss='categorical_crossentropy',
                    metrics=['sparse_categorical_accuracy'],
                    # metrics=['categorical_accuracy']
                  )
    model.summary()
    return model

x_df, y_df = read_dataset("./Dataset/Train.csv")
x_test_df, y_test_df = read_dataset("./Dataset/Test.csv", withLabel2=True)

# df_statistics(y_df)
# df_statistics(y_test_df)

x_df = select_features(x_df, y_df)
x_df = remove_correlations(x_df)
x_test_df = x_test_df[x_df.columns]

sc = preprocessing.StandardScaler()
x = sc.fit_transform(x_df)
quantile_trans = preprocessing.QuantileTransformer(output_distribution='uniform', random_state=48)
x = quantile_trans.fit_transform(x)
normaliztn = preprocessing.Normalizer(norm='l2')
x = normaliztn.fit_transform(x)

x_test_df = sc.transform(x_test_df)
x_test_df = quantile_trans.transform(x_test_df)
x_test_df = normaliztn.transform(x_test_df)

# enc = preprocessing.OneHotEncoder()
# y_df = enc.fit_transform(y_df).toarray()

# df_statistics(x_df)

X_train, X_test, y_train, y_test = train_test_split(x, y_df, test_size=0.2, random_state=0)
tuner = Hyperband(
                    build_model,
                    objective='val_sparse_categorical_accuracy',
                    max_epochs = 20,
                    factor=3,
                    directory='tuner/{}'.format(datetime.now().timestamp()),
                    )

tuner.search(X_train, y_train, epochs=20, batch_size=64*2, validation_data=(X_test, y_test),
                        callbacks=[tfk.callbacks.EarlyStopping('val_loss', patience=5)])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
print(f"""
The hyperparameter search is complete. 
num_layers {best_hps.get('num_layers')}.
first_layer_units {best_hps.get('first_layer_units')}.
learning_rate is {best_hps.get('learning_rate')}.
from_logits {best_hps.get('from_logits')}
dropout_rate {best_hps.get('dropout_rate')}
""")
for i in range(best_hps.get('num_layers')):
    print(best_hps.get(f'units_{i}'), best_hps.get(f'activation_{i}'))


results = []
best_model = None
best_acc = 0.0
for i in range(10):
 
    model = tuner.hypermodel.build(best_hps)
    
    history = History()
    data_generator = shuffle_dataset(x, y_df, splits=2, test_size=0.2)
    for X_train, X_test, y_train, y_test in data_generator:
        model.fit(X_train, y_train, epochs=24, batch_size=64*2, validation_data=(X_test, y_test),
                            shuffle=True,
                            callbacks=[tfk.callbacks.EarlyStopping('val_loss', patience=5), history])
    visualize(history)
    
    test_loss, test_accuracy = model.evaluate(x_test_df, y_test_df['Label1'], verbose=2)
    print('Exact match loss and Accuracy: {0:.2f} {1:.2f}%'.format(test_loss, test_accuracy*100))
    
    y_pred_raw = model.predict(x_test_df)
    y_pred = np.argmax(y_pred_raw, axis=1)
    y_pred__with_label2 = np.array([[i,1 if i!=6 else 0] for i in y_pred])
    y_pred = y_pred__with_label2
    
    hamming_test_accuracy = hamming_accuracy(y_test_df, y_pred)
    hamming_test_loss = 1 - hamming_test_accuracy
    
    print('Hammigng loss and Accuracy {0:.2f} {1:.2f}%'.format(hamming_test_loss, hamming_test_accuracy*100))
    
    results.append((test_loss, test_accuracy, hamming_test_loss, hamming_test_accuracy))
    if hamming_test_accuracy > best_acc:
        best_model = model
        best_acc = hamming_test_accuracy
    
arr = np.array(results)
std_variation = np.std(arr, axis=0)
mean = np.mean(arr, axis=0)

print(results)
print("std_variation", std_variation)
print("mean", mean)

y_pred_raw = best_model.predict(x_test_df)
y_pred = np.argmax(y_pred_raw, axis=1)
y_pred__with_label2 = np.array([[i,1 if i!=6 else 0] for i in y_pred])
y_pred = y_pred__with_label2
hamming_test_accuracy = hamming_accuracy(y_test_df, y_pred)
y_pred_df = pd.DataFrame(y_pred, columns=['Label1', 'Label2'])
y_pred_df['Label1'] = y_pred_df['Label1'] + 1
y_pred_df.to_csv("predictions.csv", sep=",")


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

