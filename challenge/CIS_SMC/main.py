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
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

from sklearn.datasets import make_classification

# exact match ratio
#MR = np.all(y_pred == y_true, axis=1).mean()
# def exact_match_ratio_loss(y_actual, y_pred):
#     return y_atual != y_pred

def hamming_loss(y_true, y_pred):
    special_class = 6
    temp=0
    assert(y_true.shape[1] == 1)
    for i in range(y_true.shape[0]):
        if y_true[i][0] == y_pred[i][0]:
            temp += 2
        elif y_true[i][0] == special_class and y_pred[i][0] != special_class:
            temp += 0
        else:
            temp += 1
    return temp/(y_true.shape[0] * 2)

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
def select_kbest(x,y, k=17):
    fs = SelectKBest(score_func=f_classif, k=k)
    X_selected = fs.fit_transform(x, y)
    # print(X_selected.shape)
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
# df = remove_correlations(df)
# df = remove_sparses(df, col_number=2)

df = remove_lowcor_with_label(df, 'Label1')
# df = remove_linears(df, reverse=True)
# df = df.drop(['F10', 'F12', 'F13', 'F20', 'F27'], axis=1)
        
x_df = df.iloc[:, :-1].astype('float32')
y_df = df.iloc[:, -1:].astype('int32')

t = select_kbest(x_df, np.ravel(y_df))

# sc = preprocessing.StandardScaler()
# x = sc.fit_transform(t)
x = quantile(x)

normaliztn = preprocessing.Normalizer(norm='l2')
x = normaliztn.fit_transform(x)

# df_statistics(x_df)
# x = preprocessing.normalize(x_df, norm='l2', axis=1, copy=True, return_norm=False)

X_train, X_test, y_train, y_test = train_test_split(x, y_df-1, test_size=0.2, random_state=0)


def tuner_build_model(hp):
    model = tfk.Sequential()
    model.add(tfk.layers.Dense(
        hp.Choice("input_units", values=[x.shape[1]+1, x.shape[1]*2 + 1]), 
        input_shape=(x.shape[1],), 
        activation='relu')
        ) #First Hidden Layer
    model.add(tfk.layers.Dense(108, activation='relu')) #Second  Hidden Layer
    for i in range(hp.Int("num_layers", min_value=1, max_value=3, step=1)):
        model.add(tfk.layers.Dense(hp.Int(f"units_{i}", min_value=54, max_value=108, step=16), activation='relu'))
    # model.add(tfk.layers.Dense(54, activation='relu')) #Third  Hidden Layer
    model.add(tfk.layers.Dense(10, activation='softmax')) #Output Layer
    #model Looks like:  784 input -> [50 units in layer1] ->[100 units in layer2] -> 1 output
    
    model.compile(optimizer=tfk.optimizers.Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
                  # loss='sparse_categorical_crossentropy',
                  loss = hamming_loss,
                  metrics=['sparse_categorical_accuracy'])
    model.summary()
    return model

tuner = RandomSearch(
                        tuner_build_model, 
                        objective='sparse_categorical_accuracy', 
                        max_trials = 10,
                        executions_per_trial=2, # reduce variance.
                        )

tuner.search(X_train, y_train, epochs=2, validation_data=(X_test, y_test),
                        callbacks=[tfk.callbacks.EarlyStopping('val_loss', patience=3)])
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
