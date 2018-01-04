import os
import glob
import pandas as pd
import numpy as np
from functools import partial
from keras.models import Sequential
from keras.layers import Flatten, Conv1D, MaxPooling1D, Dropout, Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


LABELS = {}
counter = iter(range(20))

def pad_and_convert(s):
    """Collect 1000 bytes from packet payload. If payload length is less than
    1000 bytes, pad zeroes at the end. Then convert to integers and normalize."""
    if len(s) < 2000:
        s += '00' * (2000-len(s))
    else:
        s = s[:2000]
    return [float(int(s[i]+s[i+1], 16)/255) for i in range(0, 2000, 2)]

def read_file(f, label):
    df = pd.read_csv(f, index_col=None, header=0)
    df.columns = ['data']
    df['label'] = label
    return df

def preprocess(path):
    files = glob.glob(os.path.join(path, '*.txt'))
    list_ = []
    for f in files:
        label = f.split('/')[-1].split('.')[0]
        LABELS[label] = next(counter)
        labelled_df = partial(read_file, label=LABELS[label])
        list_.append(labelled_df(f))
    df = pd.concat(list_, ignore_index=True)
    return df

def build_model(dropout_rate=0.01, optimizer='adam'):
    activation = 'relu'
    num_classes = len(LABELS)
    model = Sequential()
    model.add(Conv1D(512, strides=2, input_shape=(1000, 1), activation=activation, kernel_size=3, padding='same'))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(256, strides=2, activation=activation, kernel_size=3, padding='same'))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def main():
    df = preprocess(path='Dataset')
    df['data'] = df['data'].apply(pad_and_convert)
    X_train, X_test, y_train, y_test = train_test_split(df['data'], df['label'],
                                                        test_size=0.3, random_state=4)
    X_train = X_train.apply(pd.Series)
    X_test = X_test.apply(pd.Series)
    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    num_classes = len(LABELS)
    #y_train = to_categorical(y_train, num_classes)
    #y_test = to_categorical(y_test, num_classes)
    #print(y_train.shape)

    classifier = KerasClassifier(build_fn=build_model)
    parameters = {'batch_size': [32, 64, 128],
                  'epochs': [50, 100],
                  'optimizer': ['adam', 'rmsprop', 'adadelta'],
                  'dropout_rate': [0.1, 0.25, 0.5]}
    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=parameters,
                               scoring='accuracy',
                               cv=3)

    grid_result = grid_search.fit(X_train, y_train)
    best_parameters = grid_result.best_params_
    best_accuracy = grid_result.best_score_

    print('Best parameters are: {}\nBest score is: {}'.format(best_params_, best_score_))

if __name__ == '__main__':
    main()
