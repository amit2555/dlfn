import os
import glob
import pandas as pd
import numpy as np
from functools import partial
from keras.models import Sequential
from keras.layers import Flatten, Conv1D, MaxPooling1D, Dropout, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


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

def main():
    activation = 'relu'
    df = preprocess('Dataset')
    df['data'] = df['data'].apply(pad_and_convert)
    num_classes = len(LABELS)
    X_train, X_test, y_train, y_test = train_test_split(df['data'], df['label'],
                                                        test_size=0.3, random_state=4)
    X_train = X_train.apply(pd.Series)
    X_test = X_test.apply(pd.Series)
    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv1D(512, strides=2, input_shape=X_train.shape[1:], activation=activation, kernel_size=3, padding='same'))
    model.add(MaxPooling1D())
    model.add(Conv1D(256, strides=2, activation=activation, kernel_size=3, padding='same'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    result = model.fit(X_train, y_train, verbose=1, epochs=50, batch_size=32, validation_data=(X_test, y_test))

if __name__ == '__main__':
    main()
