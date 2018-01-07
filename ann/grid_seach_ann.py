import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


def build_model(optimizer):
    model = Sequential()
    model.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11)) #input_dim = number of input variables
    model.add(Dense(output_dim=6, init='uniform', activation='relu'))
    model.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,-1].values

#Encode Country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#Encode Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

## K-fold Cross-Validation
#classifier = KerasClassifier(build_fn=build_model, batch_size=10, nb_epoch=50)
#accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)

## Parameter Tuning
classifier = KerasClassifier(build_fn=build_model)
parameters = {'batch_size': [25, 32],
              'epochs': [50, 100],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

#mean = accuracies.mean()
#variance = accuracies.std()

print("Best parameters={}, \nBest accuracy={}".format(best_parameters, best_accuracy))
