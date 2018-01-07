import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout


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


model = Sequential()
model.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11)) #input_dim = number of input variables
model.add(Dropout(p=0.1))
model.add(Dense(output_dim=6, init='uniform', activation='relu'))
model.add(Dropout(p=0.1))
model.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=10, nb_epoch=50, verbose=1)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

#predicting a single observation
new_pred = model.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
print((new_pred > 0.5))

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
