# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
training_set = pd.read_excel("Final_Train.xlsx")
test_set = pd.read_excel("Final_Test.xlsx")
training_set=training_set.dropna()
X_train=training_set.iloc[:,0:6].values
Y_train = training_set.iloc[:,-1].values 
x=pd.DataFrame(X_train)
X_test=test_set.iloc[:,0:6].values
from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
le2 = LabelEncoder()

X_train[:,0] = le1.fit_transform(X_train[:,0].astype(str))

X_train[:,1] = le1.fit_transform(X_train[:,1].astype(str))

X_train[:,2] = le1.fit_transform(X_train[:,2].astype(str))

X_train[:,3] = le1.fit_transform(X_train[:,3].astype(str))

X_train[:,4] = le1.fit_transform(X_train[:,4].astype(str))

X_train[:,5] = le1.fit_transform(X_train[:,5].astype(str))

X_test[:,0] = le2.fit_transform(X_test[:,0].astype(str))

X_test[:,1] = le2.fit_transform(X_test[:,1].astype(str))

X_test[:,2] = le2.fit_transform(X_test[:,2].astype(str))

X_test[:,3] = le2.fit_transform(X_test[:,3].astype(str))

X_test[:,4] = le2.fit_transform(X_test[:,4].astype(str))

X_test[:,5] = le2.fit_transform(X_test[:,5].astype(str))
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
Y_train = Y_train.reshape((len(Y_train), 1)) 

Y_train = sc_X.fit_transform(Y_train)

Y_train = Y_train.ravel()
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100,criterion="mse")

regressor.fit(X_train,Y_train)

Y_pred = sc_X.inverse_transform(regressor.predict(X_test))
Yp=pd.DataFrame(Y_pred)

pd.DataFrame(Y_pred, columns = ['Fees']).to_excel("predictions_tree1.xlsx", index = False)
