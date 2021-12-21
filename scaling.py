from sklearn.datasets import load_wine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
dataset = load_wine()


X = dataset['data']
y = dataset['target']

model = KNeighborsClassifier()
model.fit(X, y)
p = model.predict(X)

accuracyNotScaled = accuracy_score(y, p)

print(accuracyNotScaled)

X = dataset['data']
y = dataset['target']


scaler = MinMaxScaler()
X = scaler.fit_transform(X)

model2 = KNeighborsClassifier()
model2.fit(X, y)
p = model2.predict(X)

accuracyScaled = accuracy_score(y, p)

print(accuracyScaled)