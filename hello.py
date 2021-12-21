from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

dataset = load_boston()

X = dataset['data']
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y) # riempie le 4 variabili (separa data in test e train)

model = LinearRegression() # istanzio model dalla classe LinearRegression
model.fit(X_train, y_train) # addestro il modello (model) solo con i dati train

predizione_train = model.predict(X_train)
predizione_test = model.predict(X_test) # y predetta

mea_test = mean_absolute_error(y_test, predizione_test) # errore medio della predizione (differenza tra y vera e y predetta)
mea_train = mean_absolute_error(y_train, predizione_train)
print(mea_test)
print(mea_train)