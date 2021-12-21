from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
dataset = load_iris()

X = dataset['data']
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5) # riempie le 4 variabili (separa data in test e train)

model = DecisionTreeClassifier() # albero di decisione
model.fit(X_train, y_train) # addestro il modello (model) solo con i dati train

predizione_train = model.predict(X_train)
predizione_test = model.predict(X_test) # y predetta

accuracy_train = accuracy_score(y_train, predizione_train)
accuracy_test = accuracy_score(y_test, predizione_test)

print(f'Train {accuracy_train} Test {accuracy_test}')

plot_confusion_matrix(y_test,predizione_test)
plt.show()