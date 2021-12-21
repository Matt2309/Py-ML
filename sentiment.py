import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  # separare il dataset x validazione
from sklearn.naive_bayes import BernoulliNB  # algoritmo di addestramento
from sklearn.metrics import accuracy_score  # test di accuratezza


df = pd.read_csv('movie_review.csv')

print(df.head())

X = df['text']
y = df['tag']

vectorizer = CountVectorizer(ngram_range=(1,2)) # considera anche frequenze di due parole
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=40)
model = BernoulliNB()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)


accuracy_train = accuracy_score(y_train, p_train)
accuracy_test = accuracy_score(y_test, p_test)

print(f'Train {accuracy_train} Test {accuracy_test}')

