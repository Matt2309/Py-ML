import numpy as np
import scikitplot as skplt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

def randomize(v, lab, prob=0.2): #valori ed etichette
    v2 = []
    for el in v: #per ogni elemento in valori
        if np.random.random() > prob: #valori a caso tra 0 e 1 --> se maggiore
            v2.append(el)
        else:
            v2.append((np.random.choice(lab)))
    return v2

labels = ['cronaca', 'politica', 'sport']
y = np.random.choice(labels, 1000)
p = randomize(y, labels) #utilizzo quella funzione per fargli contenere nell'80% dei casi gli elementi di y e nel restante 20 un elemento a caso tra i 3

acc = accuracy_score(y, p)
print(acc)

report = classification_report(y, p)
print(report)

skplt.metrics.plot_confusion_matrix(y, p)
plt.show()