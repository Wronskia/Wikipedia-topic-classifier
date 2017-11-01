from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import _pickle as cPickle
from sklearn.metrics import classification_report
import numpy as np

def run_bayes(category):
    print('Build Bayes model...')
    [x_train, y_train, x_val, y_val, x_test, y_test] = cPickle.load(open("dumps/tfidf" + category + ".dat", "rb"))
    NB = GaussianNB()
    NB.fit(x_train, y_train)
    y_pred = NB.predict(x_train)
    acc = accuracy_score(y_train, y_pred)
    y_pred = NB.predict(x_val)
    val_acc = accuracy_score(y_val, y_pred)

    print('acc :', acc)
    print('val acc :', val_acc)
    print(classification_report(y_val, y_pred))

    print("------------------------------")
    print("baseline results on test set: ")
    print("------------------------------")

    print(classification_report(y_test, NB.predict(x_test)))

    return NB.predict_proba(np.concatenate((x_train, x_val), axis=0))[:, 1], NB.predict_proba(x_test)[:, 1]
