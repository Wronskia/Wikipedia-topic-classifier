from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import _pickle as cPickle
from sklearn.metrics import classification_report
import numpy as np
from matplotlib import pyplot


def run_logreg(category):
    print('Build Logistic Regression Model...')
    [x_train, y_train, x_val, y_val, x_test, y_test] = cPickle.load(open("dumps/tfidf" + category + ".dat", "rb"))
    logreg = LogisticRegression(C=0.001)
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_train)
    acc = accuracy_score(y_train, y_pred)
    y_pred = logreg.predict(x_val)
    val_acc = accuracy_score(y_val, y_pred)

    print('acc :', acc)
    print('val acc :', val_acc)
    print(classification_report(y_val, y_pred))

    # You can tune the parameters of the model using the commented code below

    '''
    Cs = np.linspace(0.00001,0.1, 10)
    val_scores = list()
    scores = list()
    for C in Cs:
        logreg = LogisticRegression(C=C)
        print('C =',C)
        logreg.fit(x_train, y_train)
        y_pred = logreg.predict(x_val)
        val_acc = accuracy_score(y_val,y_pred)
        val_scores.append(val_acc)
        print('val acc :', val_acc)
        y_pred = logreg.predict(x_train)
        acc = accuracy_score(y_train,y_pred)
        scores.append(acc)
        print('acc :', acc)
    x_axis = Cs
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, scores, label='Train')
    ax.plot(x_axis, val_scores, label='Test')
    ax.legend()
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Inverse of regularization strength')
    pyplot.title('Logistic Regression Classification Accuracy')
    pyplot.savefig('acc.png')
    pyplot.show()
    '''
    return logreg.predict_proba(np.concatenate((x_train, x_val), axis=0))[:, 1], logreg.predict_proba(x_test)[:, 1], \
           y_test
