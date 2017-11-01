from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import _pickle as cPickle
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from matplotlib import pyplot


def run_svm(category):
    print('Build SVM model...')
    [x_train, y_train, x_val, y_val, x_test, y_test] = cPickle.load(open("dumps/tfidf" + category + ".dat", "rb"))
    svm = LinearSVC(C=0.0001)
    svm = CalibratedClassifierCV(svm)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_train)
    acc = accuracy_score(y_train, y_pred)
    y_pred = svm.predict(x_val)
    val_acc = accuracy_score(y_val, y_pred)

    print('acc :', acc)
    print('val acc :', val_acc)
    print(classification_report(y_val, y_pred))

    # The commented code bellow was used for tuning the parameter
    # you can see the plot on the folder plots

    '''
    Cs = np.linspace(0.0001,0.001, 50)
    max_val = 0
    max_C = 0
    val_scores = list()
    scores = list()
    for C in Cs:
        SVM = LinearSVC(C=C)
        print('C =',C)
        SVM.fit(x_train, y_train)
        y_pred = SVM.predict(x_val)
        val_acc = accuracy_score(y_val,y_pred)
        val_scores.append(val_acc)
        print('val acc :', val_acc)
        y_pred = SVM.predict(x_train)
        acc = accuracy_score(y_train,y_pred)
        scores.append(acc)
        print('acc :', acc)
        if val_acc > max_val:
            max_val = val_acc
            max_C = C
    print('Best C : ', max_C)
    print('Best val_acc : ',max_val)
    x_axis = Cs
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, scores, label='Train')
    ax.plot(x_axis, val_scores, label='Test')
    ax.legend()
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Penalty parameter C of the error term')
    pyplot.title('Linear SVC Classification Accuracy')
    pyplot.savefig('acc.png')
    pyplot.show()
    '''
    return svm.predict_proba(np.concatenate((x_train, x_val), axis=0))[:, 1], \
           svm.predict_proba(x_test)[:, 1], \
           np.concatenate((y_train, y_val), axis=0)
