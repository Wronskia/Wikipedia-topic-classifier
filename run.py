import argparse
from tfidf_dump import dump_tfidf
from svm import run_svm
from logreg import run_logreg
from bayes import run_bayes
from dump_cleaned_files import create_cleaned_files
from cnn import run_cnn
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier as xgb
from fasttext import run_fasttext
import numpy as np


def print_bold(string):
    print("\033[1m" + string + "\033[0m")


def execute(topic1, topic2, test, dump_files):
    if dump_files == "True":
        print_bold("\n"+"Downloading the datasets ..."+"\n")
        create_cleaned_files(topic1, topic2, test)

    print_bold("Dumps TFIDF features ..."+"\n")

    # category is used to specify the unique Id of the dumped model
    category = topic1 + "-" + topic2
    dump_tfidf(category)

    print("=========================================================")
    print_bold("Start Running bayes model to establish a baseline")
    print("=========================================================")


    print_bold("\n" + "Run Bayes model ..." + "\n")

    pred_train_bayes, pred_test_bayes = run_bayes(category)

    print("=========================================================")
    print_bold("Improvement of the baseline")
    print("=========================================================")


    print_bold("Run Cnn model ..."+"\n")

    pred_train_cnn, pred_test_cnn = run_cnn()

    print("--------------------------------------------------------------------------")
    print_bold("Run Fasttext model ..."+"\n")
    pred_train_fasttext, pred_test_fasttext = run_fasttext()

    print("--------------------------------------------------------------------------")

    print_bold("Run SVM model ..."+"\n")

    pred_train_svm, pred_test_svm, y_train = run_svm(category)


    print("--------------------------------------------------------------------------")

    print_bold("Run Logistic Regression model ..."+"\n")

    pred_train_logreg, pred_test_logreg, y_test = run_logreg(category)

    print("--------------------------------------------------------------------------")

    print_bold("Starting Ensemble Method")

    # using train+val for training the ensemble (training on more dataset == stronger results)
    train = np.column_stack((pred_train_svm, pred_train_logreg, pred_train_cnn, pred_train_fasttext))
    test = np.column_stack((pred_test_svm, pred_test_logreg, pred_test_cnn, pred_test_fasttext))
    model = xgb().fit(train, y_train)

    print("--------------------------------------------------------------------------")
    print_bold("Final results on the test set : ")
    print(classification_report(y_test, model.predict(test)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('topic1', help='The first topic')
    parser.add_argument('topic2', help='The second topic')
    parser.add_argument('test', help='test file')
    parser.add_argument('dump_files', help='if False will not save the cleaned dataset')
    args = parser.parse_args()
    execute(args.topic1, args.topic2, args.test, args.dump_files)


if __name__ == '__main__':
    main()
