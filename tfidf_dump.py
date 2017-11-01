from tools.input_matrix import tfidf_inputs
import _pickle as cPickle

'''
This file helps to dump tf_idf model inputs
'''


def dump_tfidf(category, val_split=0.1, max_n_words=20000):
    # Here you put the configuration you want to dump
    x_train, y_train, x_val, y_val, x_test, y_test = tfidf_inputs(val_split, max_n_words)

    # Here you dump it choosing the name and the objects to dump
    cPickle.dump([x_train, y_train, x_val, y_val, x_test, y_test],
                 open('dumps/tfidf' + category
                      + '.dat', 'wb'))
