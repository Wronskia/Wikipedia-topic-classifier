import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tools.input_matrix import embedding_inputs
from sklearn.metrics import classification_report
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt


def run_fasttext(pretrained=True, embedding_dims=50, val_split=0.2, max_n_words=20000, maxlen=1000, ngram_range=1):
    epochs = 50
    batch_size = 60

    if pretrained:
        x_test, y_test, x_train, y_train, x_val, y_val, embedding_matrix = embedding_inputs(ngram_range=ngram_range,
                                                                                            pretrained=pretrained,
                                                                                            embedding_dims=embedding_dims,
                                                                                            val_split=val_split,
                                                                                            max_n_words=max_n_words,
                                                                                            maxlen=maxlen)
    else:
        x_test, y_test, x_train, y_train, x_val, y_val, max_n_words = embedding_inputs(ngram_range=ngram_range,
                                                                                       pretrained=pretrained,
                                                                                       embedding_dims=embedding_dims,
                                                                                       val_split=val_split,
                                                                                       max_n_words=max_n_words,
                                                                                       maxlen=maxlen)

    print('Build fasttext model...')
    model = Sequential()
    # mapping vocabulary to embedding_dims dimensions
    if pretrained:
        model.add(Embedding(max_n_words + 1,
                            embedding_dims,
                            input_length=maxlen, weights=[embedding_matrix]))
    else:
        model.add(Embedding(max_n_words,
                            embedding_dims,
                            input_length=maxlen))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy fasttext')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('acc_fasttext.png')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss fasttext ')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('loss_fasttext.png')
    plt.show()

    y_pred = model.predict_classes(x_val)
    y_pred = to_categorical(y_pred)

    print("\n")
    print(classification_report(y_val, y_pred))

    return model.predict_proba(np.concatenate((x_train, x_val), axis=0))[:, 1], model.predict_proba(x_test)[:, 1]
