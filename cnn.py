import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from tools.input_matrix import embedding_inputs
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report


def run_cnn(pretrained=False, embedding_dims=50, val_split=0.2, max_n_words=20000, maxlen=1000, ngram_range=1):
    nb_epoch = 6
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

    print('Build cnn model...')
    model = Sequential()
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    if pretrained:
        model.add(Embedding(max_n_words + 1,
                            embedding_dims,
                            input_length=maxlen, weights=[embedding_matrix]))
    else:
        model.add(Embedding(max_n_words,
                            embedding_dims,
                            input_length=maxlen))

    model.add(Convolution1D(filters=55, kernel_size=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, \
                              verbose=1, mode='auto')

    checkpoint = ModelCheckpoint("ckpt/file.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=nb_epoch,
                        validation_data=(x_val, y_val), callbacks=[earlystop, checkpoint])

    y_pred = model.predict_classes(x_val)
    y_pred = to_categorical(y_pred)

    print("\n")
    print(classification_report(y_val, y_pred))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy for cnn')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # Here you can choose the name of the image containing the accuracy evolution curve per epoch
    plt.savefig('acc_cnn.png')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss for cnn')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # Here you can choose the name of the image containing the loss evolution curve per epoch
    plt.savefig('loss_cnn.png')
    plt.show()

    return model.predict_proba(np.concatenate((x_train, x_val), axis=0))[:, 1], model.predict_proba(x_test)[:, 1]
