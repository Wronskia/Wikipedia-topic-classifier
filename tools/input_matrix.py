import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import glob

'''
Creates the inputs of the TF-IDF model.

'''


def tfidf_inputs(val_split, max_n_words):
    topic = []
    y = []

    for link in glob.glob("data/link1/*.txt"):
        try:
            topic.append(open(link, 'r').read())
            y.append(0)
        except:
            print("(tf-idf features)exception occured in %s for link1 folder"% link)
            pass
        if len(topic) == 1000:
            break
    for link in glob.glob("data/link2/*.txt"):

        try:
            topic.append(open(link, 'r').read())
            y.append(1)
            if len(topic) == 2000:
                break
        except:
            print("(tf-idf features)exception occured in %s for link2 folder"% link)
            pass

    print('Found %s articles for the train/val' % len(topic))

    x_test = []
    y_test = []
    for link in glob.glob("data/test1/*.txt"):
        try:
            x_test.append(open(link, 'r').read())
            y_test.append(0)
        except:
            print("(tf-idf features)exception occured in %s for test1 folder"% link)
            pass

    for link in glob.glob("data/test2/*.txt"):
        try:
            x_test.append(open(link, 'r').read())
            y_test.append(1)
        except:
            print("(tf-idf features)exception occured in %s for test2 folder"% link)
            pass

    print('Found %s articles for the test set' % len(x_test))

    tokenizer = Tokenizer(num_words=max_n_words, filters='', lower=False)
    tokenizer.fit_on_texts(topic)
    # TODO : Maybe add the possibility to add n-grams to the tf-idf matrix
    data = tokenizer.texts_to_matrix(topic, mode='tfidf')
    x_test1 = tokenizer.texts_to_matrix(x_test, mode='tfidf')

    y = np.asarray(y)

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', y.shape)
    print('Shape of test data tensor:', x_test1.shape)


    print("\n"+"Shuffling ...")

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    y = y[indices]

    nb_validation_samples = int(val_split * data.shape[0])
    x_train = data[:-nb_validation_samples]
    y_train = y[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = y[-nb_validation_samples:]

    return x_train, y_train, x_val, y_val, x_test1, y_test


# The functions below were inspired from keras documentation


def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def embedding_inputs(ngram_range=1, pretrained=False, embedding_dims=50, val_split=0.1, max_n_words=20000, maxlen=5000):
    if pretrained:
        embeddings_index = {}
        # use of wikipedia glove pretraining
        f = open('tools/glove.6B/glove.6B.' + str(embedding_dims) + 'd.txt', 'rb')
        print('Indexing word vectors.')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word.decode("utf8")] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))

    topic = []
    y = []

    for link in glob.glob("data/link1/*.txt"):
        try:
            topic.append(open(link, 'r').read())
            y.append(0)
        except:
            print("(embedding features) exception occured in %s for link1 folder"% link)
            pass
        if len(topic) == 1000:
            break
    print("Finished loading first article")
    for link in glob.glob("data/link2/*.txt"):

        try:

            topic.append(open(link, 'r').read())
            y.append(1)
            if len(topic) == 2000:
                break
        except:
            print("(embedding features) exception occured in %s for link2 folder"% link)
            pass

    x_test = []
    y_test = []
    for link in glob.glob("data/test1/*.txt"):
        try:
            x_test.append(open(link, 'r').read())
            y_test.append(0)
        except:
            print("(embedding features)exception occured in %s for test1 folder"% link)
            pass

    for link in glob.glob("data/test2/*.txt"):
        try:
            x_test.append(open(link, 'r').read())
            y_test.append(1)
        except:
            print("(embedding features) exception occured in %s for test2 folder"% link)
            pass

    print("Finished loading second article")

    tokenizer = Tokenizer(num_words=max_n_words)
    tokenizer.fit_on_texts(topic)
    sequences = tokenizer.texts_to_sequences(topic)
    sequences_test = tokenizer.texts_to_sequences(x_test)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    if not pretrained and ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        ngram_set = set()
        for input_list in sequences:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = max_n_words + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.

        max_n_words = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        sequences = add_ngram(sequences, token_indice, ngram_range)
        sequences_test = add_ngram(sequences_test, token_indice, ngram_range)
        print('Average train sequence length: {}'.format(np.mean(list(map(len, sequences)), dtype=int)))

    data = pad_sequences(sequences, maxlen=maxlen)
    x_test = pad_sequences(sequences_test, maxlen=maxlen)

    y = to_categorical(np.asarray(y))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', y.shape)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    y = y[indices]
    print(len(y))
    nb_validation_samples = int(val_split * data.shape[0])
    x_train = data[:-nb_validation_samples]
    y_train = y[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = y[-nb_validation_samples:]
    print('Preparing embedding matrix.')
    if pretrained:
        # use wikipedia glove pretraining
        num_words = min(max_n_words, len(word_index))
        embedding_matrix = np.zeros((num_words + 1, embedding_dims))
        for word, i in word_index.items():
            if i > max_n_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    if pretrained:
        return x_test, y_test, x_train, y_train, x_val, y_val, embedding_matrix
    return x_test, y_test, x_train, y_train, x_val, y_val, max_n_words
