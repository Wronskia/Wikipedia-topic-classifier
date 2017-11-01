import re
from nltk.corpus import stopwords


def remove_stopwords(article):
    stopword_list = set(stopwords.words("english"))
    return [word for word in article.split() if word not in stopword_list]


def remove_character(article):
    return re.sub("[^a-zA-Z]+", " ", article)


def lower(string):
    return string.lower()


def clean(article):
    article = article.strip()
    article = lower(article)
    article = remove_character(article)
    article = remove_stopwords(article)
    article = ' '.join(article)
    return article
