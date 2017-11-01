from tools.cleaner import *
from extract import extract_links

"""
The cleaning/extraction of the wikipedia articles takes some time (around 30mn for 2000 articles)

This file saves the cleaned Wikipedia articles in the specified folders

topic1 articles are saved in data/link1
topic2 articles are saved in data/link2
topic1 test articles are saved in data/test1
topic2 test articles are saved in data/test2
"""


def create_cleaned_files(topic1, topic2, test):
    topic1_links, topic2_links = extract_links(topic1, topic2)

    print("Writing %s articles in data/link1 folder" % topic1)

    for link in topic1_links:
        try:
            output_content = clean(wikipedia.page(link).content)
            cleaned_article = open("data/link1" + '/' + link + '.txt', 'w')
            cleaned_article.write(output_content)
            cleaned_article.close()
        except:
            pass

    print("Writing %s articles in data/link2 folder" % topic2)

    for link in topic2_links:

        try:
            output_content = clean(wikipedia.page(link).content)
            cleaned_article = open("data/link2" + '/' + link + '.txt', 'w')
            cleaned_article.write(output_content)
            cleaned_article.close()
        except:
            pass

    print("Writing %s  test file in data/test1 folder" % topic1)

    with open(test, "rb") as f:
        for link in f:
            if link.decode("utf8").strip()[-1] == "0":
                link = link.decode("utf8").strip()[:-1]
                try:
                    output_content = clean(wikipedia.page(link).content)
                    cleaned_article = open("data/test1" + '/' + link + '.txt', 'w')
                    cleaned_article.write(output_content)
                    cleaned_article.close()
                except:
                    pass

    print("Writing %s  test file in data/test2 folder" % topic2)

    with open(test, "rb") as f:
        for link in f:
            if link.decode("utf8").strip()[-1] == "1":
                link = link.decode("utf8").strip()[:-1]
                try:
                    output_content = clean(wikipedia.page(link).content)
                    cleaned_article = open("data/test2" + '/' + link + '.txt', 'w')
                    cleaned_article.write(output_content)
                    cleaned_article.close()
                except:
                    pass
