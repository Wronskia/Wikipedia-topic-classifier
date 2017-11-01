import wikipedia
import random


# TODO: FIND A BETTER METHOD TO EXTRACT 1000 RELEVENT ARTICLES
# TODO: Have a look at https://github.com/chrisjmccormick/wiki-sim-search which uses gensim library

def extract_links(topic1, topic2):

    try:
        top1 = wikipedia.page(topic1)
        top2 = wikipedia.page(topic2)
    except:
        raise Exception('Please provide two wikipedia articles')

    topic1_links = []
    topic2_links = []

    # Method : look for list links like "List of all sports" and extract links from those
    for link in top1.links:
        if 'list' in link.lower():
            topic1_links.append(wikipedia.page(link).links)

    print("%s links downloaded" % topic1)

    for link in top2.links:
        if 'list' in link.lower():
            topic2_links.append(wikipedia.page(link).links)

    print("%s links downloaded" % topic2)

    flat_top1_links = [link for group in topic1_links for link in group]
    flat_top2_links = [link for group in topic2_links for link in group]

    try:
        sample_top1_links = random.sample(flat_top1_links, 1200)
        sample_top2_links = random.sample(flat_top2_links, 1200)
    except:
        raise Exception('Could not find enough articles')

    return sample_top1_links, sample_top2_links
