# Wikipedia article classifier

## Description

This tool performs binary classification among wikipedia articles. Given 2 article categories defined by the user, this tool extracts 1000 wikipedia articles of each category and train classifiers based on those. The extraction method presented in the code is not very good and one should rather use some more sophisticated technique to extract similar articles to the ones given by the user (for example : https://github.com/chrisjmccormick/wiki-sim-search which uses gensim library) or by running a frequency analysis by analyzing google search results or by using directly the Wiki API. Even with this poor technique, the model score quite good (94% on test set) Here, to keep this part simple, I decided to use the basic package Wikipedia https://pypi.python.org/pypi/wikipedia. Then the tool performs classifications using different algorithms and different methods to extract features. The idea is to bring different information to the ensemble method.
## Requirements

- Run `pip -r install requirements.txt`
## Features

- This tool supports :  tfidf features, pretrained word embedding features 1 and 2 grams
- For the pretrain word embedding, I used wikipedia glove : https://nlp.stanford.edu/projects/glove/ to be put in tools/glove.6B/

## Example use case

- choose two subjects for (example : Sports and Computer_Science) and first run the following

`python3 run.py Sports Computer_Science test.txt True`

- run.py takes 4 arguments :
1/  The first topic
2/  The second topic
3/  Test file defined by the user following the format of the test.txt provided here for the running example
4/  dump_files : if False will not save the cleaned dataset
                            
Actually, first run the script with dump_files = True since it takes a bit of time to download all 2000 articles and clean them. it will store the articles in data/link1 and data/link2, data/test1 and data/test2

- Since one might overfit to the validation set by tuning the parameters, all the articles in test.txt (200 articles) are not used in the train/val and comes from another distribution (I used a different method to extract them)

- In case you use the files I provide in the folder data/ please run the following `python3 run.py Sports Computer_Science test.txt False`. This will directly start running one model after the other and finally outputing the final result on the test set.



## Tuning Parameters
- for both ML models I commented the code to tune parameters and provide the graph but you can find some graphs on the plots folder
- for deep neural net models I kept the verbosity and the figures which are generated when running the run.py
- once you saved the datasets, you can play with those by tuning depending on the topics chosen.

## Running example on Computer Science and Sports

- First please unzip data.zip
- Download pretrain glove embedding https://nlp.stanford.edu/projects/glove/  and put in tools/glove.6B/
- then run  `python3 run.py Sports Computer_Science test.txt False`

## Models and Results

here are the results of the models on train/val :

| Models       | Accuracy           | Validation Acc |
| -------------|:------------------:|:-------------------:|
| Svm          | 0.998              | 0.985               |
| Logistic Reg | 0.998              | 0.99                |
| Bayes        | 0.977              | 0.93                |
| Fasttext     | 0.997              | 0.972               |
| neuralnet    | 0.9975             | 0.98                |


## Baseline Naive Bayes result on test set :

| class        | precision     | recall             | f1-score             |    support          |
| -------------|:-------------:|:------------------:|:-------------------:|:-------------------:|
| class 0      | 0.94          | 0.83               | 0.88                 | 100                 |
| class 1      | 0.85          | 0.95               | 0.90                 | 100                 |
| Avg/Total    | 0.90          | 0.89               | 0.89                 | 200                 |

## Results:  ensemble of the 4 models : Fasttext+cnn+logreg+svm

- Final results of the ensemble on the test set :

|              | precision     | recall             | f1-score            |    support          |
| -------------|:-------------:|:------------------:|:-------------------:|:-------------------:|
| class 0      | 0.99          | 0.90               | 0.94                | 100                 |
| class 1      | 0.91          | 0.99               | 0.95                | 100                 |
| Avg/Total    | 0.95          | 0.94               | 0.94                | 200                 |

