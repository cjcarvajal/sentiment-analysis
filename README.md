# sentiment-analysis

Sentiment analysis is a growing area of research on [NLP](https://nlp.stanford.edu/), it pursues the extraction of subjetive information from texts, specifically the sentiments of a person towards an aspect of an entity. In many cases the person holding the sentiment is the author of the text, but this doesn't apply to all cases, as the author may be narrating the view of a third person (or entity). The knowledge obtained by the sentiment analysis may be the input for autonomous systems or to serve as support for a decission maker. In today's world, sentiment analysis is applied on sales platforms like Amazon, to evaluate a product or even in social networks, to map the peoples opinion. This last use case may be of great value on [elections](https://www.theatlantic.com/technology/archive/2020/04/how-facebooks-ad-technology-helps-trump-win/606403/) for example, where the analysis of a live Twitter streaming may measure the popularity of a candidate, or even, to drive the speech on debates.

In this repo, you will find an example to perform sentiment analysis training a model and using it to predict sentiments on unseen texts.

## About the files used

To train and test the model, I use an existing repository on IMDB reviews [[1]](#1), please go to the [web site](http://ai.stanford.edu/~amaas/data/sentiment/) for further information. The dataset contains a column for the review text and a label column for the sentiment, being 0 a negative review and 1 a positive.

## Tech stack

What tools did I use:

* [Python](https://www.python.org/downloads/release/python-383/)
* [Pandas](https://pandas.pydata.org/) (For data manipulation)
* [nltk](https://www.nltk.org/) (To NLP preprocessing)
* [sklearn](https://scikit-learn.org/stable/) (For model training)
* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) (To remove html tags from text)

### Preprocessing

The preprocessing of data have three phases: reading data, cleaning and vectorizing text and
splitting in training, testing, and unlabeled data sets. For the first phase, I read the csv files for training and testing from the [files section](#about-the-files-used) and merge the two data sets, I use the sklearn [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) method to divide the sets.

I just read a lot of blogs entrances on sentiment analysis with the same approach: to train a model and to obtain performance metrics using a test dataset, but none of these posts explain how to use the trained model to predict values on new data (the real value of the model), in this script I used the pipeline feature, to train and store the model and then to use it with unseen data.

```
vectorizer = TfidfVectorizer ()
text_classifier = RandomForestClassifier()
training_pipe = Pipeline([('vectorizer',vectorizer),('text_classifier',text_classifier)])
```

For the data cleaning, I remove the html tags from the text, pass all text to lower case, and remove the stopwords. I used TF-IDF vectorisation to get the vector representation of each review.

### Training

I use a **RandomForestClassifier** as experiment, you could use some others [classification methods](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble) to get better results.

The results of this method are:

| Class  | Precission | Recall | f1-score | support |
| ---|------|------|------|------|
| 0  | 0.72 | 0.83  | 0.77 | 6757 |
| 1  | 0.80 | 0.68 | 0.74 | 6743 |

The accuracy of the model is around 0.755.

### Getting predictions

The **predictor.py** use the trained model to make predictions, in this script I used four reviews as examples, the model throws 1 for positive sentiments and 0 for negative ones, the results of this script are:


| Review | Sentiment |
|------|-------|
|It's simultaneously difficult to watch and yet, impossible to take your eyes off Joaquin Phoenix in this film that may become the definitive origin tale of the Joker.  | 1 |
|This was the **worst** movie I've ever seen, so bad that I hesitate to label it a 'movie' and thus reflect **shame** upon the entire medium of film.| 0 |
|The Devil Inside is yet another **dull** found footage effort with nothing, bar a mad climax, to offer audiences.| 0 |
|One of the **smartest**, most inventive movies in memory, it manages to be as endearing as it is provocative.| 1 |

Notice the phrases on the review that represent the negative and positive sentiments on each example.

### Execution

Clone this repository into your machine, download the Test.csv and Train.csv from [Kaggle](https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format) and place these files in the folder of the cloned repo. Remember you should have Python3 installed and the modules discused in [Tech stack](#tech-stack). Then execute the script with:

```
python3 sentiment_analysis.py
```

The training results results are printed in screen, after running the script you should see a file **sentiment_model.sav** which correspond to the trained model.

To run the **predictor.py** do the same:

```
python3 predictor.py
```

## References
<a id="1">[1]</a> 
Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher (2011).
Learning Word Vectors for Sentiment Analysis. 
Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, 142-150.