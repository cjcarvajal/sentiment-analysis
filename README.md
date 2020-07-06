# sentiment-analysis

A little chit chat here about the theory on sentiment analysis and NLP

## About the files used

To train and test the model, I used an existing repository on IMDB reviews [[1]](#1), please go to the [web site](http://ai.stanford.edu/~amaas/data/sentiment/) for further information. The dataset contains a column for the review text and a label column for the sentiment, being 0 a negative review and 1 a positive.

## Tech stack

What tools did I use:

* [Python](https://www.python.org/downloads/release/python-383/)
* [Pandas](https://pandas.pydata.org/) (For data manipulation)
* [nltk](https://www.nltk.org/) (To NLP preprocessing)
* [sklearn](https://scikit-learn.org/stable/) (For model training)
* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) (To remove html tags from text)

### Preprocessing

The preprocessing of data have three phases: reading data, cleaning and vectorizing text and
splitting in training, testing, and unlabeled data sets. For the first phase, I read the csv files for training and testong from the [files section](#about-the-files-used), merged the two data sets and selected a random sample for 10000 records. I created this subset for memory issues on my machine, actually I ran this experiment on a Mac with 8 GB RAM, so if you have a better machine (or make improvements on the scripts to make it more efficient) you could try running with more records.

I just read a lot of blogs entrances on sentiment analysis with the same approach: to train a model and to obtain performance metrics using a test dataset, but none of this post explain how to use the trained model to predict values on new data (the real value of the model), so I wrote the script to put an example on how to do it. For that reasons I selected a subset of 10% of the records, and remove the label, assignin it a new value for the label column of -1.

For the data cleaning, I removed the html tags from the text, pass all text to lower case, and remove the stopwords. I used TF-IDF vectorization to get the vector representation of each review.

I used the final portion of the dataset as the new data to predict, using fixed position, so I split the data set into classified and unclassified:

```
classified_features = processed_features[:-sample_size]
unclassified_features = processed_features[-sample_size:]
classified_labels = final_dataframe[final_dataframe['label']!=-1]['label']
```

Finally, I took the classified subset and splitted it into training and testing, again using fixed position on the data. This was a tradeoff instead of using the [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) (which I recommend due to the randomness of the process) because of memory issues.

### Training

I used a RandomForestClassifier as experiment, you could use some others [classification methods](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble) to get better results.

The results of this method are:

| Class  | Precission | Recall | f1-score | support |
| ---|------|------|------|------|
| 0  | 0.67 | 0.80 | 0.73 | 1467 |
| 1  | 0.76 | 0.61 | 0.68 | 1503 |

## References
<a id="1">[1]</a> 
Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher (2011).
Learning Word Vectors for Sentiment Analysis. 
Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, 142-150.