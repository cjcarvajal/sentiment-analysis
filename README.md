# sentiment-analysis

Sentiment analysis is a growing area of research on [NLP](https://nlp.stanford.edu/), it persues the extraction of subjetive information from texts, specifically the sentiments of a person towards an aspect of an entity. In many cases the person holding the sentiment is the author of the text, but this doesn't apply to all cases, as the author may be narrating the view of a third person (or entity). The knowledge obtained by the sentiment analysis may be the input for autonomous systems or to serve as support for a decission maker. In today's world, sentiment analysis is applied on sales platforms like Amazon, to evaluate a product or even in social networks, to map the peoples opinion. This last use case may be of great value on [elections](https://www.theatlantic.com/technology/archive/2020/04/how-facebooks-ad-technology-helps-trump-win/606403/) for example, where the analysis of a live Twitter streaming may size the popularity of a candidate, or even, to drive the speech on debates.

In this repo, you will find an example to perform sentiment analysis training a model and using it to predict sentiments on new texts.

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
splitting in training, testing, and unlabeled data sets. For the first phase, I read the csv files for training and testing from the [files section](#about-the-files-used), merge the two data sets and select a random sample for 10000 records. I create this subset for memory issues on my machine, actually I ran this experiment on a Mac with 8 GB RAM, so if you have a better machine (or make improvements on the script to improve efficiency) you could try running with more records.

I just read a lot of blogs entrances on sentiment analysis with the same approach: to train a model and to obtain performance metrics using a test dataset, but none of these posts explain how to use the trained model to predict values on new data (the real value of the model), so I wrote the script to put an example on how to do it. For that reasons I select a subset of 10% of the records, and remove the label, assigning a new value for the label column of -1.

```
sample_size = int(0.01 * len(merged_df))
merged_df.iloc[-sample_size:]['label'] = -1
```

For the data cleaning, I remove the html tags from the text, pass all text to lower case, and remove the stopwords. I used TF-IDF vectorisation to get the vector representation of each review.

I use the final portion of the dataset as the new data to predict, using fixed position, so I split the data set into classified and unclassified:

```
classified_features = processed_features[:-sample_size]
unclassified_features = processed_features[-sample_size:]
classified_labels = final_dataframe[final_dataframe['label']!=-1]['label']
```

Finally, I take the classified subset and split it into training and testing, again using fixed position on the data. This was a tradeoff instead of using the [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) (which I recommend due to the randomness of the process) because of memory issues.

### Training

I use a **RandomForestClassifier** as experiment, you could use some others [classification methods](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble) to get better results.

The results of this method are:

| Class  | Precission | Recall | f1-score | support |
| ---|------|------|------|------|
| 0  | 0.67 | 0.80 | 0.73 | 1467 |
| 1  | 0.76 | 0.61 | 0.68 | 1503 |

The accuracy of the model is around 0.706.

### Getting predictions

In the final part of the script, I use the trained model on the **unclassified_features**, which contains the vector representations of the reviews with no label discussed on [Preprocessing](#preprocessing) and save a new file with the text review and the predicted label. For example I get this predictions:

| Text | Label |
|------|-------|
|Wow this movie **sucked big time**. I heard this movie expresses the meaning of friendship very well. And with all the internet hype on this movie I figured what could go wrong? However the movie was just **plain bad**... | 0 |
|This Is one of **my favourite westerns**. **What a cast!** Glenn Ford plays his role In his usual mild ... | 1 |

Notice the phrases on the review that represent the negative and positive sentiments on each example.

### Execution

Clone this repository into your machine, download the Test.csv and Train.csv from [Kaggle](https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format) and place this files in the folder of the cloned repo. Remember you should have Python3 installed and the modules discused in (#tech-stack). Then execute the script with:

```
python3 sentiment_analysis.py
```

The output of the results are printed in screen, the predicted values are saved on a file **predicted_data.csv** that youl will find in the repo folder after running the process.

## References
<a id="1">[1]</a> 
Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher (2011).
Learning Word Vectors for Sentiment Analysis. 
Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, 142-150.