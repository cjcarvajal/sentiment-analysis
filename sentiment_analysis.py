import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle

# Evaluate the algorithms to select the best one for the problem and 
# use the one which best fits for you 
from sklearn.ensemble import RandomForestClassifier

language_stopwords = stopwords.words('english')
non_words = list(punctuation)

file_encoding = "utf-8"  
file_separator = ","

def process_file(file):
	return pd.read_csv(file,encoding=file_encoding,sep=file_separator)

def remove_stop_words(dirty_text):
	cleaned_text = ''
	for word in dirty_text.split():
		if word in language_stopwords or word in non_words:
			continue
		else:
			cleaned_text += word + ' '
	return cleaned_text

def remove_punctuation(dirty_string):
	for word in non_words:
		dirty_string = dirty_string.replace(word, '')
	return dirty_string

'''
def train_test_split(classified_features, classified_labels):
	train_boundary_index = int(0.3 * len(classified_features))
	X_train = classified_features[:-train_boundary_index]
	X_test = classified_features[-train_boundary_index:]
	y_train = classified_labels[:-train_boundary_index]
	y_test = classified_labels[-train_boundary_index:]
	return X_train, X_test, y_train, y_test
'''

train_df = process_file("Train.csv")
test_df = process_file("Test.csv")

merged_df = pd.concat([train_df,test_df])
# Due to memory issues, take a sample of the total dataset
merged_df = merged_df.sample(n=10000)

# Clean the dataset
# Remove all nan
final_dataframe = merged_df.dropna()
# Remove html tags
final_dataframe['text'] = final_dataframe['text'].apply(lambda text:BeautifulSoup(text).get_text())

# All to lower case
final_dataframe['text'] = final_dataframe['text'].str.lower()
# Remove punctuation and spanish stopwords
final_dataframe['text'] = final_dataframe['text'].apply(remove_punctuation).apply(remove_stop_words)

# Pipeline instantiation, first defines a vectorization method and the an algorithm to classify
vectorizer = TfidfVectorizer ()
text_classifier = RandomForestClassifier()
training_pipe = Pipeline([('vectorizer',vectorizer),('text_classifier',text_classifier)])

# Divide training and test sets
train_df, test_df = train_test_split(final_dataframe, test_size=0.3, random_state=0)

train_data = train_df['text']
train_target = train_df['label']

test_data = test_df['text']
test_target = test_df['label']

training_pipe.fit(train_data,train_target)

# Predictions over the test data
predictions = training_pipe.predict(test_data)

print(confusion_matrix(test_target,predictions))
print(classification_report(test_target,predictions))
print(accuracy_score(test_target, predictions))

# Save the model
pickle.dump(training_pipe, open('sentiment_model.sav','wb'))