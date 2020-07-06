import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Evaluate the algorithms to select the best one for the problem and 
# use the one which best fits for you 
from sklearn.ensemble import RandomForestClassifier

language_stopwords = stopwords.words('spanish')
non_words = list(punctuation)
non_words.extend(['¿', '¡'])

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

def train_test_split(classified_features, classified_labels):
	train_boundary_index = int(0.3 * len(classified_features))
	X_train = classified_features[:-train_boundary_index]
	X_test = classified_features[-train_boundary_index:]
	y_train = classified_labels[:-train_boundary_index]
	y_test = classified_labels[-train_boundary_index:]
	return X_train, X_test, y_train, y_test

train_df = process_file("Train.csv")
test_df = process_file("Test.csv")

merged_df = pd.concat([train_df,test_df])
# Due to memory issues, take a sample of the total dataset
merged_df = merged_df.sample(n=10000)

# Assing -1 label on random records to simulate not classified data
# First select a sample of 1 % of the records
sample_size = int(0.01 * len(merged_df))
merged_df.iloc[-sample_size:]['label'] = -1

new_records_df = merged_df.loc[merged_df['label']==-1]

# Clean the dataset
# Remove all nan
final_dataframe = merged_df.dropna()
# Remove html tags
final_dataframe['text'] = final_dataframe['text'].apply(lambda text:BeautifulSoup(text).get_text())

# All to lower case
final_dataframe['text'] = final_dataframe['text'].str.lower()
# Remove punctuation and spanish stopwords
final_dataframe['text'] = final_dataframe['text'].apply(remove_punctuation).apply(remove_stop_words)

# Tokenization
#TF-IDF
vectorizer = TfidfVectorizer ()
X = vectorizer.fit_transform(final_dataframe['text'])

# Separate labels and features
processed_features = X.toarray()
classified_features = processed_features[:-sample_size]
unclassified_features = processed_features[-sample_size:]
classified_labels = final_dataframe[final_dataframe['label']!=-1]['label']

# Divide training and test separete_classified_data
X_train, X_test, y_train, y_test = train_test_split(classified_features, classified_labels)

# Algorithm training
text_classifier = RandomForestClassifier()
text_classifier.fit(X_train, y_train)

# Predictions over the test data
predictions = text_classifier.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))

# Predictions over new data
predictions = text_classifier.predict(unclassified_features)

new_records_df['label'] = predictions
new_records_df.to_csv('predicted_data.csv', encoding='utf-8')