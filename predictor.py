import pickle
import texttable as tt

sentiment_model = pickle.load(open('sentiment_model.sav','rb'))

reviews = ["It's simultaneously difficult to watch and yet, impossible to take your eyes off Joaquin Phoenix in this film that may become the definitive origin tale of the Joker.",
	"This was the worst movie I've ever seen, so bad that I hesitate to label it a 'movie' and thus reflect shame upon the entire medium of film.",
	"The Devil Inside is yet another dull found footage effort with nothing, bar a mad climax, to offer audiences.",
	"One of the smartest, most inventive movies in memory, it manages to be as endearing as it is provocative."]

predictions = sentiment_model.predict(reviews)

tab = tt.Texttable()
tab.header(['Review','Sentiment'])

for row in zip(reviews,predictions):
	tab.add_row(row)

s = tab.draw()
print(s)