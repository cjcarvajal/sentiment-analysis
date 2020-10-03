import pickle
import texttable as tt

sentiment_model = pickle.load(open('sentiment_model.sav','rb'))

reviews = ["It's simultaneously difficult to watch and yet, impossible to take your eyes off Joaquin Phoenix in this film that may become the definitive origin tale of the Joker.",
	"This was the worst movie I've ever seen, so bad that I hesitate to label it a 'movie' and thus reflect shame upon the entire medium of film.",
	"Shatteringly stupid and repulsively misogynistic, 'Martyrs' mashes revenge, torture and the supernatural into one solid, quasi-religious lump.",
	"Just like the games of the era, Console Wars is bright, engaging, and frequently so fast-moving and unfocused that it might give you a headache."]

predictions = sentiment_model.predict(reviews)

tab = tt.Texttable()
tab.header(['Review','Sentiment'])

for row in zip(reviews,predictions):
	tab.add_row(row)

s = tab.draw()
print(s)