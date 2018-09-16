import pandas as pd
import csv
from emotion_predictor import EmotionPredictor

# Pandas presentation options
pd.options.display.max_colwidth = 150   # show whole tweet's content
pd.options.display.width = 200          # don't break columns
# pd.options.display.max_columns = 7      # maximal number of columns


# Predictor for Ekman's emotions in multiclass setting.
model = EmotionPredictor(classification='ekman', setting='mc', use_unison_model=True)

tweets = []
with open('/home/didier/ASONAMW/PA/Sentiment/Billboard_Music_Awards_2016_Sentiment.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        tweets.append(row[1])

predictions = model.predict_classes(tweets)
predictions.to_csv("/home/didier/ASONAMW/PA/Sentiment/Billboard_Music_Awards_2016_Emotions.txt", sep=';', encoding='utf-8')

tweets = []
with open('/home/didier/ASONAMW/PA/Sentiment/Billboard_Music_Awards_2017_Sentiment.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        tweets.append(row[1])

predictions = model.predict_classes(tweets)
predictions.to_csv("/home/didier/ASONAMW/PA/Sentiment/Billboard_Music_Awards_2017_Emotions.txt", sep=';', encoding='utf-8')


tweets = []
with open('/home/didier/ASONAMW/PA/Sentiment/MWC_2016_Sentiment.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        tweets.append(row[1])

predictions = model.predict_classes(tweets)
predictions.to_csv("/home/didier/ASONAMW/PA/Sentiment/MWC_2016_Emotions.txt", sep=';', encoding='utf-8')

tweets = []
with open('/home/didier/ASONAMW/PA/Sentiment/MWC_2017_Sentiment.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        tweets.append(row[1])

predictions = model.predict_classes(tweets)
predictions.to_csv("/home/didier/ASONAMW/PA/Sentiment/MWC_2017_Emotions.txt", sep=';', encoding='utf-8')


tweets = []
with open('/home/didier/ASONAMW/PA/Sentiment/Oscars_2016_Sentiment.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        tweets.append(row[1])

predictions = model.predict_classes(tweets)
predictions.to_csv("/home/didier/ASONAMW/PA/Sentiment/Oscars_2016_Emotions.txt", sep=';', encoding='utf-8')

tweets = []
with open('/home/didier/ASONAMW/PA/Sentiment/Oscars_2017_Sentiment.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        tweets.append(row[1])

predictions = model.predict_classes(tweets)
predictions.to_csv("/home/didier/ASONAMW/PA/Sentiment/Oscars_2017_Emotions.txt", sep=';', encoding='utf-8')


tweets = []
with open('/home/didier/ASONAMW/PA/Sentiment/Paris_Games_Week_2015_Sentiment.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        tweets.append(row[1])

predictions = model.predict_classes(tweets)
predictions.to_csv("/home/didier/ASONAMW/PA/Sentiment/Paris_Games_Week_2015_Emotions.txt", sep=';', encoding='utf-8')

tweets = []
with open('/home/didier/ASONAMW/PA/Sentiment/Paris_Games_Week_2017_Sentiment.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        tweets.append(row[1])

predictions = model.predict_classes(tweets)
predictions.to_csv("/home/didier/ASONAMW/PA/Sentiment/Paris_Games_Week_2017_Emotions.txt", sep=';', encoding='utf-8')


tweets = []
with open('/home/didier/ASONAMW/PA/Sentiment/US_Election_2008_Sentiment.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        tweets.append(row[1])

predictions = model.predict_classes(tweets)
predictions.to_csv("/home/didier/ASONAMW/PA/Sentiment/US_Election_2008_Emotions.txt", sep=';', encoding='utf-8')

tweets = []
with open('/home/didier/ASONAMW/PA/Sentiment/US_Election_2016_Sentiment.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=';')
    for row in plots:
        tweets.append(row[1])

predictions = model.predict_classes(tweets)
predictions.to_csv("/home/didier/ASONAMW/PA/Sentiment/US_Election_2016_Emotions.txt", sep=';', encoding='utf-8')
















#print(predictions, '\n')

#probabilities = model.predict_probabilities(tweets)
#print(probabilities, '\n')

#embeddings = model.embedd(tweets)
#print(embeddings, '\n')
