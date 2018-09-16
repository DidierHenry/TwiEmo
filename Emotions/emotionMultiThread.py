import pandas as pd
import csv
from emotion_predictor import EmotionPredictor
import sys

# Pandas presentation options
pd.options.display.max_colwidth = 150   # show whole tweet's content
pd.options.display.width = 200          # don't break columns
# pd.options.display.max_columns = 7      # maximal number of columns


# Predictor for Ekman's emotions in multiclass setting.
model = EmotionPredictor(classification='ekman', setting='mc', use_unison_model=True)


inputFile = sys.argv[1]
outputFile = sys.argv[2]
separator = sys.argv[3]
col = int(sys.argv[4])

tweets = []
with open(inputFile,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter= separator)
    for row in plots:
        tweets.append(row[col])

predictions = model.predict_classes(tweets)
predictions.to_csv(outputFile, sep=';', encoding='utf-8')
