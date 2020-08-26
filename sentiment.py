from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
text = input("Tell me all about your day: ")
score = analyser.polarity_scores(text)
compound = score['compound']
print(compound)

if (compound >= 0.2):
    # Positive response
    print("Sounds like you had a good day")
elif (compound <= -0.2):
    # Negative response
    print("Sounds like you had a bad day")
else:
    # Neutral response
    print("Sounds like you had quite the day")
