import nltk
import pyap
import wordcloud
from wordcloud import WordCloud
from nltk import FreqDist
import phonenumbers
import pyusps
from pyusps import address_information
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.corpus import stopwords
from nltk.corpus import words
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import omdb
from omdb import OMDBClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Movie Stuff
# Read and put together data on movie ratings, combined with userId number, movieId number, and Movie Title
df = pd.read_csv('ratings.csv', sep=',')
movie_titles = pd.read_csv('movies.csv')
df = pd.merge(df, movie_titles, on='movieId')

# Restrict data to 10 million reviews so that pivot table/matrix will work
df = df.head(10000000)
# Creates data frame with average rating of each movie
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
# Adds column for number of ratings for each movie
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()

# Use previous setup steps to now create the recommender system:

# First creates matrix with movie titles as columns, user ID as index, and ratings as values, making it
# simple to access all users' ratings of a particlar movie
movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')

# Displays the top 10 movies with the most ratings
print("Top 10 Movies by Number of Ratings:", ratings.sort_values(
    'number_of_ratings', ascending=False).head(10))

# Displays the top 10 movies with the highest average rating
print("Top 10 Movies by Avg Rating:", ratings.sort_values(
    'rating', ascending=False).head(10))

# Collects all user ratings for two specific movies (Can edit default movie HERE)
oz_user_rating = movie_matrix['Iron Man (2008)']    # Wizard of Oz, The (1939)

# Uses user entered movie as movie, or above default if invalid
title = input("What's your favorite movie (with year): ")


# Now do NLP stuff. Input lines put together to make user experience smoother
text = input("Tell me all about your day: ")

# Sentiment Analysis
analyser = SentimentIntensityAnalyzer()
score = analyser.polarity_scores(text)
compound = score['compound']

# Splits input into words
tokens = nltk.word_tokenize(text)

# Makes pairs of words with part of speech
tagged = nltk.pos_tag(tokens)

# print(tagged)
# Chunks words into named entities (phrases)
entities = nltk.chunk.ne_chunk(tagged)

# Shows named entities and parts of speech
# print(entities)
# Part 2: Identify Names of People, Places/Addresses, and Phone Numbers
people = []
places = []
# Only works for US Addresses right now
addresses = pyap.parse(text, country='US')

# Identifies phone numbers
nums = []
for match in phonenumbers.PhoneNumberMatcher(text, "US"):
    nums.append(phonenumbers.format_number(
        match.number, phonenumbers.PhoneNumberFormat.E164))

for e in entities:
    try:
        # Identify names of people
        if (e.label() == "PERSON"):
            temp = ""
            for l in e.leaves():
                temp += l[0] + " "
            people.append(temp.strip())
        # Identify Places (Non-Address)
        elif (e.label() == "GPE" or e.label() == "FACILITY"):
            temp = ""
            for l in e.leaves():
                temp += l[0] + " "
            places.append(temp.strip())
    except AttributeError:
        pass
print("People:", people)
print("Places:", places)
print("Phone Numbers:", nums)

valid_addresses = []
# Address Validation using and pyusps
for addr in addresses:
    addr_dict = addr.as_dict()
    street_addr = addr_dict['full_street']
    city = addr_dict['city']
    state = addr_dict['region1']
    my_addr = dict([
        ('address', street_addr),
        ('city', city),
        ('state', state),
    ])
    try:
        # Searches USPS resources to verify address, throws ValueError if invalid
        valid_addr = address_information.verify('021CLOVE5421', my_addr)
        valid_addresses.append(
            valid_addr['address'] + ", " + valid_addr['city'] + ", " + valid_addr['state'] + " " + valid_addr['zip5'] + "-" + valid_addr['zip4'])
    except ValueError:
        # print("Invalid Address")
        pass
print("Valid Addresses:", valid_addresses)

# Filters text for wordcloud, removing filler words like "the"
stop_words = set(stopwords.words('english'))
filtered_text = [w for w in tokens if not w in stop_words]
filtered_text = [w for w in filtered_text if len(w) > 1]
tagged = [p for p in tagged if p[0] in filtered_text]
# Converts each word to its base form so singular and plural are same
wnl = WordNetLemmatizer()
filtered_text = [wnl.lemmatize(token.lower()) for token in filtered_text]


# SPELL CHECK: If proper noun, don't spell check, otherwise do
word_list = words.words()
fixed_output = ""
count = 0
for t in filtered_text:
    highest = 0
    possibilities = []
    fixed_text = ""
    # First check if word is spelled correctly/is proper noun
    if (t.lower() in word_list or tagged[count][1] == 'NNP'):
        fixed_text = " " + t
    # Check if token is punctuation
    elif (len(t) == 1):
        fixed_text = t
    else:
        # Generates list of possibilities
        for word in word_list:
            # Using 85 as the cutoff for similarity
            if (fuzz.ratio(t.lower(), word) >= 85):
                possibilities.append((word, fuzz.ratio(t.lower(), word)))
        # From list of possibilities, picks one with highest confidence to autocorrect to
        for word in possibilities:
            if (word[1] > highest):
                fixed_text = " " + word[0]
                highest = word[1]
        # print("Did you mean:", possibilities)
        if not possibilities:
            fixed_text = " " + t
    fixed_output += fixed_text
    count += 1
print("Filtered, Lemmatized, & Spell Checked Input:" + fixed_output)
text = fixed_output

# Spell check applies to word cloud only since other named entities are all proper nouns anyway

# Splits input into words
tokens = nltk.word_tokenize(text)

# Calculates frequencies of each word in input
fdist1 = FreqDist(tokens)
print(fdist1.most_common(50))

filtered_string = ""
for w in tokens:
    filtered_string += w + " "

# Word cloud for 50 most frequent words
wordcloud = WordCloud(max_font_size=100).generate(filtered_string)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")

# Word cloud for just fruit names in the fruits taxonomy file (if any)
fruits = open("fruits.txt", "r")
fruit_list = fruits.read().split('\n')

filtered_text = [w for w in tokens if w in fruit_list]
filtered_string = ""
for w in filtered_text:
    filtered_string += w + " "

fdist1 = FreqDist(filtered_text)
print(fdist1.most_common(len(fruit_list)))

if (filtered_string):
    wordcloud2 = WordCloud(max_font_size=100).generate(filtered_string)
    plt.figure()
    plt.imshow(wordcloud2, interpolation="bilinear")
    plt.axis("off")
# Back to movie stuff
if (title in df['title'].tolist()):
    oz_user_rating = movie_matrix[title]

# Calculates correlation of all other movies to selected movie
similar_to_oz = movie_matrix.corrwith(oz_user_rating)


# Makes sure that comparison of users who did not review the selected movie does not take place
corr_oz = pd.DataFrame(similar_to_oz, columns=['Correlation'])
corr_oz.dropna(inplace=True)

# Displays top movies similar to selected movie (top 10 recommendations),
# along with number of ratings for each of the recommended movies.

# Custom message depending on sentiment
if (compound >= 0.2):
    print("Sounds like you had a great day! Since you liked " + title +
          ", maybe check out one of these movies to make tonight just as amazing!")
elif (compound <= -0.2):
    print("Sounds like you had a rough day but don't worry. Since you liked " +
          title + ", maybe check out one of these movies tonight to make things better.")
else:
    print("Sounds like you had quite the day. Since you liked " +
          title + ", maybe watch one of these movies tonight.")
corr_oz = corr_oz.join(ratings['number_of_ratings'])

# Gets and prints top recommendation (really second row of the list since first is the user entered movie)
top_rec = corr_oz[corr_oz['number_of_ratings'] > 100].sort_values(
    by='Correlation', ascending=False).iloc[1].name
print("Top Recommendation: " + top_rec)

# Prints brief synopsis of top recommendation
client = OMDBClient(apikey='da4dbe2c')
rec_list = top_rec.split('(')
# Accounts for Movies with parentheses in title
time = (rec_list[len(rec_list) - 1]).split(')')[0]
name = rec_list[0]
# Accounts for Movies that start with something like "The" or "A"
if ',' in name:
    name = name.split(',')[0]

# If a year exists (it should), it looks up with both the title and year, if not just title
if time:
    print(client.get(title=name, year=time)['plot'])
else:
    print(client.get(title=name)['plot'])

# Prints top 10 recommendations along with correlations and number of ratings
print(corr_oz[corr_oz['number_of_ratings'] > 100].sort_values(
    by='Correlation', ascending=False).head(10))

# Display word clouds last
plt.show()
fruits.close()
