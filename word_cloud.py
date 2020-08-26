import nltk
import wordcloud
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.corpus import words
from wordcloud import WordCloud
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
#from nltk.book import *
text = input("Enter text: ")

# Splits input into words
tokens = nltk.word_tokenize(text)
# print(tokens)
#tokens = text5
tags = nltk.pos_tag(tokens)
stop_words = set(stopwords.words('english'))
filtered_text = [w for w in tokens if not w in stop_words]
filtered_text = [w for w in filtered_text if len(w) > 1]
print("filtered:", filtered_text)
wnl = WordNetLemmatizer()
filtered_text = [wnl.lemmatize(token.lower()) for token in filtered_text]
tagged = [p for p in tags if p[0] in filtered_text]

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
        print("Did you mean:", possibilities)
        if not possibilities:
            fixed_text = " " + t
    fixed_output += fixed_text
    count += 1
print("Corrected Input:" + fixed_output)  # Can uncomment to debug/trace code
text = fixed_output

# Splits input into words
tokens = nltk.word_tokenize(text)


# Calculates frequencies of each word in input
fdist1 = FreqDist(tokens)
# print(fdist1)

print(fdist1.most_common(50))

filtered_string = ""
for w in tokens:
    filtered_string += w + " "

wordcloud = WordCloud(max_font_size=100).generate(filtered_string)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
