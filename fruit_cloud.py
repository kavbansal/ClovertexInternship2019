import nltk
import wordcloud
from nltk import FreqDist
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.corpus import words
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer


fruits = open("fruits.txt", "r")
fruit_list = fruits.read().split('\n')

text = input("Enter text with fruits: ")
# Splits input into words
tokens = nltk.word_tokenize(text)

# Makes pairs of words with part of speech
tagged = nltk.pos_tag(tokens)

# Used to account for plural forms and implement case insensitivity
wnl = WordNetLemmatizer()
tokens = [wnl.lemmatize(token.lower()) for token in tokens]
tagged = nltk.pos_tag(tokens)
# SPELL CHECK: If proper noun, don't spell check, otherwise do
word_list = words.words()
fixed_output = ""
count = 0
for t in tokens:
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
        #print("Did you mean:", possibilities)
        if not possibilities:
            fixed_text = " " + t
    fixed_output += fixed_text
    count += 1
# print("Corrected Input:" + fixed_output)  # Can uncomment to debug/trace code
text = fixed_output

# Splits input into words
tokens = nltk.word_tokenize(text)

# Separates words present in the taxonomy file
filtered_text = [w for w in tokens if w in fruit_list]
filtered_string = ""
for w in filtered_text:
    filtered_string += w + " "

# Calculates and displays frequencies of each keyword present
fdist1 = FreqDist(filtered_text)
print(fdist1.most_common(len(fruit_list)))

#Generate and display Word Cloud
wordcloud = WordCloud(max_font_size=100).generate(filtered_string)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
fruits.close()
