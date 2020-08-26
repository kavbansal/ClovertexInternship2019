import nltk
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.corpus import words

word_list = words.words()

input_text = input("Enter text: ")
tokens = nltk.word_tokenize(input_text)
tagged = nltk.pos_tag(tokens)
print(tokens)
print(tagged)

fixed_output = ""
count = 0
for text in tokens:
    highest = 0
    possibilities = []
    fixed_text = ""
    # First check if word is spelled correctly/is proper noun
    if (text.lower() in word_list or tagged[count][1] == 'NNP'):
        fixed_text = " " + text
    # Check if token is punctuation
    elif (len(text) == 1):
        fixed_text = text
    else:
        # Generates list of possibilities
        for word in word_list:
            # Using 85 as the cutoff for similarity
            if (fuzz.ratio(text.lower(), word) >= 85):
                possibilities.append((word, fuzz.ratio(text.lower(), word)))
        # From list of possibilities, picks one with highest confidence to autocorrect to
        for word in possibilities:
            if (word[1] > highest):
                fixed_text = " " + word[0]
                highest = word[1]
        print("Did you mean:", possibilities)
        if not possibilities:
            fixed_text = " " + text
    fixed_output += fixed_text
    count += 1
print("Output:" + fixed_output)
