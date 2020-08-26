import nltk

text = input("Enter text: ")

tokens = nltk.word_tokenize(text)
tagged = nltk.pos_tag(tokens)
nouns = [x for x in tagged if x[1][0] == 'N']

for n in nouns:
    print(n[0])
