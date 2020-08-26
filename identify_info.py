import nltk
import pyap
import phonenumbers
import pyusps
from pyusps import address_information
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.corpus import words

text = input("Enter text: ")

# Splits input into words
tokens = nltk.word_tokenize(text)

# Makes pairs of words with part of speech
tagged = nltk.pos_tag(tokens)

# print(tagged)
# Chunks words into named entities (phrases)
entities = nltk.chunk.ne_chunk(tagged)

# Shows named entities and parts of speech
print(entities)

# Part 1: Identify Nouns
# nouns = [x for x in tagged if x[1][0] == 'N']
# print("Nouns:")
# for n in nouns:
#     print(n[0])

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
        print("Did you mean:", possibilities)
        if not possibilities:
            fixed_text = " " + t
    fixed_output += fixed_text
    count += 1
print("Corrected Input:" + fixed_output)
text = fixed_output

# Splits input into words
tokens = nltk.word_tokenize(text)
# Makes pairs of words with part of speech
tagged = nltk.pos_tag(tokens)

# Chunks words into named entities (phrases)
entities = nltk.chunk.ne_chunk(tagged)

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
# print("Addresses:", addresses)
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
        print("Invalid Address")
        pass
print("Valid Addresses:", valid_addresses)
