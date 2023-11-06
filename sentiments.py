'''
Note: 
1) Keep all data files in the same folder as this script
2) Code was run in the virtual box image provided for lab 1.
2) Install the following python libraries:
    - nltk
    - autocorrect
    - demoji

'''
# import the libraries to be used

import sys
import os
import shutil
from nltk import word_tokenize, sent_tokenize, WordNetLemmatizer, download
from nltk.tokenize import TweetTokenizer
from nltk.metrics.distance import jaccard_distance 
from nltk.util import ngrams
from nltk.corpus import words, stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, FloatType, IntegerType
from autocorrect import Speller
import demoji

# download packages from nltk
download('wordnet')
download('words')
download('stopwords')
download('punkt')
download('vader_lexicon')

# set threshold for sentiment score
threshold = 0.33

# initialise the spell corrector
word_dictionary = set(words.words())

# initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# initialize spell corrector
spell = Speller(lang='en')

# stop words
stop_words = set(stopwords.words('english'))

# initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# read csv file and store string in each row as a set
with open('candidates_bjp.csv', 'r') as f:
    bjp = f.read().splitlines()
    
with open('candidates_inc.csv', 'r') as f:
    congress = f.read().splitlines()
    
# convert everything to lowercase
bjp = [line.lower() for line in bjp]
congress = [line.lower() for line in congress]

# insert string in each row into a set
bjp = set([line for line in bjp])
congress = set([line for line in congress])

# define bjp words
bjp_words = set(['modi','narendra','bjp',])
bjp_words.update(bjp)

# define congress words
congress_words = set(['congress','rahul','gandhi'])
congress_words.update(congress)

# initialize the tokenizer
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

# Convert emojis to words
def replace_emoji(text_with_emojis):
    return demoji.replace_with_desc(text_with_emojis)

# a function that takes in a string and returns a list of tokens without stopwords and emojis
def tokenize(string):
    word_list = tknzr.tokenize(replace_emoji(string))
    word_list = [word for word in word_list if word not in stop_words]
    return word_list

# a function that takes in a list of strings, checks that the string does not contain non-alphabetical characters and returns a list of strings with only alphabetical characters
def remove_non_alpha(list_of_strings):
    # initialize an empty list to store the cleaned strings
    cleaned = []
    # loop through the list of strings
    for string in list_of_strings:
        # check that the string does not contain non-alphabetical characters
        if string.isalpha():
            # lemmatize the string and append to the cleaned list
            cleaned.append(string)
    # return the cleaned list
    return cleaned  

# a function that replace word that match a abbreviations key with the abbreviations value
def replace_abb(list_of_strings, abbreviations):
    # initialize an empty list to store the cleaned strings
    cleaned = []
    # loop through the list of strings
    for string in list_of_strings:
        # if the string is a key in the abbreviations dictionary, replace it with the value
        if string in abbreviations.keys():
            # lemmatize the string and append to the cleaned list
            cleaned.append(abbreviations[string])
        else:
            cleaned.append(string)
    # return the cleaned list
    return cleaned

# a function that takes in a string, splits it into characters, if there are more than 3 consecutive characters that are the same, it removes the extra characters, else it returns the original string
def remove_repeated_characters(string):
    # split the string into characters
    characters = list(string)
    # initialize an empty list to store the cleaned characters
    cleaned = []
    # initialize a counter to count the number of consecutive characters
    count = 0
    # loop through the characters
    for i in range(len(characters)):
        # if the character is the same as the previous character, increment the counter
        if characters[i] == characters[i-1]:
            count += 1
        # if the character is not the same as the previous character, reset the counter
        else:
            count = 0
        # if the counter is less than 2, append the character to the cleaned list
        if count < 2:
            cleaned.append(characters[i])
    # join the cleaned list of characters into a string
    cleaned = ''.join(cleaned)
    # return the cleaned string
    return cleaned

# a function that takes in a list of strings and returns a list of strings with fewer repeated characters
def remove_repeated_characters_list(list_of_strings):
    return [remove_repeated_characters(string) for string in list_of_strings]

# a function that takes in a list of strings and returns a list of strings with corrected spelling
def correct_spelling(list_of_words):
    spells = [spell(word) for word in list_of_words]
    return spells   

# a function to lemmatize the words in a list of strings and return a list of lemmatized strings
def lemmatize(list_of_strings):
    # initialize an empty list to store the lemmatized strings
    lemmatized = []
    # loop through the list of strings
    for string in list_of_strings:
        # lemmatize the string and append to the lemmatized list
        lemmatized.append(lemmatizer.lemmatize(string))
    # return the lemmatized list
    return lemmatized

# a function that takes in a string and returns the compound sentiment score
def get_sentiment_score(string):
    return sid.polarity_scores(string)['compound']

# a function to round the sentiment score to nearest whole number based on whether the threshold is met
def round_sentiment_score(score):
    if score >= threshold:
        return 1
    elif score <= -1 * threshold:
        return -1
    else:
        return 0

# preparing abbreviations
# open and read abbreviations.txt
with open('abbreviations.txt', 'r') as f:
    abbreviations = f.read().splitlines()

# convert everything to lowercase
abbreviations = [line.lower() for line in abbreviations]

# split each line into a dictionary of key-value pairs
abbreviations = [line.split('-') for line in abbreviations]
# remove leading and trailing whitespace from keys and values
abbreviations = [[key.strip(), value.strip()] for key, value in abbreviations]

abbreviations = {key: value for key, value in abbreviations}

# initialize spark session
spark = SparkSession.builder.appName('data_cleaning').getOrCreate()
sc = spark.sparkContext

# read the csv files and create pyspark dataframes
reddit = spark.read.csv('Reddit_Data.csv', header=True)
twitter = spark.read.csv('Twitter_Data.csv', header=True)

# rename the first column to 'clean_text'
reddit = reddit.withColumnRenamed('clean_comment', 'clean_text')

# merge the two dataframes
merged = reddit.union(twitter)

# drop any duplicates in clean_text column
merged = merged.dropDuplicates(['clean_text'])

# drop any rows with null values
merged = merged.na.drop()

# filter for rows containing at least one of the relevant words in bjp_words set and no words in congress_words set
merged_bjp = merged.filter(merged.clean_text.rlike('|'.join(bjp_words)) & ~merged.clean_text.rlike('|'.join(congress_words)))

# filter for rows containing at least one of the relevant words in congress_words set and no words in bjp_words set
merged_congress = merged.filter(merged.clean_text.rlike('|'.join(congress_words)) & ~merged.clean_text.rlike('|'.join(bjp_words)))

parties = [merged_bjp, merged_congress]

# tokenize each row of the clean_text column and store in a new column in the dataframe
# replace word that match a abbreviations key with the abbreviations value
# replace word that has more than 3 consecutive characters with the word with only 3 consecutive characters
# use nltk spelling corrector to correct spelling
# lemmatize the words in the tokenized_text column

# register UDFs
remove_non_alpha_udf = udf(remove_non_alpha, ArrayType(StringType()))
replace_abb_udf = udf(lambda x: replace_abb(x, {}), ArrayType(StringType()))
remove_repeated_characters_udf = udf(remove_repeated_characters_list, ArrayType(StringType()))
correct_spelling_udf = udf(correct_spelling, ArrayType(StringType()))
lemmatize_udf = udf(lemmatize, ArrayType(StringType()))
word_tokenize_udf = udf(tokenize, StringType())
get_sentiment_score_udf = udf(get_sentiment_score, FloatType())
round_sentiment_score_udf = udf(round_sentiment_score, IntegerType())

# processing sentiments
for i, party in enumerate(parties):
    # apply UDFs to the dataframe in one go to avoid multiple passes
    party = party.withColumn('tokenized_text', 
                            lemmatize_udf(
                                correct_spelling_udf(
                                    remove_repeated_characters_udf(
                                        replace_abb_udf(
                                            remove_non_alpha_udf(
                                                word_tokenize_udf(
                                                    merged.clean_text
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
    # join the tokens in the tokenized_text column into a string
    party = party.withColumn('processed_text', udf(lambda x: ' '.join(x), StringType())(party.tokenized_text))

    # apply UDF to the dataframe
    party = party.withColumn('sentiment_score', get_sentiment_score_udf(party.processed_text))

    # round sentiment score to nearest whole number
    party = party.withColumn('sentiment', round_sentiment_score_udf(party.sentiment_score))

    # drop the tokenized_text column
    party = party.drop('tokenized_text')

    filename = 'bjp' if i == 0 else 'congress'

    # summarise the sentiments
    positive = party.filter(party.sentiment == 1).count()
    negative = party.filter(party.sentiment == -1).count()
    neutral = party.filter(party.sentiment == 0).count()
    total = party.count()
    net = positive - negative    
    average = net/total
    
    # summarise the sentiments
    positive_original = party.filter(party.category == 1).count()
    negative_original = party.filter(party.category == -1).count()
    neutral_original = party.filter(party.category == 0).count()
    total_original = party.count()
    net_original = positive_original - negative_original    
    average_original = net_original/total_original

    # save the summary to a text file
    with open(f'{filename}_summary.txt', 'w') as f:
        f.write('Sentiments Summary\n')
        f.write('New:\n')
        f.write(f'Positive: {positive}\nNegative: {negative}\nNeutral: {neutral}\nTotal: {total}\nNet: {net}\nAverage: {average}\n\n')
        f.write('Original:\n')
        f.write(f'Positive: {positive_original}\nNegative: {negative_original}\nNeutral: {neutral_original}\nTotal: {total_original}\nNet: {net_original}\nAverage: {average_original}\n')
    
    
    # check if merged folder exists and delete it if it does
    if os.path.exists(filename):
        shutil.rmtree(filename)

    # write the dataframe to a csv file
    party.select("processed_text","category","sentiment").write.save(filename, format="csv")

