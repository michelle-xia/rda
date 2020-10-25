#!/usr/bin/python3.8
# install pandas with pip install --user pandas
# install nltk with pip install --user nltk
# Set inputs after line 91
# Uncomment line 98 for single column analysis or line 101 for multiple column topic analysis
# Custom update vader dictionary in lines 21 and 22
# This program returns the sentiment scores for a single column of text using the VADER dictionary

import pandas as pd
import numpy as np
import nltk
import sys

nltk.download("vader_lexicon")
def get_sentiment(rating_data):
    """
    https: // github.com / cjhutto / vaderSentiment
    :return:
    """
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    # newWords = {'good': 2.0, 'down': 2.0, 'normal': 2.0, 'well': 2.0, 'negative': 2.0, 'positive': -2.0, 'cases': -2.0, 'covid': -2.0, 'spike': -2.0}
    # sid.lexicon.update(newWords)
    
    rating_data['sent_neg'] = -10
    rating_data['sent_neu'] = -10
    rating_data['sent_pos'] = -10
    rating_data['sent_compound'] = -10
    print("\nRATING DATA", rating_data, "\n")
    for i in range(len(rating_data)):
        sentence = rating_data['Sentences'][i]
        print(sentence)
        ss = sid.polarity_scores(sentence)
        print(ss['neg'])
        rating_data.loc[i, 'sent_neg'] = float(ss['neg'])
        print(rating_data.loc[i, 'sent_neg'])
        rating_data.loc[i, 'sent_neu'] = ss['neu']
        rating_data.loc[i, 'sent_pos'] = ss['pos']
        rating_data.loc[i, 'sent_compound'] = ss['compound']
    return rating_data

def parse_topics_for_sentiment(input_file, topics, topic):
    """This function takes in a file with extracted text per topic
        and saves the sentiment scores for each line within a topic in an excel file"""
    # read input
    df = pd.read_csv(input_file, encoding = 'latin1')
    df.rename(columns=lambda x: x.strip(), inplace=True)

    try:
        topics.remove(topic)
    except ValueError:
        print("Topic does not exist in list of topics")
        sys.exit(0)

    # drop other topics
    df = df.drop(columns=topics)
    print("\n",df.isnull().sum().sum(), "null columns")
    df = df.dropna()
    cleaned = df.values.tolist()
    cleaned = [val[0] for val in cleaned]

    # new DataFrame with index starting from 0
    new_df = pd.concat([pd.DataFrame([i], columns=['id']) for i in range(len(cleaned))],
            ignore_index=True)
    new_df['Sentences'] = cleaned

    # get sentiment for topic
    sentiment_data = get_sentiment(new_df)

    # write output to file
    output_name = topic + "SentimentOutput.xlsx"
    sentiment_data.to_excel(output_name, index=False)
    print("Written to", output_name)



def single_column_sentiment(input_file, output_file):
    """Input: CSV file with text in the first row
        Output: Excel file with sentiment scores for each line"""
    df = pd.read_csv(input_file, encoding = 'latin1')
    df = df.rename(columns={ df.columns[0]: "Sentences" })

    sentiment_data = get_sentiment(df)
    sentiment_data.to_excel(output_file, index = False)
    print("Written to", output_file)




if __name__ == "__main__":
    # Set inputs
    input_file = "" # input file is CSV
    topics = [] # list with names of all topics if doing topic sentiment analysis
    topic = "" # topic you want the sentiment for
    output_file = "SentimentOutput.xlsx" # excel output file

    # Uncomment line below for single column sentiment
    # single_column_sentiment(input_file, output_file)

    # Uncomment line below for multiple topic sentiment
    # parse_topics_for_sentiment(input_file, topics, topic)
