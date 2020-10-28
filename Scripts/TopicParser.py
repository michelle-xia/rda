#!/usr/bin/python3.8
# Install pandas library if not already installed
# pip install --user pandas
# This program returns each all text entries associated with a certain topic. If you do not
# have topic modeling per Doc, run LDA_reviews.py
# Set inputs after line 46
import pandas as pd


def merge_doc_with_post(text_input, doc_input, output= "merged_doctopics_and_text.csv"):
    """This function merges text with each individual doc
        input: File with text, File with each Doc and its topic modeling breakdown
        output: Merged file for each Doc with its text"""
    agg = pd.read_csv(text_input, encoding='latin1')
    agg.rename(columns=lambda x: x.strip(), inplace=True)

    doc = pd.read_csv(doc_input, encoding='latin1')
    doc.rename(columns=lambda x: x.strip(), inplace=True)

    merged = doc.merge(agg, on='id', how='outer')
    merged.to_csv(output)
    return output


def extract(inputfile, topics, tolerance = 0.5,  final_output = "ExtractedPostsPerTopic.csv"):
    """This function extracts all text related to a certain topic
        input: Merged file with LDA scores and text
        output: File with text in each topic"""
    df = pd.read_csv(inputfile, encoding='latin1')
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # topics = ['SocialJustice', 'Science', 'OnlineLearning', 'Housing', 'MSCS', 'Groups', 'COVID', 'Friends', 'Voting'] 
    # topics = ['Football', 'Voting', 'Covid']
    postdf = pd.DataFrame(index=range(len(df)), columns=[topic for topic in topics])
    print(postdf)
    for i in range(1, len(df)):
        for topic in topics:
            if df[topic][i] > tolerance:
                postdf.loc[i, topic] = df['Post'][i]

    postdf.to_csv(final_output)
    print(final_output)

    
if __name__ == "__main__":
    # Set inputs here
    input_text_file = ""  # input file with text data to merge
    input_doc_file = ""   # input file with LDA topic modeling per doc
    output_merged_file_name = ""  # optional name for Merged File created

    output_file_name  = merge_doc_with_post(input_text_file, input_text_file)

    topic_names = []  # input a list of the topics in your input_doc_file here
    final_output_file_name = "" # optional name for final file with text from each topic
    tolerance = 0.5 # optional tolerance parameter, tweak this as needed for more or less text entries within a topic

    extract(output_file_name, topic_names)
