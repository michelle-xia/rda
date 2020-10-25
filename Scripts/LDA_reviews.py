#!/usr/bin/python3.8
#Install LDA library if not already installed
#pip install --user lda
# Specify inputs in line 26
# there are two output files. Set the names of these files in lines 33 and 34
import os, csv, nltk, lda
import pandas as pd
import numpy as np
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import PunktSentenceTokenizer,RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Set inputs here
input_file = ""

id_col = 'id' # specify name of column enumerating each individual doc
text_col = 'Post' # specify name of column with text entries
ntopics= 7  # set number of topics to create

# Output names
output_text_file = ""  # set name for file with text distribution per topic
output_doc_file = ""   # set name for file with topic distribution per line

df=pd.read_csv(input_file, encoding='latin1')

# check for null values
print("Number of rows with any of the empty columns:")
print(df.isnull().sum().sum())
df=df.dropna()



word_tokenizer=RegexpTokenizer(r'\w+')
wordnet_lemmatizer = WordNetLemmatizer()
stopwords_nltk=set(stopwords.words('english'))

def tokenize_text(version_desc):
    lowercase=version_desc.lower()
    text = wordnet_lemmatizer.lemmatize(lowercase)
    tokens = word_tokenizer.tokenize(text)
    return tokens

vec_words = CountVectorizer(tokenizer=tokenize_text,stop_words=stopwords_nltk,decode_error='ignore')
total_features_words = vec_words.fit_transform(df[text_col])

print(total_features_words.shape)

model = lda.LDA(n_topics=int(ntopics), n_iter=500, random_state=1)
model.fit(total_features_words)

topic_word = model.topic_word_
doc_topic=model.doc_topic_
doc_topic=pd.DataFrame(doc_topic)
df=df.join(doc_topic)
doc=pd.DataFrame()

for i in range(int(ntopics)):
    topic="topic_"+str(i)
    doc[topic]=df.groupby([id_col])[i].mean()

doc=doc.reset_index()
topics=pd.DataFrame(topic_word)
topics.columns=vec_words.get_feature_names()
topics1=topics.transpose()

topics1.to_excel(output_text_file)
print ("Topics word distribution written to", output_text_file)
doc.to_excel(output_doc_file, index=False)
print ("Text topic distribution written to", output_doc_file)