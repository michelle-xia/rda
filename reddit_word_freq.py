import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
df = pd.read_csv('aggregate_reddit_text.csv', engine='python')
df.rename(columns=lambda x: x.strip(), inplace=True)
print(df)
            