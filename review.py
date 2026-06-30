import pandas as pd
import re
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

# Download (only first time)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Load data
df = pd.read_csv(
    "Amazon_Reviews.csv",
    encoding="latin1",
    engine="python",
    on_bad_lines="skip"
)

print("✅ Data Loaded:", df.shape)

# fix empty strings
df = df.replace(["", " "], "N/A")

df['Rating'] = (
    df['Rating']
    .astype(str)
    .str.extract(r'(\d+)')
)

df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

df['Rating'] = df['Rating'].fillna(0)

# date experience
df['Date of Experience'] = df['Date of Experience'].replace(["", " "], "N/A")
df['Date of Experience'] = df['Date of Experience'].fillna("N/A")

# link columns clean
df['Profile Link'] = df['Profile Link'].apply(
    lambda x: x if isinstance(x, str) and "/users/" in x else "N/A"
)

# reviewers column clean
df['Reviewer Name'] = df['Reviewer Name'].apply(
    lambda x: x if isinstance(x, str) and x.replace(" ", "").isalpha() else "N/A"
)

# all blank rows
df = df.dropna(how="all")

# Clean data
df.drop_duplicates(inplace=True)
df.dropna(subset=['Review Text'], inplace=True)

# NLP setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(words)

df['Cleaned_Text'] = df['Review Text'].apply(preprocess_text)

# Remove empty rows
df = df[df['Cleaned_Text'] != ""]

# SENTIMENT ANALYSIS 

sia = SentimentIntensityAnalyzer()

# VADER sentiment
def vader_sentiment(text):
    score = sia.polarity_scores(str(text))['compound']
    
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# TextBlob sentiment (for comparison)
def textblob_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply BOTH
df['Sentiment_VADER'] = df['Review Text'].apply(vader_sentiment)   # 🔥 original text pe
df['Sentiment_TextBlob'] = df['Cleaned_Text'].apply(textblob_sentiment)

# Score column
df['Sentiment_Score'] = df['Review Text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# Final sentiment (use VADER as main)
df['Final_Sentiment'] = df['Sentiment_VADER']

# TOP WORDS

all_words = " ".join(df['Cleaned_Text']).split()
print("Top 10 Words:", Counter(all_words).most_common(10))

df.to_csv("nlp_output.csv", index=False)

print("✅ NLP + ADVANCED SENTIMENT DONE!")

# Preview
print(df[['Review Text','Cleaned_Text','Sentiment_VADER','Sentiment_TextBlob','Final_Sentiment']].head())