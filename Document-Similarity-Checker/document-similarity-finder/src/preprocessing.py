from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('wordnet')

def load_documents(file_path):
    df = pd.read_csv(file_path)
    return df['text'].tolist()

def tokenize(text):
    tokens = text.split()
    return tokens

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token.lower() not in stop_words]

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_documents(documents):
    processed_docs = []
    for doc in documents:
        tokens = tokenize(doc)
        tokens = remove_stopwords(tokens)
        tokens = lemmatize(tokens)
        processed_docs.append(' '.join(tokens))
    return processed_docs

def compute_tfidf_matrix(processed_docs):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    return tfidf_matrix