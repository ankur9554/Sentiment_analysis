#importing packages
# DataFrame
import pandas as pd 

# nltk
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

# Utility
import pandas as pd
import numpy as np 
import warnings
warnings.filterwarnings('ignore')
import re
import string
import pickle
import csv
import streamlit as st

nltk.download('stopwords')
stopword = set(stopwords.words('english'))
print(stopword)
nltk.download('punkt')
nltk.download('wordnet')

def csv_to_list(uploaded_file):
    column_index = 0 # Adjust this to the appropriate column index
    # Initialize an empty list to store the column values
    column_list = []
    csvreader = csv.reader(uploaded_file)
    for row in csvreader:
        if len(row) > column_index:
            column_list.append(row[column_index])
    return column_list


def predict(vectoriser, model, text):
    # Predict the sentiment
    processes_text=[process_tweets(sen) for sen in text]
    textdata = vectoriser.transform(processes_text)
    sentiment = model.predict(textdata)

    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern = '@[^\s]+'
#Function to remove a pattern
def process_tweets(tweet):
    # Lower Casing
    tweet = tweet.lower()
    tweet=tweet[1:]
    # Removing all URls 
    tweet = re.sub(urlPattern,'',tweet)
    # Removing all @username.
    tweet = re.sub(userPattern,'', tweet) 
    #Remove punctuations
    tweet = tweet.translate(str.maketrans("","",string.punctuation))
    #tokenizing words
    final_tokens = word_tokenize(tweet)
    '''#Removing Stop Words
    final_tokens = [w for w in tokens if w not in stopword]'''
    #reducing a word to its word stem 
    wordLemm = WordNetLemmatizer()
    finalwords=[]
    for w in final_tokens:
      if len(w)>1:
        word = wordLemm.lemmatize(w)
        finalwords.append(word)
    return ' '.join(finalwords)

from pandas._libs.tslibs import vectorized



#loading pickle files   model, vectorizer
model = pickle.load(open('Pickle/SVM.pickle','rb'))
vectorizer = pickle.load(open('Pickle/vectoriser.pickle','rb'))


st.title('Social Media Post Analyzer')


uploaded_file = st.file_uploader("Upload your file here...", type=['csv'])
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file,encoding='latin1')
    st.write(dataframe)
    datalist = csv_to_list(dataframe)
    df = predict(vectorizer, model, datalist)
    st.write(df)
    Sentiment= vectorizer.transform(dataframe['clean_text'])
    prediction = model.predict(Sentiment)
    st.write(prediction)