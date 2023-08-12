import streamlit as st
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv
import pickle
import re
import string

nltk.download('stopwords')
stopword = set(stopwords.words('english'))
print(stopword)
nltk.download('punkt')
nltk.download('wordnet')


def csv_to_list(file_path):
    column_index = 0  # Column index for the desired column
    column_list = []

    try:
        with open(file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if len(row) > column_index:
                    column_list.append(row[column_index])
        return column_list
    except Exception as e:
        return str(e) 

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
    tokens = word_tokenize(tweet)
    #Removing Stop Words
    final_tokens = [w for w in tokens if w not in stopword]
    #reducing a word to its word stem 
    wordLemm = WordNetLemmatizer()
    finalwords=[]
    for w in final_tokens:
      if len(w)>1:
        word = wordLemm.lemmatize(w)
        finalwords.append(word)
    return ' '.join(finalwords)

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

#loading pickle files   model, vectorizer
model = pickle.load(open('Pickle/SVM.pickle','rb'))
vectorizer = pickle.load(open('Pickle/vectoriser.pickle','rb'))


st.title('Depression Analysis of Social Media Posts')
uploaded_file = st.file_uploader("Upload your file here...", type=['csv'])
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file,encoding='latin1')
    st.write(dataframe)
    datalist = dataframe['SentimentText'].tolist()
    # Preprocess the data
    processed_datalist = [process_tweets(sen) for sen in datalist]
    
    # Predict sentiment
    df = predict(vectorizer, model, processed_datalist)
    
    st.write(df)
