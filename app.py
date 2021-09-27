# Imports statements.
import streamlit as st
import pandas as pd
import re
import pandas as pd
import matplotlib.pyplot as plt

import scikitplot as skplt


#Data Preprocessing and Feature Engineering
# from wordcloud import WordCloud, STOPWORDS 
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score



st.title("COVID - 19 Tweets Sentimental Analysis")

st.write("#### Analyze the sentiments of people during COVID using Various Machine Learning Models along with their accuracy and results")

MODELS = ("Multinomial Naive Bayes","Logistic Regression","SVM")
    # Creating a mapping for sentiments
mapping = {'fear':0,
                'sad':1,
                'anger':2,
                'joy':3}

reverse = {
    0:'Fear',
    1:'Sad',
    2:'Anger',
    3:'Joy'
}
model_name = st.sidebar.selectbox(label="Select the Model",options = MODELS)

def get_dataset():
    df = pd.read_csv('./input/finalSentimentdata2.csv')
    
    df['sentiment'] = df['sentiment'].map(mapping)

    return df

# Loading the dataset
twitter_data = get_dataset()

# Basic cleaning step 
def clean_text_column(row):
    text = row['text'].lower()
    text = re.sub(r'[^(a-zA-Z\s)]','',text)
    text = re.sub(r'\(','',text)
    text = re.sub(r'\)','',text)
    text = text.replace('\n',' ')
    text = text.strip()
    return text

twitter_data['transformed_text'] = twitter_data.apply(clean_text_column,axis = 1)

# Stopword Removal
# These are new stopwords which i add after several model runs and found out these are irrelevant words which are created which cleaning process.
new_additions=['aren', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'let', 'll', 'mustn', 're', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn']
stopwords_set = stopwords.words('english') + new_additions
twitter_data['transformed_text'] = twitter_data.apply(lambda x : remove_stopwords(x['transformed_text']),axis = 1)

#Normalizing the words in tweets 
def normalization(tweet):
    lem = WordNetLemmatizer()
    normalized_tweet = []
    for word in tweet['transformed_text'].split():
        normalized_text = lem.lemmatize(word,'v')
        normalized_tweet.append(normalized_text)
    return " ".join(normalized_tweet)

twitter_data['transformed_text'] = twitter_data.apply(normalization,axis = 1)


# Spliting the data in train - test
msg_train, msg_test, label_train, label_test = train_test_split(twitter_data['transformed_text'],twitter_data['sentiment'], test_size=0.1,random_state = 2)

def get_model_pipeline(model_name):

    if model_name == "Multinomial Naive Bayes":
        pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words=stopwords_set)),
                    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
                        ])
    elif model_name == "Logistic Regression":
        pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words=stopwords_set)),
                    ('classifier',LogisticRegression(solver='sag')),  # train on TF-IDF vectors w/ Naive Bayes classifier
                        ])
    elif model_name == "SVM":
        pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words=stopwords_set)),
                    ('clf', OneVsRestClassifier(SVC(), n_jobs=1)),
                        ])
    
    return pipeline

pipeline = get_model_pipeline(model_name)
pipeline.fit(msg_train, label_train)

st.write(f"Model {model_name} successfully trained!")

prediction = pipeline.predict(msg_test)

accuracy = accuracy_score(label_test, prediction)

st.write(f"##### Model accuracy: %.2f"%(accuracy*100))

skplt.metrics.plot_confusion_matrix(
    label_test, 
    prediction,
    figsize=(8,8))

st.pyplot()

def test_tweet_transformation(tweet):
    text = tweet
    def clean(text):
        text = text.lower()
        text = re.sub(r'[^(a-zA-Z\s)]','',text)
        text = re.sub(r'\(','',text)
        text = re.sub(r'\)','',text)
        text = text.replace('\n',' ')
        text = text.strip()
        return text
    
    text = clean(text)
    
    text = remove_stopwords(text)

    def normalization(text):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in text.split():
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return " ".join(normalized_tweet)
    
    normalized_text = normalization(text)
    return [normalized_text]

st.write("### Please select a tweet to check the prediction")
sample_tweets = twitter_data['text'].sample(10,random_state = 0)
# print(sample_tweets)

test_tweet_pred = None
try:
    tweet = st.selectbox(label = "Select sample test Tweet",options=tuple(sample_tweets))
    # Taking test tweet from user
    test_tweet_pred = pipeline.predict(test_tweet_transformation(tweet))
    st.write(f"Test Tweet prediction: {reverse[test_tweet_pred[0]]}")
except ValueError:
    st.write("No test tweet given")

