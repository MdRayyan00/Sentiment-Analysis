# Importing Modules.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, render_template, request, redirect
from nltk.classify import NaiveBayesClassifier
from nltk import NaiveBayesClassifier as nbc
from nltk.stem import PorterStemmer, porter
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from itertools import chain
from nltk import tokenize
import nltk.classify.util
import string
import nltk

nltk.download('movie_reviews')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

app = Flask(__name__)


def tokenizer(text):
    stops = list(string.punctuation)
    tokens = []
    for word in text:
        word.lower()
        if word not in stops:
            tokens.append(word)
    return tokens


def word_feats(words):
    return dict([(word, True) for word in words])


pos_fileids = movie_reviews.fileids('pos')
neg_fileids = movie_reviews.fileids('neg')

pos_word_set = []
neg_word_set = []

for file in pos_fileids:
    raw_words = movie_reviews.words(fileids=[file])
    raw_words = tokenizer(raw_words)
    temp_set = word_feats(raw_words)

    # give positive word dictionary 'pos' tag
    pos_word_set.append((temp_set, 'pos'))

for file in neg_fileids:
    raw_words = movie_reviews.words(fileids=[file])
    raw_words = tokenizer(raw_words)
    temp_set = word_feats(raw_words)

    # give negative word dictionary 'neg' tag
    neg_word_set.append((temp_set, 'neg'))

splitter = int(len(pos_word_set) * 0.8)
# splitter2 = int(len(neg_word_set) *0.8)
# print(splitter)

train_set = pos_word_set[:splitter] + neg_word_set[:splitter]


classifier = NaiveBayesClassifier.train(train_set)


def classify_text(text):
    tokenized_text = tokenizer(text)
    featured_text = word_feats(tokenized_text)
    result = classifier.classify(featured_text)
    result_prob = classifier.prob_classify(featured_text)
    return result, result_prob.prob('pos'), result_prob.prob('neg')


@app.route('/')
def default():
    return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
def form():
    input_text = request.form['enteredText']
    # print(input_text)
    result, result_prob_pos, result_prob_neg = classify_text(input_text)
    return render_template('index.html', text=input_text, finalPos=result_prob_pos, finalNeg=result_prob_neg, result=result)


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8000, threaded=True)
