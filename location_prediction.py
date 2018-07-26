import re
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk import SnowballStemmer
import sklearn.linear_model as sk
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
import string

"""
Tokenize tweets.
"""
def tokenize(data):

    stemmer = SnowballStemmer("english")
    stop_words = text.ENGLISH_STOP_WORDS
    temp = data
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    temp = regex.sub(' ', temp)
    temp = "".join(b for b in temp if ord(b) < 128)
    temp = temp.lower()
    words = temp.split()
    no_stop_words = [w for w in words if not w in stop_words]
    stemmed = [stemmer.stem(item) for item in no_stop_words]

    return stemmed

"""
Reduces dataset to only contain tweets with location
marked as either WA or MA.
"""
def reduce_by_location(data):
    temp = data[(data.location.str.contains(r'[.]+ WA$'))
            | (data.location.str.contains(r'[.]+ MA$'))
            | (data.location.str.contains('Boston'))
            | (data.location.str.contains('Seattle'))
            | (data.location.str.contains(r'[.]+ Washington\s'))
            | (data.location.str.contains('Massachusetts'))]
    return temp

"""
Creates target variables 1: for WA and 0: MA.
"""
def map_locations(data):
    targets = []
    for location in data.location.apply(lambda x: x.encode('utf-8').strip()):
        if (r'[.]+ WA$' in location) or ('Seattle' in location) or (r'[.]+ Washington\s' in location):
            targets.append(1)
        else:
            targets.append(0)
    return np.array(targets)

"""
Balances datasets by selecting random points from
the minority class.
"""
def balance_datasets(data, targets):
    new_data = data.copy()
    if (len(targets[targets==0])) > (len(targets[targets==1])):
        points_needed = len(targets[targets==0]) - len(targets[targets==1])
        indices = np.where(targets == 1)
    else:
        points_needed = len(targets[targets==1]) - len(targets[targets==0])
        indices = np.where(targets == 0)

    np.random.shuffle(indices)
    indices = np.resize(indices, points_needed)
    new_data = new_data.append(data.iloc[indices])
    targets_to_add = targets[indices]
    new_targets = np.concatenate([targets, targets_to_add])
    return new_data, new_targets

DATA_FOLDER = 'tweet_data/'
filename = 'tweets_#superbowl.txt'

# Collect tweets from superbowl
tweets_ = []
with open(DATA_FOLDER + filename, 'r') as f:
    for row in f:
        jrow = json.loads(row)
        d = {
            'tweet': jrow['title'],
            'location': jrow['tweet']['user']['location']
        }
        tweets_.append(d)
all_data = pd.DataFrame(tweets_)

# Filter out tweets by appropriate location data
reduced_data = reduce_by_location(all_data)

# Create target label
# 0: MA 1: WA
all_targets = map_locations(reduced_data)

# Balance datset
data, train_targets = balance_datasets(reduced_data, all_targets)

# # Vectorize tweets
vectorizer = CountVectorizer(analyzer='word', stop_words='english', tokenizer=tokenize)
tfidf_transformer = TfidfTransformer()
train_counts = vectorizer.fit_transform(data.tweet)
train_tfidf = tfidf_transformer.fit_transform(train_counts)

# Truncate twitter data to 50 features
svd = TruncatedSVD(n_components=50, random_state=42)
train_reduced = svd.fit_transform(train_tfidf)

# Feature Scaling For Certain Algorithms Require Nonnegative Values
min_max_scaler = preprocessing.MinMaxScaler()
train_data = min_max_scaler.fit_transform(train_reduced)

# Perform 5-Fold CV to fit Naive Bayes Model
k=5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

accuracies = 0
for train_index, test_index in kf.split(train_data):
    X_train, X_test = train_data[train_index], train_data[test_index]
    y_train, y_test = train_targets[train_index], train_targets[test_index]

    clf = MultinomialNB().fit(X_train, y_train)
    predicted_bayes = clf.predict(X_test)
    accuracy_bayes = np.mean(predicted_bayes == y_test)
    accuracies += accuracy_bayes

print "Average CV-Accuracy of Multinomial Naive Bayes: " + str(accuracies/k)
print(classification_report(y_test, predicted_bayes))
print "Confusion Matrix:"
print(confusion_matrix(y_test, predicted_bayes))

# Perform 5-Fold CV to fit Logistic Regression
accuracies = 0
for train_index, test_index in kf.split(train_data):
    X_train, X_test = train_data[train_index], train_data[test_index]
    y_train, y_test = train_targets[train_index], train_targets[test_index]

    logit = sk.LogisticRegression().fit(X_train, y_train)
    probabilities = logit.predict(X_test)
    predicted_lr = (probabilities > 0.5).astype(int)
    accuracy_lr = np.mean(predicted_lr == y_test)
    accuracies += accuracy_lr

print "Average CV-Accuracy of Logistic Regression: " + str(accuracies/k)
print(classification_report(y_test, predicted_lr))
print "Confusion Matrix:"
print(confusion_matrix(y_test, predicted_lr))

accuracies = 0
for train_index, test_index in kf.split(train_data):
    X_train, X_test = train_data[train_index], train_data[test_index]
    y_train, y_test = train_targets[train_index], train_targets[test_index]

    # Perform 5-Fold CV to fit Logistic Regression
    linear_SVM = LinearSVC(dual=False, random_state=42).fit(X_train, y_train)
    predicted_svm = linear_SVM.predict(X_test)
    accuracy_svm = np.mean(predicted_svm == y_test)
    accuracies += accuracy_svm

print "Average CV-Accuracy of Linear SVM: " + str(accuracies/k)
print(classification_report(y_test, predicted_svm))
print "Confusion Matrix:"
print(confusion_matrix(y_test, predicted_svm))