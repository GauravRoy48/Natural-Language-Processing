#######################################################################################
# Creator     : Gaurav Roy
# Date        : 20 May 2019
# Description : The code performs Natural Language Processing algorithm on the 
#               Restaurant_Reviews.tsv. It uses Decision Tree as classification model.
#######################################################################################

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts

import re
# Remove non-significant words (eg: the, and, in, etc)
import nltk
# Download all the words that are irrelevant in a text
nltk.download('stopwords')
# Import stopwords
from nltk.corpus import stopwords
# Required for stemming step
from nltk.stem.porter import PorterStemmer

# Creating a corpus of 1000 reviews
corpus = []
for i in range(0, len(dataset)):
    # Stemming step: Keeping only root of a word
    ps = PorterStemmer()
    # Removing everything except letters. Replace deleted characters with ' '
    # Chaning string to lower case
    # Split the review string into a list of words
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]).lower().split()
    # Removing irrelevant words from review using stopwords
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # Joining back the words to form a single string
    review = ' '.join(review)
    corpus.append(review)

# Create Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

Y = dataset.iloc[:,1].values

#############################
# USING DECISION TREES
#############################

# Splitting to Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)

# Fitting Decision Tree Classifier to Training Set
# Create Classifier Here
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)

# Predicting the Test Set Results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Calculating Performance Metrics
dtc_accuracy = sum(cm.diagonal())/cm.sum()
dtc_precision = cm[1][1]/(cm[0][1] + cm[1][1])
dtc_recall = cm[1][1]/(cm[1][1]+cm[1][0])
dtc_f1score = 2*dtc_precision*dtc_recall/(dtc_precision+dtc_recall)
