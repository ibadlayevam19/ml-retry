###  Bernoulli Naive Bayes on Binary Text Data

* Setup: SMS Spam Classification using Bernoulli Naive Bayes

* Step 1: Load the dataset from this URL:
* https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

* Step 2: Convert the 'label' column into binary values: spam = 1, ham = 0

* Step 3: Use CountVectorizer with binary=True to transform the text into binary features

* Step 4: Split the dataset into training and test sets (e.g., 70/30 split)

* Step 5: Initialize and train a BernoulliNB model

* Step 6: Predict on the test set and evaluate using accuracy and confusion matrix