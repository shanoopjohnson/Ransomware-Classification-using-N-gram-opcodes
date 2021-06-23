import pickle
from sklearn.metrics import accuracy_score
import pandas
import numpy as np
import os
import glob
import csv
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, xgboost, numpy, textblob, string
from sklearn.metrics import confusion_matrix
import sklearn.metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn import datasets, svm

print("Loading the created Dataset:")
data = open('C:/Users/johns/Documents/final_year_project/dec-10- lab/main.txt').read()
#data = open('/content/drive/MyDrive/Mtech CSE FInal yr Project/main.txt').read()
labels, opcode_sequence = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    try:
        labels.append(content[0])
        opcode_sequence.append(" ".join(content[1:]))
        
    except:
        print ("number of samples",i)
        pass

    
# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['opcode_sequence'] = opcode_sequence
trainDF['label'] = labels
print("split dataset into train and test: ")
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['opcode_sequence'], trainDF['label'])

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

print("tf-idf vectorizing................. ")
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=500)
tfidf_vect_ngram.fit(trainDF['opcode_sequence'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
#features
indices = np.argsort(tfidf_vect_ngram.idf_)[::-1]
features = tfidf_vect_ngram.get_feature_names()

feature_names = tfidf_vect_ngram.get_feature_names()
dense = xvalid_tfidf_ngram.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
df.insert(0,column = "label", value = valid_y)
df.to_csv("CSV")

print("Creating voting classifier.....")
k_fold = KFold(n_splits=2)

estimator = [] 
estimator.append(('LR', LogisticRegression(solver ='lbfgs',multi_class ='multinomial', max_iter = 200))) 
estimator.append(('SVC', SVC(gamma ='auto', probability = True))) 
estimator.append(('DTC', DecisionTreeClassifier())) 
estimator.append(('RFC', ensemble.RandomForestClassifier())) 
estimator.append(('XGB', xgboost.XGBClassifier())) 
estimator.append(('AB', ensemble.AdaBoostClassifier()))
vot_hard = VotingClassifier(estimators = estimator, voting ='hard')

print("Training model using stratified K fold cross Validation........")
results=[vot_hard.fit(xtrain_tfidf_ngram, train_y).score(xvalid_tfidf_ngram, valid_y)for train, test in k_fold.split(xtrain_tfidf_ngram)]
print("Accuracy of each fold: ", results)
filename = 'voting_classifier.sav'
pickle.dump(vot_hard, open(filename, 'wb'))

print("--------------------------------------")
print("Testing the model accuracy: ")
predictions = vot_hard.predict(xvalid_tfidf_ngram)
cnf_matrix = confusion_matrix(valid_y, predictions,labels=[0, 1])
accuracy = accuracy_score(valid_y, predictions)
print("model confusion matrix: \n",cnf_matrix)
print("model accuracy: \n",accuracy)

