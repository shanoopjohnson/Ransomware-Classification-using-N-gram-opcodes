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


print("Loading the created Dataset")
data = open('C:/Users/johns/Documents/final_year_project/june-21/equal_mix.txt').read()
labels1, opcode_sequence1 = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    try:
        labels1.append(content[0])
        opcode_sequence1.append(" ".join(content[1:]))
        
    except:
        print (i)
        pass

    
print("creating dataframe using texts and lables........")
trainDF1 = pandas.DataFrame()
trainDF1['opcode_sequence1'] = opcode_sequence1
trainDF1['label1'] = labels1
valid_x1=trainDF1['opcode_sequence1']
trainDF1['label1']=pd.to_numeric(trainDF1['label1'])
valid_y1=trainDF1['label1'].tolist()

tfidf_vect_ngram1= TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=500)
tfidf_vect_ngram1.fit(trainDF1['opcode_sequence1'])
#xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram1 =  tfidf_vect_ngram1.transform(valid_x1)
#features
indices = np.argsort(tfidf_vect_ngram1.idf_)[::-1]
features = tfidf_vect_ngram1.get_feature_names()
print("feture length: ",len(features))

loaded_model = pickle.load(open("voting_classifier.sav", "rb"))
predictions1 = loaded_model.predict(xvalid_tfidf_ngram1)
cnf_matrix1 = confusion_matrix(valid_y1, predictions1,labels=[0, 1])
accuracy1 = accuracy_score(valid_y1, predictions1)
#print("prediction: ",predictions1)
print("confusion matrix: \n",cnf_matrix1)
print("accuracy: ",accuracy1)