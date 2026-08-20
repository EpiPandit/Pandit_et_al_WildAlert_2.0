
# coding: utf-8

# In[ ]:

from __future__ import division
import os as os
get_ipython().magic(u'matplotlib inline')
#get_ipython().magic(u'run Settings.py')
import re
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose

from itertools import count
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import pandas as pd
import collections
from textblob import TextBlob


import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
#from sklearn.cross_validation import
from sklearn.model_selection import StratifiedKFold ,cross_val_score, train_test_split, cross_val_predict
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import learning_curve
from pandas_ml import ConfusionMatrix
import nltk
#nltk.download()
from nltk.corpus import stopwords
stops = set(stopwords.words("english")) 


# In[ ]:


def read_data(filename):
    data = pd.read_csv(filename, low_memory=False)
    data.columns = ['WRMD_ID','Case', 'Admitted_at', 'Species', 'Organization','Address_found','Latitude', 'Longitude',
                    'Disposition', 'Dispositioned_at', 'Age', 'label', 'Reasons', 'Condition','Diagnosis', 'Notes']
    data['Admitted_at'] = pd.to_datetime(data['Admitted_at']).dt.strftime('%Y-%m-%d')
    data.set_index(pd.DatetimeIndex(data['Admitted_at']), inplace= True, drop= False)
    data['label'] = data.label.str.lower()
    df = data.label.str.split(',', expand=True) 
    df.columns = ['c1', 'c2','c3','c4','c5']
    df['c1'] = df.c1.str.strip()
    df['c2'] = df.c2.str.strip()
    df['c3'] = df.c3.str.strip()
    df['c4'] = df.c4.str.strip()
    df['c5'] = df.c4.str.strip()
    df.replace('physical inury', 'physical injury', inplace= True)
    df.replace('physical injury', 'physical injury', inplace= True)
    df.replace('neurological disease', 'neurologic disease', inplace =  True)
    df.replace('nerologic disease', 'neurologic disease', inplace =  True)
    df.replace('nekurologic disease', 'neurologic disease', inplace =  True)
    df.replace('nutritional disease', 'nutritional disease', inplace =  True)
    df.replace(' nutritional disease', 'nutritional disease', inplace =  True)
    df.replace('petrochemical exposure ', 'petrochemical exposure', inplace =  True)
    df.replace('gastrointestinal disease ', 'gastrointestinal disease', inplace =  True)
    df.replace('gastrointestinal  disease', 'gastrointestinal disease', inplace =  True)
    df.replace('physical  injury', 'physical injury', inplace= True)
    df.replace('physical trauma', 'physical injury', inplace= True)
    df.replace('nonspecifc', 'nonspecific', inplace= True)
    df.replace('nonspeciic', 'nonspecific', inplace= True)
    df.replace('neurologic diseases', 'neurologic disease', inplace= True)
    df.replace('beaced', 'stranded', inplace= True)

    df.replace('physical inury', 'physical injury', inplace= True)
    df.replace('physcial injury', 'physical injury', inplace= True)
    df.replace('phyical injury', 'physical injury', inplace= True)
    def get_predicting_condition(c):
        if not c.c2:
            return c.c1
        elif (c.c1 == 'stranded')| (c.c1 == 'physical injury'):
            return c.c2
        else:
            return c.c1


    """
    column target to predict 

    """

    df['condition_predict'] = df.apply(get_predicting_condition, axis = 1)
    data = pd.concat([data, df], axis=1)

    """
    columns which will be used to predict 

    """
    data['Condition'] = data['Condition'].fillna('empty')
    data['Reasons'] = data['Reasons'].fillna('empty')
    data['Diagnosis'] = data['Diagnosis'].fillna('empty')
    data["Condition"] = data["Condition"].map(str)+" " + data["Reasons"] +" " +data["Diagnosis"]
    
    for c in data.columns:
        if data[c].dtype == 'O':
            data[c] = data[c].str.replace('\\xa0', '')
            data[c] = data[c].str.replace('xa0', '')
            data[c] = data[c].str.replace('\\xf3', 'o')
            data[c] = data[c].str.replace('\\xe1', 'a')
            data[c] = data[c].str.replace('\\xe9', 'e')
            data[c] = data[c].str.replace('\\xe3', 'a')
            data[c] = data[c].str.strip()
            data[c] = data[c].str.replace('[^\x00-\x7F]','')
    data.Condition.replace(regex=True,inplace = True, to_replace=r'\**',value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r'\:',value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r'\,',value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r'\.',value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r'\-',value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r'\"',value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r"\'",value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r"\;",value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r"\/",value=r'')
    data.Condition = data.Condition.str.strip()

    return data

def read_data_historic(filename):
    h_data = pd.read_csv(filename)
    #h_data = pd.read_csv('C:\Users\Falco\Desktop\directory\WMRD\data\historic_data_2013-01-01_2018-05-22_above_all major centers.csv')
    h_data.columns = ['WRMD_ID','Case', 'Admitted_at', 'Species', 'Organization','Reasons', 'Address_found','Latitude', 'Longitude',
                    'Disposition', 'Dispositioned_at','Diagnosis', 'Sex','Weight', 'Age','Condition']
    h_data['Admitted_at'] = pd.to_datetime(h_data['Admitted_at']).dt.strftime('%Y-%m-%d')
    h_data.set_index(pd.DatetimeIndex(h_data['Admitted_at']), inplace= True, drop= False)
    h_data.set_index(pd.DatetimeIndex(h_data['Admitted_at']), inplace= True, drop= False)
    """
    columns which will be used to predict 

    """
    h_data['ConditionO'] = h_data['Condition'].fillna('empty')
    h_data['Reasons'] = h_data['Reasons'].fillna('empty')
    h_data['Diagnosis'] = h_data['Diagnosis'].fillna('empty')
    h_data["ConditionO"] = h_data["ConditionO"].map(str)+" " + h_data["Reasons"] +" " +h_data["Diagnosis"]

    for c in h_data.columns:
        if h_data[c].dtype == 'O':
            h_data[c] = h_data[c].str.replace('\\xa0', '')
            h_data[c] = h_data[c].str.replace('xa0', '')
            h_data[c] = h_data[c].str.replace('\\xf3', 'o')
            h_data[c] = h_data[c].str.replace('\\xe1', 'a')
            h_data[c] = h_data[c].str.replace('\\xe9', 'e')
            h_data[c] = h_data[c].str.replace('\\xe3', 'a')
            h_data[c] = h_data[c].str.strip()
            h_data[c] = h_data[c].str.replace('[^\x00-\x7F]','')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r'\**',value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r'\:',value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r'\,',value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r'\.',value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r'\-',value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r'\"',value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r"\'",value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r"\;",value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r"\/",value=r'')
    h_data.Condition = h_data.Condition.str.strip()
    return h_data
   



# In[ ]:


stops = set(stopwords.words("english"))
def review_to_words(review_text):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    # 1. 
    #print (c['Condition'])
    #review_text = c['con']
    
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    # 
    # 5. Remove stop words
    #meaningful_words = [w for w in words if not w in stops]   
    meaningful_words = words
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    
    meaningful_words = ( " ".join( meaningful_words )) 

    ### Generating tokens
    clean_condi_u = unicode(meaningful_words, 'utf8')  # convert bytes into proper unicode
    words = TextBlob(clean_condi_u).words
    ### Generating lemmas
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]


# In[ ]:

def review_to_words_2(c):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    # 1. 
    #print (c['Condition'])
    #review_text = c['con']
    
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", c['Condition']) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    
    meaningful_words = ( " ".join( meaningful_words )) 

    ### Generating tokens
    clean_condi_u = unicode(meaningful_words, 'utf8')  # convert bytes into proper unicode
    words = TextBlob(clean_condi_u).words
    ### Generating lemmas
    # for each word, take its "base form" = lemma 
    word_list =  [word.lemma for word in words]
    return ( " ".join( word_list))


def Bag_of_Words(data, method = 1):
    if method ==1:
        bow_transformer = CountVectorizer(analyzer=review_to_words, ngram_range=(1, 3), stop_words  = None).fit(data['Condition'])
        condition_bow = bow_transformer.transform(data['Condition'])
            
        tfidf_transformer = TfidfTransformer().fit(condition_bow)
        condition_tfidf = tfidf_transformer.transform(condition_bow)
        return condition_tfidf
    if method == 2:
        #data['condition_lemma'] = data.apply(review_to_words_2, axis=1)
        bow_transformer = bow_transformer2 = CountVectorizer(analyzer='word',ngram_range= (1, 2), encoding='utf8',stop_words ='english').fit(data['Condition'])
        condition_bow = bow_transformer.transform(data['Condition'])
            
        tfidf_transformer = TfidfTransformer().fit(condition_bow)
        condition_tfidf = tfidf_transformer.transform(condition_bow)
        return condition_tfidf
        
    


# In[ ]:


def RandomForest(kf, tfidf, data, params, verbose= False,normalized = False):
    rf = RandomForestClassifier(params)
    #rf = RandomForestClassifier(n_estimators=100, oob_score=False, random_state=123456)
    scores = cross_val_score(rf, tfidf, data['condition_predict'], cv=kf, scoring='accuracy')
    print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))
    y_pred = cross_val_predict(rf, tfidf, data['condition_predict'], cv = kf)
    df = pd.DataFrame({'prediction':y_pred, 'obsevred': data['condition_predict']})
    confusion_matrix = ConfusionMatrix(df.obsevred, df.prediction)
    if verbose:
        print("Confusion matrix:\n%s" % confusion_matrix)
        confusion_matrix.print_stats()
    confusion_matrix.plot(normalized=normalized, backend='seaborn', cmap='Blues',annot= True)
    print (scores)
    return scores

def NaiveBayes(kf, tfidf, data,verbose= False, normalized = False):
    NB = MultinomialNB()
    scores = cross_val_score(NB, tfidf, data['condition_predict'], cv=kf, scoring='accuracy')
    print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))
    y_pred = cross_val_predict(NB, tfidf, data['condition_predict'], cv = kf)
    df = pd.DataFrame({'prediction':y_pred, 'obsevred': data['condition_predict']})
    confusion_matrix = ConfusionMatrix(df.obsevred, df.prediction)
    if verbose:
        print("Confusion matrix:\n%s" % confusion_matrix)
        confusion_matrix.print_stats()
    confusion_matrix.plot(normalized=normalized, backend='seaborn', cmap='Blues',annot= True)
    print (scores)
    return scores

def AdaBoost(kf, tfidf, data,verbose= False, normalized = False):
    ADB = AdaBoostClassifier(n_estimators=100)
    scores = cross_val_score(ADB, tfidf, data['condition_predict'], cv=kf, scoring='accuracy')
    print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))
    y_pred = cross_val_predict(ADB, tfidf, data['condition_predict'], cv = kf)
    df = pd.DataFrame({'prediction':y_pred, 'obsevred': data['condition_predict']})
    confusion_matrix = ConfusionMatrix(df.obsevred, df.prediction)
    if verbose:
        print("Confusion matrix:\n%s" % confusion_matrix)
        confusion_matrix.print_stats()
    confusion_matrix.plot(normalized=normalized, backend='seaborn', cmap='Blues',annot= True)
    print (scores)
    return scores


def GBoosting(kf, tfidf, data,verbose= False, normalized = False):
    GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    scores = cross_val_score(GBC, tfidf, data['condition_predict'], cv=kf, scoring='accuracy')
    print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))
    y_pred = cross_val_predict(GBC, tfidf, data['condition_predict'], cv = kf)
    df = pd.DataFrame({'prediction':y_pred, 'obsevred': data['condition_predict']})
    confusion_matrix = ConfusionMatrix(df.obsevred, df.prediction)
    if verbose:
        print("Confusion matrix:\n%s" % confusion_matrix)
        confusion_matrix.print_stats()
    confusion_matrix.plot(normalized=normalized, backend='seaborn', cmap='Blues',annot= True)
    print (scores)
    return scores


# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

