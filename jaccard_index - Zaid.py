
import nltk
import numpy as np
from textblob import TextBlob
import pandas as pd 
import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold # import KFold
from sklearn.model_selection import KFold, StratifiedKFold
#Load libraries needed for classification 
from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB, MultinomialNB,GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from jaccard_index.jaccard import jaccard_index
from sklearn import svm, datasets
from nltk.stem import WordNetLemmatizer
 
lemmatizer = WordNetLemmatizer()
# A.read the course names training data & apply preprocessing into a data frame
# A.1 upload  Train data file names
path =r'D:\text-mining\trainData'
DfTrain=pd.read_excel("D:\\text-mining\\trainData\\TrainData.xlsx")
DfTest= pd.read_excel("D:\\text-mining\\testData\\testData.xlsx")

#A.3 # Construct a bag of words matrix for train/validate data.
# This will lowercase everything, and ignore all punctuation by default.
# It will also remove stop words.
train_data_vectorizer = CountVectorizer(lowercase=True, stop_words="english")
train_data_non_preprocess_vectorizer = CountVectorizer(lowercase=True)
test_data_vectorizer = CountVectorizer(lowercase=True, stop_words="english")
#DfTrain=train_dfs[0]
#B. Train Basic Pre-processing

#B.1 transform to Lower case
DfTrain['TrLowercase'] = DfTrain['TrCourse'].apply(lambda x: " ".join(x.lower() for x in x.split()))
DfTest['TsLowercase'] = DfTest['TestCourse'].apply(lambda x: " ".join(x.lower() for x in x.split()))

#B.2 Removing Punctuation
DfTrain['TrPunctuation'] = DfTrain['TrLowercase'].str.replace('[^\w\s]','')
DfTest['TsPunctuation'] = DfTest['TsLowercase'].str.replace('[^\w\s]','')


#B.3 Removal of Stop Words
#extend to append a list of stop words
ext_st_word_list=[]
ext_st_word=pd.read_csv(r'D:\text-mining\Stopwords\stopwords.csv')
ext_st_word_list=ext_st_word['Stop words']
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(ext_st_word_list)
DfTrain['TrStopWords'] = DfTrain['TrPunctuation'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
DfTest['TsStopWords'] = DfTest['TsPunctuation'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))

#B.4 Rare words removal
#remove rarely occurring words from the text. Because they’re so rare, the association between them and other words is dominated by noise
freq = pd.Series(' '.join(DfTrain['TrStopWords']).split()).value_counts()[-10:]
freqTs = pd.Series(' '.join(DfTest['TsStopWords']).split()).value_counts()[-10:]


DfTrain['TrWordsremoval'] = DfTrain['TrStopWords'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
DfTest['TsWordsremoval'] = DfTest['TsStopWords'].apply(lambda x: " ".join(x for x in x.split() if x not in freqTs))



# Lemmatize
DfTrain['TrLemmatization'] = DfTrain['TrWordsremoval'].apply(lambda x: " ".join([lemmatizer.lemmatize(word, pos='v') for word in x.split()]))
DfTest['TsLemmatization'] = DfTest['TsWordsremoval'].apply(lambda x: " ".join([lemmatizer.lemmatize(word, pos='v') for word in x.split()]))

#iterates over test rows


from jaccard_index.jaccard import jaccard_index
for test_idx, test_row in DfTest.iterrows() :
# iterates over train rows
    for train_idx, train_row in DfTrain.iterrows():
       jac_ndx=jaccard_index((test_row['TsLemmatization']), (train_row['TrLemmatization']))


       if jac_ndx>.50:
           DfTest.loc[test_idx, 'TrLemmatization']=train_row['TrLemmatization']
           DfTest.loc[test_idx, 'TrReadyToFuzzyID']=train_row['TrReadyToFuzzyID']
           DfTest.loc[test_idx,'TrCourse'] = train_row['TrCourse']
           DfTest.loc[test_idx,'JaccardIndex']=jac_ndx

DfTest.to_csv(r'D:\Text-Mining\results\Jaccard_Index.csv', index=True)
