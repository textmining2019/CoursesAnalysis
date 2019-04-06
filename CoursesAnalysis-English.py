
# coding: utf-8

# In[5]:



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
#from sklearn.naive_bayes import ComplementNB
#from bayes.classifiers import ComplementNB
#from naiveBayesClassifier.classifier import Classifier
from sklearn import svm, datasets
from nltk.stem import WordNetLemmatizer


# In[117]:



lemmatizer = WordNetLemmatizer()
# A.read the course names training data & apply preprocessing into a data frame
# A.1 upload  Train data file names
path =r'D:\MahmoudOut\Final Project\CoursesAnalysis-master'
ds1_train=pd.read_excel(path+"\TrainData.xlsx")


# In[7]:



#A.3 # Construct a bag of words matrix for train/validate data.
# This will lowercase everything, and ignore all punctuation by default.
# It will also remove stop words.
train_data_vectorizer = CountVectorizer(lowercase=True, stop_words="english")
train_data_non_preprocess_vectorizer = CountVectorizer(lowercase=True)
test_data_vectorizer = CountVectorizer(lowercase=True, stop_words="english")


# In[127]:


#get english sentences only
import re

def EnglishOnly(text):
    #print (text)
    result = re.sub(r'[^ء-ي]',' ',str(text)) #arabic
    if len(result.strip()) >0:
        return 'F'
    else:
        return 'T'


# In[128]:


ds1_train['EnglishOnly'] =ds1_train['valueName'].apply(lambda x : EnglishOnly(x))
ds1_train = ds1_train[ds1_train.EnglishOnly =='T']
len(ds1_train)


# In[129]:


#ds1_train=train_dfs[0]
#B. Train Basic Pre-processing

#B.1 transform to Lower case
ds1_train['transform_to_Lower_case'] = ds1_train['valueName'].apply(lambda x: " ".join(x.lower() for x in x.split()))
print(ds1_train.head())


# In[130]:



#B.2 Removing Punctuation
ds1_train['Removing_Punctuation'] = ds1_train['transform_to_Lower_case'].str.replace('[^\w\s]','')
print(ds1_train.head())


# In[131]:


#B.3 Removal of Stop Words
#extend to append a list of stop words
ext_st_word_list=[]
ext_st_word=pd.read_csv(path+'\stopwords.csv')
ext_st_word_list=ext_st_word['Stop words']
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(ext_st_word_list)
ds1_train['Removal_of_Stop_Words'] = ds1_train['Removing_Punctuation'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
print(ds1_train.head())


# In[132]:


#B.4 Rare words removal
#remove rarely occurring words from the text. Because they’re so rare, the association between them and other words is dominated by noise
freq = pd.Series(' '.join(ds1_train['Removal_of_Stop_Words']).split()).value_counts()[-10:]
print(freq)


# In[135]:


ds1_train['Rare_words_removal'] = ds1_train['Removal_of_Stop_Words'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
print(ds1_train.head())



# In[133]:


#Correct wrong spelling words
#pip install pyspellchecker 
    
from spellchecker import SpellChecker

spell = SpellChecker()


# In[139]:


#correct spelling 
def CorrectSpell( str ):
   #print(str)
   cStr = spell.unknown([str])
   if (len(cStr)>0) : 
        # Get the one `most likely` answer
       #print(cStr) 
       return spell.correction(cStr.pop())
   else :
       return str


# In[140]:


ds1_train['Correct_Unknown_Words'] = ds1_train['Rare_words_removal'].apply(lambda x: " ".join(CorrectSpell(word) for word in x.split()))


# In[39]:


import nltk
nltk.download('wordnet')


# In[40]:


# Lemmatize
ds1_train['Lemmatization'] = ds1_train['Rare_words_removal'].apply(lambda x: " ".join([lemmatizer.lemmatize(word, pos='v') for word in x.split()]))

