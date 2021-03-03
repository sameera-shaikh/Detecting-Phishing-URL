#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[2]:


import numpy as np # use for multi-dimensional array and matrix
import pandas as pd 
import seaborn as sns # use for high-level interface for drawing attractive and informative statistical graphics 
import matplotlib.pyplot as plt # It provides an object-oriented API for embedding plots into applications
get_ipython().run_line_magic('matplotlib', 'inline')
import time # calculate time 

from sklearn.linear_model import LogisticRegression # algo use to predict good or bad
from sklearn.naive_bayes import MultinomialNB # nlp algo use to predict good or bad

from sklearn.model_selection import train_test_split # spliting the data between feature and target
from sklearn.metrics import classification_report # gives whole report about metrics (e.g, recall,precision,f1_score,c_m)
from sklearn.metrics import confusion_matrix # gives info about actual and predict
from nltk.tokenize import RegexpTokenizer # regexp tokenizers use to split words from text  
from nltk.stem.snowball import SnowballStemmer # stemmes words
from sklearn.feature_extraction.text import CountVectorizer # create sparse matrix of words using regexptokenizes  
from sklearn.pipeline import make_pipeline # use for combining all prerocessors techniuqes and algos

from PIL import Image # getting images in notebook
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator# creates words colud
#from wordcloud import WordCloud

from bs4 import BeautifulSoup # use for scraping the data from website
from selenium import webdriver # use for automation chrome 
import networkx as nx # for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

import pickle# use to dump model 

import warnings # ignores pink warnings 
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords


# In[3]:


import sys
print(sys.executable)


# In[4]:


import os
os.chdir('C:/Users/Sameera/Desktop/dessertetion')


# In[5]:


phish_data=pd.read_csv("phishing_site.csv")


# In[6]:


phish_data.head()


# In[7]:


phish_data.tail()


# In[8]:


phish_data.info()


# In[9]:


phish_data.isnull().sum()


# In[10]:


#create a dataframe of classes counts
label_counts = pd.DataFrame(phish_data.Label.value_counts())


# In[11]:


#visualizing target_col
sns.set_style('darkgrid')
sns.barplot(label_counts.index,label_counts.Label)


# In[12]:


tokenizer = RegexpTokenizer(r'[A-Za-z]+')


# In[13]:


phish_data.URL[0]


# In[14]:


# this will be pull letter which matches to expression
tokenizer.tokenize(phish_data.URL[0]) # using first row


# In[15]:


print('Getting words tokenized ...')
t0= time.perf_counter()
phish_data['text_tokenized'] = phish_data.URL.map(lambda t: tokenizer.tokenize(t)) # doing with all rows
t1 = time.perf_counter() - t0
print('Time taken',t1 ,'sec')


# In[16]:


phish_data.sample(5)


# In[17]:


stemmer = SnowballStemmer("english") # choose a language


# In[18]:


print('Getting words stemmed ...')
t0= time.perf_counter()
phish_data['text_stemmed'] = phish_data['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')


# In[19]:


phish_data.sample(5)


# In[20]:


print('Getting joiningwords ...')
t0= time.perf_counter()
phish_data['text_sent'] = phish_data['text_stemmed'].map(lambda l: ' '.join(l))
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')


# In[21]:


phish_data.sample(5)


# In[22]:


#sliceing classes
# bad_sites = df[df.Label == 'bad']
# good_sites = df[df.Label == 'good']
bad_sites = phish_data[phish_data.Label == 'bad']
good_sites = phish_data[phish_data.Label == 'good']


# In[23]:


bad_sites.head()


# In[24]:


good_sites.head()


# In[25]:


import nltk
from nltk.corpus import stopwords

def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'com','http'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    mask = mask)
    wordcloud.generate(text)
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'green', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()


# In[26]:


data = good_sites.text_sent
data.reset_index(drop=True, inplace=True)


# In[27]:


common_text = str(data)
# common_mask = np.array(Image.open('star.png'))
plot_wordcloud(common_text, max_words=400, max_font_size=120, 
               title = 'Most common words use in good urls', title_size=15)


# In[28]:


data = bad_sites.text_sent
data.reset_index(drop=True, inplace=True)


# In[29]:


common_text = str(data)
common_mask = np.array(Image.open('star.png'))
plot_wordcloud(common_text, common_mask, max_words=400, max_font_size=120, 
               title = 'Most common words use in bad urls', title_size=15)


# In[ ]:


browser = webdriver.Chrome(r"chromedriver.exe")


# In[31]:


# pwd


# In[32]:


import os
os.chdir('C:\\Users\\Sameera\\Desktop\\dessertetion')


# In[33]:


list_urls = ['https://samewaywins.com/ccplusservice/UK/?dom=trakgobigmedia.com&cep=90-JeQ1DbTXGltUWxrdC6CkRrXUDU36P1B_0s-3Wf34AjIqVcCGtEVHSA3Jctq6EMAnDsxr2Bv9Vop1ryUSw4MKbT2x8GpDOXoFcnR1cPz4WA9uyP3w8LI91h756FvUz8gvkuPeXJI-ImH6CyItqv-u1X3LlkpGOngF_qkR7DCIyta0QigZsh6XtshiyPIdh8cMp7dzu8LEUFhRIeQyK0lvsPPHaxk7G3RLyjKHMw7CN4EDJqtMfCGxBd-8-xXYfSjT5Q1fXcfTKsfx1Dy_Wb70nLvZcEFaz_xiEiQSYTYc3XrbXHAe6cphZeGxy8Jn79udvmrnwJRuufcsMhSxI118iQG8Pjy_lCElaPor3X5Bzz6SkkNGSoXdYnzgHrFQF&lptoken=16ab10043715079748a6','chrome-extension://fheoggkfdfchfphceeifdbepaooicaho/html/site_status_block_page.html',
             'https://lts.lehigh.edu/phishing/examples','chrome-extension://fheoggkfdfchfphceeifdbepaooicaho/html/site_status_block_page.html','https://www.ezeephones.com/about-us'] 
#here i take phishing sites 
links_with_text = []


# In[34]:


for url in list_urls:
    browser.get(url)
    soup = BeautifulSoup(browser.page_source,"html.parser")
    for line in soup.find_all('a'):
        href = line.get('href')
        links_with_text.append([url, href])


# In[35]:


df = pd.DataFrame(links_with_text, columns=["from", "to"])
# good_sites = pd.DataFrame(links_with_text, columns=["from", "to"])
# bad_sites = pd.DataFrame(links_with_text, columns=["from", "to"])


# In[59]:


df.head(20)


# In[37]:


# df.info()


# In[38]:


#create cv object
cv = CountVectorizer()


# In[39]:


feature = cv.fit_transform(phish_data.text_sent) #transform all text which we tokenize and stemed


# In[40]:


feature[:5].toarray() # convert sparse matrix into array to print transformed features


# In[41]:


trainX, testX, trainY, testY = train_test_split(feature, phish_data.Label)


# In[42]:


# create lr object
lr = LogisticRegression()


# In[43]:


lr.fit(trainX,trainY)


# In[44]:


lr.score(testX,testY)


# In[45]:


Scores_ml = {}
Scores_ml['Logistic Regression'] = np.round(lr.score(testX,testY),2)


# In[46]:


print('Training Accuracy :',lr.score(trainX,trainY))
print('Testing Accuracy :',lr.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(lr.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(lr.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[47]:


# create mnb object
mnb = MultinomialNB()


# In[48]:


mnb.fit(trainX,trainY)


# In[49]:


mnb.score(testX,testY)


# In[50]:


Scores_ml['MultinomialNB'] = np.round(mnb.score(testX,testY),2)


# In[51]:


print('Training Accuracy :',mnb.score(trainX,trainY))
print('Testing Accuracy :',mnb.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(mnb.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(mnb.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[52]:


acc = pd.DataFrame.from_dict(Scores_ml,orient = 'index',columns=['Accuracy'])
sns.set_style('darkgrid')
sns.barplot(acc.index,acc.Accuracy)


# In[53]:


#So, Logistic Regression is the best fit model, Now we make sklearn pipeline using Logistic Regression


# In[54]:


pipeline_ls = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())
##(r'\b(?:http|ftp)s?://\S*\w|\w+|[^\w\s]+') ([a-zA-Z]+)([0-9]+) -- these tolenizers giving me low accuray


# In[55]:


trainX, testX, trainY, testY = train_test_split(phish_data.URL, phish_data.Label)


# In[56]:


pipeline_ls.fit(trainX,trainY)


# In[57]:


pipeline_ls.score(testX,testY)


# In[60]:


print('Training Accuracy :',pipeline_ls.score(trainX,trainY))
print('Testing Accuracy :',pipeline_ls.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(pipeline_ls.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(pipeline_ls.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[61]:


pickle.dump(pipeline_ls,open('phishing.pkl','wb'))


# In[62]:


loaded_model = pickle.load(open('phishing.pkl', 'rb'))
result = loaded_model.score(testX,testY)
print(result)


# In[ ]:




