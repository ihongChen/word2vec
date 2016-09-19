
# coding: utf-8

# In[1]:

import re
import jieba 
import pypyodbc
import pandas as pd
import jieba
from gensim import models


# In[2]:

CON = pypyodbc.connect("DRIVER={SQL Server};SERVER=dbm_public;UID=xxxx;PWD=xxxxx;DATABASE=External",unicode_results=True)


# In[3]:

sql = ur""" 
         SELECT top 5000 [標題],[內文],[推文內容] from PTT
         WHERE [看版] like 'CreditCard'
         ORDER BY [發文日期] DESC
"""
df = pd.read_sql_query(sql,CON)


# In[4]:

df.tail(10)


# In[33]:

jieba.load_userdict('../data/userdict.txt')
jieba.suggest_freq(u'現金回饋', True)


# In[34]:

with open('../data/stopwords_tw.txt') as f:
        doc = f.read()
        doc = doc.decode('utf-8')
        doc = re.sub('\r\n','\n',doc)

def get_chinese(text):    
    return ''.join(re.findall(ur'[\u4e00-\u9fa5|a-zA-Z]+',text))
    
    
STOP_WORDS = set(doc.split("\n"))


# In[35]:

my_tokenizer = lambda doc:' '.join([get_chinese(w) for w in jieba.cut(doc) if len(w)>1 and  w not in STOP_WORDS]) #
df_token = df.applymap(my_tokenizer)


# In[36]:

df_token[:10]


# In[37]:

corpus = ' '.join(df_token.apply(lambda xx:xx[u'標題']+' '+xx[u'內文']+' '+xx[u'推文內容'],1).tolist()).encode('utf8')


# In[38]:

corpus_title = ' '.join(df_token[u'標題'].tolist()).encode('utf8')
corpus_text = ' '.join(df_token[u'內文'].tolist()).encode('utf8')
corpus_push = ' '.join(df_token[u'推文內容'].tolist()).encode('utf8')
# corpus = corpus_title + corpus_text + corpus_push 


# In[39]:

print corpus[:100]


# In[40]:

# df_token_str = df_token.apply(lambda xx:' '.join(xx).encode('utf8'))


# In[41]:

# corpus = ' '.join(df_token_str.tolist())


# In[42]:

with open('pttcc_token_corpus.txt','w') as f:
    f.write(corpus)


# In[43]:

sentences = models.word2vec.LineSentence("pttcc_token_corpus.txt")


# In[44]:

model = models.word2vec.Word2Vec(sentences, size=100, window=10, min_count=10, workers=8)


# In[45]:

for word, score in model.most_similar(positive=[u"中信"],topn=20):
    print "word = %s / score = %s " % (word,score)


# In[46]:

for word, score in model.most_similar(positive=[u"永豐"],topn=20):
    print "word = %s / score = %s " % (word,score)


# In[47]:

for word, score in model.most_similar(positive=[u"GOGO", u"永豐"],negative=[u"台新"],topn=20):
    print "word = %s / score = %s " % (word,score)


# In[48]:

for word, score in model.most_similar(positive=[u"GOGO", u"花旗"],negative=[u"台新"],topn=20):
    print "word = %s / score = %s " % (word,score)


# In[49]:

for word, score in model.most_similar(positive=[u"GOGO", u"元大"],negative=[u"台新"],topn=20):
    print "word = %s / score = %s " % (word,score)


# In[50]:

for word, score in model.most_similar(positive=[u"GOGO", u"永豐"],negative=[u"台新"],topn=20):
    print "word = %s / score = %s " % (word,score)


# In[51]:

for word, score in model.most_similar(positive=[u"數位", u"中信"],negative=[u"富邦"],topn=20):
    print "word = %s, \t score = %.2f " % (word,score)


# In[52]:

sino_vec = model[u'永豐']
sino_vec.shape


# In[53]:

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)


# In[54]:

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[55]:

get_ipython().magic(u'matplotlib inline')
# import seaborn as sns


# In[76]:

del kw,cards,banks,words


# In[104]:

kw = set([u'COSTCO',u'悠遊',u'門檻',u'手續費',u'年費',          u'停車',u'現金回饋',u'旅遊',u'加油',u'電影',u'機場',u'接送'])
# 
cards = set([u'饗樂',u'數位',u'GOGO',u'愛金',u'透明',u'Me',u'學學',u'鑽金',u'生活',u'數位',u'饗樂',u'兄弟'])
banks = set([u'永豐',u'國泰',u'富邦',u'中信',u'台新',u'花旗',u'元大',u'玉山'])
words = kw | banks |cards
vectors = [model[word] for word in words]
# vectors2d = tsne.fit_transform(vectors)
pca = PCA(n_components=2)
vectors2d = pca.fit(vectors).transform(vectors)


# In[105]:


my_dpi = 96
fig = plt.figure(figsize=(1600/my_dpi, 900/my_dpi), dpi=my_dpi)
ax = fig.add_subplot(111)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
for point, word in zip(vectors2d , words):
    # plot points
    if word in banks:
        color = 'r'
        size = 100
    elif word in cards:
        color = 'b'
        size = 70
    else:
        color = 'g'
        size = 50
    plt.scatter(point[0], point[1], c=color,s=size)
    # plot word annotations
    
    
    plt.annotate(
        word, 
        xy = (point[0], point[1]),
        xytext = (-4, -6) ,
        textcoords = 'offset points',
        ha = 'right' ,
        va = 'bottom',
        size = "x-large"
        )
plt.autoscale(enable=True,tight=True,axis='x')
plt.savefig('words2map.png',dpi=my_dpi)
    
#     plt.autoscale(enable=True,tight=True,axis='x')


# In[62]:

model[u'旅遊']


# In[101]:

for word, score in model.most_similar(positive=[u"加油"],topn=10):
    print "word = %s / score = %s " % (word,score)


# In[ ]:



