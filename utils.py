
# coding: utf-8

# In[49]:

import pandas as pd
import numpy as np
import string
punc = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~' #string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
from itertools import compress
import re


# In[56]:


def textPrep(st):
    
    if (pd.isnull(st)):
        return np.nan
    # Convert the input to lower string
    st = st.lower()
    # Convert all punctuation in the string to space
    punc_table = str.maketrans({key: ' ' for key in punc})
    st = st.translate(punc_table)
    
    # Remove double space
    st = re.sub(' +', ' ',st.strip())
    
    return st

def textPrepNoLower(st):
    
    # Convert the input to lower string
    # st = st.lower()
    # Convert all punctuation in the string to space
    punc_table = str.maketrans({key: ' ' for key in punc})
    st = st.translate(punc_table)
    
    # Remove double space
    st = re.sub(' +', ' ',st.strip())
    
    return st


# In[60]:


def code2list(st):
    return [int(float(i)) for i in st.strip('[').strip(']').split(',')]


# In[80]:


def getOutVec(st, pat):
    enc = []
     
    if (pat not in [np.nan, None]):
        if st.strip().split(pat)[0] == '':
            enc.append(np.ones(len(pat.strip().split(' ')), dtype=int))
            st = st[len(pat):]
        
        st = st.lower()
        pat = pat.lower()
        pat = ' ' + pat + ' ' # Searching the whole key word
        
        for item in st.split(pat)[:-1]:
            enc.append(np.zeros(len(item.strip().split(' ')), dtype=int))
            enc.append(np.ones(len(pat.strip().split(' ')), dtype=int))
        enc.append(np.zeros(len(st.split(pat)[-1].strip().split(' ')), dtype=int))
        flat = [i for sub in enc for i in sub]
        return flat
    else:
        return np.zeros(len(st.strip().split(' ')), dtype=int)


# In[78]:


def textSeperate(st):
    if pd.isnull(st):
        return np.nan
    else:
        return st.strip().split(' ')

def text2code(st, vocab):
    vec = []
    if not (pd.isnull(st)):
        st = textPrep(st)
        for w in textSeperate(st):
            if w in vocab:
                vec.append(vocab.index(w))
            else:
                vec.append(np.NaN)

        return vec
    else:
        return np.nan


# In[61]:

from collections import OrderedDict
def getWordIdenFromOutputVector(originSt, outVec):
    listOfWord = textSeperate(originSt)
    return list(OrderedDict.fromkeys(compress(listOfWord, outVec)))


def code2text(codeList, vocab):
    wordList = [vocab[i] for i in codeList]
    return ' '.join(wordList)
