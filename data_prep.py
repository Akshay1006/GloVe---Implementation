#Input Data is of the form

['W1 W987 W675 W34',
 'W876 W312 W876',
 'W456 W9871 W65 W008',...]
 
 import pandas as pd
 import numpy as np
 
 word_list=df.word_id.values.tolist()
 
 def process_word_list(word_list):
    a=[i.split() for i in word_list]
    return a 

word_list_split=process_word_list(word_list)

from itertools import groupby

word_list_clean=[]

for elem in word_list_split:
    word_list_clean.append([x[0] for x in groupby(elem)])

print(len([item for sublist in word_list_clean for item in sublist]),
len([item for sublist in word_list_split for item in sublist]))

from nltk.tokenize import word_tokenize
from itertools import combinations
from collections import Counter

vocab=list(set(x for l in word_list_clean for x in l))

# Co Occurence Matrix Creation

co_occ={ii:Counter({jj:0 for jj in vocab if jj != ii}) for ii in vocab}
len_dict={key:len(value) for key,value in co_occ.items()}

#Running a Co-occurence matrix takes time. Try to run it on a GPU

k=3 # Size of the window to be looked upon. Configurable Parameter

for sen in word_list:
    for ii in range(len(sen)):
        if ii < k:
            c = Counter(sen[0:ii+k+1])
            del c[sen[ii]]
            co_occ[sen[ii]] = co_occ[sen[ii]] + c
        elif ii > len(sen)-(k+1):
            c = Counter(sen[ii-k::])
            del c[sen[ii]]
            co_occ[sen[ii]] = co_occ[sen[ii]] + c
        else:
            c = Counter(sen[ii-k:ii+k+1])
            del c[sen[ii]]
            co_occ[sen[ii]] = co_occ[sen[ii]] + c
            
            
co_occv1 = {ii:dict(co_occ[ii]) for ii in vocab}

# Convert the data to Pandas dataframe
df_co_occ =pd.DataFrame.from_dict(co_occurence,orient='index')

# Generate the Key for words

word_counter=Counter()
word_counter.update(vocab)

word2id={w:i for i,(w,_) in enumerate(word_counter.most_common())}
id2word={i:w for w,i in word2id.items()}

co_occurencev1={v:co_occurence[k] for k,v in word2id.items()}

from copy import deepcopy

co_occ_copy=deepcopy(co_occurencev1)
for i,j in co_occurencev1.items():
    a=deepcopy(j)
    for k,v in j.items():
        key=word2id[k]
        value=v
        a.pop(k)
        a.update({key:value})
    co_occ_copy.pop(i)
    co_occ_copy.update({i:a})
