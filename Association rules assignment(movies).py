#!/usr/bin/env python
# coding: utf-8

# In[5]:


import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.special import comb
from itertools import combinations, permutations
from apyori import apriori as apr
from mlxtend.frequent_patterns import apriori, association_rules
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder


# 1 - Business Problem

# 
# ___Prepare rules for the all the data sets 1) Try different values of support and confidence. Observe the change in number of rules for different support,confidence values 2) Change the minimum length in apriori algorithm 3) Visulize the obtained rules using different plots___

# In[6]:


df=pd.read_csv('my_movies.csv')


# In[7]:


df


# In[8]:


df.head()


# In[9]:


df1 = df.iloc[:,5:]


# In[10]:



df1.head()


# In[11]:


df1.describe().T


# In[12]:


df1.isnull().sum()


# In[13]:


df1.shape


# In[14]:


item_sets = {}


# In[15]:


te = TransactionEncoder()


# In[16]:


te_ary = te.fit(df1).transform(df1)


# In[17]:


ap = pd.DataFrame(te_ary, columns=te.columns_)


# In[18]:


ap.sum().to_frame('Frequency').sort_values('Frequency',ascending=False)[:25].plot(kind='bar',
                                                                                  figsize=(12,8),
                                                                                  title="Frequent Items")
plt.show()


# 3 - Apriori algorithm
# 

# In[19]:


ap_0_5 = {}
ap_1 = {}
ap_5 = {}
ap_1_0 = {}


# In[20]:


confidence = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


# In[21]:


def gen_rules(df,confidence,support):
    ap = {}
    for i in confidence:
        ap_i =apriori(df1,support,True)
        rule= association_rules(ap_i,min_threshold=i)
        ap[i] = len(rule.antecedents)
    return pd.Series(ap).to_frame("Support: %s"%support)


# In[22]:


confs = []


# In[23]:


for i in [0.005,0.001,0.003,0.007]:
    ap_i = gen_rules(ap,confidence=confidence,support=i)
    confs.append(ap_i)


# In[24]:


all_conf = pd.concat(confs,axis=1)


# In[25]:


all_conf.plot(figsize=(8,8),grid=True)
plt.ylabel('Rules')
plt.xlabel('Confidence')
plt.show()


# 4 - Conclusion

# In[27]:


#As shown in above graph1.Lower the Confidence level Higher the no. of rules 2.Higher the Support, lower the no. of rules.Lets try with Support 0.005 and Confidence at 0.4


# In[28]:


ap_final =  apriori(ap,0.005,True)


# In[29]:


rules_final = association_rules(ap_final,min_threshold=.4,support_only=False)


# In[30]:


rules_final[rules_final['confidence'] > 0.5]


# In[31]:


support = rules_final["support"]
confidence =  rules_final["confidence"]
lift = rules_final["lift"]


# In[32]:


fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection = '3d')
ax1.scatter(support,confidence,lift)
ax1.set_xlabel("support")
ax1.set_ylabel("confidence")
ax1.set_zlabel("lift")


# In[33]:


plt.scatter(support,confidence, c =lift, cmap = 'gray')
plt.colorbar()
plt.xlabel("support");plt.ylabel("confidence")


# In[ ]:




