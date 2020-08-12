import numpy as np
import pandas as pd
import math
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
import plotly.tools as tls
import plotly.express as px
import os


os.path.isfile('/Users/AnnaD/Desktop/avalanche')
avalanche= pd.read_csv('/Users/AnnaD/Desktop/avalanche/2CAIC_AccDB_Investigations.csv')
avalanche = avalanche.iloc[:,0:39]

# In[5]:
avalanche = avalanche.iloc[:347,:]


# In[6]:
avalanche['date_modified']=pd.to_datetime(avalanche['date_modified'])
avalanche['avi_date']=pd.to_datetime(avalanche['avi_date'])
avalanche['month'] = avalanche['avi_date'].dt.month_name()


# In[7]:


count_aspect = pd.DataFrame(avalanche.groupby('aspect').count()['avalanche']).reset_index()
count_aspect


# In[8]:


order=(['c','a','b','h','e','d','f','g'])
count_aspect['order'] = order
count_aspect.sort_values(by=['order'],inplace=True)
count_aspect = count_aspect.rename(columns={"avalanche": "Count"})


# In[9]:




fig = px.bar_polar(count_aspect, r=count_aspect['Count'], theta=['N','NE','E','SE','S','SW','W','NW'],
                   color="Count", template="ggplot2"
                   )
fig.update_layout(title_text="Count of Avalanches by Aspect")
fig.show()



# In[10]:


#January, February, March, December, April
count_month = pd.DataFrame(avalanche.groupby(['month','aspect']).count())
count_month = count_month['avalanche']
count_month = count_month.to_frame()
count_month.reset_index(inplace=True)


# In[11]:


def set_order(row):
    if row["aspect"] == 'N':
        return "a"
    elif row["aspect"] == 'NE':
        return "b"
    elif row["aspect"] == 'E':
        return "c"
    elif row["aspect"] == 'SE':
        return "d"
    elif row["aspect"] == 'S':
        return "e"
    elif row["aspect"] == 'SW':
        return "f"
    elif row["aspect"] == 'W':
        return "g"
    else:
        return "h"
count_month = count_month.assign(order = count_month.apply(set_order,axis=1))


# In[12]:


april_count = count_month[0:6]
december_count = count_month[6:13]
february_count = count_month[13:20]
january_count = count_month[20:28]
june_count = count_month[28:30]
march_count = count_month[30:38]
may_count = count_month[38:41]
november_count = count_month[41:43]
october_count = count_month[43:]


# In[13]:


#created dummy columns
february_count.loc[-1]=['February','SW', 0,'f']
february_count.index=february_count.index +1
december_count.loc[-1]=['December','SW', 0,'f']
december_count.index=december_count.index +1


# In[14]:

april_count = april_count.sort_values(['order'])
december_count = december_count.sort_values(['order'])
february_count = february_count.sort_values(['order'])
january_count = january_count.sort_values(['order'])
march_count = march_count.sort_values(['order'])
may_count = may_count.sort_values(['order'])
november_count = november_count.sort_values(['order'])
october_count = october_count.sort_values(['order'])


# In[15]:


fig = px.bar_polar(r=december_count["avalanche"],
                    theta=december_count['aspect'], )
fig.update_layout(
    title={
        'text': "December",
        'y':0.11,
        'x':0.5,
        })


# In[16]:




fig = px.bar_polar(r=january_count["avalanche"],
                    theta=january_count['aspect'], )
fig.update_layout(
    title={
        'text': "January",
        'y':0.11,
        'x':0.5,
        'xanchor': 'center',

        'yanchor': 'top'}, polar_radialaxis_ticksuffix='')

fig.show()


# In[17]:


fig = px.bar_polar(r=february_count["avalanche"],
                    theta=february_count['aspect'] )
fig.update_layout(
    title={
        'text': "February",
        'y':0.11,
        'x':0.5,
        'xanchor': 'center',

        'yanchor': 'top'})



fig.show()


# In[18]:


fig = px.bar_polar(r=march_count["avalanche"],
                    theta=march_count['aspect'], )
fig.update_layout(
    title={
        'text': "March",
        'y':0.11,
        'x':0.5,
        'xanchor': 'center',

        'yanchor': 'top'}, polar_radialaxis_ticksuffix='')

fig.show()


# In[19]:


count_angle = pd.DataFrame(avalanche.groupby('angle').count()['avalanche']).reset_index()
count_angle= count_angle.rename(columns={"avalanche": "Count"})


# In[20]:


avalanche['no_killed'].fillna(0,inplace=True)


# In[21]:


avalanche['no_killed'].value_counts()


# In[22]:


avalanche["Binary_Deaths"] = avalanche['no_killed'].apply(lambda x: 1 if x > 0 else 0)


# In[ ]:





# In[23]:


avalanche2 =avalanche.copy()


# In[24]:


avalanche2['text'] =avalanche2.apply(lambda x:'%s %s %s %s %s' % (x['description'],x['acc_sum_pub'], x['comments_pub'], x['rescue_sum_pub'],x['wx_sum_pub']),axis=1)


# In[25]:


avalanche3 = avalanche2.filter(items=['text'])


# In[26]:


avalanche3.head()


# In[27]:


avalanche = pd.concat([avalanche, avalanche3], axis='columns')
avalanche


# In[28]:





# In[29]:


text_avalanche = avalanche[['text','Binary_Deaths']]
text_avalanche


# In[ ]:


avalanche = avalanche.fillna(0,inplace=True)


# In[ ]:


X = text_avalanche['text']
y = text_avalanche['Binary_Deaths']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:


X = X.values.reshape(-1,1)
print(X_train.shape)
print(y_train.shape)


# In[ ]:


stemmer = nltk.stem.PorterStemmer()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
# 1. Instantiate
bagofwords = CountVectorizer(stop_words = "english", ngram_range =(1,4))

# 2. Fit
bagofwords.fit(X_train)

# 3. Transform
X_train = bagofwords.transform(X_train)
X_test=bagofwords.transform(X_test)


# In[ ]:


# these are now the features, they are the individual tokens
bagofwords.get_feature_names()


# In[ ]:


X_train.toarray()
text= pd.DataFrame(columns=bagofwords.get_feature_names(), data=X_train.toarray())
display(text)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# Training score
logreg.score(X_train,y_train)


# In[ ]:


coefficients = logreg.coef_
coefficients


# In[ ]:


X_test


# In[ ]:




# In[ ]:


indices = coefficients.argsort()[0]
# The words with the lowest coefficients
# most predictive of a 0 (negative review)
poscoeffword = np.array(column_names)[indices[:20]]
