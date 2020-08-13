#Import packages

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
from sklearn.preprocessing import LabelEncoder

#Import csv of data from CAIC
os.path.isfile('/Users/AnnaD/Desktop/avalanche')
avalanche= pd.read_csv('/Users/AnnaD/Desktop/avalanche/2CAIC_AccDB_Investigations.csv')
avalanche = avalanche.iloc[:,0:39]

# In[5]: used iloc to omit the last columns and rows with nan
avalanche = avalanche.iloc[:347,:]


# In[6]: change column to datetime
avalanche['date_modified']=pd.to_datetime(avalanche['date_modified'])
avalanche['avi_date']=pd.to_datetime(avalanche['avi_date'])
avalanche['month'] = avalanche['avi_date'].dt.month_name()


# In[7]: most avalanches happen on NE slopes


count_aspect = pd.DataFrame(avalanche.groupby('aspect').count()['avalanche']).reset_index()
count_aspect


# In[8]: I wanted to make a bar_polar plot and needed to sort columns


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


#I wanted to group by month in order to see changes in avalanche aspect by month
count_month = pd.DataFrame(avalanche.groupby(['month','aspect']).count())
count_month = count_month['avalanche']
count_month = count_month.to_frame()
count_month.reset_index(inplace=True)


# In[11]: in order to sort by N,NE,E,SE, etc I assigned a letter to later sort by that column


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

#I label encoded aspects
labelencoder = LabelEncoder()
avalanche['aspect'] = avalanche['aspect'].astype(str)
avalanche['travel_mode'] = avalanche['travel_mode'].astype(str)
avalanche['avi_trigger'] = avalanche['avi_trigger'].astype(str)
avalanche['avi_trigger2'] = avalanche['avi_trigger2'].astype(str)
avalanche['travel_mode'] = avalanche['travel_mode'].astype(str)
avalanche['activity'] = avalanche['activity'].astype(str)
avalanche['travel_mode_encoded']=labelencoder.fit_transform(avalanche["travel_mode"])
avalanche["aspect_encoded"]=labelencoder.fit_transform(avalanche["aspect"])
avalanche['avi_triggerencoded']=labelencoder.fit_transform(avalanche["avi_trigger"])
avalanche['avi_trigger2encoded']=labelencoder.fit_transform(avalanche["avi_trigger2"])
avalanche['travel_mode_encoded']=labelencoder.fit_transform(avalanche["travel_mode"])
avalanche['activity_encoded']=labelencoder.fit_transform(avalanche["activity"])

# In[12]: I created dataframes for each month


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


#created dummy columns and imputed 0s when months didn't have avalanches in certain aspects
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

april_count
april_count['pcnt_aspect'] =(april_count.avalanche/ april_count.avalanche.sum())*100
december_count['pcnt_aspect'] =(december_count.avalanche/ december_count.avalanche.sum())*100
february_count['pcnt_aspect'] =(february_count.avalanche/ february_count.avalanche.sum())*100
january_count['pcnt_aspect'] =(january_count.avalanche/ january_count.avalanche.sum())*100
march_count['pcnt_aspect'] =(march_count.avalanche/ march_count.avalanche.sum())*100
november_count['pcnt_aspect'] =(november_count.avalanche/ november_count.avalanche.sum())*100
october_count['pcnt_aspect'] =(october_count.avalanche/ october_count.avalanche.sum())*100


# In[15]: most of the avalances in December were mostl in NE  aspects


fig = px.bar_polar(r=december_count["avalanche"],
                    theta=december_count['aspect'], )
fig.update_layout(
    title={
        'text': "December",
        'y':0.11,
        'x':0.5,
        })


# In[16]: avalanches in January were also mostly in NE aspects

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


# In[17]: in February, avalanches were mostly in SE aspects


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


# In[18]: again, in March most of the avalanches were in NE aspects


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

avalanche['avi_month_num'] = pd.DatetimeIndex(avalanche['avi_date'])
avalanche['day_of_week'] = avalanche['avi_date'].dt.dayofweek
avalanche['weekend_ind'] = 0
avalanche.loc[avalanche['day_of_week'].isin([5, 6]), 'weekend_ind'] = 1
avalanche['int_month']= avalanche['avi_month_num'].astype(int)

#ensure all measurments are in ft





# In[20]: column to indicate whether or not there were deaths
avalanche["Deaths"] = avalanche['no_killed'].apply(lambda x: 1 if x > 0 else 0)

heatmap = avalanche.drop(columns = ['acc_id','avalanche', 'no_killed','no_injured','no_non_crit'] )

avicorr = heatmap.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(10, 8))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(avicorr, cmap='viridis_r', annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(avicorr.columns)), avicorr.columns)
#Apply yticks
plt.yticks(range(len(avicorr.columns)), avicorr.columns)
#show plot

avalanche_bymonth = avalanche.filter(['month','avalanche'])

avalanche_bymonth = avalanche_bymonth.groupby(['avalanche']).sum()
avalanche_bymonth

fig = go.Figure(
    data=[go.Bar(y=avalanche_bymonth['avalanche'])],
    layout_title_text="A Figure Displayed with fig.show()",
)
fig.show()
