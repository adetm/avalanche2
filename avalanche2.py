#Import packages
import numpy as np
import pandas as pd
import math
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import plotly.graph_objects as go
import datetime
import plotly.tools as tls
import plotly.express as px
import os
import chart_studio
chart_studio.tools.set_credentials_file(username='xxxxx', api_key='xxxxxxx')

#import CSV
avalanche= pd.read_csv('/Users/AnnaD/Desktop/avalanche/2CAIC_AccDB_Investigations (original).csv')
avalanche = avalanche.iloc[:,0:39]

#Extract information about time of avalanche
avalanche['avi_time_known'] = avalanche['avi_time_known'].astype(str)
avalanche_time = avalanche[(avalanche['avi_time_known']=='Known') | (avalanche['avi_time_known']=='Estimated')]
avalanche_time['avi_date_time'] = pd.to_datetime(avalanche_time['avi_date_time'])
avalanche_time['time'] = (avalanche_time['avi_date_time'].dt.time)
avalanche_time['time'] = pd.to_datetime(avalanche_time['time'], format='%H:%M:%S').dt.time
avalanche_time_sort = avalanche_time.sort_values(by=['time'])

#plot avalanche time distribution
fig = fig = px.histogram(avalanche_time_sort, x="time", nbins=9, title='Time of Avalanche Distribution')
fig.show()


#scatter plot
#filter df columns to simplify
rose_scatter = avalanche[['site_elev','aspect','travel_mode.1']]
#drop na
rose_scatter = rose_scatter.dropna()

#change aspect to degrees for ordering
def set_order(row):
    if row["aspect"] == 'N':
        return 0
    elif row["aspect"] == 'NE':
        return 45
    elif row["aspect"] == 'E':
        return 90
    elif row["aspect"] == 'SE':
        return 135
    elif row["aspect"] == 'S':
        return 180
    elif row["aspect"] == 'SW':
        return 225
    elif row["aspect"] == 'W':
        return 270
    elif row["aspect"] == 'NW':
        return 315
    else:
        return null

rose_scatter = rose_scatter.assign(order = rose_scatter.apply(set_order,axis=1))

#group by aspect of occurrence
count_aspect = pd.DataFrame(avalanche.groupby('aspect').count()['avalanche']).reset_index()
count_aspect = count_aspect.rename(columns={"avalanche": "Count"})

fig= px.bar_polar(count_aspect, r=count_aspect['Count'], theta=['N','NE','E','SE','S','SW','W','NW'],
                   color="Count", template="ggplot2"
                   )
fig.update_layout(title_text="Avalanches by Aspect (2009-2020)")
fig.show()
