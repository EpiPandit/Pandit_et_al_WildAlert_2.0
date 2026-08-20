
# coding: utf-8

# In[ ]:

from __future__ import division
import os as os
 
from IPython.display import HTML
import pandas as pd
import numpy as np
import os as os
from matplotlib import pyplot as plt
import seaborn as sns
from numpy import random as random



from matplotlib.colors import ListedColormap
plt.rcParams['figure.figsize'] = (12, 12)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
#plt.rcParams['savefig.dpi'] = 3*plt.rcParams['savefig.dpi']
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.loc'] = 'center left'
plt.rcParams['axes.linewidth'] = 1

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')
sns.set_style('whitegrid')
plt.close()


import re
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import count #,izip
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import pandas as pd
import collections
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
#import plotly.plotly as py
#import chart_studio.plotly as py
#import plotly.io as pio
#py.sign_in('PranavPandit', 'k6a6t505FellCQ6HzlU0')
#import cufflinks as cf



#####################################################################################################################
#####################################################################################################################


def read_data(filename):
    data = pd.read_csv(filename, encoding= 'unicode_escape')
    #data = pd.merge(data, sc_names, left_on='Species', right_on='Training_data_name', how='left')
    #data = pd.read_csv(filename, low_memory=False, skiprows= [0])
    #data.columns = ['ID', 'Case', 'Admitted_at', 'Species', 'Organization','Reasons', 'Address_found', 'Latitude', 'Longitude',
    #                'Care_rescuer', 'Disposition', 'Dispositioned_at', 'Diagnosis', 'Sex', 'Weight', 'Age', 'Condition']
    data['Admitted_at'] = pd.to_datetime(data['Admitted_at']).dt.strftime('%Y-%m-%d')
    data.set_index(pd.DatetimeIndex(data['Admitted_at']), inplace= True, drop= False)
    return data


#####################################################################################################################
#####################################################################################################################

def test_stationarity(SpeciesName, syndrome, data):
    
    vo = data[(data.ScientificName == SpeciesName) & (data.prediction_adjusted == syndrome)]
    weekly_vo = vo.resample('W')['WRMD_ID'].count()
    #Determing rolling statistics
    rolmean = weekly_vo.rolling(window=4,center=False).mean() 
    rolstd = weekly_vo.rolling(window=4,center=False).std()
    sp_data = pd.concat([weekly_vo, rolmean, rolstd], axis=1)
    sp_data.columns = ['original', 'rolling mean', 'std deviation']
    layout = go.Layout(
        title= str(SpeciesName)+': admissions, rolling mean & std deviation',
        yaxis=dict(title='# admissions'),
        xaxis=dict(title='time'))
    
    #Plot rolling statistics:
    sp_data.iplot(kind='scatter',layout=layout)
       
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(weekly_vo, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    print ("""
    
    
    Additive Decomposition of Dataset
    
    
    """)
    daily = vo.resample('15D')['WRMD_ID'].count()
    result = seasonal_decompose(daily, model='additive')
    result.plot()
    

#####################################################################################################################
#####################################################################################################################

  


def plot_anomalies(data, Species, family, syndrome, window_size, sigma, time_bin = 'W', save = False, verbose = False):
    
    #result = test_stationarity(SpeciesName=Species, data =  data)
    #result.plot()
    
    """
    options for time bin
    'D' = day,
    'M' = month,
    'W' = week,
    """
    if Species == 'None':
        vo = data[(data.family == family) & (data.prediction_adjusted == syndrome)]
    else:
        vo = data[(data.ScientificName == Species) & (data.prediction_adjusted == syndrome)]
        
    weekly_vo = vo.resample(time_bin)['WRMD_ID'].count()
    weekly_vo = pd.DataFrame(weekly_vo)
    weekly_vo.columns = ['ID']
    weekly_vo['rolling_mean'] = weekly_vo.ID.rolling(window = window_size, center = False).mean()
    weekly_vo['residual'] = weekly_vo.ID - weekly_vo.rolling_mean
    weekly_vo['std'] = weekly_vo.residual.std(axis=0)

    weekly_vo['testing_std'] = weekly_vo.residual.rolling(window= window_size, center= False).std()
    weekly_vo.testing_std.fillna(weekly_vo.testing_std.mean(), inplace= True)
    
    def identify_anomalies(c, sigma=sigma):
        if c.ID > c.rolling_mean + (sigma*c.testing_std):
            return c.ID
    weekly_vo['anomalies'] = weekly_vo.apply(identify_anomalies, axis=1)
    weekly_vo.columns = ['# admissions', 'rolling mean', 'std', 'residual', 'rolling std', 'anomalies']
    
    upper_bound = go.Scatter(
        name='upper Bound',
        x=weekly_vo.index,
        y=weekly_vo['rolling mean'] + (2*weekly_vo['rolling std']),
        mode='lines',
        marker=dict(color="#820101"),
        line=dict(width=0),
        fillcolor='#b7b7b7',
        fill='tonexty' )
    
    rolling_m = go.Scatter(
        x = weekly_vo.index,
        y = weekly_vo['rolling mean'],
        name='rolling mean',
        mode='lines',
        line=dict(color='#1f77b4'),
        fillcolor='#b7b7b7',
        fill='tonexty' )

    lower_bound = go.Scatter(
        name='lower Bound',
        x=weekly_vo.index,
        y=weekly_vo['rolling mean'] - (2*weekly_vo['rolling std']),
        marker=dict(color="#b7b7b7"),
        line=dict(width=0),
        mode='lines',)

    addmissions = go.Scatter(
        x = weekly_vo.index,
        y = weekly_vo['# admissions'],
        name='# admissions',)

    ano = go.Scatter(
        x = weekly_vo.index,
        y =weekly_vo['anomalies'] ,
        name='anomalies',
        mode='markers')

    plottingdata = [lower_bound, rolling_m,upper_bound, addmissions, ano]
    if Species == 'None':
        T = "Weekly admissions of "+ str(family)+' ('+syndrome+')',
    else:
        T = "Weekly admissions of "+ str(Species)+' ('+syndrome+')',
    print (T)
    layout = go.Layout(
        title= 'Weekly admissions',
        yaxis=dict( title='number of admissions',),  width=900, height=640
        )

    fig = go.Figure(data=plottingdata, layout=layout)
    plot_url = iplot(fig)
    if verbose:
        print('number of admissions triggering alert') 
        print(weekly_vo[weekly_vo['anomalies'].notnull()]['anomalies'])
    if save:
        name = family+Species+syndrome
        py.image.save_as(fig, filename='C:/Users/Falco/Desktop/directory/WMRD/data/'+name+'.png')
        #pio.write_image(fig, 'C:/Users/Falco/Desktop/directory/WMRD/data/'+name+'2.png')

    #from IPython.display import Image
    #Image('C:\Users\Falco\Desktop\directory\WMRD\data\TimeSeries.png')
    
    return weekly_vo 


def plot_anomalies_data_specific(df, syndrome, window_size, sigma, time_bin = 'W', save = False, verbose = False, adjusted = True):
    
    #result = test_stationarity(SpeciesName=Species, data =  data)
    #result.plot()
    
    """
    options for time bin
    'D' = day,
    'M' = month,
    'W' = week,
    
    """
    if syndrome:
        if adjusted:
            vo = df[df.prediction_adjusted == syndrome]
        else:
            vo = df[df.prediction == syndrome]
    else:
        vo = df
        
    weekly_vo = vo.resample(time_bin)['WRMD_ID'].count()
    weekly_vo = pd.DataFrame(weekly_vo)
    weekly_vo.columns = ['ID']
    weekly_vo['rolling_mean'] = weekly_vo.ID.rolling(window = window_size, center = False).mean()
    weekly_vo['residual'] = weekly_vo.ID - weekly_vo.rolling_mean
    weekly_vo['std'] = weekly_vo.residual.std(axis=0)

    weekly_vo['testing_std'] = weekly_vo.residual.rolling(window= window_size, center= False).std()
    weekly_vo.testing_std.fillna(weekly_vo.testing_std.mean(), inplace= True)
    
    def identify_anomalies(c, sigma=sigma):
        if c.ID > c.rolling_mean + (sigma*c.testing_std):
            return c.ID
    weekly_vo['anomalies'] = weekly_vo.apply(identify_anomalies, axis=1)
    weekly_vo.columns = ['# admissions', 'rolling mean', 'std', 'residual', 'rolling std', 'anomalies']
    
    upper_bound = go.Scatter(
        name='upper Bound',
        x=weekly_vo.index,
        y=weekly_vo['rolling mean'] + (2*weekly_vo['rolling std']),
        mode='lines',
        marker=dict(color="#820101"),
        line=dict(width=0),
        fillcolor='#b7b7b7',
        fill='tonexty' )
    
    rolling_m = go.Scatter(
        x = weekly_vo.index,
        y = weekly_vo['rolling mean'],
        name='rolling mean',
        mode='lines',
        line=dict(color='#1f77b4'),
        fillcolor='#b7b7b7',
        fill='tonexty' )

    lower_bound = go.Scatter(
        name='lower Bound',
        x=weekly_vo.index,
        y=weekly_vo['rolling mean'] - (2*weekly_vo['rolling std']),
        marker=dict(color="#b7b7b7"),
        line=dict(width=0),
        mode='lines',)

    addmissions = go.Scatter(
        x = weekly_vo.index,
        y = weekly_vo['# admissions'],
        name='# admissions',)

    ano = go.Scatter(
        x = weekly_vo.index,
        y =weekly_vo['anomalies'] ,
        name='anomalies',
        mode='markers')

    plottingdata = [lower_bound, rolling_m,upper_bound, addmissions, ano]
    
    T = "Weekly admissions"
    print (T)
    layout = go.Layout(
        title= 'Weekly admissions',
        yaxis=dict( title='number of admissions',),  width=900, height=640
        )

    fig = go.Figure(data=plottingdata, layout=layout)
    plot_url = iplot(fig)
    if verbose:
        print('number of admissions triggering alert') 
        print(weekly_vo[weekly_vo['anomalies'].notnull()]['anomalies'])
    if save:
        name = 'special_syndrome_study'
        py.image.save_as(fig, filename='C:/Users/Falco/Desktop/directory/WRMD_paper/outputs/'+name+'.png')
        #pio.write_image(fig, 'C:/Users/Falco/Desktop/directory/WMRD/data/'+name+'2.png')

    #from IPython.display import Image
    #Image('C:\Users\Falco\Desktop\directory\WMRD\data\TimeSeries.png')
    
    return weekly_vo 




    