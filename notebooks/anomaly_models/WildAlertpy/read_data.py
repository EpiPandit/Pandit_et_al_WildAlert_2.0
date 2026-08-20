
import re
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
import pandas as pd
import collections

"""
def read_data(filename):
    data = pd.read_csv(filename, low_memory=False)
    data.columns = ['WRMD_ID','Case', 'Admitted_at', 'Species', 'Organization','Address_found','Latitude', 'Longitude',
                    'Disposition', 'Dispositioned_at', 'Age', 'label', 'Reasons', 'Condition','Diagnosis', 'Notes']
    data['Admitted_at'] = pd.to_datetime(data['Admitted_at']).dt.strftime('%Y-%m-%d')
    data.set_index(pd.DatetimeIndex(data['Admitted_at']), inplace= True, drop= False)
    data['label'] = data.label.str.lower()
    df = data.label.str.split(',', expand=True) 
    df.columns = ['c1', 'c2','c3','c4','c5']
    df['c1'] = df.c1.str.strip()
    df['c2'] = df.c2.str.strip()
    df['c3'] = df.c3.str.strip()
    df['c4'] = df.c4.str.strip()
    df['c5'] = df.c4.str.strip()
    df.replace('physical inury', 'physical injury', inplace= True)
    df.replace('physical injury', 'physical injury', inplace= True)
    df.replace('neurological disease', 'neurologic disease', inplace =  True)
    df.replace('nerologic disease', 'neurologic disease', inplace =  True)
    df.replace('nekurologic disease', 'neurologic disease', inplace =  True)
    df.replace('nutritional disease', 'nutritional disease', inplace =  True)
    df.replace(' nutritional disease', 'nutritional disease', inplace =  True)
    df.replace('petrochemical exposure ', 'petrochemical exposure', inplace =  True)
    df.replace('gastrointestinal disease ', 'gastrointestinal disease', inplace =  True)
    df.replace('gastrointestinal  disease', 'gastrointestinal disease', inplace =  True)
    df.replace('physical  injury', 'physical injury', inplace= True)
    df.replace('physical trauma', 'physical injury', inplace= True)
    df.replace('nonspecifc', 'nonspecific', inplace= True)
    df.replace('nonspeciic', 'nonspecific', inplace= True)
    df.replace('neurologic diseases', 'neurologic disease', inplace= True)
    df.replace('beaced', 'stranded', inplace= True)

    df.replace('physical inury', 'physical injury', inplace= True)
    df.replace('physcial injury', 'physical injury', inplace= True)
    df.replace('phyical injury', 'physical injury', inplace= True)
    def get_predicting_condition(c):
        if not c.c2:
            return c.c1
        elif (c.c1 == 'stranded')| (c.c1 == 'physical injury'):
            return c.c2
        else:
            return c.c1


    df['condition_predict'] = df.apply(get_predicting_condition, axis = 1)
    data = pd.concat([data, df], axis=1)

    data['Condition'] = data['Condition'].fillna('empty')
    data['Reasons'] = data['Reasons'].fillna('empty')
    data['Diagnosis'] = data['Diagnosis'].fillna('empty')
    data["Condition"] = data["Condition"].map(str)+" " + data["Reasons"] +" " +data["Diagnosis"]
    
    for c in data.columns:
        if data[c].dtype == 'O':
            data[c] = data[c].str.replace('\\xa0', '')
            data[c] = data[c].str.replace('xa0', '')
            data[c] = data[c].str.replace('\\xf3', 'o')
            data[c] = data[c].str.replace('\\xe1', 'a')
            data[c] = data[c].str.replace('\\xe9', 'e')
            data[c] = data[c].str.replace('\\xe3', 'a')
            data[c] = data[c].str.strip()
            data[c] = data[c].str.replace('[^\x00-\x7F]','')
    data.Condition.replace(regex=True,inplace = True, to_replace=r'\**',value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r'\:',value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r'\,',value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r'\.',value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r'\-',value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r'\"',value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r"\'",value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r"\;",value=r'')
    data.Condition.replace(regex=True,inplace = True, to_replace=r"\/",value=r'')
    data.Condition = data.Condition.str.strip()

    return data
"""
"""    
def read_data(filename):
    data = pd.read_csv(filename, encoding= 'unicode_escape')
    #data = pd.merge(data, sc_names, left_on='Species', right_on='Training_data_name', how='left')
    #data = pd.read_csv(filename, low_memory=False, skiprows= [0])
    #data.columns = ['ID', 'Case', 'Admitted_at', 'Species', 'Organization','Reasons', 'Address_found', 'Latitude', 'Longitude',
    #                'Care_rescuer', 'Disposition', 'Dispositioned_at', 'Diagnosis', 'Sex', 'Weight', 'Age', 'Condition']
    data['Admitted_at'] = pd.to_datetime(data['Admitted_at']).dt.strftime('%Y-%m-%d')
    data.set_index(pd.DatetimeIndex(data['Admitted_at']), inplace= True, drop= False)
    return data
"""
def read_data(filename):
    data = pd.read_excel(filename,)
    #data = pd.merge(data, sc_names, left_on='Species', right_on='Training_data_name', how='left')
    #data = pd.read_csv(filename, low_memory=False, skiprows= [0])
    #data.columns = ['ID', 'Case', 'Admitted_at', 'Species', 'Organization','Reasons', 'Address_found', 'Latitude', 'Longitude',
    #                'Care_rescuer', 'Disposition', 'Dispositioned_at', 'Diagnosis', 'Sex', 'Weight', 'Age', 'Condition']
    data['reported_date'] = pd.to_datetime(data['reported_date']).dt.strftime('%Y-%m-%d')
    data.set_index(pd.DatetimeIndex(data['reported_date']), inplace= True, drop= False)
    #print(data.columns.tolist())
    return data

def read_data_historic(filename):
    h_data = pd.read_csv(filename, encoding='latin-1')
    #h_data = pd.read_csv('C:\Users\Falco\Desktop\directory\WMRD\data\historic_data_2013-01-01_2018-05-22_above_all major centers.csv')
    h_data.columns = ['WRMD_ID','Case', 'Admitted_at', 'Species', 'Organization','Reasons', 'Address_found','Latitude', 'Longitude',
                    'Disposition', 'Dispositioned_at','Diagnosis', 'Sex','Weight', 'Age','Condition']
    h_data['Admitted_at'] = pd.to_datetime(h_data['Admitted_at']).dt.strftime('%Y-%m-%d')
    h_data.set_index(pd.DatetimeIndex(h_data['Admitted_at']), inplace= True, drop= False)
    h_data.set_index(pd.DatetimeIndex(h_data['Admitted_at']), inplace= True, drop= False)
    """
    columns which will be used to predict 

    """
    h_data['ConditionO'] = h_data['Condition'].fillna('empty')
    h_data['Reasons'] = h_data['Reasons'].fillna('empty')
    h_data['Diagnosis'] = h_data['Diagnosis'].fillna('empty')
    h_data["ConditionO"] = h_data["ConditionO"].map(str)+" " + h_data["Reasons"] +" " +h_data["Diagnosis"]

    for c in h_data.columns:
        if h_data[c].dtype == 'O':
            h_data[c] = h_data[c].str.replace('\\xa0', '')
            h_data[c] = h_data[c].str.replace('xa0', '')
            h_data[c] = h_data[c].str.replace('\\xf3', 'o')
            h_data[c] = h_data[c].str.replace('\\xe1', 'a')
            h_data[c] = h_data[c].str.replace('\\xe9', 'e')
            h_data[c] = h_data[c].str.replace('\\xe3', 'a')
            h_data[c] = h_data[c].str.strip()
            h_data[c] = h_data[c].str.replace('[^\x00-\x7F]','')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r'\**',value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r'\:',value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r'\,',value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r'\.',value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r'\-',value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r'\"',value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r"\'",value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r"\;",value=r'')
    h_data.Condition.replace(regex=True,inplace = True, to_replace=r"\/",value=r'')
    h_data.Condition = h_data.Condition.str.strip()
    return h_data
   