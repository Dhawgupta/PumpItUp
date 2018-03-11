import pandas as pd 
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from tpot import TPOTClassifier

import lightgbm as lgb

from sklearn.linear_model import Ridge, RidgeCV

import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(name + ' done in ' + str(round(time.time() - t0)) + 's \n')
    
def get_year(data):
    return data.split('-')[0]

def get_month(data):
    return data.split('-')[1]

def get_day(data):
    return data.split('-')[2]

def convert(data):
    if data == 2:
        return 'functional'
    if data == 0:
        return 'non functional'
    if data == 1:
        return 'functional needs repair'
    
def data_preprocessing(df, mode):
    
    '''
    data['recored_year'] = data['date_recorded'].apply(get_year)
    data['recored_month'] = data['date_recorded'].apply(get_month)
    data['recored_day'] = data['date_recorded'].apply(get_day)
    
    categorical_features = list(data.select_dtypes(include=['object']).columns)
  
    for feature in categorical_features:
        data[feature] = pd.factorize(data[feature])[0]
    
    features_to_drop = ['id', 'date_recorded', 'status_group']
    
    if mode == 'train':
        X = data.drop(features_to_drop, axis=1)
        y = data['status_group']
        return X, y
    
    if mode == 'test':
        features_to_drop.remove('status_group')
        X = data.drop(features_to_drop, axis=1)
        return X  
    '''
    
    def funder_wrangler(row):  
    

        if row['funder']=='Government Of Tanzania':
            return 'gov'
        elif row['funder']=='Danida':
            return 'danida'
        elif row['funder']=='Hesawa':
            return 'hesawa'
        elif row['funder']=='Rwssp':
            return 'rwssp'
        elif row['funder']=='World Bank':
            return 'world_bank'    
        else:
            return 'other'
    
    df['funder'] = df.apply(lambda row: funder_wrangler(row), axis=1)
    
    def installer_wrangler(row):
        
        if row['installer']=='DWE':
            return 'dwe'
        elif row['installer']=='Government':
            return 'gov'
        elif row['installer']=='RWE':
            return 'rwe'
        elif row['installer']=='Commu':
            return 'commu'
        elif row['installer']=='DANIDA':
            return 'danida'
        else:
            return 'other'  
    
    df['installer'] = df.apply(lambda row: installer_wrangler(row), axis=1) 
    
    df = df.drop('subvillage', axis=1)
    
    df.public_meeting = df.public_meeting.fillna('Unknown')
    
    def scheme_wrangler(row):
    
        if row['scheme_management']=='VWC':
            return 'vwc'
        elif row['scheme_management']=='WUG':
            return 'wug'
        elif row['scheme_management']=='Water authority':
            return 'wtr_auth'
        elif row['scheme_management']=='WUA':
            return 'wua'
        elif row['scheme_management']=='Water Board':
            return 'wtr_brd'
        else:
            return 'other'

    df['scheme_management'] = df.apply(lambda row: scheme_wrangler(row), axis=1)
            
    df = df.drop('scheme_name', axis=1)
    
    df.permit = df.permit.fillna('Unknown')
    
    df.date_recorded = pd.datetime(2013, 12, 3) - pd.to_datetime(df.date_recorded)
    df.columns = ['days_since_recorded' if x=='date_recorded' else x for x in df.columns]
    df.days_since_recorded = df.days_since_recorded.astype('timedelta64[D]').astype(int)

    df = df.drop('wpt_name', axis=1)
    
    df = df.drop(['region', 'lga', 'ward'], axis=1)
    
    df = df.drop('recorded_by', axis=1)
    
    df = df.drop(['extraction_type', 'extraction_type_group'], axis=1)
    
    df = df.drop('management', axis=1)
    
    df = df.drop('management_group', axis=1)
    
    df = df.drop('payment', 1)
    
    df = df.drop('quality_group', 1)
    
    df = df.drop('quantity_group', 1)
    
    df = df.drop('source', 1)
    
    df = df.drop(['gps_height', 'longitude', 'latitude', 'region_code', 'district_code',
             'num_private', 'id'], axis=1)
    
    def construction_wrangler(row):
        if row['construction_year'] >= 1960 and row['construction_year'] < 1970:
            return '60s'
        elif row['construction_year'] >= 1970 and row['construction_year'] < 1980:
            return '70s'
        elif row['construction_year'] >= 1980 and row['construction_year'] < 1990:
            return '80s'
        elif row['construction_year'] >= 1990 and row['construction_year'] < 2000:
            return '90s'
        elif row['construction_year'] >= 2000 and row['construction_year'] < 2010:
            return '00s'
        elif row['construction_year'] >= 2010:
            return '10s'
        else:
            return 'unknown'
    
    df['construction_year'] = df.apply(lambda row: construction_wrangler(row), axis=1)
    
    
    dummy_cols = ['funder', 'installer', 'basin', 'public_meeting', 'scheme_management', 'permit',
                  'construction_year', 'extraction_type_class', 'payment_type', 'water_quality',
                  'quantity', 'source_type', 'source_class', 'waterpoint_type',
                 'waterpoint_type_group']
    
    df = pd.get_dummies(df, columns = dummy_cols)
    
    if mode == 'train':
        
        vals_to_replace = {'functional':2, 'functional needs repair':1,
                       'non functional':0}
        df['status_group_vals'] = df.status_group.replace(vals_to_replace)
        
        X = df.drop(['status_group_vals', 'status_group'], 1)
        y = df['status_group_vals']
        return X, y
    
    if mode == 'test':
        X = df
        return X
    


with timer("Reading input files"):    

    data_path = '/home/ubuntu/toxic/PumpItUp/PumpItUp-master/'
    
    train_df = pd.read_csv(data_path + 'train.csv')
    test_df = pd.read_csv(data_path + 'test.csv')
    train_label = pd.read_csv(data_path + 'train_labels.csv')
    
    train_df = pd.merge(train_df, train_label)

with timer("Data preprocessing"): 
    
    X, y = data_preprocessing(train_df, 'train')
    test_df = data_preprocessing(test_df, 'test')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
    
pipeline_optimizer = TPOTClassifier(generations=100, population_size=100, cv=5,
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')
