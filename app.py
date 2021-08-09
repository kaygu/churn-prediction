import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print(f'Start of {__name__}')
    df = pd.read_csv('BankChurners.csv')
    df = df.iloc[:,:-2] # Remove 2 last columns
    df.drop('CLIENTNUM', axis=1, inplace=True)
    print(f'There are {df.shape[0]} rows and {df.shape[1]} columns in this dataset')
    print(df.info())
    # Tranform object dtypes to category
    objs = df.select_dtypes(['object'])
    for col in objs.columns:
        df[col] = df[col].astype('category')
    cat = df.select_dtypes('category')
    for col in cat.columns:
        print(f'\n{col} :\n', df[col].value_counts())
    #print('\n\nNumber unique value :\n', df.nunique())
    #print('\n\nNumber of Unknown values :\n', df.isin(['Unknown']).sum())
    '''
    print(df.describe())
    for c in df.columns:
        print(c, df[c].unique())
    print(df.duplicated(subset=['CLIENTNUM']).count())
    #print(df.isnull().sum())
    '''
    #print(df.value_counts())
    print(f'\nChurn rate : {len(df[df["Attrition_Flag"] == "Attrited Customer"]) / df.shape[0] * 100:.3f}%')
    #print(f'\nEducation Level :\n{df["Education_Level"].value_counts()}\n') # Pie chart ?
    #print(f'\nIncome Category :\n{df["Income_Category"].value_counts()}\n')
    print(df.dtypes)
