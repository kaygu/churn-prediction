import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print(f'Start of {__name__}')
    df = pd.read_csv('BankChurners.csv')
    print(df.shape)
    print(df.head())
    #print(df.isnull().sum())
