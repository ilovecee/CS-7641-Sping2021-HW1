# This module is for data preprocessing and exploratory data analysis
# Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Define a function to preprocess the HR turnover data
def turnover_prep (filename):
    # Read data
    df = pd.read_csv (filename)

    #'salary' and 'department' are categorical features, transform to numerical
    df['salary'] = df['salary'].astype('category')
    df['salary'] = df['salary'].cat.reorder_categories(['low', 'medium', 'high'])
    df['salary'] = df['salary'].cat.codes

    # 'department'
    dum = pd.get_dummies (df['department'])
    dum = dum.drop (dum.columns[-1], axis = 1)
    df = df.drop ('department', axis = 1)
    df = df.join(dum)

    # drop the target feature: 'churn
    X = df.drop('churn', axis = 1)
    y = df['churn']

    return X, y





def wine_prep (filename):
    # read the data
    df = pd.read_csv (filename)

    X = df.drop('quality', axis = 1)
    y = df['quality']
    y[y <= 5] = 1
    y[y > 5] = 0
    y = y.astype('bool')


    return X, y