"""
Ridge model for predicting bike rental demand for Capital Bike Share
in Washington D.C.
"""
import pandas as pd
from scipy import stats
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

DF = pd.read_csv('train_dataset.csv')

def create_timefs(df):
    """
    Creates hour and month column in the dataframe
    """
    df = df.copy()
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    df['month'] = pd.to_datetime(df['datetime']).dt.month
    return df

DF = create_timefs(DF)
DF = DF.drop(['datetime'], axis=1)
z = np.abs(stats.zscore(DF))
DF = DF[(z < 3).all(axis=1)]

feature_list = ['temp', 'atemp', 'workingday', 'hour', 'month',
                'weather', 'humidity']
y = np.log1p(DF["count"])

X_train = (DF[feature_list])
y_train = y

linear_m = make_pipeline(PolynomialFeatures(degree=8), Ridge(max_iter=3000, alpha=0.0001, normalize=True))
linear_m.fit(X_train, y_train)
