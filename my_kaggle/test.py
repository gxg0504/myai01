# -*- coding: UTF-8 -*-
# %matplotlib inline

import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    return np.nan
def replace_titles(x):
    title=x['Title']
    if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Master']:
        return 'Master'
    elif title in ['Countess', 'Mme','Mrs']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms','Miss']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    elif title =='':
        if x['Sex']=='Male':
            return 'Master'
        else:
            return 'Miss'
    else:
        return title
title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']
				
label = train['Survived'] # 目标列

# 接下来我们对每个特征进行一下分析：
train.groupby(['Pclass'])['PassengerId'].count().plot(kind='bar')
plt.show()
train.groupby(['SibSp'])['PassengerId'].count().plot(kind='bar')
plt.show()
train.groupby(['Parch'])['PassengerId'].count().plot(kind='bar')
train.groupby(['Embarked'])['PassengerId'].count().plot(kind='bar')
train.groupby(['Sex'])['PassengerId'].count().plot(kind='bar')
