"""
1) Explores the data using histograms
2) Analyses using logistic regression
3) Uses KNN to make a prediction of the survival rate

"""

import pandas
import numpy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf 

data = pandas.read_csv('train.csv', low_memory=False)
datatest = pandas.read_csv('test.csv', low_memory= False)

# data['Sex'] = data['Sex'].astype('category')
# sns.countplot(x = 'Sex', data = data)
# plt.show()

# print (data['Sex'].describe())

# Make age round
data['Age'] = data['Age'].apply(lambda x: round(x, 0))
datatest['Age'] = data['Age'].apply(lambda x: round(x, 0))
# print (data['Age'][803])


# bin the age into a few groups
data['AgeGroup'] = pandas.cut(data['Age'], 5)
print (data['AgeGroup'].head(5))


# count plot on gender vis-a-vis whether they survived
sns.countplot(x = 'Sex', hue = 'Survived', data = data)
plt.show()

# count plot on passenger class vis-a-vis whether they survived
plt.figure(2)
sns.countplot(x = 'Pclass', hue = 'Survived', data = data)
plt.show()

# countplot on Age Group vis-a-vis whether they survived
plt.figure(3)
sns.countplot(x = 'AgeGroup', hue = 'Survived', data = data)
plt.show()

# # Centre the Variables before conducting justice
# 
# def centrevar(x):
#     data[x] = (data[x] - data[x].mean())
# 
# centrevar('Pclass')
# centrevar('Sex')

# Logistics regression
lreg1 = smf.logit(formula = 'Survived ~ Pclass + Sex + AgeGroup + Parch', data = data).fit()
print (lreg1.summary())
print ("")

# odds ratios
print ("Odds Ratios")
print (numpy.exp(lreg1.params))

# odd ratios with 95% confidence intervals
params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))

# change Gender to Categorical Integers
def GenderInt(row):
    if row['Sex'] == "male":
        return 0
    elif row['Sex'] == "female":
        return 1
data['Sex'] = data.apply(lambda row: GenderInt(row), axis = 1)
datatest['Sex'] = datatest.apply(lambda row: GenderInt(row), axis = 1)

def HandleMissingValues(dataset, col):
    dataset[col] = dataset[col].fillna(dataset[col].median())

HandleMissingValues(data, 'Age')
HandleMissingValues(data, 'Pclass')
HandleMissingValues(datatest, 'Age')
HandleMissingValues(datatest, 'Pclass')
HandleMissingValues(datatest, 'Sex')
HandleMissingValues(data, 'Sex')   
HandleMissingValues(datatest, 'Parch')
HandleMissingValues(data, 'Parch')   
HandleMissingValues(datatest, 'SibSp')
HandleMissingValues(data, 'SibSp')

from sklearn.neighbors import KNeighborsClassifier
TrainingData = np.array(data[['Age','Pclass', 'Sex', 'Parch']])
TrainingScores = np.array(data['Survived'])
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(TrainingData, TrainingScores)
TestData = np.array(datatest[['Age','Pclass', 'Sex','Parch']])
datatest['Survived'] = neigh.predict(TestData)
# print (datatest)
datasubmission = datatest[['PassengerId', 'Survived']]


datasubmission.to_csv('Results.csv', sep = ',', index = False)
