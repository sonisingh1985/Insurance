import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

 
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

insuranceDF = pd.read_csv('insurance2.csv')
print(insuranceDF.head())

insuranceDF.info()

corr = insuranceDF.corr()
print(corr)
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.show()
dfTrain = insuranceDF[:1000]
dfTest = insuranceDF[1000:1300]
dfCheck = insuranceDF[1300:]

trainLabel = np.asarray(dfTrain['insuranceclaim'])
trainData = np.asarray(dfTrain.drop('insuranceclaim',1))
testLabel = np.asarray(dfTest['insuranceclaim'])
testData = np.asarray(dfTest.drop('insuranceclaim',1))

means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)
 
trainData = (trainData - means)/stds
testData = (testData - means)/stds

insuranceCheck = LogisticRegression()
insuranceCheck.fit(trainData, trainLabel)

accuracy = insuranceCheck.score(testData, testLabel)
print("accuracy = ", accuracy * 100, "%")

coeff = list(insuranceCheck.coef_[0])
labels = list(dfTrain.drop('insuranceclaim',1).columns)
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')
plt.show()
joblib.dump([insuranceCheck, means, stds], 'insurance01Model.pkl')

insuranceLoadedModel, means, stds = joblib.load('insurance01Model.pkl')
accuracyModel = insuranceLoadedModel.score(testData, testLabel)
print("accuracy = ",accuracyModel * 100,"%")

print(dfCheck.head(38))

sampleData = dfCheck[2:3]
 
# prepare sample  
sampleDataFeatures = np.asarray(sampleData.drop('insuranceclaim',1))
sampleDataFeatures = (sampleDataFeatures - means)/stds
 
# predict 
predictionProbability = insuranceLoadedModel.predict_proba(sampleDataFeatures)
prediction = insuranceLoadedModel.predict(sampleDataFeatures)
print('Insurance Claim Probability:', predictionProbability)
print('Insurance Claim Prediction:', prediction)

