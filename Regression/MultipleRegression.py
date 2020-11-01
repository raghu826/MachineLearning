import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
# matplotlib inline

df = pd.read_csv(r'C:\Users\telelink\Contacts\Desktop\irisDataSet.csv')
cdf = df.drop(columns=['Unnamed: 0'])
newData = cdf.rename(columns={"Sepal.Length": "SepalLength","Petal.Length": "PetalLength",
                      "Sepal.Width": "SepalWidth","Petal.Width": "PetalWidth"})
print(newData.head())

msk = np.random.rand(len(newData)) < 0.8
train = newData[msk]
test = newData[~msk]

# plt.scatter(train.SepalLength, train.PetalWidth,  color='blue')
# plt.xlabel("SepalLength")
# plt.ylabel("PetalWidth")

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['SepalLength', 'SepalWidth']])
y = np.asanyarray(train[['PetalWidth']])
regr.fit(x, y)
print('Coefficients: ', regr.coef_)   # The coefficients


y_hat= regr.predict(test[['SepalLength', 'SepalWidth']])
x = np.asanyarray(test[['SepalLength', 'SepalWidth']])
y = np.asanyarray(test[['PetalWidth']])
print("Residual sum of squares: %.2f"% np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

#plt.show()