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

msk = np.random.rand(len(newData)) < 0.7
train = newData[msk]
test = newData[~msk]

plt.scatter(train.SepalLength, train.PetalWidth,  color='blue')
plt.xlabel("SepalLength")
plt.ylabel("PetalWidth")

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['SepalLength']])
train_y = np.asanyarray(train[['PetalWidth']])
regr.fit(train_x, train_y)
# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

#plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")


test_x = np.asanyarray(test[['SepalLength']])
test_y = np.asanyarray(test[['PetalWidth']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_))

plt.show()