import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
# matplotlib inline

df = pd.read_csv(r'C:\Users\telelink\Contacts\Desktop\irisDataSet.csv')
print(df.head())

dummies = pd.get_dummies(df['Species'])
print(dummies)
# Create a subset of the Iris data that contains only the sepal length attribute, and only the setosa and
# virginica classes. Draw a scatterplot showing the attribute on the x-axis and the class on the y-axis.

cdf = df[['SepalLength', 'Species']]
cdf = cdf[cdf['Species'].isin(['setosa', 'virginica'])]
cdf = cdf.reset_index(drop=True)
plt.scatter(cdf.SepalLength, cdf.Species, c='b')
plt.xlabel('Sepal length')
plt.ylabel('Species')

# dummies = pd.get_dummies(df['Species'])
# print(dummies)

msk = np.random.rand(len(df)) < 0.7
train = df[msk]
test = df[~msk]

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

plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-g')
#plt.plot(train_x, regr.predict(train_x), '-r')

test_x = np.asanyarray(test[['SepalLength']])
test_y = np.asanyarray(test[['PetalWidth']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_))

plt.show()
