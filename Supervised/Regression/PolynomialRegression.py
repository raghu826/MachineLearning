import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score
#matplotlib inline

df = pd.read_csv(r'C:\Users\telelink\Contacts\Desktop\irisDataSet.csv')
cdf = df.drop(columns=['Unnamed: 0'])
newData = cdf.rename(columns={"Sepal.Length": "SepalLength","Petal.Length": "PetalLength",
                      "Sepal.Width": "SepalWidth","Petal.Width": "PetalWidth"})
#print(newData.head())

plt.scatter(newData.PetalLength, newData.PetalWidth,  color='blue')
plt.xlabel("PetalLength")
plt.ylabel("PetalWidth")

msk = np.random.rand(len(newData)) < 0.8
train = newData[msk]
test = newData[~msk]

train_x = np.asanyarray(train[['PetalLength']])
train_y = np.asanyarray(train[['PetalWidth']])

test_x = np.asanyarray(test[['PetalLength']])
test_y = np.asanyarray(test[['PetalWidth']])


poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
train_x_poly

clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
# The coefficients
print('Coefficients: ', clf.coef_)
print('Intercept: ', clf.intercept_)

plt.scatter(train.PetalLength, train.PetalWidth,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0] + clf.coef_[0][1]*XX + clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("PetalLength")
plt.ylabel("PetalWidth")

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_, test_y))

plt.show()
