import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
# matplotlib inline

df = pd.read_csv(r'C:\Users\telelink\Contacts\Desktop\CanadaIncome.csv')
print(df.head())

plt.scatter(df.Year, df.PerCapitaIncome, c='r', marker='*')
plt.xlabel('Year')
plt.ylabel('PerCapitaIncome')

regr = linear_model.LinearRegression()
regr.fit(df[['Year']], df.PerCapitaIncome)
# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)
plt.plot(df.Year, regr.predict(df[['Year']]), '-b')

print(regr.predict([[2020]]))
print("R2-score: %.2f" % r2_score(df.PerCapitaIncome, regr.predict(df[['Year']])))

plt.show()

