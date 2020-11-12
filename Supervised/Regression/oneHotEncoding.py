import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing


df = pd.read_csv(r'C:\Users\telelink\Contacts\Desktop\carPrices.csv')
df = df.rename(columns={'Car Model' : 'CarModel', 'Age(yrs)' : 'years'} )

plt.scatter(df.Mileage, df.Price, c='b')
plt.xlabel('Mileage')
plt.ylabel('Price')

le = preprocessing.LabelEncoder()
df.CarModel = le.fit_transform(df.CarModel)
print(df)

X = df[['CarModel', 'Mileage', 'years']].values
Y = df['Price'].values

reg = linear_model.LinearRegression()
reg.fit(X,Y)
print('benz: ',reg.predict([[2, 45000, 4]]))
print('BMW: ',reg.predict([[1, 86000, 7]]))
print(reg.score(X,Y))
plt.show()


