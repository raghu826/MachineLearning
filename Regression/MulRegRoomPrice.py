import pandas as pd
import matplotlib.pyplot as plt
from word2number import w2n
from sklearn import linear_model

df = pd.read_csv(r'C:\Users\telelink\Contacts\Desktop\hiring.csv')
df.experience = df.experience.fillna("zero")
df.experience = df.experience.apply(w2n.word_to_num)

median = df['test_score'].median()
df['test_score'] = df['test_score'].fillna(median)
print(df)
plt.scatter(df.experience, df.salary, c= 'r')

reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score', 'sportScore\t']], df['salary'])
print('coefficient:', reg.coef_ ,'intercept:', reg.intercept_)
#plt.plot(df.experience, reg.predict(df[['experience']]), '-b')
print(reg.predict([[2,9,6]]))

plt.show()
