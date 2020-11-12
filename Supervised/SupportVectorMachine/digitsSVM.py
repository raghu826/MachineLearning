from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import *

digits = load_digits()
print(dir(digits))
df = pd.DataFrame(digits.data, digits.target)
df['target'] = digits.target

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis='columns'), df.target, test_size=0.2)

# using rbf kernal
rbf_model = SVC(kernel='rbf')
rbf_model.fit(X_train, y_train)
print('kernal rbf accuracy : ', rbf_model.score(X_test, y_test))
y_predicted = rbf_model.predict(X_test)

# Implemented Confusion Matrix'
cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

# using linear kernel
linear_model = SVC(kernel='linear')
linear_model.fit(X_train, y_train)
print('kernal linear accuracy : ', linear_model.score(X_test, y_test))

plt.show()
