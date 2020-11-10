from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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

# using linear kernel
linear_model = SVC(kernel='linear')
linear_model.fit(X_train, y_train)
print('kernal rbf accuracy : ', linear_model.score(X_test, y_test))
