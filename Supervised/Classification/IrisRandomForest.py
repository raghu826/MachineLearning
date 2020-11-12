from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

iris = load_iris()
print(dir(iris))

df = pd.DataFrame(iris.data, columns= iris.feature_names)
df['target'] = iris.target
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), iris.target, test_size=0.3)
model = RandomForestClassifier(n_estimators=20) # n_estimators indicates number of decision trees
model.fit(X_train, y_train)
print('model accuracy', model.score(X_test, y_test))
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)

plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True)

plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()




