import pandas as pd
from sklearn import tree
from sklearn import model_selection

df = pd.read_csv(r'C:\Users\telelink\Contacts\Desktop\DataSets\Titanic.csv')
df = df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns')
print(df.columns)

inputs = df.drop(['Survived'], axis='columns')
inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2}) # mapping male n female to values 1 n 2
inputs.Age = inputs.fillna(inputs.Age.mean()) # cleaning the data for empty spots
target = df[['Survived']]

print(inputs.head(6))
print(target.head())

X_train, X_test, y_train, y_test = model_selection.train_test_split(inputs, target, test_size=0.3)

decision = tree.DecisionTreeClassifier()
decision.fit(X_train, y_train)
print('Accuracy : ', decision.score(X_test, y_test))

print(decision.predict([[1, 2, 38.0, 71.2833]]))



