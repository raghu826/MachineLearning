from sklearn.datasets import load_iris
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# loading iris dataset
iris = load_iris()
print(dir(iris))
df = pd.DataFrame(iris.data, columns= iris.feature_names)
df = df.drop(['sepal length (cm)', 'sepal width (cm)'], axis= 'columns')

# plotting the data to visualize
plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c= 'r')
plt.xlabel('petal length')
plt.ylabel('petal width')

km = KMeans(n_clusters=3)       # applying k Mean
y_predicted = km.fit_predict(df)
df['cluster'] = y_predicted
print(df.head())

print(km.cluster_centers_)  # to mark the centroids of cluster
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c='black', marker = '+')

df1 = df[df.cluster==0] # creating diff dataFrames for better visualisation
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue', label='setosa')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color='green', label='versicolor')
plt.scatter(df3['petal length (cm)'], df3['petal width (cm)'], color='yellow', label='virginica')
plt.legend()

k_range = range(1,10)  # elbow method to find the better k value
sse= []                # array to store
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df)
    sse.append(km.inertia_)

plt.plot(k_range, sse)
plt.xlabel('k')
plt.ylabel('sse')

plt.show()