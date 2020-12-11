import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv(r'C:\Users\telelink\Contacts\Desktop\DataSets\irisDataSet.csv')
df = df.drop(['SepalLength', 'SepalWidth', 'Species'], axis='columns')

# plt.scatter(df.PetalLength, df.PetalWidth)  just to see how the clusters would be
# plt.xlabel('PetalLength')
# plt.ylabel('PetalWidth')
# plt.title('ScatterPlot')

len = range(1, 11)     # Elbow Method to find out number of clusters
sse = []               # squared error
for k in len:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42)
    km.fit_predict(df[['PetalLength', 'PetalWidth']])
    #km.fit_predict(X)
    sse.append(km.inertia_)

# plt.plot(len, sse, label = 'clusters')   # plotting elbow curve to finalize no.of clusters
# plt.xlabel('No.of Clusters')
# plt.ylabel('squared error')
# plt.title('finding k')

km = KMeans(n_clusters=3, init='k-means++', random_state=42)  # training the kMeans Model
y_pred = km.fit_predict(df[['PetalLength', 'PetalWidth']])
print('y_pred:', y_pred)   # predicted values stored in y_pred
centres = km.cluster_centers_
print('Cluster-centres:', centres)

df['cluster'] = y_pred   # add y_pred(clusters) values to dataframe
print(df.head())

df1 = df[df.cluster == 0]  # created different dataframe for each cluster
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
plt.scatter(df1.PetalLength, df1['PetalWidth'], color='green', label='Cluster1')
plt.scatter(df2.PetalLength, df2['PetalWidth'], color='red', label='Cluster2')
plt.scatter(df3.PetalLength, df3['PetalWidth'], color='black', label='Cluster3')
plt.scatter(centres[:, 0], centres[:, 1], label='Centres', s = 300,  marker= '*')
plt.legend()
plt.show()
