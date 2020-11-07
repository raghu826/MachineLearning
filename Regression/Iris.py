import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import seaborn as sns

irisData = pd.read_csv(r'C:\Users\telelink\Contacts\Desktop\irisDataSet.csv')
df = irisData.drop(columns=['Unnamed: 0'])      # to delete columns
# To change Column names use rename
df = df.rename(columns={"Sepal.Length": "SepalLength","Petal.Length": "PetalLength",
                      "Sepal.Width": "SepalWidth","Petal.Width": "PetalWidth"})
print(df.columns)
print(df.head())
#print(df.describe()) # To get the stats of the DataSet

# To plot PetalLength n SepalLength
plt.scatter(df.PetalLength, df.SepalLength, color = 'red')
plt.xlabel("PetalLength")
plt.ylabel("SepalLength")
plt.title("normal Plot")

# To plot indicating species with different colors using seaborn
sns.set_style('whitegrid')
sns.FacetGrid(df, hue="Species", height=4).map(plt.scatter,"PetalLength", "SepalLength").add_legend()
plt.xlabel("PetalLength")
plt.ylabel("SepalLength")
plt.title("Different colors in same plot")

plt.show()





