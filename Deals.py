#menggunakan dataset Deals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


dataset = pd.read_csv('Deals.csv', delimiter=';')
print('Sample Data:')
print(dataset.head())

#untuk meubah nilai padaa kolom "Gender" menjadi representasi numerik
le = LabelEncoder()
dataset['Gender'] = le.fit_transform(dataset['Gender'])
dataset['Payment Method'] = le.fit_transform(dataset['Payment Method'])

#labelk pada kolom "Future Customer"
dataset['Future Customer'] = le.fit_transform(dataset['Future Customer'])

x = dataset.iloc[:, :4]

kMeans = KMeans(n_clusters=3)
labels = kMeans.fit_predict(x)

cols = dataset.columns

plt.scatter(x.loc[labels == 0, cols[0]],
            x.loc[labels == 0, cols[1]],
            s=100, c='purple',
            label='Cluster 1')
plt.scatter(x.loc[labels == 1, cols[0]],
            x.loc[labels == 1, cols[1]],
            s=100, c='orange',
            label='Cluster 2')
plt.scatter(x.loc[labels == 2, cols[0]],
            x.loc[labels == 2, cols[1]],
            s=100, c='green',
            label='Cluster 3')

plt.scatter(kMeans.cluster_centers_[:, 0],
            kMeans.cluster_centers_[:, 1],
            s=100, c='red',
            label='Centroids')

plt.legend()
plt.xlabel(cols[0])
plt.ylabel(cols[1])
plt.title('KMeans Clustering')
plt.show()
