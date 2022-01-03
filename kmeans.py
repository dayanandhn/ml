#Import The libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

#Generate our dataset
dataset=make_blobs(n_samples=200,
                   centers=4,
                   n_features=2,
                   cluster_std=1.6,
                    random_state=45)

points=dataset[0]
print(points[:1])

#kmeans and fitting the data 
kmeans=KMeans(n_clusters=4)
kmeans.fit(points)

plt.scatter(dataset[0][:,0],dataset[0][:,1])
plt.scatter(points[:,0],points[:,1])

clusters=kmeans.cluster_centers_
print(clusters)

y_ks=kmeans.fit_predict(points)

plt.scatter(points[y_ks == 0,0],points[y_ks == 0,1],s=50,color='red')
plt.scatter(points[y_ks == 1,0],points[y_ks == 1,1],s=50,color='green')
plt.scatter(points[y_ks == 2,0],points[y_ks == 2,1],s=50,color='yellow')
plt.scatter(points[y_ks == 3,0],points[y_ks == 3,1],s=50,color='cyan')

plt.scatter(clusters[0][0],clusters[0][1],marker='*',s=500,color="black")
plt.scatter(clusters[1][0],clusters[1][1],marker='*',s=500,color="black")
plt.scatter(clusters[2][0],clusters[2][1],marker='*',s=500,color="black")
plt.scatter(clusters[3][0],clusters[3][1],marker='*',s=500,color="black")
plt.show()