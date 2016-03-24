import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import datasets
import numpy as np
import pandas as pd
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from scipy.cluster.vq import kmeans,kmeans2,vq

iris = datasets.load_iris()
X = iris.data
y = iris.target

#original undecomposed plot
# plt.figure()
# plt.scatter(iris.data[0:150, 1], iris.data[0:150, 2], c=y, cmap=plt.cm.Paired)
# plt.show()
# plt.clf()

lda = LDA(n_components = 4)
x = lda.fit_transform(X, y)

nb = KNeighborsClassifier(n_neighbors = 3)
nb.fit(x, y) 

score = cross_val_score(nb, x, y, scoring = 'accuracy', cv = 10)
print np.mean(score)
#.9333

klist = range(1,11)
KM = [ kmeans(x, k) for k in klist ]
WCSS = [ v for (c,v) in KM ]

#making elbow plot
# plt.figure()
# plt.scatter(klist, WCSS)
# plt.xlabel('Number of clusters')
# plt.ylabel('Ave. WCSS')
# plt.title('Elbow Plot')
# plt.show()
# plt.clf()
#elbow at 3

#classifying the observations with KM2 and plotting
#looks more clean than the original
KM3 = kmeans2(x, 3, minit='points')
plt.figure()
plt.scatter(iris.data[0:150, 1], iris.data[0:150, 2], c=KM3[1], cmap=plt.cm.Paired)
plt.show()
plt.clf()

