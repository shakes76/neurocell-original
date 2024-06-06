'''
Test the scikit-learn simple dataset creator for solving simple toy problems
'''
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

#parameters
samples = 128
state = 1 #set None for random state, or a fixed number for fixed dataset.

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plt.title("Multi-class, two informative features, one cluster", fontsize="small")
X1, Y1 = make_classification(
    n_samples=samples, n_features=3, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=3, random_state=state
)
ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], marker="o", c=Y1, s=25, edgecolor="k")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
