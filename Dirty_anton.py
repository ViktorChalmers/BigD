import numpy.random as rand
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def optimusPrime(X):
    prime = np.zeros(np.shape(X))

    for i in range(len(X[0])):
        prime[:,i] = (X[:,i]-np.mean(X[:,i]))/np.sqrt(np.std(X[:,i]))

    return prime

def dicksnballs(data_size = 1000):
    ball1 = [[rand.normal(loc=0, scale=2.0, size=None), rand.normal(loc=10, scale=5.0, size=None)] for _ in
              range(int(data_size/4))]
    ball2 = [[rand.normal(loc=10, scale=2, size=None), rand.normal(loc=10, scale=5.0, size=None)] for _ in
              range(int(data_size/4))]
    midgard = [[rand.normal(loc=5, scale=2, size=None), rand.normal(loc=10, scale=2.0, size=None)] for _ in
              range(int(data_size/4))]
    dick1 = [[rand.normal(loc=5, scale=2.0, size=None), rand.normal(loc=5, scale=5.0, size=None)] for _ in
              range(int(data_size/4))]
    dick2 = [[rand.normal(loc=15, scale=2.0, size=None), rand.normal(loc=5, scale=5.0, size=None)] for _ in
              range(int(data_size/4))]
    class1 = np.array(ball1 + ball2)
    class2 = np.array(dick1+dick2)
    return [class1, class2]

def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
    h = 0.02
    x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolors='k');

data_size=1000


'''
class1=[[rand.normal(loc=0, scale=2.0, size=None),rand.normal(loc=5, scale=2.0, size=None)] for _ in range(data_size)]
class2=[[rand.normal(loc=5, scale=2, size=None),rand.normal(loc=0, scale=1.0, size=None)] for _ in range(data_size)]
'''

[class1, class2] = dicksnballs()

'''
class1 = [[rand.beta(0.1, 3),rand.beta(5, 3)] for _ in range(data_size)]
class2 = [[rand.beta(5, 2),rand.beta(10, 10)] for _ in range(data_size)]
'''

#X = np.array(class1 + class2)
class1=np.array(class1)
class2=np.array(class2)
X = np.append(class1,class2,axis = 0)

Y = np.zeros(len(X))
for i in range(len(class1)):
    Y[i] = 1
#X=optimusPrime(X)
'''
Observation of each class is drawn from a normal distribution (same as LDA).
QDA assumes that each class has its own covariance matrix (different from LDA).
'''
print(Y)

#for i in range(batch_size):
clf = QDA().fit(X,Y)
clf_cart = tree.DecisionTreeClassifier().fit(X, Y)
pipeline_QDA = make_pipeline(StandardScaler().fit(X, Y), clf)
pipeline_cart = make_pipeline(StandardScaler().fit(X, Y), clf_cart)

predictions_QDA = pipeline_QDA.predict(X)
predictions_cart = pipeline_cart.predict(X)

scores_QDA = cross_val_score(pipeline_QDA, X, Y, cv=10, n_jobs=1)
scores_cart = cross_val_score(pipeline_cart, X, Y, cv=10, n_jobs=1)
print(f'QDA scores: {scores_QDA} \n'
      f'cart scores: {scores_cart}')
f1_QDA = f1_score(Y, predictions_QDA, average=None)
f1_cart = f1_score(Y, predictions_cart, average=None)

print('Cross Validation accuracy: QDA: %.3f +/- %.3f ' % (np.mean(scores_QDA), np.std(scores_QDA))+', Cart:  %.3f +/- %.3f' %(np.mean(scores_cart), np.std(scores_cart)))

print(f'F1 scores QDA: {f1_QDA}, Cart: {f1_cart}')

plt.plot(class1[:, 0], class1[:, 1], 'o', label="class1")
plt.plot(class2[:, 0], class2[:, 1], '*', label="class2")
plt.title('Cross Validation accuracy: QDA: %.3f +/- %.3f ' % (np.mean(scores_QDA), np.std(scores_QDA))+', Cart:  %.3f +/- %.3f' %(np.mean(scores_cart), np.std(scores_cart)))
plt.legend()
#plt.figure()
plot_decision_boundary(clf_cart, X, Y, cmap='Paired_r')
plt.title('CART accuracy:  %.3f +/- %.3f' %(np.mean(scores_cart), np.std(scores_cart)))
#plt.figure()
plot_decision_boundary(clf, X, Y, cmap='Paired_r')
plt.title('Accuracy: QDA: %.3f +/- %.3f ' % (np.mean(scores_QDA), np.std(scores_QDA)))
plt.show()