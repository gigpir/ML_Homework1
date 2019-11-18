#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created By Luigi Pirisi

"""

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib.colors import Normalize


# FUNCTIONS
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


# Plot a line based on the x and y axis value list.
def draw_line(plotName):
    # List to hold x values.
    x_number_values = ACCs[:, 0]

    # List to hold y values.
    y_number_values = ACCs[:, 1]

    #   Plot also the points
    plt.scatter(x_number_values, y_number_values, c="red", s=30)

    # Plot the number in the list and set the line thickness.
    plt.plot(x_number_values, y_number_values, linewidth=3)

    # Set the line chart title and the text font size.
    plt.title(plotName+" accuracy on validation set", fontsize=16)

    # Set x axes label.
    plt.xlabel("C value", fontsize=10)

    # Set y axes label.
    plt.ylabel("Accuracy", fontsize=10)
    # set axis dimensions
    plt.xlim(0.0001, 2000)
    plt.ylim(min(y_number_values)-.1, 1)
    plt.xscale("log")
    # Set the x, y axis tick marks text size.
    plt.tick_params(axis='both', labelsize=9)
    #SaveThePlot
    plt.savefig(plotName+"_Accuracy_val.PNG")
    # Display the plot in the matplotlib's viewer.
    plt.show()


def print_section(text):
    text = text.upper()
    size = len(text)
    print("\n", end='')
    for ind in range(size * 3):
        print("-", end='')
    print("\n", end='')
    for ind in range(size):
        print("-", end='')
    print(text, end='')
    for ind in range(size):
        print("-", end='')
    print("\n", end='')
    for ind in range(size * 3):
        print("-", end='')
    print("\n")


# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


X, y = load_wine(return_X_y=True)

X_2Dsel = X[:, :2]  # make a selection of the first 2 coloumns



# build train validation and test in proportion 5:2:3
X_train, X_test, y_train, y_test = train_test_split(X_2Dsel, y, test_size=0.3, random_state=19)
# Preprocessing: standardize the set from the train+val set
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=2 / 7, random_state=19)

####################################################################
####################################################################
####################################################################
####################################################################
####################################################################

print_section("SVM with linear kernel")
for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    # clf = LinearSVC(C=c,max_iter=10000000)
    clf = svm.SVC(kernel='linear', C=c)
    clf.fit(X_train, y_train)
    # print(clf.coef_)
    # print("\n")

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plt.figure()

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X0, X1, c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("SVM with linear kernel (c=%.3f)" % (c), fontsize=16)

    # Evaluate the method on the validation set
    pred_test = clf.predict(X_val)

    # Prepare the "subtitle"
    # subtitle="Prediction accuracy for the validation dataset with k="+str(n_neighbors)
    acc = metrics.accuracy_score(y_val, pred_test)


    plt.xlabel("X1", fontsize=12)
    plt.ylabel("X2", fontsize=12)
    if (c == 0.001):
        ACCs = [c, acc]
    else:
        ACCs = np.vstack([ACCs, [c, acc]])
    plt.savefig("SVM_linear_C"+str(c)+".png")
    plt.show()

if __name__ == '__main__':
    draw_line("SVM-linear")
bestC = ACCs[np.argmax(ACCs[:, 1])][0]  # C that produce max accuracy
print("The best value for C is " + '{:.3}'.format(bestC)+" with an accuracy on the validation set equal to "+'{:.2%}'.format(np.max(ACCs[:,1])))
#retrain the method with best C on train+val set
clf = svm.SVC(kernel='linear', C=bestC)
clf.fit(np.append(X_train,X_val, axis=0) , np.append(y_train,y_val, axis=0))
#predict test set
pred_test = clf.predict(X_test)
acc = metrics.accuracy_score(y_test, pred_test)
print("The accuracy for bestC=" + '{:.3}'.format(bestC) + " on the test set is " + '{:.2%}'.format(acc) + "\n")


# Plot the decision boundaries. I'm assuming that i've to plot the test point with the region coming from training
X0, X1 = X_test[:, 0], X_test[:, 1]
xx, yy = make_meshgrid(X0, X1)

plt.figure()

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# Plot also the TEST points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(X_test[:, 0].min() - 1, X_test[:, 0].max() + 1)
plt.ylim(X_test[:, 1].min() - 1, X_test[:, 1].max() + 1)
plt.title("SVM with Linear kernel (BestC=%.3f) \nTest points plotted" % (bestC),fontsize=16)
plt.xlabel("X1", fontsize=12)
plt.ylabel("X2", fontsize=12)
plt.savefig("SVM_linear_C_Best.PNG")
plt.show()


####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
print_section("SVM with RBF kernel")

for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    # clf = LinearSVC(C=c,max_iter=10000000)
    clf = svm.SVC(kernel='rbf', gamma='auto', C=c)
    clf.fit(X_train, y_train)
    # print(clf.coef_)
    # print("\n")

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plt.figure()

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points
    plt.scatter(X0, X1, c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("SVM with RBF kernel (gamma=auto) (c=%.3f)" % (c), fontsize=16)
    plt.xlabel("X1", fontsize=12)
    plt.ylabel("X2", fontsize=12)
    # Evaluate the method on the validation set
    pred_test = clf.predict(X_val)

    # Prepare the "subtitle"
    # subtitle="Prediction accuracy for the validation dataset with k="+str(n_neighbors)
    acc = metrics.accuracy_score(y_val, pred_test)
    if (c == 0.001):
        ACCs = [c, acc]
    else:
        ACCs = np.vstack([ACCs, [c, acc]])
    plt.savefig("SVM_rbf_C"+str(c)+".PNG")
    plt.show()

if __name__ == '__main__':
    draw_line("SVM-rbf")

bestC = ACCs[np.argmax(ACCs[:, 1])][0]  # C that produce max accuracy
print("The best value for C is " + '{:.3}'.format(bestC)+" with an accuracy on the validation set equal to "+'{:.2%}'.format(np.max(ACCs[:,1])))
#retrain the method with best C on train+val set
clf = svm.SVC(kernel='rbf', C=bestC)
clf.fit(np.append(X_train,X_val, axis=0) , np.append(y_train,y_val, axis=0))
#predict test set
pred_test = clf.predict(X_test)
acc = metrics.accuracy_score(y_test, pred_test)
print("The accuracy for C=" + '{:.3}'.format(bestC) + " on the test set is " + '{:.2%}'.format(acc) + "\n")

# Plot the decision boundaries. I'm assuming that i've to plot the test point with the region coming from training
X0, X1 = X_test[:, 0], X_test[:, 1]
xx, yy = make_meshgrid(X0, X1)

plt.figure()

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# Plot also the TEST points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(X_test[:, 0].min() - 1, X_test[:, 0].max() + 1)
plt.ylim(X_test[:, 1].min() - 1, X_test[:, 1].max() + 1)
plt.title("SVM with RBF kernel (BestC=%.3f) (gamma=auto)\nTest points plotted" % (bestC),fontsize=16)
plt.xlabel("X1", fontsize=12)
plt.ylabel("X2", fontsize=12)
plt.savefig("SVM_RBF_C_Best_Gamma_auto.PNG")
plt.show()


# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_range = np.logspace(-2, 6, 9)
gamma_range = np.logspace(-4, 4, 9)
classifiers = []
ACCs = []
bestParams = [0.0, 0.0]
bestAcc = 0.0
i=0
ACCs=np.zeros((len(C_range),len(gamma_range)))
for i,C in enumerate(C_range):
    for j,gamma in enumerate(gamma_range):
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_train, y_train)
        classifiers.append((C, gamma, clf))
        pred_test = clf.predict(X_val)
        acc = metrics.accuracy_score(y_val, pred_test)
        ACCs[i][j]=acc
        if (acc > bestAcc):
            bestAcc = acc
            bestParams = [C, gamma]

# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.
## We extract just the scores
#scores = ACCs.reshape(len(C_range),len(gamma_range))
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(ACCs, interpolation='nearest', cmap=plt.cm.hot,norm=MidpointNormalize(vmin=0.2, midpoint=0.70))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.savefig("RBF_GridSearch_HeatMap.png")
plt.show()


# Evaluate the best parameters on the test set.
clf = SVC(C=bestParams[0], gamma=bestParams[1])
clf.fit(np.append(X_train,X_val, axis=0) , np.append(y_train,y_val, axis=0))
pred_test = clf.predict(X_test)
acc = metrics.accuracy_score(y_test, pred_test)
print("GridSearch found best accuracy for C=" + '{:.3}'.format(bestParams[0]) + " and gamma=" + '{:.2}'.format(bestParams[1]))
output = "The parameters above produce on the test set an accuracy equal to " + '{:.2f}'.format(acc * 100)+"%"
print(output)

# Plot the decision boundaries. I'm assuming that i've to plot the test point with the region coming from training
X0, X1 = X_test[:, 0], X_test[:, 1]
xx, yy = make_meshgrid(X0, X1)

plt.figure()

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# Plot also the TEST points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(X_test[:, 0].min() - 1, X_test[:, 0].max() + 1)
plt.ylim(X_test[:, 1].min() - 1, X_test[:, 1].max() + 1)
plt.title("SVM with RBF kernel (c=%.3f) (gamma=%.3f) \nBest parameters - Test points plotted" % (bestParams[0], bestParams[1]),
          fontsize=16)
plt.xlabel("X1", fontsize=12)
plt.ylabel("X2", fontsize=12)
plt.savefig("SVM_rbf_C"+str(bestParams[0])+"_gamma"+str(bestParams[1])+"best.PNG")
plt.show()
####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
# [16] K FOLD
print_section("K-fold")
# Merge the training and validation split. You should now have 70% training and 30% test data.


X_trainVal, y_trainVal = np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val))
k = 5
k_Fold = KFold(n_splits=k)
ACCs = (np.size(C_range), np.size(gamma_range))
ACCs = np.zeros(ACCs)
for indTrain, indVal in k_Fold.split(X_trainVal, y_trainVal):
    for i, C in enumerate(C_range):
        for j, gamma in enumerate(gamma_range):
            clf = SVC(C=C, gamma=gamma)
            clf.fit(X_trainVal[indTrain], y_trainVal[indTrain])
            pred_test = clf.predict(X_trainVal[indVal])
            acc = metrics.accuracy_score(y_trainVal[indVal], pred_test)
            ACCs[i, j] += acc

# obtain mean value of accuracy matrix
ACCs /= k


# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.
## We extract just the scores
#scores = ACCs.reshape(len(C_range),len(gamma_range))
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(ACCs, interpolation='nearest', cmap=plt.cm.hot,norm=MidpointNormalize(vmin=0.2, midpoint=0.70))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.savefig("RBF-Kfold_GridSearch_HeatMap.png")
plt.show()


# obtain index of max accuracy value
max_i, max_j = int(np.argmax(ACCs) / np.size(gamma_range)), int(np.argmax(ACCs) % np.size(C_range))
# obtain C and Gamma for max accuracy value
best_C, best_gamma = C_range[max_i], gamma_range[max_j]
output = "The best accuracy performed on the validation set with K Fold = " + '{:d}'.format(
    k) + " is " + '{:.2f}'.format(np.max(ACCs) * 100) + "% \nFound with C=" + '{:.3f}'.format(
    best_C) + " and gamma="'{:.3f}'.format(best_gamma)
print(output)
# Retrain the model with best parameters on the whole train and val set and evaluate it on the test set
clf = SVC(C=best_C, gamma=best_gamma)
clf.fit(X_trainVal, y_trainVal)
pred_test = clf.predict(X_test)
acc = metrics.accuracy_score(y_test, pred_test)
output = "The parameters above produce on the test set an accuracy equal to " + '{:.2f}'.format(np.max(ACCs) * 100)+"%"
print(output)
X0, X1 = X_test[:, 0], X_test[:, 1]
xx, yy = make_meshgrid(X0, X1)

plt.figure()

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# Plot also the TEST points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(X_test[:, 0].min() - 1, X_test[:, 0].max() + 1)
plt.ylim(X_test[:, 1].min() - 1, X_test[:, 1].max() + 1)
plt.title("SVM with 5-fold RBF kernel (c=%.3f) (gamma=%.3f) \nBest parameters - Test points plotted" % (best_C, best_gamma),
          fontsize=16)
plt.xlabel("X1", fontsize=12)
plt.ylabel("X2", fontsize=12)
plt.savefig("SVM_k-fold_rbf_C"+str(bestParams[0])+"_gamma"+str(bestParams[1])+"best.PNG")
plt.show()
## #############################################################################
## Visualization
##
## draw visualization of parameter effects
#
# plt.figure(figsize=(8, 6))
##xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
# X0, X1 = X_train[:, 0], X_train[:, 1]
# xx, yy = make_meshgrid(X0, X1)
#    
# for (k, (C, gamma, clf)) in enumerate(classifiers):
#    # evaluate decision function in a grid
#    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#    Z = Z.reshape(xx.shape)
#    # visualize decision function for these parameters
#    plt.subplot(len(C_range), len(gamma_range), k + 1)
#    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
#              size='medium')
#
#    # visualize parameter's effect on decision function
#    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
#    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,edgecolors='k')
#    plt.xticks(())
#    plt.yticks(())
#    plt.axis('tight')

# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.


# C_range = np.logspace(-2, 10, 13)
# gamma_range = np.logspace(-9, 3, 13)
#
# param_grid = dict(gamma=gamma_range, C=C_range)
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
#
# grid.fit(X_train, y_train)
#
# print("The best parameters are %s with a score of %0.2f\n"% (grid.best_params_, grid.best_score_))
#
## plot the scores of the grid
#
## We extract just the scores
# scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),len(gamma_range))
#
## Draw heatmap of the validation accuracy as a function of gamma and C
##
## The score are encoded as colors with the hot colormap which varies from dark
## red to bright yellow. As the most interesting scores are all located in the
## 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
## as to make it easier to visualize the small variations of score values in the
## interesting range while not brutally collapsing all the low score values to
## the same color.
#
# plt.figure(figsize=(8, 6))
# plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
# plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
# plt.xlabel('gamma')
# plt.ylabel('C')
# plt.colorbar()
# plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
# plt.yticks(np.arange(len(C_range)), C_range)
# plt.title('Validation accuracy')
# plt.show()
#
# pred_test = grid.best_estimator_.predict(X_test)
# acc=metrics.accuracy_score(y_test, pred_test)
# print("The accuracy for C=" + '{:.3}'.format(bestC) +" on the test set is "+ '{:.2%}'.format(acc)+"\n")

################################


## Take the first two features. We could avoid this by using a two-dim dataset
# X, y = load_wine(return_X_y=True)
# X=X[:,:2]
#
## we create an instance of SVM and fit out data. We do not scale our
## data since we want to plot the support vectors
# C = 1.0  # SVM regularization parameter
# models = (svm.SVC(kernel='linear', C=C),
#          svm.LinearSVC(C=C, max_iter=10000),
#          svm.SVC(kernel='rbf', gamma=0.7, C=C),
#          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
# models = (clf.fit(X, y) for clf in models)
#
## title for the plots
# titles = ('SVC with linear kernel',
#          'LinearSVC (linear kernel)',
#          'SVC with RBF kernel',
#          'SVC with polynomial (degree 3) kernel')
#
## Set-up 2x2 grid for plotting.
##fig, sub = plt.subplots(2, 2)
##plt.subplots_adjust(wspace=0.4, hspace=0.4)
#
# X0, X1 = X[:, 0], X[:, 1]
# xx, yy = make_meshgrid(X0, X1)
#
# for clf, title in zip(models, titles):
#    plt.figure()
#    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#    Z = Z.reshape(xx.shape)
#    #plt.contourf(xx, yy, Z)
#    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
#    plt.scatter(X0, X1, c=y, cmap=cmap_bold, s=20, edgecolors='k')
#    plt.xlim(xx.min(), xx.max())
#    plt.ylim(yy.min(), yy.max())
#    plt.xlabel('Sepal length')
#    plt.ylabel('Sepal width')
#    plt.xticks(())
#    plt.yticks(())
#    plt.title(title)
#    plt.show()
#
#
#
#
