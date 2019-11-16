# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
import numpy as np
import sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn import metrics

X, y = load_wine(return_X_y=True)
weight="uniform"


Xsel=X[:,:2] #make a selection of the first 2 coloumns

#Preprocessing: standardize the set
scaler=StandardScaler()
scaler.fit(Xsel)
Xsel = scaler.transform(Xsel)

#build train validation and test in proportion 5:2:3
X_train, X_test, y_train, y_test = train_test_split(Xsel, y, test_size=0.3, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=2/7, random_state=1)

for n_neighbors in [1,3,5,7]:
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
    clf.fit(X_train, y_train)

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    h = .02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("KNN classification (k = %i, weights = '%s')" % (n_neighbors, weight),fontsize=16)
    
    # Evaluate the method on the validation set
    pred_test = clf.predict(X_val)
    
    #Prepare the "subtitle"
    #subtitle="Prediction accuracy for the validation dataset with k="+str(n_neighbors)
    acc=metrics.accuracy_score(y_val, pred_test)
    #subtitle='\nAccuracy='+'{:.2%}'.format(acc)
    
    #plt.xlabel(subtitle, fontsize=12)

    plt.savefig('KNN_plot_k'+str(n_neighbors)+'.png')

    plt.show()

    if(n_neighbors==1):
        ACCs=[n_neighbors, acc]
    else:
        ACCs = np.vstack([ACCs,[n_neighbors, acc]])


# Plot a line based on the x and y axis value list.
def draw_line():

    # List to hold x values.
    x_number_values = ACCs[:, 0]

    # List to hold y values.
    y_number_values = ACCs[:, 1]
    
    #   Plot also the points
    plt.scatter(x_number_values, y_number_values,c="red", s=30  )
    
    # Plot the number in the list and set the line thickness.
    plt.plot(x_number_values, y_number_values, linewidth=3)
    
    # Set the line chart title and the text font size.
    plt.title("Accuracy on validation set", fontsize=16)

    # Set x axes label.
    plt.xlabel("k value", fontsize=10)

    # Set y axes label.
    plt.ylabel("Accuracy", fontsize=10)
    #set axis dimensions
    plt.xlim(0, 9)
    plt.ylim(0.4, 1)
    
    # Set the x, y axis tick marks text size.
    plt.tick_params(axis='both', labelsize=9)

    plt.savefig('KNN-accuracy_validation_set.png')
    # Display the plot in the matplotlib's viewer.
    plt.show()

if __name__ == '__main__':
    draw_line()

bestK = np.int(ACCs[np.argmax(ACCs[:,1])][0]) #K that produce max accuracy

print("The best K found is ",bestK)

clf = neighbors.KNeighborsClassifier(bestK, weights=weight) 
clf.fit(X_train, y_train)

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
h = .02  # step size in the mesh
x_min, x_max = Xsel[:, 0].min() - 1, Xsel[:, 0].max() + 1
y_min, y_max = Xsel[:, 1].min() - 1, Xsel[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(Xsel[:, 0], Xsel[:, 1], c=y, cmap=cmap_bold,edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i, weights = '%s')" % (bestK, weight),fontsize=16)
    
# Evaluate the method on the test set
pred_test = clf.predict(X_test)
    
#Prepare the "subtitle"
#subtitle="Prediction accuracy for the test dataset with k="+str(n_neighbors)
acc=metrics.accuracy_score(y_test, pred_test)
subtitle='\nAccuracy='+'{:.2%}'.format(acc)
    
plt.xlabel(subtitle, fontsize=12)
plt.savefig('KNN-plot_bestK.png')
plt.show()
