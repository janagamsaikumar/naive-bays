import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
# read the dataset into the spyder 
dataset=pd.read_csv(r'C:\Users\saikumar\Desktop\AMXWAM data science\class24_nov 14,2020\Social_Network_Ads.csv')

# seperate dependent and independent variables
X=dataset.iloc[:,2:-1].values
y=dataset.iloc[:,-1].values

# check the dataset for any null values
dataset.notnull().sum()
# do a feature scaling 
sc=StandardScaler()
X=sc.fit_transform(X)

# split my dataset into train and test 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# fit gaussianNB to my training model
Gnb=GaussianNB()
Gnb.fit(X_train,y_train)
# predict
y_pred=Gnb.predict(X_test)

# check the measures using confusion matrix
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
cr=classification_report(y_test,y_pred)

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, Gnb.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (age vs salary)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

X_pred=Gnb.predict(X_train)
X_cm=confusion_matrix(y_train,X_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, Gnb.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes ( age vs salary)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()










