# Red-Wine-Quality-Data-set
# Data Exploratory Analysis, Principle Component Analysis &amp; Prediction and Model selection :  
# 1. Logistic Regression 
# 2. Decision Trees 
# 3. Naive Bayes 
# 4. Random Forests 
# 5. SVM


### RED WINE QUALITY DATASET ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

data = pd.read_csv('C:\\Users\\Siddharth\\Downloads\\winequality-red.csv')
print(data)

#print(data.head())
print("\n")

print(data.corr)

print(data.columns)

print(data.info())

print(data['quality'].unique())

print(data['quality'])

#Check correleation between the variables using Seaborn's pairplot.
sns.pairplot(data)
plt.show()

#count of each target variable
from collections import Counter
print(Counter(data['quality']))

#count of the target variable
sns.countplot(x='quality', data=data)
plt.show()

#Plot a boxplot to check for Outliers
#Target variable is Quality. So will plot a boxplot each column against target variable
sns.boxplot('quality', 'fixed acidity', data = data)
plt.show()

#######

sns.boxplot('quality', 'volatile acidity', data = data)
plt.show()

sns.boxplot('quality', 'citric acid', data = data)
plt.show()

sns.boxplot('quality', 'residual sugar', data = data)
plt.show()

sns.boxplot('quality', 'chlorides', data = data)
plt.show()

sns.boxplot('quality', 'free sulfur dioxide', data = data)
plt.show()

sns.boxplot('quality', 'total sulfur dioxide', data = data)
plt.show()

sns.boxplot('quality', 'density', data = data)
plt.show()

sns.boxplot('quality', 'pH', data = data)
plt.show()

sns.boxplot('quality', 'sulphates', data = data)
plt.show()

sns.boxplot('quality', 'alcohol', data = data)
plt.show()

####
#boxplots show many outliers for quite a few columns. Describe the dataset to get a better idea on what's happening
#print(data.describe())
#fixed acidity - 25% - 7.1 and 50% - 7.9. Not much of a variance. Could explain the huge number of outliers
#volatile acididty - similar reasoning
#citric acid - seems to be somewhat uniformly distributed
#residual sugar - min - 0.9, max - 15!! Waaaaay too much difference. Could explain the outliers.
#chlorides - same as residual sugar. Min - 0.012, max - 0.611
#free sulfur dioxide, total suflur dioxide - same explanation as above

print(data.quality.unique())

print(Counter(data['quality']))

#next we shall create a new column called Review. This column will contain the values of 1,2, and 3.
#1 - Bad
#2 - Average
#3 - Excellent
#This will be split in the following way.
#1,2,3 --> Bad
#4,5,6,7 --> Average
#8,9,10 --> Excellent
#Create an empty list called Reviews
reviews = []
for i in data['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
data['Reviews'] = reviews

#view final data
print(data.columns)

print(data['Reviews'].unique())

print(Counter(data['Reviews']))

x = data.iloc[:,:11]
y = data['Reviews']

#print(data.shape)

sc = StandardScaler()
x = sc.fit_transform(x)

#view the scaled features
print(x)

pca = PCA()
x_pca = pca.fit_transform(x)

#plot the graph to find the principal components
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()
plt.show()

#AS per the graph, we can see that 8 principal components attribute for 90% of variation in the data.
#we shall pick the first 8 components for our prediction.
pca_new = PCA(n_components=8)
x_new = pca_new.fit_transform(x)

print(x_new)


x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.25)


#pca.explained_variance_ratio_

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


## LOGISTIC REGRESSION

lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_predict = lr.predict(x_test)

#print confusion matrix and accuracy score
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_acc_score = accuracy_score(y_test, lr_predict)
print(lr_conf_matrix)
print(lr_acc_score*100)

############## DECISION TREE

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_predict = dt.predict(x_test)

#print confusion matrix and accuracy score
dt_conf_matrix = confusion_matrix(y_test, dt_predict)
dt_acc_score = accuracy_score(y_test, dt_predict)
print(dt_conf_matrix)
print(dt_acc_score*100)

################### NAIVE BAYES ######################

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
nb_predict=nb.predict(x_test)

#print confusion matrix and accuracy score
nb_conf_matrix = confusion_matrix(y_test, nb_predict)
nb_acc_score = accuracy_score(y_test, nb_predict)
print(nb_conf_matrix)
print(nb_acc_score*100)

#####################RANDOM FOREST#################

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_predict=rf.predict(x_test)

#print confusion matrix and accuracy score
rf_conf_matrix = confusion_matrix(y_test, rf_predict)
rf_acc_score = accuracy_score(y_test, rf_predict)
print(rf_conf_matrix)
print(rf_acc_score*100)

###########SVM#######################

#we shall use the rbf kernel first and check the accuracy
lin_svc = SVC()
lin_svc.fit(x_train, y_train)
lin_svc=rf.predict(x_test)

#print confusion matrix and accuracy score
lin_svc_conf_matrix = confusion_matrix(y_test, rf_predict)
lin_svc_acc_score = accuracy_score(y_test, rf_predict)
print(lin_svc_conf_matrix)
print(lin_svc_acc_score*100)

# 98.5% accuracy wit RBF Kernel! Same as Random Forest! Let's try the linear kernel now and see if it improves our accuracy in any way.

######## LINEAR KERNEL ###################

rbf_svc = SVC(kernel='linear')
rbf_svc.fit(x_train, y_train)
rbf_svc=rf.predict(x_test)

rbf_svc_conf_matrix = confusion_matrix(y_test, rf_predict)
rbf_svc_acc_score = accuracy_score(y_test, rf_predict)
print(rbf_svc_conf_matrix)
print(rbf_svc_acc_score*100)

# The same accuracy! So we can see that the SVC and the Random Forest give us good prediction accuracy for the Wine Classification problem.
# We can further improve accuracy by fine-tuning the parameters of each classifier.
# Hope you found this github useful! Pleae leave in comments in case of any questions, concerns, and feedback! Thank you :)





