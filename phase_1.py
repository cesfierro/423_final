import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('tumor_data_d.csv')

# Data Exploration
print(df.info())
print(df.isna().sum().sort_values())

"""Preprocessing Steps:"""
# Handle missing values:
# Create a new DF and fill the N/A values with the mean,
# df.mean() needs the numeric-only parameter because the 'Label' column is not numeric
mean_df = df.fillna(df.mean(numeric_only=True))

# This print is to ensure the N/A values were filled
print(mean_df.isna().sum().sort_values())

# Handle duplication:
mean_df.drop_duplicates(inplace=True)
print("Original DF Entries: ", df.shape[0], "\nNew DF Entries: ", mean_df.shape[0])

# To scale the data, first I will seperate the Label column from the rest of the numeric data to scale it
X = mean_df.drop('Label', axis=1).values
y = mean_df['Label'].values

# Split the data according to Part A, to be able to use the scalar object
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20,stratify=y)

# Instatiate a scalar object, then fit and transform the training data. and transform the test data
scale = StandardScaler()
X_train_scl = scale.fit_transform(X_train)
X_test_scl = scale.transform(X_test)

"""Part A (classification):"""
# Steps 1 & 2 were done during preprocessing
# Before I begin step 3, I will test the number of neighbors to use in my knn model
# Create dictionaries to store training and testing accuracies, create an np array from 1-20 to test those values as neighbors
train_acc = {}
test_acc = {}
neighbors = np.arange(1,21)
# for every neighbor value from 1-20
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor) # create a knn model with that amount of neighbors
    knn.fit(X_train_scl, y_train) # fit the scaled x data
    #store the training and testing accuracies
    train_acc[neighbor] = knn.score(X_train_scl, y_train) 
    test_acc[neighbor] = knn.score(X_test_scl, y_test)
# Create a graph to plot accuracies and visualize which amount of neighbors is best
#plt.figure(figsize=(8,6))
#plt.title("KNN: Varying Number of Neighbors")
# Plot both lines, the neighbors array (1-20) is the X-axis, the value of accuracy is the y-axis
#plt.plot(neighbors, train_acc.values(), label="Training Accuracy")
#plt.plot(neighbors, test_acc.values(), label="Testing Accuracy")
plt.legend()
#plt.xlabel("Number of Neighbors")
#plt.ylabel("Accuracy")
#plt.show()
# split the data into 5 different sets to test, then create both our models and test them
kf = KFold(n_splits=5, shuffle=True, random_state=20)
knn = KNeighborsClassifier(n_neighbors=3)
logreg= LogisticRegression()
# Step 3 - perform cross validation on both of our models using the training data
knn_cv = cross_val_score(knn, X_train_scl, y_train, cv=kf)
logreg_cv = cross_val_score(logreg, X_train_scl, y_train, cv=kf)
print("KNN: CV mean and standard deviation scores: ", np.mean(knn_cv), np.std(knn_cv))
print("Logistic Regression: CV mean and standard deviation scores: ", np.mean(logreg_cv), np.std(logreg_cv))
# Step 4
knn.fit(X_train_scl, y_train) # Fit data into knn model
knn_y_pred = knn.predict(X_test_scl)
logreg.fit(X_train_scl, y_train)
logreg_y_pred = logreg.predict(X_test_scl)
knn_class_rep = classification_report(y_test, knn_y_pred)
logreg_class_rep = classification_report(y_test, logreg_y_pred)
knn_conf_mat = confusion_matrix(y_test, knn_y_pred)
logreg_conf_mat = confusion_matrix(y_test, logreg_y_pred)

print("KNN Confusion Matrix:\n", knn_conf_mat)
print("Logistic Regression Confusion Matrix:\n", logreg_conf_mat)
print("KNN Classification Report:\n", knn_class_rep)
print("Logistic Regression Classification Report:\n", logreg_class_rep)

# Step 5 get the test set accuracies using .score() on our models
print("KNN Test Set Accuracy: ", knn.score(X_test_scl, y_test))
print("Logistic Regression Test Set Accuracy: ", logreg.score(X_test_scl, y_test))
#plt.title("Comparing CV results of our Models")
#plt.boxplot([knn_cv, logreg_cv], labels=["KNN", "Logistic Regression"])
#plt.show()

"""Part B (regression)"""
# 1 Create feature and target arrays
Xb = mean_df.drop(["area_se", "Label"], axis=1).values
yb = mean_df['area_se'].values
# 2 split the data
Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.2, random_state=20)

# 3 Building and Evaluating Model
reg = LinearRegression()
reg.fit(Xb_train, yb_train)
yb_pred = reg.predict(Xb_test)
print("Regression R Squared: ", reg.score(Xb_test, yb_test))