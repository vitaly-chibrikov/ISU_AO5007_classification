from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from methods import csv_reader

class_name="PHONE_CONFIRMED"

################ Read the data #############

X_train=csv_reader("out_X_train_Numbers_2024-05-15_v1.csv", delimeter=';')
X_test =csv_reader("out_X_test_Numbers_2024-05-15_v1.csv", delimeter=';')
Y_train=X_train[class_name]
Y_test =X_test[class_name]
X_train=X_train.drop(labels=class_name, axis=1)
X_test =X_test.drop(labels=class_name, axis=1)

################ Prepare the model #########

neigh = KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute")

################ Fit and predict ###########
neigh.fit(X_train, Y_train)
y_pred = neigh.predict(X_test)

################ Print the results #########
print(y_pred)
print("Confusion matrix:")
print(metrics.confusion_matrix(Y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
print("Precision:", metrics.precision_score(Y_test, y_pred))
print("Recall:", metrics.recall_score(Y_test, y_pred))
print("F1:", metrics.f1_score(Y_test, y_pred))

