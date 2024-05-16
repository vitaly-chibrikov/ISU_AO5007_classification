import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn import metrics

from methods import csv_reader

class_name="PHONE_CONFIRMED"

################ Read the data #############
X_train=csv_reader("out_X_train_Numbers_2024-05-15_v1.csv", delimeter=';')
X_test =csv_reader("out_X_test_Numbers_2024-05-15_v1.csv", delimeter=';')
Y_train=X_train[class_name]
Y_test =X_test[class_name]
X_train=X_train.drop(labels=class_name, axis=1)
X_test =X_test.drop(labels=class_name, axis=1)

main_components=len(X_train.columns)

################ Prepare the model #########
model=Sequential()
model.add(Dense(64, input_dim=main_components, activation="softmax"))
model.add(Dense(16, activation="softmax"))
model.add(Dense(16, activation="softmax"))
model.add(Dense(1, activation="sigmoid"))

model.summary()

################ Learn from the data ###########

tf.keras.utils.plot_model(model,"exp.png",
                          show_shapes=True,
                          show_layer_names=True)

#Creating loss function:
#BinaryCrossentropy computes the cross-entropy loss 
#between true labels and predicted labels.
loss=tf.keras.losses.BinaryCrossentropy()

#Creating optimizer: 
#Adam (Adaptive Moment Estimation) optimization 
#is a stochastic gradient descent method.
opt = Adam(learning_rate=0.0001)

model.compile(loss=loss,
              optimizer=opt,
              metrics=[])

history= model.fit(
    X_train,
    Y_train,
    epochs=100,
    verbose=2,
    validation_data=(X_test,Y_test)
    )

y_pred = model.predict(X_test).round()

################ Print the results #########
print(y_pred)
print("Confusion matrix:")
print(metrics.confusion_matrix(Y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(Y_test, y_pred))
print("Precision:", metrics.precision_score(Y_test, y_pred))
print("Recall:", metrics.recall_score(Y_test, y_pred))
print("F1:", metrics.f1_score(Y_test, y_pred))
