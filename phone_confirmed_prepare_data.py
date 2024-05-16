import pandas as pd
from sklearn.model_selection import train_test_split

from methods import csv_reader
from methods import column_filter
from methods import missing_values
from methods import replace_rare_values
from methods import row_filter
from methods import csv_writer
from methods import one_to_many
from methods import undersample
from methods import remove_outliers

############## Field for prediction #############

class_name="PHONE_CONFIRMED"

################## Read data ####################

X = csv_reader("users-2024-05-10_2022.csv", delimeter=';')

################ Remove unusable #33############

X=column_filter(X, ["ID","UTM_CAMPAING","UTM_TERM","REAL_EMAIL_SET"])

print(X)
################ Impute missing ################
X=missing_values(X)

#### Remove raws without business value ########
X=row_filter(X, "TYPE", "teacher", "exclude")

############# Lower cardinality ################
replace_rare_values(X["UTM_SOURCE"], 1000)
replace_rare_values(X["UTM_MEDIUM"], 1000)

print(X.shape)

############# Categiries to numbers ############
X=one_to_many(X)
print(X)

################ Split the data ################

X_train,X_test=train_test_split(X,test_size=0.30, random_state=1)
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)

############### Remove outliers ###############
X_train=remove_outliers(X_train, class_name)

print("X shape after otliers removal: ", X_train.shape)

########### Undersample the train set #########

X_train=undersample(X_train, class_name)

Y_train=X_train[class_name]
print("Y_train balance after undersample: ")
print(pd.Series(list(Y_train)).value_counts())

############### Save datasets #################

csv_writer(X_train, "out_X_train_Numbers_2024-05-15_v1.csv")
csv_writer(X_test, "out_X_test_Numbers_2024-05-15_v1.csv")