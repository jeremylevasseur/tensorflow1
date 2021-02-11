import pandas as pd
from sklearn import datasets
import tensorflow as tf
import itertools

def get_input_fn(data_set, num_epochs=None, n_batch = 128, shuffle=True):    
         return tf.estimator.inputs.pandas_input_fn(       
         x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),       
         y = pd.Series(data_set[LABEL].values),       
         batch_size=n_batch,          
         num_epochs=num_epochs,       
         shuffle=shuffle)


COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]

training_set = pd.read_csv("./boston_train/boston_train.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
test_set = pd.read_csv("./boston_train/boston_test.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("./boston_train/boston_predict.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)

FEATURES = ["crim", "zn", "indus", "nox", "rm",				
                 "age", "dis", "tax", "ptratio"]

LABEL = "medv"

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

estimator = tf.estimator.LinearRegressor(    
        feature_columns=feature_cols,   
        model_dir="train")

estimator.train(input_fn=get_input_fn(training_set,                                       
                                           num_epochs=None,                                      
                                           n_batch = 128,                                      
                                           shuffle=False),                                      
                                           steps=1000)
