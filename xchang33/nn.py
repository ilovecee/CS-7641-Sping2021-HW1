# import dependencies
from preprocess import turnover_prep, wine_prep
from models import nor_nn
import pandas as pd
import numpy as np


'''
# Read and preprocess the data
X, y = turnover_prep('turnover.csv')

# Drop the less important features and rebuild the tree
X_lean = X[['satisfaction', 'time_spend_company', 'evaluation', 'number_of_projects', 'average_montly_hours']]
y_lean = y
'''

# Read wine data
X, y = wine_prep ('winequality_white.csv')

# Train the model
# model_nn, X_train, X_test, y_train, y_test = nor_nn (X_lean, y_lean, nb_epoch = 10, batch_size = 300)
nor_nn (X, y, nb_epoch = 10, batch_size = 400)

'''
# Test accuracy
y_test_pred = model_nn.predict_classes (X_test, verbose = 0)
test_acc = np.sum (y_test == y_test_pred, axis = 0) / X_test.shape[0]
print (test_acc)
print (y_test_pred)
'''