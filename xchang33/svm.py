# import dependencies
from preprocess import turnover_prep, wine_prep
from models import svm, grid_search, svm_kernel_graph
import numpy as np
import pandas as pd 

'''
# read turnover data
X, y = turnover_prep('turnover.csv')
'''


# Prepare the wine data
X, y = wine_prep ('winequality_white.csv')



# Train SVM
result, model_knn, X_train, y_train = svm (X, y)

performance_df = pd.DataFrame.from_dict(result, orient='index')

print (performance_df)



# Accuracy with different kernels
l = ['linear', 'poly', 'rbf', 'sigmoid']
svm_kernel_graph(X,y,l)
