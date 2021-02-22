# import dependencies
from preprocess import turnover_prep, wine_prep
from models import knn, k_graph
import numpy as np
import pandas as pd 

'''
# read turnover data
X, y = turnover_prep('turnover.csv')
'''

# read wine data
X, y = wine_prep ('winequality_white.csv')


# Train KNN

result, model_knn, X_train, y_train = knn(X, y, 10)

performance_df = pd.DataFrame.from_dict(result, orient='index')

print (performance_df)


'''
# Explore different accuracy with different k value
k_graph (X, y, 21)
'''