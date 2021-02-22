# import dependencies
from preprocess import turnover_prep, wine_prep
from models import decision_tree, ada_dt, ada_iteration_graph
from sklearn.tree import export_graphviz
import pandas as pd

'''
# Read the turnover dataa
X, y = turnover_prep('turnover.csv')
'''

# Read the wine data
X, y = wine_prep ('winequality_white.csv')


'''
# First need to build a weak tree
result, model_dt, X_train, y_train = decision_tree (X, y
                                                      ,max_depth = 1
                                                      ,min_samples_leaf = 10
                                                      )

# See the perfomance of the weak tree
result = pd.DataFrame.from_dict(result, orient='index')
print (result)


# Export the weak tree
export_graphviz (model_dt, out_file = 'weak_dt')
'''


# Build an adaboost classifier using the weak tree as the estimator
result, model_ada, X_train, y_train = ada_dt (X, y, iteration = 900, lr = 0.05)



# See the perfomance of the weak tree
result = pd.DataFrame.from_dict(result, orient='index')
print (result)


'''
l = [i for i in range (100, 1100, 100)]

ada_iteration_graph (X, y, l)

'''

