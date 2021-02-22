# import dependencies
from preprocess import turnover_prep, wine_prep
from models import decision_tree
from models import tree_viz
from models import grid_search
from sklearn.tree import export_graphviz
from sklearn.preprocessing import scale
from models import find_most_critical_attr
import pandas as pd

'''
# Prepare the turnover data
X, y = turnover_prep('turnover.csv')

'''


# Prepare the wine data
X, y = wine_prep ('winequality_white.csv')

# Scale the data for use
X_scaled = scale (X)

#print (X.head())


'''
# Drop the less important features and rebuild the tree, for turnover
X_lean = X[['satisfaction', 'time_spend_company', 'evaluation', 'number_of_projects', 'average_montly_hours']]
y_lean = y
'''

# Drop the less important features and rebuild the tree, for turnover
X_lean = X[['alcohol', 'volatile acidity', 'free sulfur dioxide']]
y_lean = y


# Train decision tree
result, model_name, X_train, y_train = decision_tree (X_lean, y_lean
                                                      ,max_depth = 5
                                                      ,min_samples_leaf = 150
                                                      )

result = pd.DataFrame.from_dict(result, orient='index')
print (result)


'''
# Visualize the tree
tree_viz(model_name, X.columns)
'''


# GridSearch for best parameter
depth = [i for i in range(5, 21, 1)]
leaves = [i for i in range (50, 500, 50)]
pars = dict(max_depth = depth, min_samples_leaf = leaves)
best_pars = grid_search (model_name,X_train, y_train, pars, cv = 5)
print (best_pars)



'''
# export the tree into dot files and go to http://www.webgraphviz.com/ to visualize it
export_graphviz(model_name,"wine_tree_pruned.dot")
'''

'''
# Find the most important feature
attr_imp = find_most_critical_attr (model_name, X)
print (attr_imp)
'''
