# import dependencies
import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import pydotplus
from keras.models import Sequential
from keras.layers.core import Dense





# For reproducibility
ran_state = 42
np.random.seed(42)


# Decision Tree
def decision_tree (X, y, max_depth = None, min_samples_leaf = 1, class_weight = None):
    start = timeit.default_timer()
    # The default criterion for information gain is 'gini'
    # Initialize a decision tree clf
    model_dt = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf, class_weight = class_weight)

    # Hold out set (validation set)
    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size = 0.33, random_state = ran_state)

    # Fit
    model_dt.fit(X_train, y_train)

    # 5-fold CV and averaging the CV scores
    cv_score = cross_val_score (model_dt, X_train, y_train, cv = 5)
    y_pred = model_dt.predict (X_hold)

   
    # SCORES
    avg_cv_score = np.mean(cv_score)
    std_cv_score = np.std(cv_score)
    train_score = model_dt.score (X_train, y_train)
    test_score = model_dt.score (X_hold, y_hold)
    precision = precision_score (y_hold, y_pred)
    recall = recall_score (y_hold, y_pred)
    f1_score = (2 * precision * recall) / (precision + recall)
    roc_score = roc_auc_score (y_hold, y_pred)

        
    stop = timeit.default_timer()
    runtime = (stop - start)
    # construct a score dictionary
    keys = ['CV_Score_AVG', 'CV_Score_STD', 'Train_Score', 'Test_Score', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC_SCORE', 'Runtime']
    values = [avg_cv_score,
                    std_cv_score,
                    train_score,
                    test_score,
                    precision,
                    recall,
                    f1_score,
                    roc_score,
                    runtime]

    performance_dict = dict(zip(keys, values))

    return performance_dict, model_dt, X_train, y_train




# Use decision tree to find the most important attribute
def find_most_critical_attr (model, df):
    feature_importances = model.feature_importances_
    feature_list = list (df)
    relative_importances = pd.DataFrame (index = feature_list, data = feature_importances, columns = ['importance'])
    relative_importances.sort_values (by = 'importance', ascending = False)

    return relative_importances




# Visualize the tree
def tree_viz (model, feature_names):
    dot_data = StringIO()
    export_graphviz(model,
                        out_file = dot_data,
                        feature_names = feature_names,
                        class_names = 'churn',
                        filled = True, rounded = True,
                        impurity = False)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('tree_before_pruning.pdf')



# GridSearch
def grid_search (model,X_train, y_train, pars, cv = 5):
    par_search = GridSearchCV(model, pars, cv = cv)
    par_search.fit (X_train, y_train)

    return par_search.best_params_






# K-NN
def knn(X, y, k):
    start = timeit.default_timer()

    # Build a pipeline of scaling and initialization
    steps = [('scaler', StandardScaler()),
             ('knn', KNeighborsClassifier(n_neighbors = k))]
    pipeline = Pipeline(steps)

    #holdout sets for CV
    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size = 0.33, random_state = ran_state)

    model_knn = pipeline.fit(X_train, y_train)

    # 5-fold CV and averaging the CV scores
    cv_score = cross_val_score (model_knn, X_train, y_train, cv = 5)
    y_pred = model_knn.predict (X_hold)

    # SCORES
    avg_cv_score = np.mean(cv_score)
    std_cv_score = np.std(cv_score)
    train_score = model_knn.score (X_train, y_train)
    test_score = model_knn.score (X_hold, y_hold)
    precision = precision_score (y_hold, y_pred)
    recall = recall_score (y_hold, y_pred)
    f1_score = (2 * precision * recall) / (precision + recall)
    roc_score = roc_auc_score (y_hold, y_pred)


    stop = timeit.default_timer()
    runtime = (stop - start)

    # construct a score dictionary
    keys = ['CV_Score_AVG', 'CV_Score_STD', 'Train_Score', 'Test_Score', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC_SCORE', 'Runtime']
    values = [avg_cv_score,
                std_cv_score,
                train_score,
                test_score,
                precision,
                recall,
                f1_score,
                roc_score,
                runtime]

    performance_dict = dict(zip(keys, values))

    return performance_dict, model_knn, X_train, y_train



# KNN Graph with different k
def k_graph (X, y, k):
    neighbors = np.arange(1, k)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    for i, k in enumerate(neighbors):
        result, model_knn, X_train, y_train = knn (X, y, k)
        train_accuracy[i] = result['Train_Score']
        test_accuracy[i] = result['Test_Score']

    # Generate the plot
    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1, k))
    plt.show()





# SVM
def svm (X, y, C = 1, gamma = 'auto', kernel = 'rbf'):
    start = timeit.default_timer()

    # Setup the pipeline
    steps = [('scaler', StandardScaler()),
             ('svm', SVC(C = C, gamma = gamma, kernel = kernel, random_state = ran_state))]
    pipeline = Pipeline(steps)

    # Hold out set (validation set)
    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size = 0.33, random_state = ran_state)

    # Fit
    model_svm = pipeline.fit(X_train, y_train)

    # 5-fold CV and averaging the CV scores
    cv_score = cross_val_score (model_svm, X_train, y_train, cv = 5)
    y_pred = model_svm.predict (X_hold)


    # SCORES
    avg_cv_score = np.mean(cv_score)
    std_cv_score = np.std(cv_score)
    train_score = model_svm.score (X_train, y_train)
    test_score = model_svm.score (X_hold, y_hold)
    precision = precision_score (y_hold, y_pred)
    recall = recall_score (y_hold, y_pred)
    f1_score = (2 * precision * recall) / (precision + recall)
    roc_score = roc_auc_score (y_hold, y_pred)


    stop = timeit.default_timer()
    runtime = (stop - start)


    # construct a score dictionary
    keys = ['CV_Score_AVG', 'CV_Score_STD', 'Train_Score', 'Test_Score', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC_SCORE', 'Runtime']
    values = [avg_cv_score,
                std_cv_score,
                train_score,
                test_score,
                precision,
                recall,
                f1_score,
                roc_score,
                runtime]

    performance_dict = dict(zip(keys, values))

    return performance_dict, model_svm, X_train, y_train




# svm graph with different kernel function
def svm_kernel_graph (X, y, l):

    train_accuracy = np.empty(len(l))
    test_accuracy = np.empty(len(l))

    for i, kernel in enumerate(l):
        result, model_svm, X_train, y_train = svm (X, y, kernel = kernel)
        train_accuracy[i] = result['Train_Score']
        test_accuracy[i] = result['Test_Score']

    # Generate the plot
    plt.title('SVM with various kernels')
    plt.plot(l, test_accuracy, label = 'Testing Accuracy', )
    plt.plot(l, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.xlabel('Kernel Names')
    plt.ylabel('Accuracy')
    plt.show()



# 'Adaboosting' the decision tree
def ada_dt (X, y, iteration, lr):
    start = timeit.default_timer()
    # initialize the estimator
    dtree = DecisionTreeClassifier (criterion = 'gini', max_depth = 1)
    # build the adaboostclassifier with estimator
    adabst_tree = AdaBoostClassifier (base_estimator = dtree
                                    ,n_estimators = iteration
                                    ,learning_rate = lr
                                    ,random_state = ran_state)

    # split the dataset into training and testing
    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size = 0.33, random_state = ran_state) 

    # Fit
    adabst_tree.fit (X_train, y_train)

    # 5-fold CV and averaging the CV scores
    cv_score = cross_val_score (adabst_tree, X_train, y_train, cv = 5)
    y_pred = adabst_tree.predict (X_hold)


    # SCORES
    avg_cv_score = np.mean(cv_score)
    std_cv_score = np.std(cv_score)
    train_score = adabst_tree.score (X_train, y_train)
    test_score = adabst_tree.score (X_hold, y_hold)
    precision = precision_score (y_hold, y_pred)
    recall = recall_score (y_hold, y_pred)
    f1_score = (2 * precision * recall) / (precision + recall)
    roc_score = roc_auc_score (y_hold, y_pred)


    stop = timeit.default_timer()
    runtime = (stop - start)


    # construct a score dictionary
    keys = ['CV_Score_AVG', 'CV_Score_STD', 'Train_Score', 'Test_Score', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC_SCORE', 'Runtime']
    values = [avg_cv_score,
                std_cv_score,
                train_score,
                test_score,
                precision,
                recall,
                f1_score,
                roc_score,
                runtime]

    performance_dict = dict(zip(keys, values))

    return performance_dict, adabst_tree, X_train, y_train





# Draw performance graph on various iterations
def ada_iteration_graph (X, y, l):

    train_accuracy = np.empty(len(l))
    test_accuracy = np.empty(len(l))

    for i, iter in enumerate(l):
        result, model_ada, X_train, y_train = ada_dt (X, y, iteration = iter, lr = 0.05)
        train_accuracy[i] = result['Train_Score']
        test_accuracy[i] = result['Test_Score']

    # Generate the plot
    plt.title('Adaboost with various iterations')
    plt.plot(l, test_accuracy, label = 'Testing Accuracy', )
    plt.plot(l, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.xlabel('Iterations (number of estimators)')
    plt.ylabel('Accuracy')
    plt.show()




# a normal Adam neural network
def nor_nn (X, y, nb_epoch, batch_size):

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = ran_state)

    # Build the model
    model = Sequential()
    model.add(Dense(5, input_dim = X_train.shape[1], init = 'uniform', activation = 'relu'))
    model.add(Dense(5, init = 'uniform', activation = 'relu'))
    model.add(Dense(5, init = 'uniform', activation = 'relu'))
    model.add(Dense(5, init = 'uniform', activation = 'relu'))
    model.add(Dense(2, init = 'uniform', activation = 'sigmoid'))

    # Compile model
    model.compile (loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # Fit the model
    model.fit (X_train, y_train, epochs = nb_epoch, batch_size = batch_size, verbose = 2)

    #return model, X_train, X_test, y_train, y_test
