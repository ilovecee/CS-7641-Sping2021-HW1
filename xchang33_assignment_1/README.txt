I. Code Env

    Environment: Python 3.6.6 |Anaconda custom (64-bit)|

    Python environment and dependencies (requirements) need to be installed:
	
    numpy
    pandas
    timeit
    matplotlib.pyplot
    sklearn
    pydotplus
    keras
    theano


    * The default backend of keras is tensorflow, to set it to theano, please find the "keras.json" file and copy it to: "C:\Users\[xxx]\.keras\"
	    for more details please see https://keras.io/backend/


	
II. Code Files
	
    The code consists of three types of files:

	    1. preprocess.py -- preprocess the two datasets before feeding to the models;
	    2. models.py -- all algorithms and graphing functions are defined in this module;
	    3. run files : {decision_tree.py; knn.py; svm.py, nn.py, adaboost.py} -- run different lines of codes in these files to call 
                        models and graphing functions in models.py; generates results

					
					
III. Datasets
					
	The data sources are:
		1. turnover.csv
		2. winequality_white.csv

	* Please keep .py files and the .csv files in the same dir

	

IV. Supporting_docs:

	1. Many trees visualized and stored in .dot files. They are for your reference, the code generates the trees too. Go to http://www.webgraphviz.com/ to visualize the trees
	2. A word version of my analysis report is contained for editing purposes;
	3. keras.json file, set up the backend as 'theano'
	



 