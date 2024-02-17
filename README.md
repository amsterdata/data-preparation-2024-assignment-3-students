# Programming Assignment 3

This assignment is using Python as programming language. We suggest that you use **Python3.9** and install the required libraries exactly like specified in [requirements.txt](requirements.txt) via a [virtualenv](https://docs.python.org/3/library/venv.html) to avoid potential incompatibilities.


## Task 1 - Defining sklearn pipelines (2 points)

First, we ask you to implement an ML pipeline for a fictitious e-commerce use case. We provide a small data set from an e-commerce platform, which contains products from two categories: Kitchen and Jewelry. The dataset contains two kinds of input files with tab- separated data:

 * Product data (id, category, product title), e.g.: `daa54754-af9c-41c0-b542-fe5eabc5919c Kitchen Bodum French Press Coffeemaker`
 * Reviews (id, rating, review_text), e.g.: `daa54754-af9c-41c0-b542-fe5eabc5919c 5 Great!`

We ask you to prepare the data and create a training pipeline for a classifier that predicts the category from the product title, rating and review text. For that, you have to write python in the file [task1_reviews_pipeline.py](task1_reviews_pipeline.py). Please implement the following methods:

 1. The method `load_and_integrate_data` should read the partitioned input files for products and reviews from the [amazon_reviews](data/amazon_reviews) folder. This method must return a single pandas DataFrame with the whole dataset.

 1. The method `define_training_pipeline` should setup an [sklearn pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to encode the data into features and train a model to predict the category. Additionally, we ask you to make sure that your pipelines uses 5-fold cross-validation to select the hyperparameters of your model (among 3-5 candidate hyperparameter values). We strongly recommend that you use appropriate abstractions from scikit-learn here.

Note that your pipeline should reach an accuracy of over 90% on the test data. You can use the unit test [test_task1.py](test_task1.py) to help you with development here. You can execute it via `pytest test_task1.py -s`.


## Task 2 - Provenance and fairness in pipelines (3 points)

We provide a fully working [ML pipeline](task2_provenance_and_fairness.py) for you which trains a classifier to predict which reviews are helpful. In this task, we ask you to write the pipeline to compute important meta information.

 * **Group Fairness (1 point)** -- Compute the fairness of the pipeline with respect to third party reviews. In particular, compute the predictive parity metric (the difference in true positive rates) between reviews from a third party and reviews not from a third party.

 * **Provenance (2 points)** -- Compute which records from the ratings and products relation are used to train the classifier. Compute two boolean arrays with a dimensionality similar to the cardinality of the relations, where the entry at position i denotes whether the i-th record is included in the training data of the classifier. Let's for example say there are 100 rows of products in the input data. The code might only use some of them for training. The others might be filtered out earlier or used for testing.
   So your task is to compute a boolean array with 100 entries and only set a particular entry to true of the corresponding product row is used for training.

You can use the unit test [test_task2.py](test_task2.py) to help you with development here. You can execute it via `pytest test_task2.py -s`.

## Task 3 - Learning to Impute Missing Values with an Estimator/Transformer (4 points)

In the final task, we ask you to implement your own Estimator/Transformer in the file [task3_learned_imputation.py](task3_learned_imputation.py). Your  Estimator/Transformer should use ML to impute missing values. 

 * Your implementation will be supplied with a pandas Dataframe in its `fit` method, which has missing values in a particular `target_column`. At fitting time, your Estimator/Transformer should learn how to impute the missing values in this column, based on the remaining data in the dataframe. You can assume that the target column is always categorical, and that the remaining columns in the data are either categorical, numerical or contain text.

 * In the `transform` method, your Estimator/Transformer will be supplied with a pandas DataFrame again, in which it has to fill the missing values in the target column. 

Note that your pipeline should reach an accuracy of over 80% on two evaluation datasets which contain data about a product catalog and bikesharing. You can use the unit test [test_task3.py](test_task3.py) to help you with development here. You can execute it via `pytest test_task3.py -s`.


### (1 point) Code quality

We additionally ask you to write correct, efficient, well-structured and easy to-read code. Furthermore, your code should run without issues on the machines of the TAs.


## Grading

The TAs will execute and review your code for this assignment, and will deduct points for missing, incorrect or inefficient code. Your final grade for this assignment is the sum of the points achieved across all tasks. Note that we provide the unit tests to make it easier for you to develop your solution, but that passing the unit tests does not mean that the solution is necessarily correct or efficient.
