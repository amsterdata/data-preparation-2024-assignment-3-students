import numpy as np
from example_estimator_transformer import CenterAllFeaturesEstimator


def test_estimator():
    matrix = np.array([[1, 0, 6],
                       [2, 0, 3],
                       [3, 0, 9]])

    more_matrix_data = np.array([[2, 0, 3],
                                 [1, 0, 9]])

    print("Input matrix\n", matrix)
    centering = CenterAllFeaturesEstimator()
    centering.fit(matrix)
    print("Is fitted?", centering.is_fitted_)
    print("Column means:", centering.column_means_)

    centered_matrix = centering.transform(matrix)
    print("\nCentered matrix\n", centered_matrix)

    print("\nMore input data\n", more_matrix_data)
    centered_more = centering.transform(more_matrix_data)
    print("Centered\n", centered_more)
