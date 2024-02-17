import numpy as np
from task2_provenance_and_fairness import execute_pipeline


def test_task2():

    predictive_parity, ratings_usage, products_usage = execute_pipeline(1234)
    assert np.sum(ratings_usage) == 18017
    assert np.sum(products_usage) == 2859
    assert (predictive_parity - 0.088355) < 0.00001

    predictive_parity, ratings_usage, products_usage = execute_pipeline(5678)
    assert np.sum(ratings_usage) == 28595
    assert np.sum(products_usage) == 3583
    assert (predictive_parity - 0.001381) < 0.00001
