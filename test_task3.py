import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from task3_learned_imputation import CategoricalImputer


def imputation(seed, data, target_column, num_values_to_delete):
    np.random.seed(seed)

    missing_indices = np.random.permutation(data.index)[:num_values_to_delete]
    true_values = data.loc[missing_indices, [target_column]]
    data.loc[missing_indices, [target_column]] = np.nan

    imputed_data = CategoricalImputer(target_column=target_column).fit_transform(data)
    imputed_values = imputed_data.loc[missing_indices, [target_column]]

    return accuracy_score(true_values, imputed_values)


def test_task3():

    bikes = pd.read_csv('data/bikes/day.csv')
    accuracy = imputation(1234, bikes.copy(deep=True), 'season', 100)
    print(f'Accuracy for bikes with seed 1234 is {accuracy}')
    assert accuracy > 0.8

    accuracy = imputation(5678, bikes.copy(deep=True), 'season', 100)
    print(f'Accuracy for bikes with seed 5678 is {accuracy}')
    assert accuracy > 0.8

    catalog = pd.read_csv('data/catalog/mae.csv')
    accuracy = imputation(1234, catalog.copy(deep=True), 'color', 500)
    print(f'Accuracy for catalog with seed 1234 is {accuracy}')
    assert accuracy > 0.8

    accuracy = imputation(5678, catalog.copy(deep=True), 'color', 500)
    print(f'Accuracy for catalog with seed 5678 is {accuracy}')
    assert accuracy > 0.8





