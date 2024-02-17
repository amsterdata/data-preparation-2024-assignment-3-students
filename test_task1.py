import numpy as np
from task1_reviews_pipeline import load_and_integrate_data, define_training_pipeline
from sklearn.model_selection import train_test_split


def run_reviews_pipeline(seed, test_size):
    np.random.seed(seed)
    data = load_and_integrate_data()
    labels = data.pop('category')
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=seed)
    sklearn_pipeline = define_training_pipeline()
    model_with_transformations = sklearn_pipeline.fit(train_data, train_labels)
    accuracy = model_with_transformations.score(test_data, test_labels)
    print(f"Accuracy on test data is {accuracy}")
    return accuracy


def test_task1():
    accuracy = run_reviews_pipeline(1234, 0.2)
    assert accuracy > 0.9

    accuracy = run_reviews_pipeline(5678, 0.3)
    assert accuracy > 0.9
