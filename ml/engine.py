from ml.lib import accuracy_metric, rmse_metric
from ml.lib import train_test_split
from ml.lib import cross_validation_split
from ml.models import simple_linear_regression

# Evaluate an algorithm using a train/test split
def train_test_evaluate(dataset, algorithm, split=0.60, *args):
    """
    Trains a machine learning algorithm on a dataset, splits it into training and testing sets,
    evaluates the algorithm's performance on the testing set, and returns the accuracy metric.

    Parameters:
    - dataset: The dataset to train and evaluate the algorithm on.
    - algorithm: The machine learning algorithm to use for training and evaluation.
    - split: The ratio of the dataset to use for training. Default is 0.60.
    - *args: Additional arguments to pass to the algorithm.

    Returns:
    - accuracy: The accuracy metric of the algorithm on the testing set.
    """
    train, test = train_test_split(dataset, split)
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(train, test_set, *args)
    actual = [row[-1] for row in test]
    accuracy = accuracy_metric(actual, predicted)
    return accuracy

# Evaluate an algorithm using a cross-validation split
def cross_val_evaluate(dataset, algorithm, n_folds, *args):
    """
    Perform cross-validation evaluation on a dataset using a given algorithm.

    Parameters:
    - dataset (list): The dataset to be used for cross-validation.
    - algorithm (function): The algorithm to be evaluated.
    - n_folds (int): The number of folds to use for cross-validation.
    - *args: Additional arguments to be passed to the algorithm.

    Returns:
    - scores (list): A list of accuracy scores for each fold.

    """
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def simple_linear_regression_evaluate(dataset, split=0.6):
    """
    Evaluates the performance of a simple linear regression model on a dataset.

    Args:
        dataset (list): The dataset to evaluate the model on.
        split (float, optional): The ratio of training set to test set. Defaults to 0.6.

    Returns:
        float: The root mean squared error (RMSE) of the model's predictions.
    """
    test_set = list()
    for row in dataset:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = simple_linear_regression(dataset, test_set)
    actual = [row[-1] for row in dataset]
    rmse = rmse_metric(actual, predicted)
    return rmse