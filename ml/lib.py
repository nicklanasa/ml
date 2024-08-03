"""Lib functions."""
from csv import reader
from math import sqrt
from random import randrange

def load_csv(filename):
    """Load from csv file."""
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset

def str_column_to_float(dataset, column):
    """Convert string column to float."""
    for row in dataset:
        row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):
    """Convert string column to integer."""
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def dataset_minmax(dataset):
    """
    Calculate the minimum and maximum values for each column in a dataset.

    Args:
        dataset (list): The dataset containing the data.

    Returns:
        list: A list of lists, where each inner list contains the minimum and maximum values for a column.
    """
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

def normalize_dataset(dataset, minmax):
    """
    Normalize the dataset using the provided minmax values.

    Parameters:
        dataset (list): The dataset to be normalized.
        minmax (list): The minmax values for each feature in the dataset.

    Returns:
        None
    """
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def column_means(dataset):
    """
    Calculate the mean value for each column in the dataset.

    Parameters:
        dataset (list): A 2D list representing the dataset.

    Returns:
        list: A list containing the mean value for each column in the dataset.
    """
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means

def column_stdevs(dataset, means):
    """
    Calculate column standard deviations.

    The standard deviation describes the average spread
    of values from the mean. It can be calculated as the
    square root of the sum of the squared difference
    between each value and the mean and dividing by the
    number of values minus 1.
    """
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i]-means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
    return stdevs

def standardize_dataset(dataset, means, stdevs):
    """Standardize dataset."""
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]

def train_test_split(dataset, split=0.60):
    """A resampling method to split a dataset into a train and test set."""
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy

def cross_validation_split(dataset, folds=3):
    """
    Split a dataset into k folds.

    Parameters:
    - dataset (list): The dataset to be split.
    - folds (int): The number of folds to create. Default is 3.

    Returns:
    - dataset_split (list): A list of k folds, where each fold is a list of data points.
    """
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy_metric(actual, predicted):
    """
    Calculate the accuracy of a classification model.

    Parameters:
    actual (list): A list of the actual labels.
    predicted (list): A list of the predicted labels.

    Returns:
    float: The accuracy of the model in percentage.

    Examples:
    >>> actual = [1, 0, 1, 1, 0]
    >>> predicted = [1, 1, 0, 1, 0]
    >>> accuracy_metric(actual, predicted)
    60.0
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def confusion_matrix(actual, predicted):
    """
    Calculate a confusion matrix.

    Parameters:
    actual (list): A list of actual class labels.
    predicted (list): A list of predicted class labels.

    Returns:
    tuple: A tuple containing the unique class labels and the confusion matrix.

    The confusion matrix is a square matrix where each row represents the actual class labels
    and each column represents the predicted class labels. The value at matrix[i][j] represents
    the number of instances where the actual class label is i and the predicted class label is j.
    """
    unique = set(actual)
    matrix = [list() for x in range(len(unique))]
    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for i in range(len(actual)):
        x = lookup[actual[i]]
        y = lookup[predicted[i]]
        matrix[x][y] += 1
    return unique, matrix

def mae_metric(actual, predicted):
    """
    Calculate the Mean Absolute Error (MAE) metric.

    Parameters:
    actual (list): The actual values.
    predicted (list): The predicted values.

    Returns:
    float: The MAE metric value.
    """
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += abs(predicted[i] - actual[i])
    return sum_error / float(len(actual))

def rmse_metric(actual, predicted):
    """
    Calculate the Root Mean Squared Error (RMSE) between the actual and predicted values.

    Parameters:
    actual (list): The list of actual values.
    predicted (list): The list of predicted values.

    Returns:
    float: The RMSE value.
    """
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += (predicted[i] - actual[i]) ** 2
    sum_error = sum_error / float(len(predicted))
    return sqrt(sum_error)

def mean(values):
    """
    Calculate the mean of a list of values.

    Parameters:
        values (list): A list of numeric values.

    Returns:
        float: The mean of the values.
    """
    return sum(values) / float(len(values))

# Calculate the variance of a list of numbers
# The variance is the sum squared difference for each value from the mean value.
def variance(values, mean):
    """
    Calculate the variance of a list of values.

    Parameters:
        values (list): A list of numerical values.
        mean (float): The mean value of the list.

    Returns:
        float: The variance of the values.
    """
    return sum([(x-mean)**2 for x in values])

def covariance(x, mean_x, y, mean_y):
    """
    Calculate the covariance between two variables.

    Parameters:
        x (list): The first variable.
        mean_x (float): The mean of the first variable.
        y (list): The second variable.
        mean_y (float): The mean of the second variable.

    Returns:
        float: The covariance between the two variables.
    """
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

def coefficients(dataset):
    """
    Calculate the coefficients of a linear regression model.

    Parameters:
        dataset (list): A list of tuples representing the dataset. Each tuple should contain two elements,
                        where the first element is the independent variable and the second element is the
                        dependent variable.

    Returns:
        list: A list containing the intercept (b0) and slope (b1) coefficients of the linear regression model.
    """
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]

def predict(row, coefficients):
    """
    Make a prediction with the linear regression model.

    Parameters:
        row (list): A list of values representing the input data.
        coefficients (list): A list of coefficients for the linear regression model.

    Returns:
        float: The predicted value.
    """
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return yhat

def coefficients_stg(train, l_rate, n_epoch):
    """
    Calculate the coefficients for the stochastic gradient descent algorithm.

    Parameters:
    train (list): The training dataset.
    l_rate (float): The learning rate.
    n_epoch (int): The number of epochs.

    Returns:
    list: The calculated coefficients.

    """
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            sum_error += error**2
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
        print(f">epoch={epoch}, lrate={l_rate}, error={sum_error}")
    return coef
