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
    """Find the min and max values for each column."""
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

def normalize_dataset(dataset, minmax):
    """Rescale dataset columns to the range 0-1."""
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def column_means(dataset):
    """Calculate column means. Sum all of all the values for a column divided
    by the total number of values.
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
    """Split a dataset into k folds."""
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
    """Calculate a accuracy metric"""
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def confusion_matrix(actual, predicted):
    """Calculate a confusion matrix"""
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
    """Calculate mean absolute error"""
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += abs(predicted[i] - actual[i])
    return sum_error / float(len(actual))

def rmse_metric(actual, predicted):
    """Calculate the Root Mean Squared Error"""
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += (predicted[i] - actual[i]) ** 2
    sum_error = sum_error / float(len(predicted))
    return sqrt(sum_error)

# Calculate the mean value of a list of numbers
def mean(values):
    return sum(values) / float(len(values))

# Calculate the variance of a list of numbers
# The variance is the sum squared difference for each value from the mean value.
def variance(values, mean):
    return sum([(x-mean)**2 for x in values])

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

# Calculate coefficients
def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]
