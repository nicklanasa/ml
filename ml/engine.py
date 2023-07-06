from ml.lib import accuracy_metric, rmse_metric
from ml.lib import train_test_split
from ml.lib import cross_validation_split
from ml.models import simple_linear_regression

# Evaluate an algorithm using a train/test split
def train_test_evaluate(dataset, algorithm, split=0.60, *args):
  train, test = train_test_split(dataset, split)
  test_set = list()
  for row in test:
    row_copy = list(row)
    row_copy[-1] = None
    test_set.append(row_copy)
  predicted = algorithm(train, test_set, *args)
  actual = [row[-1] for row in test]
  accuracy = accuracy_metric(actual, predicted)
  print(accuracy)
  return accuracy

# Evaluate an algorithm using a cross-validation split
def cross_val_evaluate(dataset, algorithm, n_folds, *args):
  folds = cross_validation_split(dataset, n_folds)
  # print((len(folds[0][0])))
  scores = list()
  for fold in folds:
    # create copy of list of folds
    train_set = list(folds)
    # remove the held out fold
    train_set.remove(fold)
    # flatten to one long list of rows to match algo expected of training data
    train_set = sum(train_set, [])
    test_set = list()
    # create test set
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
  test_set = list()
  for row in dataset:
    row_copy = list(row)
    row_copy[-1] = None
    test_set.append(row_copy)
  predicted = simple_linear_regression(dataset, test_set)
  print(predicted)
  actual = [row[-1] for row in dataset]
  rmse = rmse_metric(actual, predicted)
  return rmse