from random import seed
from ml.lib import load_csv, str_column_to_float

from ml.models import zero_rule_algorithm_classification
from ml.models import zero_rule_algorithm_regression
from ml.models import simple_linear_regression

from ml.engine import train_test_evaluate
from ml.engine import cross_val_evaluate
from ml.engine import simple_linear_regression_evaluate

def test_evaluate_test_train():
  # Test the train/test harness
  seed(1)
  # load and prepare data
  filename = "datasets/pima-indians-diabetes.csv"
  dataset = load_csv(filename)
  for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
  # evaluate algorithm
  split = 0.6
  accuracy = train_test_evaluate(dataset, zero_rule_algorithm_classification, split)
  # print("Accuracy: %.3f%%"  % (accuracy))
  assert round(accuracy, 3) == 67.427

def test_evaluate_cross_val():
  seed(1)
  filename = "datasets/pima-indians-diabetes.csv"
  dataset = load_csv(filename)
  for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
  
  folds = 5
  scores = cross_val_evaluate(dataset, zero_rule_algorithm_classification, folds)

  assert round(scores[0], 3) == 62.092
  assert round(scores[1], 3) == 64.706
  assert round(scores[2], 3) == 64.706
  assert round(scores[3], 3) == 64.706
  assert round(scores[4], 3) == 69.281

def test_evaluate_linear_regression():
  dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]

  split = 0.6
  rmse = simple_linear_regression_evaluate(dataset, split)
  assert rmse == 0.692820323027551