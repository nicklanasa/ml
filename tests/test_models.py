from random import seed
from ml.lib import load_csv, str_column_to_float
from ml.models import random_algorithm
from ml.models import zero_rule_algorithm_classification
from ml.models import zero_rule_algorithm_regression
from ml.models import simple_linear_regression

def test_random_algorithm():
  seed(1)
  train = [[0], [1], [0], [1], [0], [1]]
  test = [[None], [None], [None], [None]]
  predictions = random_algorithm(train, test)
  # print(predictions)
  assert len(predictions) == 4

def test_zero_rule_classification():
  seed(1)
  train = [[0], [0], [0], [0], [1], [1]]
  test = [[None], [None], [None], [None]]
  predictions = zero_rule_algorithm_classification(train, test)

  zeros = 0
  for i in range(len(predictions)):
      if predictions[i] == 0:
          zeros += 1

  assert zeros == 4, "There are 4 zeros"

def test_zero_rule_regression():
  seed(1)
  train = [[10], [15], [12], [15], [18], [20]]
  test = [[None], [None], [None], [None]]
  predictions = zero_rule_algorithm_regression(train, test)
  
  for pred in predictions:
     assert pred == 15.0, "Every value is 15 in the array"

def test_another_zero_rule_regression():
  seed(1)
  train = [[2], [3], [4], [8], [9], [6]]
  test = [[None], [None], [None], [None]]
  predictions = zero_rule_algorithm_regression(train, test)

  for pred in predictions:
    assert round(pred, 2) == 5.33, "Every value is 5.33 in the array"