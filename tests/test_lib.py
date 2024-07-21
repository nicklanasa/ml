from random import seed
from ml.lib import accuracy_metric
from ml.lib import confusion_matrix
from ml.lib import mae_metric
from ml.lib import rmse_metric
from ml.lib import train_test_split
from ml.lib import cross_validation_split
from ml.lib import load_csv, str_column_to_float
from ml.lib import mean, variance, covariance, coefficients

def test_rmse():
  actual = [0.1, 0.2, 0.3, 0.4, 0.5]
  predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
  rmse_error = rmse_metric(actual, predicted)
  assert round(rmse_error, 3) == 0.009

def test_mae():
  actual = [0.1, 0.2, 0.3, 0.4, 0.5]
  predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
  mae = mae_metric(actual, predicted)
  assert round(mae, 3) == 0.008 

def test_confusion_matrix():
  # Test confusion matrix with integers
  actual = [0,0,0,0,0,1,1,1,1,1]
  predicted = [0,1,1,0,0,1,0,1,1,1]
  unique, matrix = confusion_matrix(actual, predicted)
  # print(unique)
  # print(matrix)

  assert unique.pop() == 0
  assert unique.pop() == 1  

  assert matrix[0][0] == 3, "3 0's were predicted correctly"
  assert matrix[0][1] == 2, "2 0's were predicted wrong"

  assert matrix[1][0] == 1, "1 1's were predicted wrong"
  assert matrix[1][1] == 4, "4 1's were predicted correctly"

  # print("(P)" + " ".join(str(x) for x in unique))
  # print("(A)---")
  # for i, x in enumerate(unique):
  #   print("%s| %s" % (x, " ".join(str(x) for x in matrix[i])))

def test_accuracy_metric():
  actual = [0,0,0,0,0,1,1,1,1,1]
  predicted = [0,1,0,0,0,1,0,1,1,1]
  accuracy = accuracy_metric(actual, predicted)
  assert accuracy == 80.0, "Should be 80.0"

def test_kfold():
  seed(1)
  dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
  folds = cross_validation_split(dataset, 4)
  assert len(folds) == 4, "Should be 4 folds"
  
def test_tain():
  seed(1)
  dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
  train, test = train_test_split(dataset)
  assert len(train) == 6, "Should be length 6"
  assert len(test) == 4, "Should be length 4"

def test_load_data():
  filename = "datasets/pima-indians-diabetes.csv"
  dataset = load_csv(filename)
  assert len(dataset[0]) == 9, "Should be 9 cols"
  assert dataset[0][0] == "6", "First column in first row is 6 as a str"

  str_column_to_float(dataset, 0)
  assert dataset[0][0] == 6.0, "First column in first row is 6.0, converted to float"

def test_mean():
  # calculate mean and variance
  dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
  x = [row[0] for row in dataset]
  y = [row[1] for row in dataset]
  mean_x, mean_y = mean(x), mean(y)
  assert mean_x == 3.0, "mean for first column in array is 3.0"
  assert mean_y == 2.8, "mean for second column in array is 2.8"

def test_variane():
  # calculate variance
  dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
  x = [row[0] for row in dataset]
  y = [row[1] for row in dataset]
  mean_x, mean_y = mean(x), mean(y)
  var_x, var_y = variance(x, mean_x), variance(y, mean_y)
  assert var_x == 10.0, "variance for data in 1 column for each row in array is 10.0"
  assert mean_x == 3.0, "variance for data in 2 column for each row in array is 8.8"

def test_covariance():
  dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
  x = [row[0] for row in dataset]
  y = [row[1] for row in dataset]
  mean_x, mean_y = mean(x), mean(y)
  covar = covariance(x, mean_x, y, mean_y)
  assert covar == 8.0, "covariance is 8.0"