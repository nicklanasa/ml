from random import randrange
from ml.lib import coefficients

# Generate random predictions
def random_algorithm(train, test):
    output_values = [row[-1] for row in train]
    unique = list(set(output_values))
    predicted = list()
    for row in test:
      index = randrange(len(unique))
      predicted.append(unique[index])
    return predicted

# zero rule algorithm for classification
# assumes the last column is the predicted value
# counts the occurences of the predictions
# and outputs a list with the most 
def zero_rule_algorithm_classification(train, test):
  output_values = [row[-1] for row in train]
  # produces the class value with the most occurrences
  prediction = max(set(output_values), key=output_values.count)
  predicted = [prediction for i in range(len(test))]
  return predicted

# zero rule algorithm for regression
def zero_rule_algorithm_regression(train, test):
    output_values = [row[-1] for row in train]
    prediction = sum(output_values) / float(len(output_values))
    predicted = [prediction for i in range(len(test))]
    return predicted

# Simple linear regression algorithm
def simple_linear_regression(train, test):
    predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
      yhat = b0 + b1 * row[0]
      predictions.append(yhat)
    return predictions