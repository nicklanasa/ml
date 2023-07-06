# Load a csv
from csv import reader
from ml.lib import str_column_to_float, load_csv

filename = 'datasets/pima-indians-diabetes.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns' .format(filename, len(dataset), len(dataset[0])))

print(dataset[0])

# convert string columns to float
for i in range(len(dataset[0])):
  str_column_to_float(dataset, i)
print(dataset[0])
