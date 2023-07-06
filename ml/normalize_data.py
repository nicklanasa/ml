# Load a csv
from ml.lib import str_column_to_float, load_csv, dataset_minmax, normalize_dataset, column_means, column_stdevs, standardize_dataset

filename = 'datasets/pima-indians-diabetes.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns' .
      format(filename, len(dataset), len(dataset[0])))

# convert string columns to float
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)

print(dataset[0])

# Calculate min and max for each column
minmax = dataset_minmax(dataset)
print(minmax)

# Normalize columns
normalize_dataset(dataset, minmax)
print(dataset[0])

# Standard deviation
means = column_means(dataset)
stdevs = column_stdevs(dataset, means)

print("means")
print(means)
print("stdevs")
print(stdevs)

# Standardize dataset
print("standardize dataset")
standardize_dataset(dataset, means, stdevs)
print(dataset[0])
