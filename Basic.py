import pandas as pd

# Load Iris.csv (make sure Iris.csv is in the same folder)
data = pd.read_csv("Iris.csv")

print("First 5 rows of Iris dataset:")
print(data.head())
