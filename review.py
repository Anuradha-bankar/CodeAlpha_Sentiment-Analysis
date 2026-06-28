import pandas as pd

df = pd.read_csv("Amazon_Reviews.csv.zip", encoding="latin-1")
print(df.head())