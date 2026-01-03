import pandas as pd

df = pd.read_csv("data/clinical/clinical_data.csv")

print("\nColumns:")
print(df.columns)

LABEL = "label"   

print(f"\nUnique labels in '{LABEL}':")
print(df[LABEL].unique())
print(f"\nCount: {len(df[LABEL].unique())}")

print("\nFirst 10 labels:")
print(df[LABEL].head(10))

print("\nDataset head:")
print(df.head())
