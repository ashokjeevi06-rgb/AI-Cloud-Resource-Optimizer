import pandas as pd
import os

file_path = "dataset/cloud_workload_dataset.csv"

if not os.path.exists(file_path):
    print("ERROR: CSV file not found!")
    print("Expected location:", file_path)
else:
    print("CSV file found!")

    data = pd.read_csv(file_path)

    print("\n========== DATASET DETAILS ==========")
    print("Number of records:", data.shape[0])
    print("Number of columns:", data.shape[1])

    print("\n========== COLUMN NAMES ==========")
    for column in data.columns:
        print(column)

    print("\n========== FIRST 5 ROWS ==========")
    print(data.head())

    print("\n========== MISSING VALUES ==========")
    print(data.isnull().sum())

    print("\n========== DATASET READY ==========")