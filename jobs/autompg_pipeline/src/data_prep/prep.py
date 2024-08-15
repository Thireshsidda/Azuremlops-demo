# import os
# import argparse

# import logging
# import mlflow
# from pathlib import Path

# import pandas as pd

# from sklearn.model_selection import train_test_split

# # input and output arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--data", type=str, help="path to input data")
# parser.add_argument("--train_data", type=str, help="path to train data")
# parser.add_argument("--test_data", type=str, help="path to test data")
# args = parser.parse_args()

# # Start Logging
# mlflow.start_run()

# print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

# arr = os.listdir(args.data)
# print(arr)

# # Initialize an empty list to store DataFrames
# df = []

# for filename in arr:
#     print("reading file: %s ..." % filename)
#     with open(os.path.join(args.data, filename), "r") as handle:
#         input_df = pd.read_csv((Path(args.data) / filename))
#         df.append(input_df)

# # Concatenate the list of DataFrames into a single DataFrame
# df = pd.concat(df)

# # Now you can split the DataFrame into train and test sets
# train_df, test_df = train_test_split(df, test_size=0.3, random_state=4)

# # Save the train and test DataFrames to CSV files
# train_df.to_csv((Path(args.train_data) / "train_data.csv"), index=False)
# test_df.to_csv((Path(args.test_data) / "test_data.csv"), index=False)

# # Stop Logging
# mlflow.end_run()


# import logging
# import mlflow
# import pandas as pd
# from pathlib import Path
# import os
# from sklearn.model_selection import train_test_split
# from azureml.fsspec import AzureMachineLearningFileSystem
# from azure.ai.ml import MLClient
# from azure.identity import DefaultAzureCredential

# # Start Logging
# mlflow.start_run()

# # Azure Blob Storage setup
# ml_client = MLClient.from_config(credential=DefaultAzureCredential())
# uri = "azureml://subscriptions/173ef898-5b47-4b9f-af61-8086de765cf8/resourcegroups/default_resource_group/workspaces/test_workspace_azure_ml/datastores/workspaceblobstore"

# fs = AzureMachineLearningFileSystem(uri)
# paths = fs.glob('/LocalUpload/fbcd3c1fe9cde162b392dbeb1238f9ac/data/*.csv')

# # Read all CSV files from Azure Blob Storage into a single DataFrame
# df = pd.concat((pd.read_csv(fs.open(path)) for path in paths))

from azureml.core import Workspace, Dataset, Datastore

subscription_id = '173ef898-5b47-4b9f-af61-8086de765cf8'
resource_group = 'default_resource_group'
workspace_name = 'test_workspace_azure_ml'

workspace = Workspace(subscription_id, resource_group, workspace_name)

datastore = Datastore.get(workspace, "workspaceblobstore")
dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'LocalUpload/fbcd3c1fe9cde162b392dbeb1238f9ac/data/carmpg.csv'))
df = dataset.to_pandas_dataframe()

# Split the DataFrame into train and test sets
train_df, test_df = train_test_split(df, test_size=0.3, random_state=4)

# Ensure the output directories exist
train_data_path = 'train_data'
test_data_path = 'test_data'
os.makedirs(train_data_path, exist_ok=True)
os.makedirs(test_data_path, exist_ok=True)

# Save the train and test DataFrames to CSV files
train_df.to_csv(Path(train_data_path) / "train_data.csv", index=False)
test_df.to_csv(Path(test_data_path) / "test_data.csv", index=False)

# Stop Logging
mlflow.end_run()
