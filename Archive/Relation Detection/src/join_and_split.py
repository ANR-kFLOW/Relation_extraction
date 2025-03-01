import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Concatenate CSV files
folder_path = '/data/Youss/RE/Binary_classification'
all_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
dfs = [pd.read_csv(os.path.join(folder_path, file)) for file in all_files]
concatenated_df = pd.concat(dfs)

# Step 2: Set 'num_rs' to 1 where > 0
concatenated_df['num_rs'] = concatenated_df['num_rs'].apply(lambda x: 1 if x > 0 else 0)

# Step 3: Split into Train, Validation, and Test sets
train_val_df, test_df = train_test_split(concatenated_df, test_size=0.2, stratify=concatenated_df['num_rs'], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df['num_rs'], random_state=42)

# Optional: Save the split datasets to separate CSV files
train_df.to_csv('train.csv', index=False)
val_df.to_csv('validation.csv', index=False)
test_df.to_csv('test.csv', index=False)

