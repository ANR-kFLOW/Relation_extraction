import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pandas as pd

def concatenate_csv_files(path):
    files = [file for file in os.listdir(path) if file.endswith('.csv')]
    if not files:
        print("No CSV files found in the given path.")
        return None
    
    dfs = []
    for file in files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    concatenated_df = pd.concat(dfs, ignore_index=True)
    return concatenated_df

# Provide the path where your CSV files are located
path = "/data/Youss/RE/new_data/our_previous_annotations/our_data"
concatenated_df = concatenate_csv_files(path)

if concatenated_df is not None:
    print(concatenated_df)
    # Optionally, you can save the concatenated dataframe to a new CSV file
    # concatenated_df.to_csv("concatenated_data.csv", index=False)
#
#df=pd.read_csv('/data/Youss/RE/new_data/our_previous_annotations/joined_original_transformed.csv')
concatenated_df['num_rs'] = concatenated_df['num_rs'].astype(int)
df=concatenated_df
# Create 'relation' column based on the first word in the index column before '_'
# Create 'relation' column based on the first word in the index column before '_'
df['relation'] = df['index'].apply(lambda x: x.split('_')[0])


# Splitting the DataFrame into train, test, and validation sets
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['relation'], random_state=42)
test_df, val_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['relation'], random_state=42)
train_df = train_df.drop(columns=['relation'])
test_df = test_df.drop(columns=['relation'])
val_df = val_df.drop(columns=['relation'])
# Displaying the shapes of train, test, and validation sets
print("Train set shape:", train_df.shape)
print("Test set shape:", test_df.shape)
print("Validation set shape:", val_df.shape)

folder_path = "/data/Youss/RE/new_data/our_previous_annotations/"
joined_augmented_df=pd.concat([test_df, val_df], ignore_index=True)
# Save the DataFrames to CSV files
train_df.to_csv(folder_path + "train_augmented.csv", index=False)
#test_df.to_csv(folder_path + "test_augmented.csv", index=False)
joined_augmented_df.to_csv(folder_path + "validation_augmented.csv", index=False)
