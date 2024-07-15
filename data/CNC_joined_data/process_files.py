import os
import pandas as pd


## Define the path to the main directory
#main_directory = '/data/Youss/RE/Relation_extraction/data/Joined_data'
#
## Iterate through each folder in the main directory
#for root, dirs, files in os.walk(main_directory):
#    for file in files:
#        # Check if the file is a CSV file
#        if file.endswith('.csv'):
#            # Construct the full file path
#            file_path = os.path.join(root, file)
#
#            # Read the CSV file into a DataFrame
#            df = pd.read_csv(file_path)
#
#            # Check if the 'relation' column exists and drop it
#            if 'relation' in df.columns:
#                df.drop(columns=['relation'], inplace=True)
#
#            # Drop rows where the 'num_rs' column is equal to 0
#            if 'num_rs' in df.columns:
#                df = df[df['num_rs'] != 0]
#
#            # Construct the new file name with '_without_relations'
#            new_file_name = file.replace('.csv', '_without_relations.csv')
#            new_file_path = os.path.join(root, new_file_name)
#
#            # Save the modified DataFrame to the new file
#            df.to_csv(new_file_path, index=False)
#
#print("All files have been processed and saved with '_without_relations' in their names.")


# Define the file paths
#train_path = "/data/Youss/RE/Relation_extraction/data/Joined_data/News_data_augmented/sequence_classification/train_augmented_sequence_classification_without_relations.csv"
#validation_path = "/data/Youss/RE/Relation_extraction/data/Joined_data/News_data_augmented/sequence_classification/validation_augmented_sequence_classification_without_relations.csv"
##test_path = "/data/Youss/RE/Relation_extraction/data/Joined_data/News_data/sequence_classification/test_sequence_classification.csv"
#
## Define the label mapping
#label_mapping = {
#    '0': 0,
#    'cause': 1,
#    'enable': 2,
#    'intend': 3,
#    'prevent': 4
#}
#
## Function to transform labels in a given file
#def transform_labels(file_path):
#    df = pd.read_csv(file_path)
#
#    # Apply the label mapping
#    df['label'] = df['label'].map(label_mapping)
#
#    # Save the transformed DataFrame back to the same file
#    df.to_csv(file_path, index=False)
#    print(f"Transformed labels and saved: {file_path}")
#
## Transform labels in each file
#transform_labels(train_path)
#transform_labels(validation_path)
##transform_labels(test_path)

#import os
#import pandas as pd
#
## Source and destination directories
#src_dir = '/data/Youss/RE/Relation_extraction/data/Joined_data/News_data/sequence_classification'
#dst_dir = '/data/Youss/RE/Relation_extraction/data/Joined_data/News_data/sequence_classification/without_0'
#
## Create the destination directory if it doesn't exist
#os.makedirs(dst_dir, exist_ok=True)
#
## Iterate over all files in the source directory
#for filename in os.listdir(src_dir):
#    if filename.endswith('.csv'):
#        # Read the CSV file
#        src_filepath = os.path.join(src_dir, filename)
#        df = pd.read_csv(src_filepath)
#
#        # Remove rows where the label is 0
#        df_filtered = df[df['label'] != 0]
#
#        # Save the cleaned DataFrame to the destination directory
#        dst_filepath = os.path.join(dst_dir, filename)
#        df_filtered.to_csv(dst_filepath, index=False)
#
#print("Processing complete!")
import os
import pandas as pd

# Destination directory
dst_dir = '/data/Youss/RE/Relation_extraction/data/Joined_data/News_data/sequence_classification/without_0'

# Iterate over all files in the destination directory
for filename in os.listdir(dst_dir):
    if filename.endswith('.csv'):
        # Read the CSV file
        filepath = os.path.join(dst_dir, filename)
        df = pd.read_csv(filepath)
        
        # Replace each value in the label column by subtracting 1
        df['label'] = df['label'] - 1
        
        # Save the modified DataFrame back to the same directory
        df.to_csv(filepath, index=False)

print("Label adjustment complete!")
