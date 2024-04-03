import pandas as pd
import re
import os
import csv
folder_path = '/data/Youss/RE/new_data/our_previous_annotations'

 # Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

 # Print the length of each CSV file
for csv_file in csv_files:
     file_path = os.path.join(folder_path, csv_file)
     with open(file_path, 'r') as file:
         csv_data = file.readlines()
         print(f"Length of {csv_file}: {len(csv_data)}")
#
# csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
# columns_to_keep = ['corpus', 'doc_id', 'sent_id', 'eg_id', 'index', 'text', 'causal_text_w_pairs', 'num_rs']
#
# # Process each CSV file
# for csv_file in csv_files:
#     file_path = os.path.join(folder_path, csv_file)
#
#     # Read the CSV file
#     with open(file_path, 'r') as file:
#         reader = csv.DictReader(file)
#         header = reader.fieldnames
#
#         # Keep only the specified columns
#         data_to_write = [{col: row[col] for col in columns_to_keep} for row in reader]
#
#     # Write the modified data back to the same file
#     with open(file_path, 'w', newline='') as file:
#         writer = csv.DictWriter(file, fieldnames=columns_to_keep)
#
#         # Write the header
#         writer.writeheader()
#
#         # Write the modified data
#         writer.writerows(data_to_write)
#
#     print(f"Processed {csv_file}")

#folder_path = '/data/Youss/RE/data/prob_avg_typ_data'  # Your folder path
#
## Function to process CSV files
#def process_csv(file_path):
#    # Read CSV file
#    df = pd.read_csv(file_path)
#
#    # Check if 'causal_text_w_pairs' column exists
#    if 'causal_text_w_pairs' in df.columns:
#        # Define a function to handle replacements within text
#        def add_brackets(text):
#            return f'["{text}"]'
#
#        def replace_and_modify(text):
#            # Remove all single and double quotes
#            modified_text = re.sub(r'[\'\\"]', '', text)
#            # Add a double quote after the opening bracket and before the closing bracket
#            modified_text = re.sub(r'\[', '[\"', modified_text)
#            modified_text = re.sub(r'\]', '\"]', modified_text)
#
#            return modified_text
#
#        def remove_outer_empty_lists(text):
#            # Define a pattern to match ["content"] and capture the content inside the brackets
#            pattern = r'\["([^"]*)"\]'
#            # Replace occurrences of ["content"] with the captured content inside the brackets
#            return re.sub(pattern, r'\1', text)
#
#        # Apply the replacement function to the 'causal_text_w_pairs' column
#        df['causal_text_w_pairs'] = df['causal_text_w_pairs'].apply(replace_and_modify)
#        df['causal_text_w_pairs'] = df['causal_text_w_pairs'].apply(lambda x: x.replace('[""]', '[]'))
#        df['causal_text_w_pairs'] = df['causal_text_w_pairs'].apply(lambda x: x.replace('\'|`', ''))
#        df['causal_text_w_pairs'] = df['causal_text_w_pairs'].apply(remove_outer_empty_lists)
#        df['causal_text_w_pairs'] = df['causal_text_w_pairs'].apply(lambda x: x.replace('\"\"', '\"'))
#        df['causal_text_w_pairs'] = df['causal_text_w_pairs'].apply(add_brackets)
#        df = df[(df['causal_text_w_pairs'].str.count('<ARG0>') != 0) & (df['causal_text_w_pairs'].str.count('<ARG1>') != 0)]
#
#        # Save the modified dataframe back to the CSV file
#        df.to_csv(file_path, index=False)
#
## Iterate through files in the folder
#for file in os.listdir(folder_path):
#    if file.endswith('.csv'):
#        file_path = os.path.join(folder_path, file)
#        process_csv(file_path)
