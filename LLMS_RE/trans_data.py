import os
import re
import pandas as pd

# Define the path to the directory containing the CSV files
input_directory = '/Users/youssrarebboud/Desktop/RE/Relation_extraction/LLMS_RE/data/'

# Define the function to extract text for subject and object
def extract_text(sentence):
    arg0 = re.search(r'<ARG0>(.*?)</ARG0>', sentence).group(1)
    arg1 = re.search(r'<ARG1>(.*?)</ARG1>', sentence).group(1)
    return arg0, arg1

# Define the function to extract label from index
def extract_label_from_index(index):
    if 'cnc' in index:
        return 'cause'
    label = index.split('_')[0]
    if label.endswith('s'):
        label = label[:-1]
    return label

# Process each CSV file in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        # Read the CSV file
        file_path = os.path.join(input_directory, filename)
        df = pd.read_csv(file_path)

        # Create new columns
        df['text'] = df['text']
        df['subject'], df['object'] = zip(*df['causal_text_w_pairs'].apply(extract_text))
        df['relation'] = df['index'].apply(extract_label_from_index)

        # Select only the required columns
        new_df = df[['text', 'subject', 'object', 'relation']]

        # Create new filename
        new_filename = os.path.splitext(filename)[0] + '_new.csv'
        new_file_path = os.path.join(input_directory, new_filename)

        # Save the new DataFrame to a CSV file
        new_df.to_csv(new_file_path, index=False)

print("Processing complete.")

# # Define the path to the directory containing the CSV files
# input_directory = '/Users/youssrarebboud/Desktop/RE/Relation_extraction/LLMS_RE/data/'
#
# # Function to remove <SIG0> and </SIG0> tags
# def remove_tags(text):
#     return re.sub(r'</?SIG0>', '', text)
#
# # Process each CSV file in the directory
# for filename in os.listdir(input_directory):
#     if filename.endswith('.csv'):
#         # Read the CSV file
#         file_path = os.path.join(input_directory, filename)
#         df = pd.read_csv(file_path)
#
#         # Remove <SIG0> and </SIG0> tags from all columns
#         for column in df.columns:
#             df[column] = df[column].astype(str).apply(remove_tags)
#
#         # Save the modified DataFrame back to the same CSV file
#         df.to_csv(file_path, index=False)
#
# print("Tag removal complete.")
