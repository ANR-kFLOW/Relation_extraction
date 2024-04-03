import pandas as pd

# Read the two CSV files
# df1 = pd.read_csv('/data/Youss/RE/data/prob_max_typ_data/dev_our_data_plus_atomic.csv')
# df2 = pd.read_csv('/data/Youss/CausalNewsCorpus/data/V2/train_mixed_data.csv')
#
# # Get the common columns
# common_columns = list(set(df1.columns) & set(df2.columns))
#
# # Keep only the common columns in df1
# df1 = df1[common_columns]
# #
# # # Save the result back to the file
# # df1.to_csv('/data/Youss/RE/data/prob_max_typ_data/dev_our_data_plus_atomic_filtered_columns.csv', index=False)
# print(df1.columns)


# Load the dataframes
# train_our_dt_with_sim = pd.read_csv('/data/Youss/RE/data/prob_avg_typ_data/train_our_data.csv')
# non_common_sense_data = pd.read_csv('/data/Youss/RE/data/archived/our_data_train.csv')
# # train_our_dt_with_sim['relation']=train_our_dt_with_sim['']
# non_common_sense_data['relation']=non_common_sense_data['index'].str.split('_').str[0]
# # Create a new dataframe to store the updated data
# updated_non_common_sense_data = non_common_sense_data.copy()
#
# # Assuming the common column is 'relation'
# common_column = 'relation'
#
# # Initialize indices
# train_index = 0
# non_common_sense_index = 0
#
# # Iterate over rows in train_our_dt_with_sim
# for index, row in train_our_dt_with_sim.iterrows():
#     # Get the relation and Max_Similarity from train_our_dt_with_sim
#     relation = row['relation']
#     max_similarity = row['Average_Similarity']
#
#     # Find all occurrences of the relation in updated_non_common_sense_data
#     match_indices = updated_non_common_sense_data.index[updated_non_common_sense_data[common_column] == relation].tolist()
#
#     # Check if there are matches
#     if match_indices:
#         # Iterate through all matches and add Max_Similarity to the updated dataframe
#         for match_index in match_indices:
#             updated_non_common_sense_data.loc[match_index, 'Average_Similarity'] = max_similarity
#
#         # Update the indices for both dataframes
#         train_index = index + 1
#         non_common_sense_index = match_indices[-1] + 1
#
#     else:
#         # If no match is found, break the loop
#         break
#
# # Save the updated dataframe to a new CSV file with the original indexes
# updated_non_common_sense_data.to_csv('/data/Youss/RE/data/prob_avg_typ_data/updated_our_data.csv', index=True)
#

original_test=pd.read_csv('/Users/youssrarebboud/Documents/GitHub/Prompts Based Data Augmentation for Event relation extraction/Prompt-based-data-augmentation-for-event-and-event-relation-classification/data/original_test.csv')
# import pandas as pd
# from nltk import word_tokenize
#
 # Assuming your DataFrame is called original_test
 # Replace the column names as needed based on your actual DataFrame structure

 # Function to find the indices of the tokens corresponding to trigger1 and effect
def find_event_indices(tags):
     tokens = word_tokenize(tags)
     trigger_indices = [i for i, tag in enumerate(tokens) if tag == 'trigger1']
     effect_indices = [i for i, tag in enumerate(tokens) if tag == 'effect']
     return trigger_indices, effect_indices

 # Function to extract words from the given indices
def extract_words(sentence, indices):
     tokens = word_tokenize(sentence)
     return ' '.join(tokens[i] if 0 <= i < len(tokens) else '' for i in indices)

 # Apply the function row-wise to find event indices and filter rows based on conditions
def process_row(row):
     trigger_indices, effect_indices = find_event_indices(row['tag_2'])

     # Discard rows with no trigger1 or effect or with non-successive trigger1 or effect
     if not trigger_indices or not effect_indices or (
             max(trigger_indices) - min(trigger_indices) != len(trigger_indices) - 1) or (
             max(effect_indices) - min(effect_indices) != len(effect_indices) - 1):
         return pd.Series([None, None])

     # Extract words corresponding to event1 and event2 indices
     event1_words = extract_words(row['sentence'], trigger_indices)
     event2_words = extract_words(row['sentence'], effect_indices)

     # Return the words corresponding to event1 and event2
#     return pd.Series([event1_words, event2_words])
#
# # Apply the function to each row of the DataFrame
# original_test[['event1', 'event2']] = original_test.apply(process_row, axis=1)
#
 # Drop rows where event1 and event2 are not both present
original_test = original_test.dropna(subset=['event1', 'event2'])
print(original_test)
 # Save the DataFrame to a new CSV file
original_test.to_csv('processed_data_tst.csv', index=False)
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# # Read the CSV file
# file_path = '/data/Youss/RE/refining/scripts/parsed_complete_train_atomic_tmp.csv'
# file = pd.read_csv(file_path)
# print(file)
# # Split the data into train, dev, and test sets
# train, dev_test = train_test_split(file, test_size=0.3, random_state=42)
# dev, test = train_test_split(dev_test, test_size=0.5, random_state=42)
# print(len(train), len(dev), len(test))
# # Define the new file paths
# train_path = '/data/Youss/RE/data/archived/train.csv'
# dev_path = '/data/Youss/RE/data/archived/dev.csv'
# test_path = '/data/Youss/RE/data/archived/test.csv'
# final_columns = ['corpus', 'doc_id', 'sent_id', 'eg_id', 'index', 'text', 'causal_text_w_pairs', 'num_rs']
# train = train[final_columns]
# test = test[final_columns]
# dev = dev[final_columns]
# # Save the split DataFrames to the new CSV files
# train.to_csv(train_path, index=False)
# dev.to_csv(dev_path, index=False)
# test.to_csv(test_path, index=False)
# import pandas as pd
# import random
# # Load the CSV file
#file_path = '/data/Youss/RE/data/thres_max_typ_data/train_our_data_sampled_threshold.csv'
#df = pd.read_csv(file_path)
#
# # Get unique values in the 'index' column
# unique_values = df['index'].unique()
#
# # Set the random seed for reproducibility
# random.seed(42)
#
# # Initialize an empty DataFrame to store the sampled data
# sampled_data = pd.DataFrame()
#
# # Sample an equal number of rows for each unique value in the 'index' column
# for value in unique_values:
#     value_rows = df[df['index'] == value]
#     sampled_rows = value_rows.sample(n=6512 // len(unique_values), random_state=42)
#     sampled_data = pd.concat([sampled_data, sampled_rows])
#
# # Specify the output path
# output_path = '/data/Youss/RE/data/filledvsnot_filled/equal_sampled_train_atomic.csv'
#
# # Save the equal sampled data to a new CSV file
# sampled_data.to_csv(output_path, index=False)
#
# print(f"Equal sampled data saved to {output_path}")
# import pandas as pd
# import random
#
# # Load the dev_atomic.csv file
# dev_file_path = '/data/Youss/CausalNewsCorpus/dev_atomic.csv'
# dev_df = pd.read_csv(dev_file_path)
#
# # Get unique values in the 'index' column
# unique_values_dev = dev_df['index'].unique()
#
# # Set the random seed for reproducibility
# random.seed(42)
#
# # Initialize an empty DataFrame to store the sampled data for dev
# dev_sampled_data = pd.DataFrame()
#
# # Sample 1550 rows for dev
# for value in unique_values_dev:
#     value_rows = dev_df[dev_df['index'] == value]
#     sampled_rows = value_rows.sample(n=1550 // len(unique_values_dev), random_state=42)
#     dev_sampled_data = pd.concat([dev_sampled_data, sampled_rows])
#
# # Specify the output path for dev sampled data
# dev_output_path = '/data/Youss/RE/data/filledvsnot_filled/dev_sampled.csv'
#
# # Save the dev sampled data to a new CSV file
# dev_sampled_data.to_csv(dev_output_path, index=False)
#
# print(f"Dev sampled data saved to {dev_output_path}")
#
# # Initialize an empty DataFrame to store the sampled data for test
# test_sampled_data = pd.DataFrame()
#
# # Sample 1075 rows for test
# for value in unique_values_dev:
#     sampled_rows = value_rows.sample(n=1075 // len(unique_values_dev), random_state=42)
#     test_sampled_data = pd.concat([test_sampled_data, sampled_rows])
#
# # Specify the output path for test sampled data
# test_output_path = '/data/Youss/RE/data/filledvsnot_filled/test_sampled.csv'
#
# # Save the test sampled data to a new CSV file
# test_sampled_data.to_csv(test_output_path, index=False)
#
# print(f"Test sampled data saved to {test_output_path}")
df = pd.read_csv('/data/Youss/RE/data/archived/dev_atomic.csv')
print(len(df))
df = df[~df['text'].str.contains('NONE')]
print(len(df))
#df = df[~df['index'].str.contains('train')]
d_tmp=pd.DataFrame()
d_tmp['causal_text_w_pairs'] = df['causal_text_w_pairs'].apply(eval)
# #
# #
# # Calculate the length of the lists and store in a new column
df['num_rs'] = d_tmp['causal_text_w_pairs'].apply(len)
final_columns = ['corpus', 'doc_id', 'sent_id', 'eg_id', 'index', 'text', 'causal_text_w_pairs', 'num_rs']
df=df[final_columns]
df['doc_id'] = ['train' + str(i) for i in range(len(df))]

# # result_df=df.groupby('num_rs', group_keys=False).apply(lambda x: x.sample(50))
#d_tmp['doc_id'] = df['doc_id'].astype(str)
d_tmp['sent_id'] = df['sent_id'].astype(str)
#d_tmp['index'] = df['index'].astype(str)
# # Add an incremental number to every row
d_tmp['incremental_number'] = range(1, len(d_tmp) + 1)
#
#
# # Add the incremental number to 'doc_id' and 'sent_id'
#df['doc_id'] = d_tmp['doc_id'] + d_tmp['incremental_number'].astype(str)
df['sent_id'] = d_tmp['sent_id'] + d_tmp['incremental_number'].astype(str)
#df['index'] = d_tmp['index'] + d_tmp['incremental_number'].astype(str)
## # import re
## # # Define the pattern to match '[' or ']' or single quotes
## # pattern = re.compile(r"\[|\]|'")
## #
## # # Function to process a single string
## # def process_string(s):
## #     return "['" + re.sub(pattern, "", s) + "']"
## #
## # # Apply the modification to the whole column
## # df.loc[df['num_rs'] == 1, 'causal_text_w_pairs'] = df.loc[df['num_rs'] == 1, 'causal_text_w_pairs'].apply(process_string)
## #
## # for index, row in df.iterrows():
## #     if row['num_rs'] == 0:
## #         df.at[index, 'causal_text_w_pairs'] = []
## #
## # def process_string(s):
## #     # Your processing logic here
## #     return s.replace("['\"", "['").replace("\"']", "']")
## #
## # # Assuming df is your DataFrame
## # # Apply process_string and the additional replacements only when num_rs is equal to 1
## # df.loc[df['num_rs'] == 1, 'causal_text_w_pairs'] = df.loc[df['num_rs'] == 1, 'causal_text_w_pairs'].apply(process_string)
## # Replace '[' and ']' with '[\' and \']'
## # df.loc[df['num_rs'] == 1, 'causal_text_w_pairs'] = df.loc[df['num_rs'] == 1, 'causal_text_w_pairs'].str.replace('\[', "['").str.replace('\]', "']")
## # df.loc[df['num_rs'] == 1, 'causal_text_w_pairs'] = df.loc[df['num_rs'] == 1, 'causal_text_w_pairs'].str.replace('\["', "['")
## # df.loc[df['num_rs'] == 1, 'causal_text_w_pairs'] = df.loc[df['num_rs'] == 1, 'causal_text_w_pairs'].str.replace('"\]', "']")
##
## # Convert back to integers if needed
#df['doc_id'] = df['doc_id'].astype(str)
#df['sent_id'] = df['sent_id'].astype(str)
#df['index'] = df['index'].astype(str)
#df['eg_id'] = df['eg_id'].astype(str)
#df['causal_text_w_pairs'] = df['causal_text_w_pairs'].astype(str)
#df['corpus'] = df['corpus'].astype(str)



df.to_csv('/data/Youss/RE/data/atomic_dev.csv', index=False)
#for index, row in df.iterrows():
#    if row['index'].startswith('0'):
#        df.at[index, 'causal_text_w_pairs'] = []
#        df.at[index, 'num_rs'] = 0
#df.to_csv(file_path, index=False)
df_train=pd.read_csv('/data/Youss/RE/data/atomic_train.csv')
combined_df = pd.concat([df, df_train], ignore_index=True)
print(len(combined_df))
combined_df.to_csv('atomic_complete.csv', index=False)
