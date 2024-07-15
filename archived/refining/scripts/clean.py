import pandas as pd
import re
import spacy
# file=pd.read_csv('/Users/youssrarebboud/Desktop/RE/refining/train_typed_our_data.csv')
#
# file['relation']=file['index']
#
# print(len(file))
# pattern = r".*Event\s*[12] Type: (.+)"
#
#
# # Create a function to apply the replacement to each row
# def replace_event_type(row):
#     if pd.notna(row["Event1_Type"]):
#         match = re.search(pattern, row["Event1_Type"])
#         if match:
#             row["Event1_Type"] = match.group(1)
#
#     if pd.notna(row["Event2_Type"]):
#         match = re.search(pattern, row["Event2_Type"])
#         if match:
#             row["Event2_Type"] = match.group(1).replace(">", "").split(")")[0]
#
#     return row
#
# file = file.apply(replace_event_type, axis=1)
#
# # Assuming you have the file dataframe
#
# for index, row in file.iterrows():
#     if ')' in str(row["Event1_Type"]):
#         file.at[index, "Event1_Type"] = row["Event1_Type"].split(')')[0]
#
#     if ')' in str(row["Event2_Type"]):
#         file.at[index, "Event2_Type"] = row["Event2_Type"].split(')')[0]
#
# # Save the modified dataframe as a CSV file

# file2=pd.read_csv('/data/Youss/RE/refining/GT_atomic/typed_atomic_train.csv')
# # # Convert the index in file to strings and extract the string part
# file.index = file.index.astype(str)
# # # # Extract the alphanumeric part from the index in file
# # # # Extract the text before the first underscore (_) in the index in file
# # # # Extract the text before the first underscore (_) in the "index" column
# file['stripped_index'] = file['index'].str.split('_').str[0]
# file['relation']=file['stripped_index']
# # # Create the type_relation column
# file['type_relation'] =  [str(a) + ' '+ str(b)  + ' '+str(c) for a, b,c in zip(file['Event1_Type'], file['relation'], file['Event2_Type'])]
# #
# file.to_csv('train_typed_our_data.csv', index=False)
# file2.to_csv('/data/Youss/RE/refining/GT_atomic/typed_atomic_train.csv', index=False)

# df=pd.read_csv('/Users/youssrarebboud/Documents/GitHub/Prompts Based Data Augmentation for Event relation extraction/Prompt-based-data-augmentation-for-event-and-event-relation-classification/data/original_test.csv')
# print(df.columns)
# file2=pd.read_csv('/Users/youssrarebboud/Desktop/CausalNewsCorpus/data/V2/dev_atomic.csv')
# print(len(file2))
#
# # Task 1: Filter rows with 'none' in the "text" column
# filtered_data = file2[file2['text'].str.lower().str.contains('none') == False]
#
# # Task 2: Randomly select 1600 samples while preserving the same portion in the "index" column
# total_samples = len(filtered_data)
# sample_size = 1600
#
# # Calculate the portion to maintain
# portion_to_maintain = sample_size / total_samples
#
# # Use the "sample" method with a specified portion
# random_samples = filtered_data.sample(frac=portion_to_maintain, random_state=42)
#
# # Ensure you have exactly 1600 samples (in case the portion calculation is not exact)
# if len(random_samples) < sample_size:
#     additional_samples = filtered_data.sample(n=sample_size - len(random_samples), random_state=42)
#     random_samples = pd.concat([random_samples, additional_samples])
#
# # Reset the index
# random_samples.reset_index(drop=True, inplace=True)
#
# # Now, 'random_samples' contains the randomly selected 1600 samples without 'none' in the "text" column
# print(random_samples)
#
# random_samples.to_csv('test_atomic_filtered_sampled.csv')
# Function to extract events

# df= pd.read_csv('/data/Youss/RE/data/prob_avg_typ_data/train_our_data.csv')
# # Filter rows where the 'sentence' column is not empty
# df = df[df['sentence'].notna()]
#
# # df=pd.read_csv('/Users/youssrarebboud/Desktop/RE/refining/scripts/typed_atomic_train.csv')
# nlp = spacy.load("en_core_web_sm")
# # def get_root(sentence):
# #     doc = nlp(sentence)
# #     for token in doc:
# #         if token.dep_ == "ROOT":
# #             return token.lemma_
# #     return None
#
# def get_root(sentence):
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(sentence)
#     for token in doc:
#         if token.dep_ == "ROOT" :
#             return token.text
#     return None
#
# def extract_verbs_with_spacy(sentence):
#
#     # Process the sentence with spaCy
#     doc = nlp(sentence)
#
#     # Extract verbs from the sentence
#     verbs = [token.text for token in doc if token.pos_ == "VERB"]
#     if verbs:
#         return verbs[0]
#     else:
#         print('no verb')
#         print(sentence)
#         return sentence
#
#
#
# #
# def add_backslashes(text):
#     return re.sub(r'[\'"]', ' ', text)
# # # Function to transform the text
# def transform_text(text, event1, event2):
#
#     text=add_backslashes(text)
#
#     transformed_text = re.sub(re.escape(event1), f'<ARG0>{event1}</ARG0>', text, flags=re.IGNORECASE)
#     transformed_text = re.sub(re.escape(event2), f'<ARG1>{event2}</ARG1>', transformed_text, flags=re.IGNORECASE)
#
#     transformed_text=transformed_text.strip()
#
#
#     return f'["{transformed_text}"]'
# #
# # Apply transformation and create new columns
# df['index'] = df.apply(lambda row: f"{row['relation']}_{'our_data'}_{'0'}_{'0'}", axis=1)
# df['corpus'] = 'atomic'
# df['causal_text_w_pairs'] = df.apply(lambda row: transform_text(row['sentence'], get_root(row['event1']), get_root(row['event2'])), axis=1)
# df['num_rs'] = 1
# df['doc_id']='id'
# # df['sent_id']=df['sentence'].apply(lambda x: hash(x))
# df['sent_id']=0
# df['text']=df['Text']
# df['eg_id']=0
# # df['label']=df['relation']
# df['text'] = df['text'].str.replace('___', '')
# # df = df[~df['text'].str.contains('none')]
# # print(df)
#
#
# # Select the desired columns in the final transformed DataFrame
# final_columns = ['corpus', 'doc_id', 'sent_id', 'eg_id', 'index', 'text', 'causal_text_w_pairs', 'num_rs']
# transformed_df = df[final_columns]
#
# # Print the transformed DataFrame
#
# # Filter rows where causal_text_w_pairs contain either <ARG0> or <ARG1> tag, but not both
#
# filtered_rows = transformed_df[
#     (transformed_df['causal_text_w_pairs'].str.count('<ARG0>') == 0) |
#     (transformed_df['causal_text_w_pairs'].str.count('<ARG1>') == 0)
# ]
# print(filtered_rows)
# # filtered_rows.to_csv('filterd.csv')
# # Get the indices of the rows to be removed
# indices_to_remove = filtered_rows.index
#
# # Drop the rows from the transformed_df DataFrame
# transformed_df = transformed_df.drop(indices_to_remove)
# #
# transformed_df.to_csv(('atomic_train_filtered.csv'))
# print('transformed_df',transformed_df)
# transformed_df.to_csv('/data/Youss/RE/data/prob_avg_typ_data/train_our_data_transformed.csv')

# Create a new DataFrame to store results for each row
# results_df = pd.DataFrame(columns=['corpus', 'doc_id', 'sent_id', 'eg_id', 'index', 'text', 'causal_text_w_pairs', 'num_rs'])
#
# # Iterate through each row
# for index, row in df.iterrows():
#     # Apply transformation and create new columns for the current row
#     row['index'] = f"{row['label']}_{'our_data'}_{'0'}_{'0'}"
#     row['corpus'] = 'atomic'
#     row['causal_text_w_pairs'] = transform_text(row['sentence'], get_root(row['event1']), get_root(row['event2']))
#     row['num_rs'] = 0
#     row['doc_id'] = 'id'
#     row['sent_id'] = 0
#     row['eg_id'] = 0
#     row['text'] = row['sentence'].replace('___', '')
#
#     # Check if the current row meets the filtering criteria
#     if (row['causal_text_w_pairs'].count('<ARG0>') != 0) and (row['causal_text_w_pairs'].count('<ARG1>') != 0):
#         # Save the result for the current row in the results DataFrame
#         results_df = results_df.append(row, ignore_index=True)
#
#         results_df.to_csv('transformed_test.csv', index=False)
train_our_dt_with_sim=pd.read_csv('/data/Youss/RE/data/prob_avg_typ_data/updated_our_data.csv')
# Split the DataFrame into three based on the "relation" column
print(train_our_dt_with_sim.columns)
train_our_dt_with_sim['relation']=train_our_dt_with_sim['relation']
cause_df = train_our_dt_with_sim[train_our_dt_with_sim['relation'] == 'cause']
intend_df = train_our_dt_with_sim[train_our_dt_with_sim['relation'] == 'intend']
rest_df = train_our_dt_with_sim[~train_our_dt_with_sim['relation'].isin(['cause', 'intend'])]
# atomic=pd.read_csv('/data/Youss/RE/data/archived/test.csv')
def weighted_sample(df):
    # Filter the rows where Max_Similarity is less than 0.8
    filtered_df = df[df['Average_Similarity'] < 0.8]

    # Calculate the percentage of rows that meet the condition
    percentage_negative_df = (len(filtered_df) / len(df)) * 100
    print(int(percentage_negative_df)/100)

    print(f"Percentage of rows with Max_Similarity < 0.8 in the df: {percentage_negative_df:.2f}%")
    weighted_sample = df.sample(n=int(len(df)*(1-int(percentage_negative_df)/100)), weights="Average_Similarity")

    return weighted_sample

def sample_threshold(df, threshold):
    return df[df['Max_Similarity'] < threshold]

#Sample by probability
# samples_df_cause=weighted_sample(cause_df)
# samples_df_intend=weighted_sample(intend_df)
# print(samples_df_cause)
# print(samples_df_intend)
#
#sample by threashold
samples_df_cause=weighted_sample(cause_df)
samples_df_intend=weighted_sample(intend_df)


# samples_df_cause = samples_df_cause[final_columns]
# samples_df_intend = samples_df_intend[final_columns]
# rest_df=rest_df[final_columns]
# Concatenate the three DataFrames
train_our_data_sampled_CI_thre = pd.concat([samples_df_intend, samples_df_cause, rest_df], ignore_index=True)
train_our_data_sampled_CI_thre['text']=train_our_data_sampled_CI_thre['sentence']
final_columns = ['corpus', 'doc_id', 'sent_id', 'eg_id', 'index', 'text', 'causal_text_w_pairs', 'num_rs']
# train_our_data_sampled_CI_thre = train_our_data_sampled_CI_thre[train_our_data_sampled_CI_thre['text'].notna()]
train_our_data_sampled_CI_thre=train_our_data_sampled_CI_thre[final_columns]
train_our_data_sampled_CI_thre['num_rs']=1
print(len(train_our_data_sampled_CI_thre))
train_our_data_sampled_CI_thre.to_csv('/data/Youss/RE/data/prob_avg_typ_data/train_our_data.csv', index=False)


