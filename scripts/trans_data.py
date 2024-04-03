import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
import spacy

## Load the English language model
#nlp = spacy.load("en_core_web_sm")
##nltk.download('punkt')
df1=pd.read_csv('/data/Youss/RE/new_data/our_previous_annotations/enables_stud.csv')
df2=pd.read_csv('/data/Youss/RE/new_data/our_previous_annotations/prevents_stud.csv')
df=pd.concat([df1, df2])
#
##df = df[df['label'] != 0]
##
##df['text'] = df['sentence'].apply(lambda x: re.sub(r'\"|\'|\'|\.|\$|\,', '', x))
##df['text'] = df['sentence'].apply(lambda x: re.sub(r'\"|\.', '', x))
##
#def extract_trigger_index(tag, trigger):
##    tokens = word_tokenize(tag)
##    tokens=tag.split()
#    tokens = tag.split()
#    try:
#        return tokens.index(trigger)
#    except ValueError:
#        return 0
##
### Extract index of trigger1
#df['trigger1_index'] = df['tag_2'].apply(lambda x: extract_trigger_index(x, 'trigger1'))
## Extract index of trigger2
#df['trigger2_index'] = df['tag_2'].apply(lambda x: extract_trigger_index(x, 'effect'))
#
## Function to extract trigger token
#def extract_trigger_token(text, index):
#    print(index)
#
##    words = word_tokenize(text)
##    words=text.split()
#    doc = nlp(text)
#    words = [token.text for token in doc]
#    print(len(words))
#    if int(index)<len(words):
#        return words[int(index)] if index is not None  else None
#    else:
#        return words[-1]
#
#
##for index, row in df.iterrows():
##    if row['index'].startswith('0'):
##        df.at[index, 'causal_text_w_pairs'] = []
##        df.at[index, 'num_rs'] = 0
#
#df['event1'] = df.apply(lambda row: extract_trigger_token(row['sentence'], row['trigger1_index']), axis=1)
#df['event2'] = df.apply(lambda row: extract_trigger_token(row['sentence'], row['trigger2_index']), axis=1)
#
#print(df)
#df.to_csv('/data/Youss/RE/data/our_train_val.csv', index=False)
#
#df.to_csv('original_data_tr.csv', index=False)
#import pandas as pd
#import re
#
#df=pd.read_csv('concatenated_output_with_labels.csv')
#df['label']=df['prevents']
#
#
#
#
# # Function to create text with tags
#def create_text_w_pair(row):
#     text = row['sentence']
#     event_one = row['event1'].lower()
#     event_two = row['event2'].lower()
#
#     text_with_tags = text.lower().replace(event_one, f'<ARG0>{event_one}</ARG0>').replace(event_two, f'<ARG1>{event_two}</ARG1>')
#
#     # Replace quotes in the middle of the text and wrap in square brackets
#     text_with_tags =  '[\'' + text_with_tags.replace('"', '').replace("'", '').strip() +'\']'
#
#     return text_with_tags
#
#
#df['causal_text_w_pairs'] = df.apply(create_text_w_pair, axis=1)
# # columns_to_remove = ['text_w_pair', 'Unnamed: 0.3', 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0','sentence']
# # df = df.drop(columns=columns_to_remove, errors='ignore')
# df['text']=df['sentence']
# df['num_rs']=0
# df['corpus']='youssra'
# print(df.columns)
# #
# print(df['num_rs'])
#
# # Function to check if both tags are present
# def has_both_tags(text):
#     return '<ARG0>' in text and '<ARG1>' in text
#
# # Mark rows for deletion if they don't have both tags
# rows_to_delete = []
# for index, row in df.iterrows():
#     if not has_both_tags(row['text_w_pairs']):
#         rows_to_delete.append(index)
#
# # Remove marked rows from the DataFrame
# df = df.drop(rows_to_delete)
#
# # Reset the index
# df.reset_index(drop=True, inplace=True)
#
# print(df)
#
#import pandas as pd
#import ast
#import re
#
## Sample data from your original dataset
#import pandas as pd
#import re
#
#import pandas as pd
#import pandas as pd
#import re
#
#
df=pd.read_csv("/data/Youss/RE/new_data/our_previous_annotations/GPT.csv")
def add_backslashes(text):
     return re.sub(r'[\'"]', '', str(text))


def transform_text(text, event1, event2):

     text=add_backslashes(text)

     transformed_text = re.sub(re.escape(event1), f'<ARG0>{event1}</ARG0>', text, flags=re.IGNORECASE)
     transformed_text = re.sub(re.escape(event2), f'<ARG1>{event2}</ARG1>', transformed_text, flags=re.IGNORECASE)
          
#     transformed_text = re.sub(re.escape(signal), f'<SIG0>{signal}</SIG0>', transformed_text, flags=re.IGNORECASE)

     transformed_text=transformed_text.strip()


     return f'["{transformed_text}"]'

 # Apply transformation and create new columns
df['index'] = df.apply(lambda row: f"{row['label']}_{'GPT'}_{'0'}_{'0'}", axis=1)
df['corpus'] = 'GPT3'
df['causal_text_w_pairs'] = df.apply(lambda row: transform_text(str(row['sentence']), str(row['trigger1']), str(row['trigger2'])), axis=1)

#df['causal_text_w_pairs'] ='[]'
df['sent_id']= 0

df['text']=df['sentence']
df['eg_id']=0
 # df['label']=df['relation']
df['text'] = df['text'].str.replace('___', '')
#df = df[~df['text'].str.contains('none')]
#df = df[~df['text'].str.contains('NONE')]
print(len(df))
#df = df[~df['index'].str.contains('train')]
d_tmp=pd.DataFrame()
d_tmp['causal_text_w_pairs'] = df['causal_text_w_pairs'].apply(eval)
# #
# #
# # Calculate the length of the lists and store in a new column
#df['num_rs'] = d_tmp['causal_text_w_pairs'].apply(len)
df['num_rs']=1
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
df['index'] = df['index'] + d_tmp['incremental_number'].astype(str)



 #
 # # Select the desired columns in the final transformed DataFrame
#final_columns = ['corpus', 'doc_id', 'sent_id', 'eg_id', 'index', 'text', 'causal_text_w_pairs', 'num_rs']
#transformed_df = df[final_columns]
#print('printing the atomic transformed columns')
#print(transformed_df.columns)
  # Print the transformed DataFrame
 
  # Filter rows where causal_text_w_pairs contain either <ARG0> or <ARG1> tag, but not both

transformed_df=df
filtered_rows = transformed_df[
      (transformed_df['causal_text_w_pairs'].str.count('<ARG0>') == 0) |
      (transformed_df['causal_text_w_pairs'].str.count('<ARG1>') == 0) | (transformed_df['causal_text_w_pairs'].str.count('<ARG0>') > 1)|
            (transformed_df['causal_text_w_pairs'].str.count('<ARG1>') > 1)
  ]
 
  # Get the indices of the rows to be removed
indices_to_remove = filtered_rows.index
for index, row in df.iterrows():
    if row['index'].startswith('0'):
        df.at[index, 'causal_text_w_pairs'] = []
        df.at[index, 'num_rs'] = 0
#  # Drop the rows from the transformed_df DataFrame
transformed_df = transformed_df.drop(indices_to_remove)
final_columns = ['corpus', 'doc_id', 'sent_id', 'eg_id', 'index', 'text', 'causal_text_w_pairs', 'num_rs']
transformed_df = transformed_df[final_columns]
print(transformed_df)
transformed_df.to_csv('/data/Youss/RE/new_data/our_previous_annotations/GPT_tr.csv',index=False)
