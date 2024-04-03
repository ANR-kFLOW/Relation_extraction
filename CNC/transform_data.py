import pandas as pd
df=pd.read_csv('/data/Youss/zephyr/Prevents/prevent_f.csv')
# Apply transformation and create new columns
df['index'] = df.apply(lambda row: f"{'prevent'}_{'DA'}_{'0'}_{'0'}", axis=1)
df['corpus'] = 'AD'
df['causal_text_w_pairs'] = df['text']
df['num_rs'] = 1
df['doc_id']='id'
# df['sent_id']=df['sentence'].apply(lambda x: hash(x))
df['sent_id']=0
df['text']=df['text'].str.replace(r'<ARG\d+>', '').str.replace(r'</ARG\d+>', '').str.replace(r'</SIG\d+>', '').str.replace(r'<SIG\d+>', '')
df['eg_id']=0
# df['label']=df['relation']
# df['text'] = df['text'].str.replace('___', '')
# df = df[~df['text'].str.contains('none')]
# print(df)


# Select the desired columns in the final transformed DataFrame
final_columns = ['corpus', 'doc_id', 'sent_id', 'eg_id', 'index', 'text', 'causal_text_w_pairs', 'num_rs']
transformed_df = df[final_columns]

# Print the transformed DataFrame

# Filter rows where causal_text_w_pairs contain either <ARG0> or <ARG1> tag, but not both

filtered_rows = transformed_df[
    (transformed_df['causal_text_w_pairs'].str.count('<ARG0>') == 0) |
    (transformed_df['causal_text_w_pairs'].str.count('<ARG1>') == 0)
]
print(filtered_rows)
# filtered_rows.to_csv('filterd.csv')
# Get the indices of the rows to be removed
indices_to_remove = filtered_rows.index

# Drop the rows from the transformed_df DataFrame
transformed_df = transformed_df.drop(indices_to_remove)
#
transformed_df.to_csv(('prevent_A.csv'))
