import pandas as pd
final_columns = ['corpus', 'doc_id', 'sent_id', 'eg_id', 'index', 'text', 'causal_text_w_pairs', 'num_rs']
file1=pd.read_csv('/data/Youss/RE/CausalNewsCorpus/data/AD/prevent_A.csv')
# file1=file1[final_columns]
file2=pd.read_csv('/data/Youss/RE/CausalNewsCorpus/data/AD/enbale_A.csv')

atomic=pd.read_csv('/data/Youss/RE/CausalNewsCorpus/data/AD/train_atomic.csv')
# print(len(atomic))
# atomic = atomic[~atomic['text'].str.contains('NONE')]
# atomic=atomic[final_columns]
# print(file2)
# # file3=pd.read_csv('/Users/youssrarebboud/Desktop/RE/scripts/train_mixed_data.csv')
#
# # Concatenate the dataframes
merged_data = pd.concat([file1, file2,atomic], ignore_index=True)

# merged_data=merged_data[final_columns]
# print(len(merged_data))
# merged_data['num_rs'] = merged_data.apply(lambda row: 1 if 'cnc' not in str(row['index']) else row['num_rs'], axis=1)# # Save the merged data to a new CSV file
merged_data.to_csv('/data/Youss/RE/CausalNewsCorpus/data/AD/Train_CS.csv', index=False)



# # Randomly select 1000 samples
# sampled_data = merged_data.sample(n=1000, random_state=42)  # Set a seed for reproducibility
# print(sampled_data)
# # Save the sampled data to a new CSV file
# sampled_data.to_csv('sampled_data.csv', index=False)
