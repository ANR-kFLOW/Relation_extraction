import pandas as pd
#final_columns = ['corpus', 'doc_id', 'sent_id', 'eg_id', 'index', 'text', 'causal_text_w_pairs', 'num_rs']
file1=pd.read_csv('/data/Youss/CausalNewsCorpus/data/V2/train_subtask2_grouped.csv')
file2=pd.read_csv('/data/Youss/RE/new_data/new_train_CS_GPT2.csv')


#file1=pd.read_csv('/data/Youss/RE/new_data/our_previous_annotations/enables.csv')
#file2=pd.read_csv('/data/Youss/RE/new_data/our_previous_annotations/prevents.csv')
#file3=pd.read_csv('/data/Youss/RE/new_data/our_previous_annotations/intends.csv')
#file4=pd.read_csv('/data/Youss/RE/new_data/our_previous_annotations/intend_left.csv')
#
#file2=pd.read_csv('/data/Youss/RE/new_data/our_previous_annotations/our_data/validation.csv')
#file3=pd.read_csv('/data/Youss/RE/new_data/our_previous_annotations/prevents_stud.csv')
#negative=pd.read_csv('/data/Youss/RE/new_data/our_previous_annotations/negative_Samples.csv')
#negative = negative.head(72)
#atomic=pd.read_csv('/data/Youss/CausalNewsCorpus/data/V2/train_atomic.csv')
#print(len(atomic))
#atomic = atomic[~atomic['text'].str.contains('NONE')]
#atomic=atomic[final_columns]
#print(file2)
## file3=pd.read_csv('/Users/youssrarebboud/Desktop/RE/scripts/train_mixed_data.csv')

# Concatenate the dataframes
merged_data = pd.concat([file1, file2], ignore_index=True)

#merged_data=merged_data[final_columns]
print(len(merged_data))
#merged_data['num_rs'] = merged_data.apply(lambda row: 1 if 'cnc' not in str(row['index']) else row['num_rs'], axis=1)# # Save the merged data to a new CSV file
merged_data.to_csv('/data/Youss/RE/new_data/new_train_CS_GPT3.csv', index=False)



# # Randomly select 1000 samples
# sampled_data = merged_data.sample(n=1000, random_state=42)  # Set a seed for reproducibility
# print(sampled_data)
# # Save the sampled data to a new CSV file
# sampled_data.to_csv('sampled_data.csv', index=False)
