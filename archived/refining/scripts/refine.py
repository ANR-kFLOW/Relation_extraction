import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained BERT model and tokenizer on the GPU
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Load the common sense and non-common sense datasets
common_sense_data = pd.read_csv('/data/Youss/RE/refining/GT_atomic/typed_atomic_train.csv')
non_common_sense_data = pd.read_csv('/data/Youss/RE/data/mixed_dev_hparsed.csv')
# Filter rows where either 'Events2 Type' or 'Events1 Type' is not equal to 'empty'
non_common_sense_data = non_common_sense_data[(non_common_sense_data['Event1_Type'] != 'empty') | (non_common_sense_data['Event1_Type'] != 'empty')]
def create_relation(row):
    if 'cnc' in row['index'] and len(row['causal_text_w_pairs']) > 0:
        return 'cause'
    else:
        return row['index'].rstrip('0123456789_')

# Create the 'relation' column using the apply function with the custom function
non_common_sense_data['relation'] = non_common_sense_data.apply(create_relation, axis=1)
# non_common_sense_data['relation']=non_common_sense_data['label']
non_common_sense_data['type_relation'] =  [str(a) + ' '+ str(b)  + ' '+str(c) for a, b,c in zip(non_common_sense_data['Event1_Type'], non_common_sense_data['relation'], non_common_sense_data['Event2_Type'])]
print(non_common_sense_data['type_relation'])
# Group the datasets by their 'relation' column
common_sense_groups = common_sense_data.groupby('relation')

# Initialize an empty DataFrame to store the results
result_df = pd.DataFrame(columns=['Text', 'Max_Similarity', 'Relation'])

# Iterate through non-common sense data and calculate similarity for each row
for relation, non_common_sense_row in non_common_sense_data.iterrows():
    print('non_common_sense_row')
    relation_value = non_common_sense_row['relation']
    
    if relation_value in common_sense_groups.groups:
        common_sense_group = common_sense_groups.get_group(relation_value)

        # Calculate BERT embeddings for both datasets on GPU
        common_sense_embeddings = [model(tokenizer(text, return_tensors='pt').input_ids.to(device)).last_hidden_state.mean(dim=1).detach().cpu().numpy() for text in common_sense_group['type_relation']]
        non_common_sense_embedding = model(tokenizer(non_common_sense_row['type_relation'], return_tensors='pt').input_ids.to(device)).last_hidden_state.mean(dim=1).detach().cpu().numpy()

        # Flatten the embeddings to 2D arrays
        common_sense_embeddings_flat = [embedding.flatten() for embedding in common_sense_embeddings]
        non_common_sense_embedding_flat = non_common_sense_embedding.flatten()

        # Calculate cosine similarity on CPU
        similarity_scores = cosine_similarity([non_common_sense_embedding_flat], common_sense_embeddings_flat)
        print('similarity_scores', similarity_scores)

        # Find the maximum similarity
        max_similarity = similarity_scores.max()

        # Add the maximum similarity to the non-common sense DataFrame
        non_common_sense_data.at[relation, 'Max_Similarity'] = max_similarity
        non_common_sense_data.at[relation, 'Relation'] = relation_value
        non_common_sense_data.to_csv('similarity_relation_type_max_dev_mixed.csv', index=False)

# Save the resulting non-common sense DataFrame to a CSV file with the relation in the filename
