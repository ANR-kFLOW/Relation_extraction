import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize an empty DataFrame to store the results
result_df = pd.DataFrame(columns=['Text', 'Max_Similarity', 'Relation'])
# Load a pre-trained BERT model and tokenizer on the GPU
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

common_sense_data = pd.read_csv('/data/Youss/RE/refining/GT_atomic/typed_atomic_train.csv')
non_common_sense_data = pd.read_csv('/data/Youss/llama/saved_types/typed_original_test.csv')
non_common_sense_data['relation']=non_common_sense_data['label']
non_common_sense_data['type_relation'] =  [str(a) + ' '+ str(b)  + ' '+str(c) for a, b,c in zip(non_common_sense_data['Event1_Type'], non_common_sense_data['relation'], non_common_sense_data['Event2_Type'])]
# Group the datasets by their 'relation' column

# Group the datasets by their 'relation' column
common_sense_groups = common_sense_data.groupby('relation')

# Set the number of top similarity scores to consider (change this according to your needs)
top_n = 5

# Iterate through non-common sense data and calculate similarity for each row
for _, non_common_sense_row in non_common_sense_data.iterrows():
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
        print(similarity_scores)

        # Get the indices of the top n similarity scores
        top_n_indices = np.argpartition(similarity_scores, -top_n, axis=1)[:, -top_n:]

        # Get the average of the top n similarity scores
        average_similarity = np.take_along_axis(similarity_scores, top_n_indices, axis=1).mean()

        # Add the average similarity to the result DataFrame
        result_df = result_df.append({'Text': non_common_sense_row['sentence'], 'Average_Similarity': average_similarity, 'Relation': relation_value}, ignore_index=True)

        # Save the resulting DataFrame to a CSV file
        result_df.to_csv('average_similarity_relation_type_test.csv', index=False)

