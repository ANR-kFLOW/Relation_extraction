# import spacy
# import sys
# sys.executable
#
#
# # Load a pre-trained spaCy model (e.g., en_core_web_md)
# nlp = spacy.load("en_core_web_sm")
#
#
# # Example texts from your databases (replace with your data)
# common_sense_data = ["youssra is fine", "this works I guess"]
# normal_data = ["guessing only", "youssra is fine I guess"]
#
# # Generate embeddings for common sense data
# common_sense_embeddings = [nlp(text).vector for text in common_sense_data]
#
# # Generate embeddings for normal data
# normal_embeddings = [nlp(text).vector for text in normal_data]
#
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Initialize a list to store the maximum similarity scores
# max_similarity_scores = []
#
# for normal_vector in normal_embeddings:
#     # Calculate cosine similarity between the normal vector and all common sense vectors
#     similarities = cosine_similarity([normal_vector], common_sense_embeddings)
#
#     # Find the maximum similarity score and its corresponding index
#     max_similarity = max(similarities[0])
#     max_similarity_index = similarities[0].argmax()
#
#     # Store the maximum similarity score and the corresponding common sense data
#     max_similarity_scores.append((max_similarity, common_sense_data[max_similarity_index]))
#
# import pandas as pd
#
# # Create a DataFrame with your normal data
# normal_df = pd.DataFrame({"Text": normal_data})
#
# # Add a new column for maximum similarity scores and corresponding common sense data
# normal_df["Max_Similarity_Score"] = [score[0] for score in max_similarity_scores]
# normal_df["Common_Sense_Text"] = [score[1] for score in max_similarity_scores]
#
# # Display or save the updated DataFrame
# print(normal_df)
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load a pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example datasets (replace with your actual data)
common_sense_data = ["Common sense text 1", "Common sense text 2"]
non_common_sense_data = ["Non-common sense text 1", "Non-common sense text 2"]

# Calculate BERT embeddings for both datasets
common_sense_embeddings = [model(tokenizer(text, return_tensors='pt').input_ids).last_hidden_state.mean(dim=1).detach().numpy() for text in common_sense_data]
non_common_sense_embeddings = [model(tokenizer(text, return_tensors='pt').input_ids).last_hidden_state.mean(dim=1).detach().numpy() for text in non_common_sense_data]

# Flatten the embeddings to 2D arrays
common_sense_embeddings_flat = [embedding.flatten() for embedding in common_sense_embeddings]
non_common_sense_embeddings_flat = [embedding.flatten() for embedding in non_common_sense_embeddings]

# Calculate cosine similarity
similarity_matrix = cosine_similarity(non_common_sense_embeddings_flat, common_sense_embeddings_flat)

# Find the maximum similarity for each row in the non-common sense dataset
max_similarity_scores = similarity_matrix.max(axis=1)

# Add a new column to the non-common sense dataset with the maximum similarity scores
non_common_sense_df = pd.DataFrame({'Text': non_common_sense_data, 'Max_Similarity': max_similarity_scores})

# Print or save the resulting DataFrame
print(non_common_sense_df)
