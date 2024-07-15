import pandas as pd

# # Read the two CSV files
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
import pandas as pd

# Load the dataframes
train_our_dt_with_sim = pd.read_csv('/data/Youss/RE/refining/scripts/average_similarity_relation_sentence.csv')
non_common_sense_data = pd.read_csv('/data/Youss/RE/refining/train_typed_our_data.csv')
train_our_dt_with_sim['relation']=train_our_dt_with_sim['Relation']
# Create a new dataframe to store the updated data
updated_non_common_sense_data = non_common_sense_data.copy()

# Assuming the common column is 'relation'
common_column = 'relation'

# Initialize indices
train_index = 0
non_common_sense_index = 0

# Iterate over rows in train_our_dt_with_sim
for index, row in train_our_dt_with_sim.iterrows():
    # Get the relation and avg_similarity from train_our_dt_with_sim
    relation = row['relation']
    avg_similarity = row['Average_Similarity']

    # Find the first occurrence of the relation in updated_non_common_sense_data
    match_index = updated_non_common_sense_data.index[updated_non_common_sense_data[common_column] == relation].tolist()

    # Check if there is a match
    if match_index:
        # Add avg_similarity to the updated dataframe at the matched index
        updated_non_common_sense_data.loc[match_index[0], 'Average_Similarity'] = avg_similarity

        # Update the indices for both dataframes
        train_index = index + 1
        non_common_sense_index = match_index[0] + 1

    else:
        # If no match is found, break the loop
        break

# Save the updated dataframe to a new CSV file with the original indexes
updated_non_common_sense_data.to_csv('/data/Youss/RE/refining/updated_train_typed_our_data.csv', index=True)


