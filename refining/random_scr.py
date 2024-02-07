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
# Assuming the common column is 'relation'
common_column = 'relation'

# Iterate over rows in train_our_dt_with_sim
for index, row in train_our_dt_with_sim.iterrows():
    # Get the relation and Max_Similarity from train_our_dt_with_sim
    relation = row['relation']
    print(relation)
    max_similarity = row['Max_Similarity']

    # Find the first occurrence of the relation in non_common_sense_data
    match_index = non_common_sense_data.index[non_common_sense_data[common_column] == relation].tolist()
    print(match_index)

    # Check if there is a match
    if match_index:
        # Add Max_Similarity to the initial dataframe at the matched index
        non_common_sense_data.loc[match_index[0], 'Max_Similarity'] = max_similarity

        # Update the non_common_sense_data dataframe and the train_our_dt_with_sim dataframe
        non_common_sense_data = non_common_sense_data.loc[match_index[0]+1:]
        train_our_dt_with_sim = train_our_dt_with_sim.loc[index+1:]

    else:
        # If no match is found, break the loop
        break

# Display the updated dataframe
print(non_common_sense_data)

# Save the updated dataframe to a new CSV file
non_common_sense_data.to_csv('/data/Youss/RE/refining/updated_train_typed_our_data.csv', index=False)
