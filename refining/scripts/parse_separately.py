import pandas as pd
import re

# Sample DataFrame
df = pd.read_csv('/data/Youss/llama/saved_types/typed_dev_mixed_data1.csv')

# Function to extract types separately from the given column
# Function to extract types separately from the given column
# def extract_types(event):
#     event1_matches = re.findall(r'Events1 Type:(.*?)Events2 Type:', event, re.DOTALL)
#     event2_matches = re.findall(r'Events2 Type:(.*)', event, re.DOTALL)
#
#     event1 = event1_matches[-1].strip() if event1_matches else None
#     event2 = event2_matches[-1].strip() if event2_matches else None
#
#     return event1, event2
#
#
# # Apply the function to create new columns for all rows
# df[['type1', 'type2']] = df['Event1_Type'].apply(extract_types).apply(pd.Series)
#
# # Print rows where either type1 or type2 is not empty
# non_empty_rows = df[(df['type1'].notna()) | (df['type2'].notna())]
#
#
# non_empty_rows.to_csv('non_empty.csv')
#
#
# # Filter rows where num_rs > 0
# filtered_df = df[df['num_rs'] > 0]

import re

#def extract_event_types(sentence):
#    # Find the index of the trigger phrase
#    trigger_phrase = "Sure, I'd be happy to help!"
#    start_index = sentence.find(trigger_phrase)
#
#    # Extract text after the trigger phrase
#    remaining_text = sentence[start_index + len(trigger_phrase):]
#
#    # Extract matches for patterns like "Event1 Type" and "Event2 Type"
#    event_matches = re.findall(r'\bEvent(\d) Type:\s*(.*?)\s*(?=\bEvent[12] Type:|$)', remaining_text)
#
#    # Initialize dictionaries for Event1 and Event2
#    event_1_dict = {}
#    event_2_dict = {}
#
#    # Fill dictionaries based on matches
#    for match in event_matches:
#        event_number, event_type = match
#        if event_number == '1':
#            event_1_dict[event_number] = event_type
#        elif event_number == '2':
#            event_2_dict[event_number] = event_type
#
#    # Print the extracted events
#    print(f"Events1 Type: {event_1_dict.get('1', 'No Match')}")
#    print(f"Events2 Type: {event_2_dict.get('2', 'No Match')}")

def extract_event_types(result_text, num_rs):
    event_1_type = "empty"
    event_2_type = "empty"

    if num_rs == 0:
        return event_1_type, event_2_type

    # Finding the index where to start searching for event types
    inst_index = result_text.find("[/INST]")
    if inst_index == -1:
        return event_1_type, event_2_type

    # Extracting types for event 1 and event 2 after the '[/INST]' marker
    pattern = re.compile(r'(Event Type:|Type:)\s*(.*)', re.IGNORECASE)
    matches = re.findall(pattern, result_text[inst_index:])
    event_type_pattern = re.compile(r'Events\d Type:\n\n((?:\*.*\n)*)', re.IGNORECASE)
    matches2 = re.findall(event_type_pattern, result_text)
    if num_rs == 1:
        if len(matches) >= 1:
            event_1_type = matches[0][1].strip()

        if len(matches) >= 2:
            event_2_type = matches[1][1].strip()
    if num_rs > 1:

        if len(matches2) >= 1:
            event_1_type = matches2[0].strip()

        if len(matches2) >= 2:
            event_2_type = matches2[1].strip()

    return event_1_type, event_2_type

#def extract_event_types(result_text, num_rs):
#    event_1_type = "empty"
#    event_2_type = "empty"
#    event_type_pattern = re.compile(r'Events\d Type:\n\n((?:\*.*\n)*)', re.IGNORECASE)
#    if num_rs == 0:
#        return event_1_type, event_2_type
#
#    matches = re.findall(event_type_pattern, result_text)
#    if num_rs > 1:
#
#        if len(matches) >= 1:
#            event_1_type = matches[0].strip()
#
#        if len(matches) >= 2:
#            event_2_type = matches[1].strip()
#    
#
#    return event_1_type, event_2_type

# Apply the function to each row of the DataFrame
# Apply the function to the DataFrame
df[['Events1 Type', 'Events2 Type']] = df.apply(lambda row: pd.Series(extract_event_types(row['Event1_Type'], row['num_rs'])), axis=1)


# Save the DataFrame to disk
df.to_csv('parsed_dev.csv', index=False)
