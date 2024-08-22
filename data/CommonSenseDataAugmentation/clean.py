import pandas as pd
import re
df=pd.read_csv('extracted_sentences.csv')
df.columns=['text']
#df['text'] = df['text'].apply(lambda x: re.sub(r'\b\d+\.?\b|\.', '', x))


def parse_row(row):
    # Define the pattern to match
    pattern_len_df = r'.*len df \d+.*'
    pattern_end_tag = r'</s>'
    pattern_arg0 = r'<ARG0>'
    
    # If match found for "len df \d*", remove the entire line plus preceding text
    match_len_df = re.search(pattern_len_df, row)
    if match_len_df:
        row = row[match_len_df.end():].strip()  # Extract the text after the match and trim whitespace
    
    # If match found for "</s>", remove it along with all text that comes after it
    match_end_tag = re.search(pattern_end_tag, row)
    if match_end_tag:
        row = row[:match_end_tag.start()].strip()  # Extract the text before the match and trim whitespace
    
    # Check for two occurrences of "<ARG0>"
    match_arg0 = re.findall(pattern_arg0, row)
    if len(match_arg0) >= 2:
        return ''  # Return an empty string to remove the row
    
    # Remove all text before the first occurrence of "<ARG0>"
    match_arg0 = re.search(pattern_arg0, row)
    if match_arg0:
        row = row[match_arg0.start():]  # Extract the text starting from the first occurrence of <ARG0>
    
    return row

# Apply the function to the DataFrame
df['text'] = df['text'].apply(parse_row)
# Remove rows with empty strings
df = df[df['text'] != '']

# Remove duplicate rows
df.drop_duplicates(inplace=True)
print(len(df))
df = df[df['text'].str.contains(r'<ARG0>.*</ARG0>.*<ARG1>.*</ARG1>', regex=True)]
print(len(df))


#df = df[~df['text'].str.contains(r'\n{3,}', regex=True)]
# Count tokens using split
#df['num_tokens'] = df['text'].apply(lambda x: len(x.split()))
#
## Filter rows with more than 50 tokens
#df = df[df['num_tokens'] <= 20].drop(columns=['num_tokens'])

df.to_csv('clean2.csv')
print(df)
