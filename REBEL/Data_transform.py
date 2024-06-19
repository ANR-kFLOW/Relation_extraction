import pandas as pd
import  re
import random
def convert_to_rebel(data):

    data = data.rename(columns={"text": "context"})
    data['triplets'] = '<triplet> ' + data['event0'] + ' <subj> ' + data['event1'] + ' <obj> ' + data['label'] #Add the suitable format
    data = data[['context', 'triplets']]
    return data


# Function to extract text between tags
def extract_text(sentence):
    arg0 = re.search(r'<ARG0>(.*?)</ARG0>', sentence).group(1)
    arg1 = re.search(r'<ARG1>(.*?)</ARG1>', sentence).group(1)
    return arg0, arg1

# Function to fill event0 and event1 columns with random words from text column
# Define the extract_events function
def extract_events(df):
    events0 = []
    events1 = []
    relations = []
    texts = []

    for index, row in df.iterrows():
        if row['num_rs'] > 0:
            for sentence in eval(row['causal_text_w_pairs']):
                event0 = re.search(r'<ARG0>(.*?)</ARG0>', sentence).group(1)
                event1 = re.search(r'<ARG1>(.*?)</ARG1>', sentence).group(1)
                relation = row['relation']

                # Remove <SIG0> and </SIG0> strings if found
                event0 = event0.replace('<SIG0>', '').replace('</SIG0>', '')
                event1 = event1.replace('<SIG0>', '').replace('</SIG0>', '')

                events0.append(event0)
                events1.append(event1)
                relations.append(relation)
                texts.append(row['text'])

    return events0, events1, relations, texts

# Assuming you have a DataFrame named 'df'
# Create a new DataFrame from the lists returned by extract_events function



# Drop the original column if needed
# df.drop('causal_text_w_pairs', axis=1, inplace=True)




train=pd.read_csv('/data/Youss/RE/Relation_extraction/data/Joined_data/News_data/test.csv')

df = train[~train['relation'].str.contains('0')]
print((len(df)))
events0, events1, relations, texts = extract_events(df)
new_df = pd.DataFrame({
    'event0': events0,
    'event1': events1,
    'label': relations,
    'text': texts
})


# train['label']=train['index'].str.split('_').str[0]
# train['trigger1']=extract_events(train)[0]
# train['trigger1']=extract_events(train)([1])
# Apply function to each row and assign to new columns
# Apply function to relevant rows




new_df.to_csv('try.csv',index=False)
train_transformed=convert_to_rebel(new_df)
print(train_transformed)
train_transformed.to_csv('/data/Youss/RE/REBEL/data/news_data_with_cnc/test.csv',index=False)

# trans_train=convert_to_rebel(train)
