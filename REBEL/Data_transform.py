import pandas as pd
import  re
import random
def convert_to_rebel(data):
    data['label']=data['index'].str.split('_').str[0]
    # data = data[data.label != str(0)]
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
def fill_random(row):
    if row['num_rs'] > 0:
        event0, event1 = extract_text(row['causal_text_w_pairs'])
    else:
        print('here')
        words = row['text'].split()
        random.shuffle(words)
        event0 = words.pop()
        print(event0)
        event1 = words.pop()
    return pd.Series([event0, event1], index=['event0', 'event1'])


# Drop the original column if needed
# df.drop('causal_text_w_pairs', axis=1, inplace=True)




train=pd.read_csv('/data/Youss/RE/Relation_extraction/data/atomic/atomic_train.csv')
# train['label']=train['index'].str.split('_').str[0]
# train['trigger1']=extract_events(train)[0]
# train['trigger1']=extract_events(train)([1])
# Apply function to each row and assign to new columns
# Apply function to relevant rows
train[['event0', 'event1']] = train.apply(fill_random, axis=1)
train_transformed=convert_to_rebel(train)
print(train_transformed)
train_transformed.to_csv('/data/Youss/RE/REBEL/data/atomic/atomic_train.csv',index=False)

# trans_train=convert_to_rebel(train)
