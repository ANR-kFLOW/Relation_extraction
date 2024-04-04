import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report

train=pd.read_csv('/data/Youss/RE/CausalNewsCorpus/data/mixed_data/mixed_CS_our_train_data.csv')
val=pd.read_csv('/data/Youss/RE/CausalNewsCorpus/data/mixed_data/dev_mixed_data.csv')
test=pd.read_csv('/data/Youss/RE/CausalNewsCorpus/data/mixed_data/test_our_data.csv')



train['label'] = train['index'].str.split('_').str[0]
val['label'] = val['index'].str.split('_').str[0]
test['label'] = test['index'].str.split('_').str[0]


# Checking conditions and updating 'label' accordingly
train.loc[train['index'].str.contains('cnc'), 'label'] = train.loc[train['index'].str.contains('cnc'), 'num_rs'].apply(lambda x: 'cause' if x != 0 else '0')
val.loc[val['index'].str.contains('cnc'), 'label'] = val.loc[val['index'].str.contains('cnc'), 'num_rs'].apply(lambda x: 'cause' if x != 0 else '0')
test.loc[test['index'].str.contains('cnc'), 'label'] = test.loc[test['index'].str.contains('cnc'), 'num_rs'].apply(lambda x: 'cause' if x != 0 else '0')

# Keep only 'text' and 'label' columns in the dataset
train = train[['text', 'label']]
val = val[['text', 'label']]
test = test[['text', 'label']]

train.to_csv('train_st1.csv')
val.to_csv('val_st1.csv')
test.to_csv('test_st1.csv')
print('done')
# Assuming train is your DataFrame containing the 'label' column
class_counts = train['label'].value_counts()
print("Number of samples for each class:")
print(class_counts)
# For validation set
val_class_counts = val['label'].value_counts()
print("Number of samples for each class in validation set:")
print(val_class_counts)

# For test set
test_class_counts = test['label'].value_counts()
print("Number of samples for each class in test set:")
print(test_class_counts)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
import torch
le = preprocessing.LabelEncoder()
le.fit(train.label)
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(label_mapping)
labels_train=le.transform(train.label)
labels_val=le.transform(val.label)
labels_test=le.transform(test.label)
text_train = train.text.values
val_train = val.text.values
test_train = test.text.values
token_id = []
attention_masks = []



print('heeereeeee')
#
def preprocessing(input_text, tokenizer):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 512,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
    )

def preprocess_data(samples, tokenizer):
    token_ids = []
    attention_masks = []
    for sample in samples:
        encoding_dict = preprocessing(sample, tokenizer)
        token_ids.append(encoding_dict['input_ids'])
        attention_masks.append(encoding_dict['attention_mask'])
    return token_ids, attention_masks

# for sample in text:
#   encoding_dict = preprocessing(sample, tokenizer)
#   token_id.append(encoding_dict['input_ids'])
#   attention_masks.append(encoding_dict['attention_mask'])
train_token_ids, train_attention_masks = preprocess_data(train['text'], tokenizer)
val_token_ids, val_attention_masks = preprocess_data(val['text'], tokenizer)
test_token_ids, test_attention_masks = preprocess_data(test['text'], tokenizer)
train_token_ids = torch.cat(train_token_ids, dim=0)
val_token_ids = torch.cat(val_token_ids, dim=0)
test_token_ids = torch.cat(test_token_ids, dim=0)
# Concatenate attention masks
train_attention_masks = torch.cat(train_attention_masks, dim=0)
val_attention_masks = torch.cat(val_attention_masks, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)

# Convert labels to tensors
train_labels = torch.tensor(labels_train)
val_labels = torch.tensor(labels_val)
test_labels = torch.tensor(labels_test)
# Convert token IDs and attention masks to TensorDataset
train_dataset = TensorDataset(train_token_ids, train_attention_masks,train_labels)
val_dataset = TensorDataset(val_token_ids, val_attention_masks,val_labels)
test_dataset = TensorDataset(test_token_ids, test_attention_masks,test_labels)

# Define batch size
batch_size = 8

# Create DataLoaders
train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)

val_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=batch_size
)

test_dataloader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=batch_size
)

# token_id = torch.cat(token_id, dim = 0)
# attention_masks = torch.cat(attention_masks, dim = 0)
# labels = torch.tensor(labels)
# val_ratio = 0.15
#
# # Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf
# batch_size = 8
#
# # Indices of the train and validation splits stratified by labels
# train_idx, val_idx = train_test_split(
#     np.arange(len(labels)),
#     test_size = val_ratio,
#     shuffle = True,
#     stratify = labels)
#
#
# # Train and validation sets
# train_set = TensorDataset(token_id[train_idx],
#                           attention_masks[train_idx],
#                           labels[train_idx])
#
# val_set = TensorDataset(token_id[val_idx],
#                         attention_masks[val_idx],
#                         labels[val_idx])
#
# # Prepare DataLoader
# train_dataloader = DataLoader(
#             train_set,
#             sampler = RandomSampler(train_set),
#             batch_size = batch_size
#         )
#
# validation_dataloader = DataLoader(
#             val_set,
#             sampler = SequentialSampler(val_set),
#             batch_size = batch_size
#         )
# test_dataloader = DataLoader(
#             test_set,
#             sampler = SequentialSampler(test_set),
#             batch_size = batch_size
#         )

def b_tp(preds, labels):
  '''Returns True Positives (TP): count of correct predictions of actual class 1'''
  return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_fp(preds, labels):
  '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
  return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_tn(preds, labels):
  '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
  return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_fn(preds, labels):
  '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
  return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_metrics(preds, labels):
  '''
  Returns the following metrics:
    - accuracy    = (TP + TN) / N
    - precision   = TP / (TP + FP)
    - recall      = TP / (TP + FN)
    - specificity = TN / (TN + FP)
  '''
  preds = np.argmax(preds, axis = 1).flatten()
  labels = labels.flatten()
  tp = b_tp(preds, labels)
  tn = b_tn(preds, labels)
  fp = b_fp(preds, labels)
  fn = b_fn(preds, labels)

  b_accuracy = (tp + tn) / (tp+tn+fp+fn) if  (tp+tn+fp+fn)>0 else ('(tp+tn+fp+fn) is 0 which is weird')
  b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
  b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
  b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
  return b_accuracy, b_precision, b_recall, b_specificity
