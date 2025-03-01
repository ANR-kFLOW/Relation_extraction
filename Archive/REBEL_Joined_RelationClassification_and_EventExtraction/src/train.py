import pandas as pd
import nltk
nltk.download('punkt')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # , BertTokenizerFast
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, precision_recall_fscore_support
import re
import io
LEARNING_RATE = 0.000025
EPOCHS = 10
BATCH_SIZE = 8
SEED = 1

SAVE_PATH = '2stft_75_old.pth'


class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df):
        txt = df['context'].tolist()
        self.texts = tokenizer(txt, padding='max_length', max_length=128, truncation=True, return_tensors="pt")

        labels = df['triplets'].to_list()
        self.labels = tokenizer(labels, padding='max_length', max_length=128, truncation=True, return_tensors="pt")

    def __len__(self):
        return len(self.labels['input_ids'])

    def get_batch_data(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.texts.items()}
        return item

    def get_batch_labels(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.labels.items()}
        return item

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
def train_loop(model, df_train, df_val):
    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)




    # optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # create a scheduler that reduces the learning rate by a factor of 0.1 every 10 epochs
    # scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    if use_cuda:
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    best_metric = 0

    for epoch_num in range(EPOCHS):

        model.train()

        total_loss_train = 0

        for train_data, train_label in tqdm(train_dataloader):
            
            train_label = train_label['input_ids'].to(device)

            mask = train_data['attention_mask'].to(device)
            input_id = train_data['input_ids'].to(device)

            optimizer.zero_grad()

            loss = model(input_id, mask, labels=train_label).loss

            total_loss_train += loss.item()
            print(total_loss_train)

            loss.backward()  # Update the weights
            optimizer.step()  # Notify optimizer that a batch is done.
            optimizer.zero_grad()  # Reset the optimer

        model.eval()

        total_loss_val = 0
        pred = []
        gt = []

        for val_data, val_label in val_dataloader:
            val_label = val_label['input_ids'].to(device)
            mask = val_data['attention_mask'].to(device)
            input_id = val_data['input_ids'].to(device)

            loss = model(input_id, mask, labels=val_label).loss
            total_loss_val += loss.item()

            outputs = model.generate(input_id)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)

            labels = tokenizer.batch_decode(val_label, skip_special_tokens=False)

            gt = gt + extract_triplets(labels, gold_extraction=True)
            pred = pred + extract_triplets(outputs, gold_extraction=False)

            del outputs, labels
        combined_metric = 0

        scores, precision, recall, f1 = re_score(pred, gt, 'relation')
        combined_metric += scores["ALL"]["Macro_f1"]

        scores, precision, recall, f1 = re_score(pred, gt, 'subject')
        combined_metric += scores["ALL"]["Macro_f1"]

        scores, precision, recall, f1 = re_score(pred, gt, 'object')
        combined_metric = (combined_metric + scores["ALL"]["Macro_f1"]) / 3

        best_metric = check_best_performing(model, best_metric, combined_metric, SAVE_PATH)
        del scores, precision, recall, f1

        # adjust the learning rate using the scheduler
        # scheduler.step()

        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .6f} | Val_Loss: {total_loss_val / len(df_val): .6f}')


def extract_triplets(texts, gold_extraction, prediction=False):
    triplets = []
    for text in texts:
        try:
            text = ''.join(text).replace('<s>', '').replace('</s>', '').replace('<pad>', '')
            relation = ''
            for token in text.split():
                if token == "<triplet>":
                    current = 't'
                    if relation != '':
                        triplets.append((subject.strip(), relation.strip(), object_.strip()))
                        relation = ''
                    subject = ''
                elif token == "<subj>":
                    current = 's'
                    if relation != '':
                        triplets.append((subject.strip(), relation.strip(), object_.strip()))
                    object_ = ''
                elif token == "<obj>":
                    current = 'o'
                    relation = ''
                else:
                    if current == 't':
                        subject += ' ' + token
                    elif current == 's':
                        object_ += ' ' + token
                    elif current == 'o':
                        relation += ' ' + token
            triplets.append((subject.strip(), relation.strip(), object_.strip()))
        except:
            if gold_extraction:
                print("Gold labels should always be extracted correctly. Exiting")
                sys.exit()
            triplets.append(("Invalid", "Invalid", "Invalid"))

    if prediction ==True: #This is to make sure not more than 1 set of triplets are extracted
        return [triplets[0]]

    return triplets


def re_score(predictions, ground_truths, type):
    print(type)
    # print(predictions)
    # print('\\\\\\\\\\\\')
    # print(ground_truths)
    # print('\\\\\\\\\\\\')
    """Evaluate RE predictions
    Args:
        predictions (list) :  list of list of predicted relations (several relations in each sentence)
        ground_truths (list) :    list of list of ground truth relations
        type (str) :          the kind of evaluation (relation, subject, object) """
    if type == 'relation':
        vocab = ['cause', 'enable', 'prevent', 'intend']
        predictions = [pred[1] for pred in predictions]
        ground_truths = [gt[1] for gt in ground_truths]

    elif type == 'subject':
        predictions = [pred[0] for pred in predictions]
        ground_truths = [gt[0] for gt in ground_truths]
        # vocab = ['Invalid'] #Create the vocabulary of possible tags
        vocab = np.unique(ground_truths).tolist()

    elif type == 'object':
        predictions = [pred[2] for pred in predictions]
        ground_truths = [gt[2] for gt in ground_truths]
        # vocab = ['Invalid']
        vocab = np.unique(ground_truths).tolist()

    scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in vocab + ["ALL"]}

    # Count GT relations and Predicted relations
    n_sents = len(ground_truths)
    n_rels = n_sents  # Since every 'sentence' has only 1 relation
    n_found = n_sents

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(predictions, ground_truths):
        for entity in vocab:

            if pred_sent == entity:
                pred_entities = {pred_sent}
            else:
                pred_entities = set()

            if gt_sent == entity:
                gt_entities = {gt_sent}

            else:
                gt_entities = set()

            scores[entity]["tp"] += len(pred_entities & gt_entities)
            scores[entity]["fp"] += len(pred_entities - gt_entities)
            scores[entity]["fn"] += len(gt_entities - pred_entities)

    # Compute per relation Precision / Recall / F1
    for entity in scores.keys():
        if scores[entity]["tp"]:
            scores[entity]["p"] = 100 * scores[entity]["tp"] / (scores[entity]["fp"] + scores[entity]["tp"])
            scores[entity]["r"] = 100 * scores[entity]["tp"] / (scores[entity]["fn"] + scores[entity]["tp"])
        else:
            scores[entity]["p"], scores[entity]["r"] = 0, 0

        if not scores[entity]["p"] + scores[entity]["r"] == 0:
            scores[entity]["f1"] = 2 * scores[entity]["p"] * scores[entity]["r"] / (
                    scores[entity]["p"] + scores[entity]["r"])
        else:
            scores[entity]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[entity]["tp"] for entity in vocab])
    fp = sum([scores[entity]["fp"] for entity in vocab])
    fn = sum([scores[entity]["fn"] for entity in vocab])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in vocab])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in vocab])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in vocab])

    # print(f"RE Evaluation in *** {mode.upper()} *** mode")

    if type == 'relation':
        print(
            "processed {} sentences with {} entities; found: {} relations; correct: {}.".format(n_sents, n_rels,
                                                                                                n_found,
                                                                                                tp))
        print(
            "\tALL\t TP: {};\tFP: {};\tFN: {}".format(
                scores["ALL"]["tp"],
                scores["ALL"]["fp"],
                scores["ALL"]["fn"]))
        print(
            "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
                precision,
                recall,
                f1))
        print(
            "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
                scores["ALL"]["Macro_p"],
                scores["ALL"]["Macro_r"],
                scores["ALL"]["Macro_f1"]))

        for entity in vocab:
            print("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
                entity,
                scores[entity]["tp"],
                scores[entity]["fp"],
                scores[entity]["fn"],
                scores[entity]["p"],
                scores[entity]["r"],
                scores[entity]["f1"],
                scores[entity]["tp"] +
                scores[entity][
                    "fp"]))

    else:
        print(f"Macro F1 for {type}: {scores['ALL']['Macro_f1']:.4f}")
        print(f"Micro F1 for {type}: {scores['ALL']['f1']:.4f}")

    return scores, precision, recall, f1


def calc_acc(predictions, gold):
    num_ner = len(predictions)  # The total number of entities
    acc_subj_correct = 0
    acc_obj_correct = 0

    for pred, gt in zip(predictions, gold):
        if pred[0] == gt[0]:  # The subjects match
            acc_subj_correct += 1

        if pred[2] == gt[2]:  # The objects match
            acc_obj_correct += 1

    acc_subj_correct = acc_subj_correct / num_ner
    acc_obj_correct = acc_obj_correct / num_ner

    print(f"acc subject: {acc_subj_correct} acc object: {acc_obj_correct}")

    return acc_subj_correct, acc_obj_correct


def check_best_performing(model, best_metric, new_metric, PATH):
    if new_metric > best_metric:
        torch.save(model, PATH)
        print("New best model found, saving...")
        best_metric = new_metric
    return best_metric

def get_text_from_ids(input_ids, tokenizer):
    """Decode input_ids to text using the tokenizer."""
    return tokenizer.batch_decode(input_ids, skip_special_tokens=True)


def get_bio_tags(text, entities):
    """Generate BIO tags for a given text based on entities."""
    words = text.split()
    bio_tags = ['O'] * len(words)

    for entity in entities:
        if len(entity) == 2:
            continue  # Skip if entity is incomplete (only relation without subject or object)
        print(entity)
        subject, relation, object = entity
        if subject in text and object in text:
            subject_start = text.index(subject)
            object_start = text.index(object)

            subject_words = subject.split()
            object_words = object.split()

            for i, word in enumerate(words):
                if text.index(word) >= subject_start and text.index(word) < subject_start + len(subject):
                    bio_tags[i] = 'B-SUB' if text.index(word) == subject_start else 'I-SUB'
                if text.index(word) >= object_start and text.index(word) < object_start + len(object):
                    bio_tags[i] = 'B-OBJ' if text.index(word) == object_start else 'I-OBJ'

    return bio_tags


def re_score_bio(predictions, ground_truths, texts, tokenizer):
    """Evaluate RE predictions with BIO tagging.
    Args:
        predictions (list) :  list of list of predicted relations (several relations in each sentence)
        ground_truths (list) :    list of list of ground truth relations
        texts (list) :       list of original sentences
        tokenizer :          tokenizer to decode input ids to text
    """
    # Decode input ids to texts if not already done
    # if isinstance(texts[0], list):
    #     texts = get_text_from_ids(texts, tokenizer)

    # Generate BIO tags for predictions and ground truths


    pred_bio_tags = [get_bio_tags(text, preds) for text, preds in zip(texts, predictions)]

    print(pred_bio_tags)
    gt_bio_tags = [get_bio_tags(text, gts) for text, gts in zip(texts, ground_truths)]

    # Initialize counts for TP, FP, and FN
    tp, fp, fn = 0, 0, 0

    # Calculate TP, FP, and FN
    for pred_tags, gt_tags in zip(pred_bio_tags, gt_bio_tags):
        for p_tag, g_tag in zip(pred_tags, gt_tags):
            if p_tag == g_tag and p_tag != 'O':
                tp += 1
            elif p_tag != g_tag and p_tag != 'O':
                fp += 1
            elif p_tag != g_tag and g_tag != 'O':
                fn += 1

    # Calculate precision, recall, and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def test_model(data, path_to_model):
    with open('2stft_75_old.pth', 'rb') as f:
        buffer = io.BytesIO(f.read())

    # Load the model from the buffer
    model = torch.load(buffer).to(device)

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained('Babelscape/rebel-large')
    # model = torch.load(path_to_model).to(device)

    test_dataset = DataSequence(data)
    test_dataloader = DataLoader(test_dataset, batch_size=4)
    model.eval()

    pred = []
    gt = []
    texts=[]


    for val_data, val_label in test_dataloader:
        test_label = val_label['input_ids'].to(device)
        mask = val_data['attention_mask'].to(device)
        input_id = val_data['input_ids'].to(device)
        # text=val_data['text']
        # print(val_data.keys())

        outputs = model.generate(input_id)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        labels = tokenizer.batch_decode(test_label, skip_special_tokens=False)

        gt = gt + extract_triplets(labels, gold_extraction=True)
        pred = pred + extract_triplets(outputs, gold_extraction=False)
        texts=texts+get_text_from_ids(input_id,tokenizer)

        del outputs, labels
    print(len(pred))
    print(len(gt))
    print(len(texts))
    # Create the pred DataFrame
    pred_data = []
    for i, triple in enumerate(pred):
        text = texts[i]
        subject, relation, obj = triple
        pred_data.append([text, subject, relation, obj])

    pred_df = pd.DataFrame(pred_data, columns=['text', 'subject', 'relation', 'object'])

    # Create the gt DataFrame
    gt_data = []
    for i, triple in enumerate(gt):
        text = texts[i]
        subject, relation, obj = triple
        gt_data.append([text, subject, relation, obj])

    gt_df = pd.DataFrame(gt_data, columns=['text', 'subject', 'relation', 'object'])

    # Save to CSV
    pred_df.to_csv('pred.csv', index=False)
    gt_df.to_csv('gt.csv', index=False)
    # scores, precision, recall, f1 = re_score(pred, gt, 'relation')
    # scores, precision, recall, f1 = re_score(pred, gt, 'subject')
    # scores, precision, recall, f1 = re_score(pred, gt, 'object')
    # precision, recall, f1 = re_score_bio(pred, gt, texts, tokenizer)
    # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    score_BIO(gt_df,pred_df)




def create_bio_tagging(text, subject, object_):
    # Remove commas from the text, subject, and object
    text = str(text).replace(',', '')
    subject = str(subject).replace(',', '')
    object_ = str(object_).replace(',', '')

    words = text.split()
    bio_tags = ['O'] * len(words)

    # Create a helper function to find the index of a substring in a list of words
    def find_sublist(sublist, main_list):
        sublen = len(sublist)
        for i in range(len(main_list) - sublen + 1):
            if main_list[i:i + sublen] == sublist:
                return i
        return -1

    subject_words = subject.split()
    object_words = object_.split()

    # Find and tag the subject
    subject_start_idx = find_sublist(subject_words, words)
    if subject_start_idx != -1:
        bio_tags[subject_start_idx] = 'B-C'
        for i in range(1, len(subject_words)):
            bio_tags[subject_start_idx + i] = 'I-C'

    # Find and tag the object
    object_start_idx = find_sublist(object_words, words)
    if object_start_idx != -1:
        bio_tags[object_start_idx] = 'B-E'
        for i in range(1, len(object_words)):
            bio_tags[object_start_idx + i] = 'I-E'

    return ' '.join(bio_tags)


def score_BIO(ground_truth, predictions):
    # Load the CSV files
    # ground_truth = pd.read_csv(gt_path)
    # predictions = pd.read_csv(pred_path)

    # Add BIO tagging columns to both DataFrames
    ground_truth['BIO_tagging'] = ground_truth.apply(
        lambda row: create_bio_tagging(row['text'], row['subject'], row['object']), axis=1)
    predictions['BIO_tagging'] = predictions.apply(
        lambda row: create_bio_tagging(row['text'], row['subject'], row['object']), axis=1)

    # Calculate precision, recall, and F1 score for BIO tagging
    gt_bio_tags = ground_truth['BIO_tagging'].apply(lambda x: x.split()).tolist()
    pred_bio_tags = predictions['BIO_tagging'].apply(lambda x: x.split()).tolist()

    all_gt_tags = [tag for sublist in gt_bio_tags for tag in sublist]
    all_pred_tags = [tag for sublist in pred_bio_tags for tag in sublist]

    precision = precision_score(all_gt_tags, all_pred_tags, average='weighted',
                                labels=['B-C', 'I-C', 'B-E', 'I-E', 'O'])
    recall = recall_score(all_gt_tags, all_pred_tags, average='weighted', labels=['B-C', 'I-C', 'B-E', 'I-E', 'O'])
    f1 = f1_score(all_gt_tags, all_pred_tags, average='weighted', labels=['B-C', 'I-C', 'B-E', 'I-E', 'O'])

    print(f"BIO Tagging - Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Calculate precision, recall, and F1 score for relations
    gt_relations = ground_truth['relation'].tolist()
    pred_relations = predictions['relation'].tolist()

    precision_rel = precision_score(gt_relations, pred_relations, average='weighted')
    recall_rel = recall_score(gt_relations, pred_relations, average='weighted')
    f1_rel = f1_score(gt_relations, pred_relations, average='weighted')


    print(f"Relations - Precision: {precision_rel}, Recall: {recall_rel}, F1 Score: {f1_rel}")

    report = classification_report(all_gt_tags, all_pred_tags)
    print(report)
    # Create mappings for replacements
    replacement_dict = {
        'B-C': 'subject',
        'I-C': 'subject',
        'B-E': 'object',
        'I-E': 'object',
        'O': 'O'
    }

    # Function to replace labels
    def replace_labels(labels, mapping):
        return [mapping[label] for label in labels]

    # Replace labels in gt and pred
    gt_replaced = replace_labels(all_gt_tags, replacement_dict)
    pred_replaced = replace_labels(all_pred_tags, replacement_dict)

    # Compute the classification report
    report = classification_report(gt_replaced, pred_replaced, target_names=['subject', 'object', 'O'])
    print(report)
    # Compute micro average
    precision, recall, f1, _ = precision_recall_fscore_support(gt_replaced, pred_replaced,
                                                               average='weighted')
    print(precision,recall,f1)

    # # Save the DataFrames with BIO tagging columns
    # ground_truth.to_csv('ground_truth_with_bio.csv', index=False)
    # predictions.to_csv('predictions_with_bio.csv', index=False)



def make_predictions(texts, path_to_model):
    """

    :param texts: List of sentences
    :param path_to_model: The path to the model
    :return: List of original sentences and their predictions
    """

    predictions = []

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = torch.load(path_to_model).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('Babelscape/rebel-large')

    results = []
    for sentence in texts:

        encoding = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = model.generate(**encoding, do_sample=True)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)

        results+= (extract_triplets(outputs, gold_extraction=False, prediction=True))

    return texts, results

if __name__ == "__main__":
    #data = pd.read_csv('drive/MyDrive/rebel_format_v2.csv')
    # df_train, df_val = train_test_split(data, test_size=0.1, random_state=SEED)
    df_train = pd.read_csv('/data/Youss/RE/REBEL/data/CS_our_data_mixed.csv')
    # df_train = df_train[~df_train['triplets'].str.contains('0')]
    df_val = pd.read_csv('/data/Youss/RE/REBEL/data/news_data_with_cnc/validation_augmented.csv')
    # df_val = df_val[~df_val['triplets'].str.contains('0')]
    #del data

    model_checkpoint = "Babelscape/rebel-large"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    print('I am calling the train loop')

    # train_loop(model, df_train, df_val)

    test_data = pd.read_csv('/data/Youss/RE/REBEL/data/news_data_with_cnc/test.csv')
    # test_data = test_data[~test_data['triplets'].str.contains('0')]
    with open('2stft_75_old.pth', 'rb') as f:
        buffer = io.BytesIO(f.read())

    # Load the model from the buffer
    model = torch.load(buffer).to(device)
    test_model(test_data, model)
