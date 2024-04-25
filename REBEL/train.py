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
import io
LEARNING_RATE = 0.000025
EPOCHS = 10
BATCH_SIZE = 8
SEED = 1
SAVE_PATH = 'model_our_data_gpt_augmented.pth'


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
    """Evaluate RE predictions
    Args:
        predictions (list) :  list of list of predicted relations (several relations in each sentence)
        ground_truths (list) :    list of list of ground truth relations
        type (str) :          the kind of evaluation (relation, subject, object) """
    if type == 'relation':
        vocab = ['cause', 'enable', 'prevent', 'intend','0']
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


def test_model(data, path_to_model):
    with open('model_our_data_gpt_augmented.pth', 'rb') as f:
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


    for val_data, val_label in test_dataloader:
        test_label = val_label['input_ids'].to(device)
        mask = val_data['attention_mask'].to(device)
        input_id = val_data['input_ids'].to(device)

        outputs = model.generate(input_id)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        labels = tokenizer.batch_decode(test_label, skip_special_tokens=False)

        gt = gt + extract_triplets(labels, gold_extraction=True)
        pred = pred + extract_triplets(outputs, gold_extraction=False)

        del outputs, labels

    scores, precision, recall, f1 = re_score(pred, gt, 'relation')
    scores, precision, recall, f1 = re_score(pred, gt, 'subject')
    scores, precision, recall, f1 = re_score(pred, gt, 'object')


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
    df_train = pd.read_csv('/data/Youss/RE/REBEL/data/train_augmented.csv')
    df_val = pd.read_csv('/data/Youss/RE/REBEL/data/validation_augmented.csv')
    #del data

    model_checkpoint = "Babelscape/rebel-large"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    print('I am calling the train loop')

    train_loop(model, df_train, df_val)

    test_data = pd.read_csv('/data/Youss/RE/REBEL/data/our_data/test.csv')
    with open('model_our_data_gpt_augmented.pth', 'rb') as f:
        buffer = io.BytesIO(f.read())

    # Load the model from the buffer
    model = torch.load(buffer).to(device)
    test_model(test_data, model)
