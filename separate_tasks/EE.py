import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
# Configuration
MODEL_NAME = "dslim/bert-large-NER"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
BIO_LABELS = {"O": 0, "B-SUBJ": 1, "I-SUBJ": 2, "B-OBJ": 3, "I-OBJ": 4}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Dataset class
class BIOTaggingDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.texts = self.data['text'].tolist()
        self.subjects = self.data['subject'].tolist()
        self.objects = self.data['object'].tolist()
        self.relations = self.data['relation'].tolist()

    def bio_tagging(self, text, subject, obj, relation):
        tokens = tokenizer.tokenize(text)
        labels = ["O"] * len(tokens)

        if relation != "no_relation":
            for ent, tag in [(subject, "SUBJ"), (obj, "OBJ")]:
                if isinstance(ent, str) and ent.strip():  # Ensure ent is a non-empty string
                    ent_tokens = tokenizer.tokenize(ent)
                    for i in range(len(tokens) - len(ent_tokens) + 1):
                        if tokens[i:i + len(ent_tokens)] == ent_tokens:
                            labels[i] = f"B-{tag}"
                            for j in range(1, len(ent_tokens)):
                                labels[i + j] = f"I-{tag}"
                            break

        if len(tokens) > MAX_LEN:
            tokens = tokens[:MAX_LEN]
            labels = labels[:MAX_LEN]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        label_ids = [BIO_LABELS[label] for label in labels]

        padding_length = MAX_LEN - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        label_ids += [BIO_LABELS["O"]] * padding_length

        return input_ids, attention_mask, label_ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids, attention_mask, label_ids = self.bio_tagging(self.texts[idx], self.subjects[idx], self.objects[idx],
                                                                self.relations[idx])
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


# Model definition
class NERModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(NERModel, self).__init__()
        self.bert = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath) # Ensure the CSV has 'text', 'subject', 'object' columns
    return df


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), \
                                                batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, target_names=BIO_LABELS.keys())
    print(report)
    return f1


def train():

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataset = BIOTaggingDataset(
            load_data("/data/Youss/RE/TACL/end_to_end_models/Bert_based_classification/train.csv"))
        dev_dataset = BIOTaggingDataset(
            load_data("/data/Youss/RE/TACL/end_to_end_models/Bert_based_classification/dev.csv"))
        test_dataset = BIOTaggingDataset(
            load_data("/data/Youss/RE/TACL/end_to_end_models/Bert_based_classification/test.csv"))


        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = NERModel(MODEL_NAME, len(BIO_LABELS)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        best_f1 = 0.0
        best_model_path = "bert-large-NER_news.pth"

        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
            for batch in progress_bar:
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), \
                                                    batch['labels'].to(device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1, len(BIO_LABELS)), labels.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

            print("Evaluating on Dev Set:")
            f1 = evaluate(model, dev_loader, device)
            print(f1)

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with F1: {best_f1}")

        print("Testing on Test Set:")
        evaluate(model, test_loader, device)




if __name__ == "__main__":
    train()

