import pandas as pd
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup,BertForSequenceClassification
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
path_to_data='/data/Youss/RE/TACL/separate_tasks/data/'
# Load the dataset
train_data = pd.read_csv(path_to_data + 'train.csv')
val_data = pd.read_csv(path_to_data + 'dev.csv')
test_data = pd.read_csv(path_to_data + 'test.csv')

print("Length of train_data:", len(train_data))
print("Length of val_data:", len(val_data))
print("Length of test_data:", len(test_data))

# Print distribution of label column for each dataset
print("\nDistribution of label column for train_data:")
print(train_data['label'].value_counts(normalize=True))

print("\nDistribution of label column for val_data:")
print(val_data['label'].value_counts(normalize=True))

print("\nDistribution of label column for test_data:")
print(test_data['label'].value_counts(normalize=True))

# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define parameters
MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5

## Initialize tokenizer and model
#tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
def load_model_and_tokenizer(model_name, num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model


model_name = 'roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# Prepare dataloaders
train_dataset = CustomDataset(train_data['text'], train_data['label'], tokenizer, MAX_LEN)
val_dataset = CustomDataset(val_data['text'], val_data['label'], tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(train_loader) * EPOCHS)

# Define loss function
loss_fn = nn.CrossEntropyLoss()
best_f1_score = 0.0
best_model_path = None
# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=False)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

        progress_bar.set_postfix({'train_loss': train_loss / (len(progress_bar))})

    # Validation loop
    model.eval()
    val_loss = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_accuracy = accuracy_score(true_labels, predictions)
    val_classification_report = classification_report(true_labels, predictions)
    val_f1 = f1_score(true_labels, predictions)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print("Validation Classification Report:")
    print(val_classification_report)
    print(f"Validation F1-score: {val_f1:.4f}")

    # Save the best model based on validation F1-score
    if val_f1 > best_f1_score:
        best_f1_score = val_f1
        best_model_path = f"Roberta_news.pt"
        torch.save(model.state_dict(), best_model_path)
# Test loop
test_dataset = CustomDataset(test_data['text'], test_data['label'], tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

# Load the best model
best_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
best_model.load_state_dict(torch.load(best_model_path))
best_model.to(device)


best_model.eval()
test_predictions = []
test_true_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = best_model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)
        test_predictions.extend(predicted.cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())

# Calculate test accuracy and classification report
test_accuracy = accuracy_score(test_true_labels, test_predictions)
test_classification_report = classification_report(test_true_labels, test_predictions)

print(f"Best Model - Test Accuracy: {test_accuracy:.4f}")
print("Best Model - Test Classification Report:")
print(test_classification_report)
