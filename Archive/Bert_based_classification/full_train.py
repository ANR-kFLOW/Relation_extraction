import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, RobertaPreTrainedModel
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
# Define label mappings
label_map_bio = {"O": 0, "B-SUBJ": 1, "I-SUBJ": 2, "B-OBJ": 3, "I-OBJ": 4}
label_map_relation = {"enable": 0, "cause": 1, "intend": 2, "prevent": 3, "no_relation": 4}

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

# Define model checkpoint path
best_model_path = "best_model_combined.pth"


class MultiHeadRoBERTa(RobertaPreTrainedModel):
    def __init__(self, config, num_relation_types=5, num_bio_labels=5, max_seq_length=128):
        super().__init__(config)
        self.roberta = RobertaModel(config)

        self.max_seq_length = max_seq_length

        # Relation existence classification (binary)
        self.relation_head = nn.Linear(config.hidden_size, 2)

        # Relation type classification (multiclass)
        self.relation_type_head = nn.Linear(config.hidden_size, num_relation_types)

        # BIO tagging classification (token classification)
        self.token_classification_head = nn.Linear(config.hidden_size, num_bio_labels)

        # Softmax for token classification
        self.softmax = nn.Softmax(dim=-1)

        # Loss functions
        self.loss_fn_relation = nn.CrossEntropyLoss()
        self.loss_fn_type = nn.CrossEntropyLoss()
        self.loss_fn_bio = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels_relation=None, labels_type=None, labels_bio=None):
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # print(f"\nBatch Size: {batch_size}")

        # Get transformer outputs
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
        pooled_output = outputs.pooler_output  # CLS token output for classification

        # print(f"sequence_output shape: {sequence_output.shape}")  # (batch_size, seq_length, hidden_size)
        # print(f"pooled_output shape: {pooled_output.shape}")  # (batch_size, hidden_size)

        # Compute relation existence (binary classification)
        relation_logits = self.relation_head(pooled_output).float()  # Ensure float32
        relation_probs = torch.softmax(relation_logits, dim=-1)
        relation_preds = torch.argmax(relation_probs, dim=-1)  # Shape: (batch_size,)

        # print(f"relation_logits shape: {relation_logits.shape}")  # (batch_size, 2)
        # print(f"relation_preds shape: {relation_preds.shape}")  # (batch_size,)

        loss = 0

        # Initialize tensors for outputs
        type_logits = torch.full(
            (batch_size, self.relation_type_head.out_features), 0, device=device, dtype=torch.float32
        )  # Default: No Relation (index 4)

        span_logits = torch.full(
            (batch_size, self.max_seq_length, self.token_classification_head.out_features), 0, device=device,
            dtype=torch.float32
        )  # Default: All "O" (index 0)

        # print(f"Initialized type_logits shape: {type_logits.shape}")  # (batch_size, num_relation_types)
        # print(f"Initialized span_logits shape: {span_logits.shape}")  # (batch_size, seq_length, num_bio_labels)

        # Compute loss for relation existence classification
        if labels_relation is not None:
            labels_relation = labels_relation.to(device)
            loss += self.loss_fn_relation(relation_logits, labels_relation)

        # Process each sample independently
        for i in range(batch_size):
            if relation_preds[i] == 1:  # If relation exists for this sample
                type_logits[i] = self.relation_type_head(pooled_output[i]).float()  # Ensure float32
                span_logits[i] = self.softmax(
                    self.token_classification_head(sequence_output[i]).float()
                )  # Ensure float32

            else:  # If no relation, set fixed values
                type_logits[i, 4] = 1.0  # One-hot for "no_relation"
                span_logits[i, :, 0] = 1.0  # Set all tokens to "O" (index 0)

        # Ensure correct shape for token classification logits
        span_logits = span_logits.view(batch_size, self.max_seq_length, -1)  # (batch_size, seq_length, num_bio_labels)

        # print(f"Final type_logits shape: {type_logits.shape}")  # (batch_size, num_relation_types)
        # print(f"Final span_logits shape: {span_logits.shape}")  # (batch_size, seq_length, num_bio_labels)

        # Compute loss for relation type and BIO tagging
        if labels_relation is not None and (relation_preds == 1).any():
            has_relation_mask = labels_relation == 1  # Mask for samples with relation

            if labels_type is not None:
                labels_type = labels_type.to(device)
                loss += self.loss_fn_type(
                    type_logits[has_relation_mask].view(-1, type_logits.shape[-1]),
                    labels_type[has_relation_mask].view(-1),
                )

            if labels_bio is not None:
                labels_bio = labels_bio.to(device)
                loss += self.loss_fn_bio(
                    span_logits.view(-1, span_logits.shape[-1]),
                    labels_bio.view(-1),
                )

        print(f"Final Loss: {loss}")

        return {
            "loss": loss,
            "relation_logits": relation_logits,
            "relation_preds": relation_preds,
            "relation_type_logits": type_logits,
            "span_logits": span_logits,  # Now correctly shaped
        }

class RelationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.texts = [self.preprocess_text(text) for text in dataframe['text'].tolist()]
        self.subjects = [self.preprocess_entity(subject) for subject in dataframe['subject'].tolist()]
        self.objects = [self.preprocess_entity(obj) for obj in dataframe['object'].tolist()]
        self.relation_types = dataframe['relation'].tolist()
        self.relation_exists = [0 if relation == "no_relation" else 1 for relation in self.relation_types]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def preprocess_text(self, text):
        """Preprocess text input by stripping spaces and ensuring consistency."""
        return str(text)

    def preprocess_entity(self, entity):
        """Preprocess entity (subject or object) to ensure consistency."""
        return str(entity)

    def tokenize_and_align_labels(self, text, subject, object, relation_type):
        """
        Tokenizes text and aligns BIO labels manually for a slow tokenizer.
        If the relation type is "no_relation", BIO labels are set to all "O".
        """
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze().tolist()
        attention_mask = encoded["attention_mask"].squeeze().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # Default all tokens to "O"
        labels = ["O"] * len(tokens)

        # If relation type is NOT "no_relation", align subject and object labels
        if relation_type != "no_relation":
            # Tokenize subject and object separately
            subject_tokens = self.tokenizer.tokenize(subject)
            object_tokens = self.tokenizer.tokenize(object)

            # Align subject
            for i in range(len(tokens) - len(subject_tokens) + 1):
                if tokens[i:i + len(subject_tokens)] == subject_tokens:
                    labels[i] = "B-SUBJ"
                    for j in range(1, len(subject_tokens)):
                        labels[i + j] = "I-SUBJ"

            # Align object
            for i in range(len(tokens) - len(object_tokens) + 1):
                if tokens[i:i + len(object_tokens)] == object_tokens:
                    labels[i] = "B-OBJ"
                    for j in range(1, len(object_tokens)):
                        labels[i + j] = "I-OBJ"

        # Convert labels to numeric
        numeric_labels = [label_map_bio[label] for label in labels]
        numeric_labels = numeric_labels[:self.max_len]  # Truncate if needed
        numeric_labels += [0] * (self.max_len - len(numeric_labels))  # Pad if needed

        return input_ids, attention_mask, numeric_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text, subject, object = self.texts[idx], self.subjects[idx], self.objects[idx]
        relation_type = self.relation_types[idx]

        input_ids, attention_mask, numeric_labels = self.tokenize_and_align_labels(text, subject, object, relation_type)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels_relation": torch.tensor(self.relation_exists[idx], dtype=torch.long),
            "labels_type": torch.tensor(label_map_relation[relation_type], dtype=torch.long),
            "labels_bio": torch.tensor(numeric_labels, dtype=torch.long)
        }

from tqdm import tqdm

def train_and_save_best_model(model, train_loader, dev_loader, epochs=10, learning_rate=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for batch in tqdm_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_relation = batch["labels_relation"].to(device)
            labels_type = batch["labels_type"].to(device)
            labels_bio = batch["labels_bio"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels_relation=labels_relation,
                labels_type=labels_type,
                labels_bio=labels_bio
            )

            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            tqdm_loader.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_loss:.4f}")

        validation_results = evaluate_model(model, dev_loader, device)
        val_f1 = (validation_results["relation"]["f1"] + validation_results["relation_type"]["f1"] +
                  validation_results["bio"]["f1"]) / 3

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with average F1-score: {best_f1:.4f}")

    print("Training complete.")


def load_best_model(model, model_path):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Best model loaded.")
    else:
        print("No saved model found, using current model.")
    return model


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_labels_relation, all_preds_relation = [], []
    all_labels_type, all_preds_type = [], []
    all_labels_bio, all_preds_bio = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_relation = batch["labels_relation"].to(device)
            labels_type = batch["labels_type"].to(device)
            labels_bio = batch["labels_bio"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels_relation=labels_relation,
                labels_type=labels_type,
                labels_bio=labels_bio
            )

            loss = outputs["loss"]
            total_loss += loss.item()

            relation_preds = torch.argmax(outputs["relation_logits"], dim=-1).cpu().numpy()
            type_preds = torch.argmax(outputs["relation_type_logits"], dim=-1).cpu().numpy()
            bio_preds = torch.argmax(outputs["span_logits"], dim=-1).cpu().numpy().flatten()

            all_labels_relation.extend(labels_relation.cpu().numpy())
            all_preds_relation.extend(relation_preds)
            all_labels_type.extend(labels_type.cpu().numpy())
            all_preds_type.extend(type_preds)
            all_labels_bio.extend(labels_bio.cpu().numpy().flatten())
            all_preds_bio.extend(bio_preds)

    avg_loss = total_loss / len(dataloader)

    relation_acc = accuracy_score(all_labels_relation, all_preds_relation)
    relation_report = precision_recall_fscore_support(all_labels_relation, all_preds_relation, average="binary")

    type_acc = accuracy_score(all_labels_type, all_preds_type)
    type_report = precision_recall_fscore_support(all_labels_type, all_preds_type, average="macro")

    bio_acc = accuracy_score(all_labels_bio, all_preds_bio)
    bio_report = precision_recall_fscore_support(all_labels_bio, all_preds_bio, average="macro")

    print(f"Validation Loss: {avg_loss:.4f}")
    print("Relation Classification Report:")
    print(
        f"Accuracy: {relation_acc:.4f}, Precision: {relation_report[0]:.4f}, Recall: {relation_report[1]:.4f}, F1: {relation_report[2]:.4f}")

    print("Relation Type Classification Report:")
    print(
        f"Accuracy: {type_acc:.4f}, Precision: {type_report[0]:.4f}, Recall: {type_report[1]:.4f}, F1: {type_report[2]:.4f}")

    print("BIO Tagging Classification Report:")
    print(
        f"Accuracy: {bio_acc:.4f}, Precision: {bio_report[0]:.4f}, Recall: {bio_report[1]:.4f}, F1: {bio_report[2]:.4f}")

    return {
        "validation_loss": avg_loss,
        "relation": {"accuracy": relation_acc, "precision": relation_report[0], "recall": relation_report[1],
                     "f1": relation_report[2]},
        "relation_type": {"accuracy": type_acc, "precision": type_report[0], "recall": type_report[1],
                          "f1": type_report[2]},
        "bio": {"accuracy": bio_acc, "precision": bio_report[0], "recall": bio_report[1], "f1": bio_report[2]},
    }

train_df = pd.read_csv('/data/Youss/RE/Bert_based_classification/combined.csv')
dev_df = pd.read_csv('/data/Youss/RE/Bert_based_classification/dev.csv')

train_dataset = RelationDataset(train_df, tokenizer)
dev_dataset = RelationDataset(dev_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)
model = MultiHeadRoBERTa.from_pretrained("roberta-large")
train_and_save_best_model(model, train_loader, dev_loader, epochs=10, learning_rate=2e-5)


# Load test dataset
test_df = pd.read_csv('/data/Youss/RE/Bert_based_classification/test.csv')
test_dataset = RelationDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiHeadRoBERTa.from_pretrained("roberta-large")

model = load_best_model(model, best_model_path)
model.to(device)

print("Final Test Results:")
test_results = evaluate_model(model, test_loader, device)
