import pandas as pd
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer, RobertaPreTrainedModel, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
# Define label mappings
label_map_bio = {"O": 0, "B-SUBJ": 1, "I-SUBJ": 2, "B-OBJ": 3, "I-OBJ": 4}
label_map_relation = {"enable": 0, "cause": 1, "intend": 2, "prevent": 3, "no_relation": 4}

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

        print(f"\nBatch Size: {batch_size}")

        # Get transformer outputs
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
        pooled_output = outputs.pooler_output  # CLS token output for classification

        print(f"sequence_output shape: {sequence_output.shape}")  # (batch_size, seq_length, hidden_size)
        print(f"pooled_output shape: {pooled_output.shape}")  # (batch_size, hidden_size)

        # Compute relation existence (binary classification)
        relation_logits = self.relation_head(pooled_output).float()  # Ensure float32
        relation_probs = torch.softmax(relation_logits, dim=-1)
        relation_preds = torch.argmax(relation_probs, dim=-1)  # Shape: (batch_size,)

        print(f"relation_logits shape: {relation_logits.shape}")  # (batch_size, 2)
        print(f"relation_preds shape: {relation_preds.shape}")  # (batch_size,)

        loss = 0

        # Initialize tensors for outputs
        type_logits = torch.full(
            (batch_size, self.relation_type_head.out_features), 0, device=device, dtype=torch.float32
        )  # Default: No Relation (index 4)

        span_logits = torch.full(
            (batch_size, self.max_seq_length, self.token_classification_head.out_features), 0, device=device,
            dtype=torch.float32
        )  # Default: All "O" (index 0)

        print(f"Initialized type_logits shape: {type_logits.shape}")  # (batch_size, num_relation_types)
        print(f"Initialized span_logits shape: {span_logits.shape}")  # (batch_size, seq_length, num_bio_labels)

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

        print(f"Final type_logits shape: {type_logits.shape}")  # (batch_size, num_relation_types)
        print(f"Final span_logits shape: {span_logits.shape}")  # (batch_size, seq_length, num_bio_labels)

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


import torch
from torch.utils.data import Dataset

label_map_bio = {"O": 0, "B-SUBJ": 1, "I-SUBJ": 2, "B-OBJ": 3, "I-OBJ": 4}
label_map_relation = {"enable": 0, "cause": 1, "intend": 2, "prevent": 3, "no_relation": 4}

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
        return str(text).strip()

    def preprocess_entity(self, entity):
        """Preprocess entity (subject or object) to ensure consistency."""
        return str(entity).strip().lower()

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


# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

# Load datasets separately
train_df = pd.read_csv('/data/Youss/RE/Bert_based_classification/combined.csv')
dev_df = pd.read_csv('/data/Youss/RE/Bert_based_classification/dev.csv')
test_df = pd.read_csv('/data/Youss/RE/Bert_based_classification/test.csv')

train_dataset = RelationDataset(train_df, tokenizer)
dev_dataset = RelationDataset(dev_df, tokenizer)
test_dataset = RelationDataset(test_df, tokenizer)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


def compute_metrics(pred):
    """
    Computes evaluation metrics for:
    - Relation existence (binary classification)
    - Relation type (multiclass classification, only for relation=1)
    - BIO tagging (token classification, only for relation=1)
    """
    labels = pred.label_ids
    preds = pred.predictions

    print("\n===== Debugging Shapes in compute_metrics =====")

    # **Print labels and predictions structure**
    print(f"Labels Structure: {type(labels)} | Length: {len(labels)}")
    print(f"Predictions Structure: {type(preds)} | Length: {len(preds)}")

    # **Relation existence classification**
    labels_relation = np.array(labels[0])  # Shape: (batch_size,)
    preds_relation = np.argmax(np.array(preds[0]), axis=-1)  # Convert logits to predictions

    print(f"labels_relation shape: {labels_relation.shape}")  # (batch_size,)
    print(f"preds_relation shape: {preds_relation.shape}")  # (batch_size,)

    relation_acc = accuracy_score(labels_relation, preds_relation)
    relation_metrics = precision_recall_fscore_support(labels_relation, preds_relation, average="binary",
                                                       zero_division=0)

    # **Fix for Relation Type Classification**
    if len(preds) > 1 and preds[1] is not None:
        labels_type = np.array(labels[1])  # Expected shape: (batch_size,)

        # **Ensure preds_type is an array**
        preds_type = np.array(preds[1])
        if preds_type.shape == ():  # If it's a scalar (np.int64), convert to array
            preds_type = np.array([preds_type])

        print(f"labels_type shape: {labels_type.shape}")  # (batch_size,)
        print(f"preds_type shape: {preds_type.shape}")  # (batch_size,)

        type_acc = accuracy_score(labels_type, preds_type)
        type_metrics = precision_recall_fscore_support(labels_type, preds_type, average="macro", zero_division=0)
    else:
        type_acc, type_metrics = 0.0, (0, 0, 0, 0)

    # **BIO Tagging classification**
    if len(preds) > 2 and preds[2] is not None:
        labels_bio = np.array(labels[2])  # Shape: (batch_size, seq_length)
        preds_bio = np.argmax(np.array(preds[2]), axis=-1)  # Convert logits to predictions (batch_size, seq_length)

        print(f"labels_bio shape (before flattening): {labels_bio.shape}")  # (batch_size, seq_length)
        print(f"preds_bio shape (before flattening): {preds_bio.shape}")  # (batch_size, seq_length)

        # **Flatten both for computing metrics**
        labels_bio = labels_bio.flatten()
        preds_bio = preds_bio.flatten()

        print(f"labels_bio shape (flattened): {labels_bio.shape}")  # (batch_size * seq_length,)
        print(f"preds_bio shape (flattened): {preds_bio.shape}")  # (batch_size * seq_length,)

        # **Ensure consistent shape before accuracy computation**
        min_length = min(len(labels_bio), len(preds_bio))
        labels_bio, preds_bio = labels_bio[:min_length], preds_bio[:min_length]

        print(f"labels_bio shape (final after trimming): {labels_bio.shape}")  # Ensure equal lengths
        print(f"preds_bio shape (final after trimming): {preds_bio.shape}")  # Ensure equal lengths

        bio_acc = accuracy_score(labels_bio, preds_bio)
        bio_metrics = precision_recall_fscore_support(labels_bio, preds_bio, average="macro", zero_division=0)
    else:
        bio_acc, bio_metrics = 0.0, (0, 0, 0, 0)

    print("===== End of Debugging Shapes =====\n")


    return {
        "relation_accuracy": relation_acc,
        "relation_precision": relation_metrics[0],
        "relation_recall": relation_metrics[1],
        "relation_f1": relation_metrics[2],

        "type_accuracy": type_acc,
        "type_precision": type_metrics[0],
        "type_recall": type_metrics[1],
        "type_f1": type_metrics[2],

        "bio_accuracy": bio_acc,
        "bio_precision": bio_metrics[0],
        "bio_recall": bio_metrics[1],
        "bio_f1": bio_metrics[2],
        "avg_f1": (bio_metrics[2] + type_metrics[2]+ relation_metrics[2])/3
    }


# Training arguments
training_args = TrainingArguments(
    output_dir="./relation_extraction/combined",  # ✅ Saves the best model here
    evaluation_strategy="epoch",  # ✅ Evaluate at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,  # ✅ Loads the best model at the end of training
    save_strategy="epoch",  # ✅ Saves at each epoch
    save_total_limit=1,  # ✅ Keeps only the best checkpoint, deletes older ones
    metric_for_best_model="avg_f1",  # ✅ Uses F1-score to select the best model
    greater_is_better=True,  # ✅ Higher F1-score is better
)


# Load model
model = MultiHeadRoBERTa.from_pretrained("roberta-large").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,  # ✅ Updated metrics
)

# Train model
trainer.train()

# # Evaluate model
# results = trainer.evaluate(test_dataset)
# print("Test Set Accuracy:", results["eval_accuracy"])

from sklearn.metrics import classification_report

# def evaluate_on_test_set(model, test_dataset, batch_size=8):
#     """
#     Evaluates the trained model on the test dataset and prints:
#     - Precision, Recall, F1-score for each class in relation existence, relation type, and BIO tagging.
#     - Macro-averaged Precision, Recall, F1-score for each category.
#     """
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
#
#     model.eval()  # Set model to evaluation mode
#     all_preds_relation, all_labels_relation = [], []
#     all_preds_type, all_labels_type = [], []
#     all_preds_bio, all_labels_bio = [], []
#
#     with torch.no_grad():
#         for batch in test_dataloader:
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels_relation = batch["labels_relation"].to(device)
#             labels_type = batch["labels_type"].to(device)
#             labels_bio = batch["labels_bio"].to(device)
#
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#
#             preds_relation = torch.argmax(outputs["relation_logits"], dim=-1).cpu().numpy()
#             preds_type = torch.argmax(outputs["relation_type_logits"], dim=-1).cpu().numpy()
#             preds_bio = torch.argmax(outputs["span_logits"], dim=-1).cpu().numpy()
#
#             all_preds_relation.extend(preds_relation)
#             all_labels_relation.extend(labels_relation.cpu().numpy())
#
#             all_preds_type.extend(preds_type)
#             all_labels_type.extend(labels_type.cpu().numpy())
#
#             all_preds_bio.extend(preds_bio.flatten())
#             all_labels_bio.extend(labels_bio.cpu().numpy().flatten())
#
#     # **PRINT EVALUATION RESULTS**
#     print("\n🔹 **Relation Existence Classification Report**")
#     print(classification_report(all_labels_relation, all_preds_relation, target_names=["No Relation", "Has Relation"], zero_division=0))
#
#     print("\n🔹 **Relation Type Classification Report**")
#     relation_labels = list(label_map_relation.keys())  # Get relation type names
#     print(classification_report(all_labels_type, all_preds_type, target_names=relation_labels, zero_division=0))
#
#     print("\n🔹 **BIO Tagging Classification Report**")
#     bio_labels = list(label_map_bio.keys())  # Get BIO tag names
#     print(classification_report(all_labels_bio, all_preds_bio, target_names=bio_labels, zero_division=0))
#
#     return {
#         "relation_report": classification_report(all_labels_relation, all_preds_relation, output_dict=True, zero_division=0),
#         "type_report": classification_report(all_labels_type, all_preds_type, output_dict=True, zero_division=0),
#         "bio_report": classification_report(all_labels_bio, all_preds_bio, output_dict=True, zero_division=0),
#     }
#
# # **RUN EVALUATION**
# test_results = evaluate_on_test_set(model, test_dataset)

from transformers import Trainer
import torch

def evaluate_test_set(test_dataset, trainer):
    """
    Evaluates the model on the test set and prints detailed metrics.
    """
    print("\nEvaluating on Test Set...")

    # Run evaluation using Trainer
    test_results = trainer.evaluate(test_dataset)

    # Print all computed metrics
    print("\n===== Test Set Evaluation Results =====")
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")

    return test_results

test_results = evaluate_test_set(test_dataset, trainer)


def predict_relation(model, tokenizer, text):
    """
    Predicts the relation existence, type, and BIO tagging for a given text.
    """
    model.eval()  # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize input text
    encoded = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Extract relation existence prediction
    relation_pred = torch.argmax(outputs["relation_logits"], dim=-1).item()

    # If relation exists, extract type and BIO labels
    if relation_pred == 1:
        relation_type = torch.argmax(outputs["relation_type_logits"], dim=-1).item()
        span_predictions = torch.argmax(outputs["span_logits"], dim=-1).squeeze().tolist()
    else:
        relation_type = "no_relation"
        span_predictions = [0] * 128  # All tokens are "O" if no relation exists (numeric index 0)

    # Convert BIO indices back to labels
    bio_labels = {0: "O", 1: "B-SUBJ", 2: "I-SUBJ", 3: "B-OBJ", 4: "I-OBJ"}
    decoded_bio = [bio_labels[idx] for idx in span_predictions]  # No more KeyError!

    return {
        "relation_exists": relation_pred,
        "relation_type": relation_type,
        "bio_labels": decoded_bio
    }

text = "the earthquacke has causes so many deads in Japan"
prediction = predict_relation(model, tokenizer, text)

print("\n===== Prediction Results =====")
print(f"Relation Exists: {prediction['relation_exists']}")
print(f"Relation Type: {prediction['relation_type']}")
print(f"BIO Labels: {prediction['bio_labels']}")
