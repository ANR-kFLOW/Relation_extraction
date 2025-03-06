import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from torch.nn import functional as F
from sklearn.metrics import classification_report
import os


MODEL_NAME = "FacebookAI/roberta-large"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


train_file = "/data/Youss/RE/TACL/end_to_end_models/Bert_based_classification/combined.csv"
dev_file = "/data/Youss/RE/TACL/end_to_end_models/Bert_based_classification/dev.csv"
test_file = "/data/Youss/RE/TACL/end_to_end_models/Bert_based_classification/test.csv"

df_train = pd.read_csv(train_file).sample(6790)
df_dev = pd.read_csv(dev_file).sample(620)
df_test = pd.read_csv(test_file).sample(630)

df_train = df_train.rename(columns={"text": "text", "relation": "label"})
df_dev = df_dev.rename(columns={"text": "text", "relation": "label"})
df_test = df_test.rename(columns={"text": "text", "relation": "label"})


label2id = {label: idx for idx, label in enumerate(pd.concat([df_train, df_dev, df_test])['label'].unique())}
print("Original labels:", df_train["label"].unique())
print("Encoded labels:", df_train["label"].map(label2id).unique())
id2label = {idx: label for label, idx in label2id.items()}
df_train['label'] = df_train['label'].map(label2id)
df_dev['label'] = df_dev['label'].map(label2id)
df_test['label'] = df_test['label'].map(label2id)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_inputs["label"] = examples["label"]
    return tokenized_inputs



dataset = {
    "train": Dataset.from_pandas(df_train),
    "validation": Dataset.from_pandas(df_dev),
    "test": Dataset.from_pandas(df_test),
}

tokenized_datasets = {split: dataset[split].map(tokenize_function, batched=True) for split in dataset}



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

train_dataset = tokenized_datasets["train"]
validation_dataset = tokenized_datasets["validation"]

epochs = 10
batch_size = 8
best_val_loss = float("inf")
best_model_path = "./new_roberta_rc_combined"


from tqdm import tqdm

for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(range(0, len(train_dataset), batch_size), desc=f"Epoch {epoch + 1}", leave=False)

    for i in progress_bar:
        batch = train_dataset.select(range(i, min(i + batch_size, len(train_dataset))))
        inputs = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt").to(device)
        labels = torch.tensor(batch["label"], dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch + 1}: Training Loss = {avg_train_loss}")

    model.eval()
    val_loss = 0
    for i in range(0, len(validation_dataset), batch_size):
        batch = validation_dataset.select(range(i, min(i + batch_size, len(validation_dataset))))
        inputs = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt").to(device)
        labels = torch.tensor(batch["label"]).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(validation_dataset)
    print(f"Epoch {epoch + 1}: Validation Loss = {avg_val_loss}")


    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print("Saving new best model...")
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)


# del model
# torch.cuda.empty_cache()


print("Loading best model for evaluation...")
model = AutoModelForSequenceClassification.from_pretrained(best_model_path).to(device)
model.eval()


test_texts = dataset["test"]["text"]
test_labels = torch.tensor(dataset["test"]["label"]).to(device)

batch_size = 8
all_preds = []
all_labels = []

for i in range(0, len(test_texts), batch_size):
    batch_texts = test_texts[i : i + batch_size]
    batch_labels = test_labels[i : i + batch_size]

    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(batch_labels.cpu().numpy())


report = classification_report(all_labels, all_preds, target_names=[id2label[i] for i in range(len(id2label))])
print(report)


with open("classification_report_combined_roberta.txt", "w") as f:
    f.write(report)


# del model
# torch.cuda.empty_cache()
