from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from utils import load_emotions

# Load emotion labels
emotions = load_emotions("data/emotions")
num_labels = len(emotions)

# Load TSV dataset
data_files = {
    "train": "data/train.tsv",
    "validation": "data/dev.tsv",
    "test": "data/test.tsv"
}
dataset = load_dataset("csv", data_files=data_files, delimiter="\t", column_names=["text", "labels"])

# Label preprocessing
def preprocess_label(example):
    label = example.get("labels")
    if label is None or label == "":
        example["labels"] = 0  # default to 0
    else:
        try:
            example["labels"] = int(label.split(",")[0])
        except:
            example["labels"] = 0
    return example

dataset = dataset.map(preprocess_label)

# Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# Training configuration (compatible with older transformers)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir="./logs"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Save model
trainer.save_model("bert-goemotions")
tokenizer.save_pretrained("bert-goemotions")
