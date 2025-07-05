from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import classification_report
from utils import load_emotions
import torch

model = BertForSequenceClassification.from_pretrained("bert-goemotions")
tokenizer = BertTokenizer.from_pretrained("bert-goemotions")
model.eval()

emotions = load_emotions("data/emotions")

# Load test set
dataset = load_dataset(
    "csv",
    data_files="data/test.tsv",
    delimiter="^I",
    column_names=["text", "labels"]
)

texts = dataset["train"]["text"]
true_labels = [
    int(label.split(",")[0]) if isinstance(label, str) else 0
    for label in dataset["train"]["labels"]
]

preds = []
for i in range(0, len(texts), 16):
    batch = texts[i:i + 16]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        batch_preds = torch.argmax(outputs.logits, dim=1)
        preds.extend(batch_preds.tolist())

print(classification_report(true_labels, preds, target_names=emotions))
