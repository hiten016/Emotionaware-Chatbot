from transformers import BertTokenizer, BertForSequenceClassification
from utils import load_emotions
import torch

model = BertForSequenceClassification.from_pretrained("bert-goemotions")
tokenizer = BertTokenizer.from_pretrained("bert-goemotions")
model.eval()

emotions = load_emotions("data/emotions")

text = input("Enter text: ")

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

print(f"Predicted Emotion: {emotions[prediction]}")
