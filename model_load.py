import torch
from transformers import BertTokenizer, BertForSequenceClassification

emotion_list = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

model = BertForSequenceClassification.from_pretrained("emotion_bert_model")
tokenizer = BertTokenizer.from_pretrained("emotion_bert_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_emotions(text, threshold=0.3):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        predicted = [emotion_list[i] for i, p in enumerate(probs) if p > threshold]
        return predicted

def get_cls_embedding(text):
    with torch.no_grad():
        encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model.bert(**encoded)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
