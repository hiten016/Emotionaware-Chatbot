import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from model_load import get_cls_embedding

train_data = pd.read_csv("emotion_data/train.tsv", sep='\t', header=None, names=['Text', 'Labels', 'ID'])
with open("emotion_data/emotions.txt") as f:
    emotion_list = f.read().splitlines()

def idx2emotion(idx_str):
    return [emotion_list[int(i)] for i in idx_str.split(',') if i.isdigit()]

train_data['Emotions'] = train_data['Labels'].apply(idx2emotion)
sample_df = train_data.sample(1000, random_state=42).reset_index(drop=True)

embeddings, labels = [], []
for _, row in sample_df.iterrows():
    emb = get_cls_embedding(row['Text'])
    embeddings.append(emb)
    labels.append(row['Emotions'][0] if row['Emotions'] else 'neutral')

X = np.array(embeddings)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='tab20', s=50)
plt.title("PCA of [CLS] Embeddings from Fine-Tuned BERT")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
