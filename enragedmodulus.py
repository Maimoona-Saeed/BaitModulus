#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip uninstall -y peft accelerate diffusers datasets gradio')


# In[2]:


get_ipython().system('pip install --quiet      sentence-transformers==2.6.1      transformers==4.37.2      huggingface_hub==0.20.3      faiss-cpu pandas numpy scikit-learn streamlit')


# In[4]:


import time, gc
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline # Ensure pipeline is imported
from sklearn.metrics import f1_score
import streamlit as st

print("pipeline imported successfully:", 'pipeline' in locals())

start_time = time.time()

# Load dataset (Sentiment140 subset)
csv_path = '/training.1600000.processed.noemoticon.csv'
df = pd.read_csv(csv_path, encoding='latin-1', header=None,
                 names=['sentiment', 'id', 'date', 'query', 'user', 'text'],
                 on_bad_lines='skip', engine='python')
df = df[['text', 'sentiment']].sample(2000, random_state=42).reset_index(drop=True)


# In[5]:


# Demo metadata
df['timestamp'] = pd.to_datetime('now')
df['city'] = np.random.choice(['Lahore', 'New York', 'London', 'Tokyo'], size=len(df))
df['country'] = np.random.choice(['Pakistan', 'USA', 'UK', 'Japan'], size=len(df))
df['age_group'] = np.random.choice(['18-24', '25-34', '35-44'], size=len(df))

# Embeddings
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embed_model.encode(df['text'].tolist(), batch_size=128, show_progress_bar=False)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


# In[ ]:


# Emotion classifier
emotion_model = pipeline(
    'zero-shot-classification',
    model='facebook/bart-large-mnli',
    device=0 if torch.cuda.is_available() else -1
)

emotion_to_brain = {
    'anger': 'amygdala', 'fear': 'amygdala', 'sadness': 'insula',
    'joy': 'nucleus accumbens', 'disgust': 'insula', 'surprise': 'prefrontal cortex'
}

def score_post_batch(texts):
    emotions_list = emotion_model(
        texts,
        candidate_labels=['anger', 'fear', 'sadness', 'joy', 'disgust', 'surprise'],
        multi_label=True,
        batch_size=64 # Increased batch size for potential optimization
    )
    results = []
    for emotions in emotions_list:
        emo_dict = dict(zip(emotions['labels'], emotions['scores']))
        rage_score = min((emo_dict.get('anger',0) + emo_dict.get('disgust',0)), 1.0)
        arousal = (emo_dict.get('anger',0) + emo_dict.get('fear',0) + emo_dict.get('surprise',0)) / 3
        brain_region = emotion_to_brain[max(emo_dict, key=emo_dict.get)] if max(emo_dict.values()) > 0.5 else None
        results.append({'rage_score': rage_score, 'emotions': emo_dict, 'arousal': arousal, 'brain_region': brain_region})
    return results

# Apply batch processing
batch_size = 64 # Define batch size
scores_list = []
for i in range(0, len(df), batch_size):
    batch_texts = df['text'].iloc[i:i+batch_size].tolist()
    scores_list.extend(score_post_batch(batch_texts))

df['scores'] = scores_list


def generate_campaign(row):
    if row['scores']['rage_score'] < 0.3 or row['scores']['emotions'].get('joy',0) > 0.5:
        emo = max(row['scores']['emotions'], key=row['scores']['emotions'].get)
        return f"Target {row['city']}, {row['country']} with {emo}-driven campaign for {row['age_group']}."
    return "Avoid: High rage potential."

df['campaign'] = df.apply(generate_campaign, axis=1)


# In[ ]:


# Eval
true_rage = [1 if s['emotions']['anger'] > 0.5 else 0 for s in df['scores'].sample(20, random_state=42)]
pred_rage = [1 if s['rage_score'] > 0.5 else 0 for s in df['scores'].sample(20, random_state=42)]
f1 = f1_score(true_rage, pred_rage)


# In[ ]:


# Streamlit dashboard
st.title("BaitModulus Dashboard")
st.subheader("Metrics")
st.write(f"F1 Score (Rage Detection): {f1:.2f}")
st.write(f"Runtime: {(time.time() - start_time):.2f} seconds")
st.dataframe(df[['text','city','country','age_group','scores','campaign']].head())
st.line_chart(df.set_index('timestamp')['scores'].apply(lambda x: x['rage_score']).head(100))

del embed_model, emotion_model, embeddings
gc.collect()


# In[ ]:


get_ipython().system('pip install numpy<2')


# In[ ]:


get_ipython().system('pip install --upgrade huggingface_hub')


# In[ ]:


get_ipython().system('pip install "numpy<2"')

