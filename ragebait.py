import time
import gc
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os

# Start timing
start_time = time.time()

# Download dataset (run once in Colab) with error check
if not os.path.exists('/content/training.1600000.processed.noemoticon.csv'):
    os.system('pip install kaggle -q')
    os.system('mkdir -p ~/.kaggle && mv /content/kaggle.json ~/.kaggle/')  # Upload kaggle.json manually
    os.system('kaggle datasets download -d kazanova/sentiment140 -p /content/ --unzip -q')
    if not os.path.exists('/content/training.1600000.processed.noemoticon.csv'):
        raise FileNotFoundError("Dataset download failed. Check Kaggle setup or network.")

# Load dataset
csv_path = '/content/training.1600000.processed.noemoticon.csv'
try:
    df = pd.read_csv(csv_path, encoding='latin-1', header=None,
                     names=['sentiment', 'id', 'date', 'query', 'user', 'text'],
                     on_bad_lines='skip', engine='python')
    df = df[['text', 'sentiment']].sample(300, random_state=42).reset_index(drop=True)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Demo metadata
df['timestamp'] = pd.to_datetime('now')
df['city'] = np.random.choice(['Lahore', 'New York', 'London', 'Tokyo'], size=len(df))
df['country'] = np.random.choice(['Pakistan', 'USA', 'UK', 'Japan'], size=len(df))
df['age_group'] = np.random.choice(['18-24', '25-34', '35-44'], size=len(df))

# Emotion classifier with fallback
try:
    emotion_model = pipeline(
        'zero-shot-classification',
        model='distilbert-base-uncased-finetuned-mnli',
        device=0 if torch.cuda.is_available() else -1
    )
except Exception as e:
    print(f"Model 'distilbert-base-uncased-finetuned-mnli' failed: {e}. Falling back to 'facebook/bart-large-mnli'.")
    emotion_model = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', device=-1)

emotion_to_brain = {
    'anger': 'amygdala', 'fear': 'amygdala', 'sadness': 'insula',
    'joy': 'nucleus accumbens', 'disgust': 'insula', 'surprise': 'prefrontal cortex'
}

def score_post_batch(texts):
    try:
        emotions_list = emotion_model(
            texts,
            candidate_labels=['anger', 'fear', 'sadness', 'joy', 'disgust', 'surprise'],
            multi_label=True,
            batch_size=256
        )
        results = []
        for emotions in emotions_list:
            emo_dict = dict(zip(emotions['labels'], emotions['scores']))
            rage_score = min((emo_dict.get('anger', 0) + emo_dict.get('disgust', 0)), 1.0)
            arousal = (emo_dict.get('anger', 0) + emo_dict.get('fear', 0) + emo_dict.get('surprise', 0)) / 3
            brain_region = emotion_to_brain[max(emo_dict, key=emo_dict.get)] if max(emo_dict.values()) > 0.5 else None
            results.append({'rage_score': rage_score, 'emotions': emo_dict, 'arousal': arousal, 'brain_region': brain_region})
        return results
    except Exception as e:
        print(f"Scoring failed: {e}")
        return [{} for _ in texts]

# Apply batch processing
batch_size = 256
scores_list = []
for i in range(0, len(df), batch_size):
    batch_texts = df['text'].iloc[i:i + batch_size].tolist()
    scores_list.extend(score_post_batch(batch_texts))

df['scores'] = scores_list

def generate_campaign(row):
    if row['scores'].get('rage_score', 0) < 0.3 or row['scores'].get('emotions', {}).get('joy', 0) > 0.5:
        emo = max(row['scores'].get('emotions', {}), key=row['scores'].get('emotions', {}).get) if row['scores'].get('emotions', {}) else 'neutral'
        return f"Target {row['city']}, {row['country']} with {emo}-driven campaign for {row['age_group']}."
    return "Avoid: High rage potential."

df['campaign'] = df.apply(generate_campaign, axis=1)

# Evaluation
try:
    true_rage = [1 if s.get('emotions', {}).get('anger', 0) > 0.5 else 0 for s in df['scores'].sample(50, random_state=42)]
    pred_rage = [1 if s.get('rage_score', 0) > 0.5 else 0 for s in df['scores'].sample(50, random_state=42)]
    f1 = f1_score(true_rage, pred_rage)
    precision = precision_score(true_rage, pred_rage)
    recall = recall_score(true_rage, pred_rage)
    cm = confusion_matrix(true_rage, pred_rage)
except Exception as e:
    print(f"Evaluation failed: {e}")
    f1, precision, recall, cm = 0.0, 0.0, 0.0, [[0, 0], [0, 0]]

# Emotion distribution for graphing
emotion_df = pd.DataFrame(df['scores'].tolist()).melt(value_name='score')
emotion_df = emotion_df.groupby('variable')['score'].mean().reset_index()

# Streamlit dashboard
st.title("BaitModulus Dashboard")
st.subheader("Metrics")
st.write(f"F1 Score (Rage Detection): {f1:.2f}")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"Runtime: {(time.time() - start_time):.2f} seconds")

# Graphs
st.subheader("Emotion Distribution")
fig1, ax1 = plt.subplots()
sns.barplot(x='variable', y='score', data=emotion_df, ax=ax1)
ax1.set_title('Average Emotion Scores')
st.pyplot(fig1)

st.subheader("Confusion Matrix")
fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title('Rage Prediction Confusion Matrix')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')
st.pyplot(fig2)

st.dataframe(df[['text', 'city', 'country', 'age_group', 'scores', 'campaign']].head())
st.line_chart(df.set_index('timestamp')['scores'].apply(lambda x: x['rage_score']).head(100))

# Cleanup
del emotion_model
gc.collect()

# Optional upgrades (uncomment if needed locally)
# os.system('pip install --upgrade huggingface_hub')
# os.system('pip install "numpy<2"')
