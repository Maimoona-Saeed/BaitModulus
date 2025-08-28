#  BaitModulus — Rage-Bait Sentiment & Emotion Analysis

BaitModulus is an NLP pipeline designed to analyze **rage-bait on social media**.  
It detects **emotions, toxicity, demographics, and geography**, and even maps emotions to **neuroscience brain regions** for storytelling.  

This is not a toy sentiment classifier — it’s a **flagship ML project** combining:
- **Transformers (DistilBERT, MiniLM)**
- **Sentence Embeddings + FAISS retrieval**
- **Toxicity detection (Detoxify)**
- **Neuroscience-inspired mappings**
- **Dashboards/plots for insights**

---

##  Features
- Scrape or load tweets / posts (tested with [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140))
- Emotion detection → anger, fear, sadness, joy, disgust, surprise
- Toxicity scoring → how provocative is the content?
- Rage score → composite of emotion × toxicity
- Demographic & geo enrichment (mocked for demo if no metadata)
- Brain region mapping (e.g., anger → amygdala)
- Visualizations → heatmaps, rage density per region
- Scalable to **200k+ records**

---

## 🛠️ Tech Stack
- **Python 3.12**
- **PyTorch 2.2.2+cpu**
- **Transformers (Hugging Face)**
- **SentenceTransformers (MiniLM-L6-v2)**
- **FAISS (Facebook AI Similarity Search)**
- **Detoxify**
- **Pandas / NumPy / Scikit-learn / Matplotlib**

---

##  Installation

Clone repo and install dependencies:

```bash
git clone https://github.com/yourusername/baitmodulus.git
cd baitmodulus

# Install requirements
pip install -r requirements.txt
