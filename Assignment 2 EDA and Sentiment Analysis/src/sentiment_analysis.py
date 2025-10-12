import pandas as pd
import numpy as np
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE = Path(__file__).resolve().parent.parent
in_path = BASE / "export" / "merged_clean.csv"
out_csv = BASE / "export" / "sentiment_results.csv"
out_metrics = BASE / "export" / "metrics.txt"

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv(in_path)
df = df.dropna(subset=["review_content"])
df["review_content"] = df["review_content"].astype(str).str.strip()
df = df[df["review_content"].str.len() > 0].copy()

df["tokens"] = df["review_content"].str.lower().apply(word_tokenize)

sw = set(stopwords.words("english"))
for w in ["not", "no", "never"]:
    sw.discard(w)
df["tokens"] = df["tokens"].apply(lambda toks: [w for w in toks if w.isalpha() and w not in sw])

lemmatizer = WordNetLemmatizer()
df["tokens"] = df["tokens"].apply(lambda toks: [lemmatizer.lemmatize(w) for w in toks])

df["clean_text"] = df["tokens"].apply(lambda toks: " ".join(toks))


df["polarity"] = df["clean_text"].apply(lambda x: TextBlob(x).sentiment.polarity)
df["subjectivity"] = df["clean_text"].apply(lambda x: TextBlob(x).sentiment.subjectivity)

def classify(p):
    if p >= 0.2:
        return "Positive"
    elif p <= -0.2:
        return "Negative"
    else:
        return "Neutral"

df["predicted_sentiment"] = df["polarity"].apply(classify)

df["actual_sentiment"] = df["review_type"].map({"Fresh": "Positive", "Rotten": "Negative"})

mask = df["predicted_sentiment"].isin(["Positive", "Negative"]) & df["actual_sentiment"].isin(["Positive", "Negative"])
y_true = df.loc[mask, "actual_sentiment"]
y_pred = df.loc[mask, "predicted_sentiment"]

if len(y_true) > 0:
    acc = accuracy_score(y_true, y_pred)
    cls_rep = classification_report(y_true, y_pred, digits=3)
    cm = confusion_matrix(y_true, y_pred, labels=["Positive", "Negative"])
else:
    acc, cls_rep, cm = None, "N/A", None

cols_to_keep = [c for c in ["rotten_tomatoes_link","movie_title","genres","review_date","critic_name","publisher_name"] if c in df.columns]
out_df = df[cols_to_keep + ["review_type","actual_sentiment","review_content","clean_text","polarity","subjectivity","predicted_sentiment"]]
out_df.to_csv(out_csv, index=False)

with open(out_metrics, "w", encoding="utf-8") as f:
    f.write("=== Sentiment Analysis (TextBlob + NLTK) ===\n")
    f.write(f"Total rows: {len(df)}\n")
    f.write(f"Evaluated rows: {int(mask.sum())}\n\n")
    if acc is not None:
        f.write(f"Accuracy: {acc:.3f}\n\n")
        f.write("Classification Report:\n")
        f.write(cls_rep + "\n\n")
        f.write("Confusion Matrix [Positive, Negative]:\n")
        f.write(str(cm) + "\n")
    else:
        f.write("No valid samples to evaluate.\n")

print(f"Results saved to: {out_csv}")
print(f"Metrics saved to: {out_metrics}")
