import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
in_csv = BASE / "export" / "sentiment_results.csv"
fig_dir = BASE / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(in_csv)

counts = (
    df["predicted_sentiment"]
    .value_counts()
    .reindex(["Positive", "Neutral", "Negative"])
    .fillna(0)
)
plt.figure()
counts.plot(kind="bar", color=["green", "gray", "red"])
plt.title("Predicted Sentiment Distribution")
plt.xlabel("Sentiment Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(fig_dir / "sentiment_distribution.png", dpi=150)
plt.close()
print("sentiment_distribution.png")

plt.figure()
plt.hist(df["polarity"].dropna(), bins=30, color="skyblue")
plt.title("Sentiment Polarity Histogram")
plt.xlabel("Polarity Score (-1 to 1)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(fig_dir / "sentiment_histogram_polarity.png", dpi=150)
plt.close()
print("entiment_histogram_polarity.png")