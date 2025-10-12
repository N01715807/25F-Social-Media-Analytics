import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent

print("=== Starting A2 Sentiment Analysis Pipeline ===")

print("\n[1/3] Cleaning and merging data...")
subprocess.run(["python", str(BASE / "clean_merge.py")], check=True)

print("\n[2/3] Performing sentiment analysis...")
subprocess.run(["python", str(BASE / "sentiment_analysis.py")], check=True)

print("\n[3/3] Generating visualizations...")
subprocess.run(["python", str(BASE / "visualization.py")], check=True)

print("\nAll tasks completed successfully. Check 'export/' and 'figures/' folders.")