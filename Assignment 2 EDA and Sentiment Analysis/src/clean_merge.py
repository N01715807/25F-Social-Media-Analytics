import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
reviews_path = BASE / "data" / "rotten_tomatoes_critic_reviews.csv"
movies_path = BASE / "data" / "rotten_tomatoes_movies.csv"
reviews = pd.read_csv(reviews_path)
movies = pd.read_csv(movies_path, usecols=['rotten_tomatoes_link','movie_title','genres'])

print("---Before cleaning:---")
print("Critic Reviews Data")
print(reviews.isna().sum())
print(reviews.shape)
print("Movies Data")
print(movies.isna().sum())
print(movies.shape)

print("---cleaning reviews:---")
reviews = reviews.dropna(subset=['review_content', 'rotten_tomatoes_link'])
print("---After dropna---")
print(reviews.shape)

reviews['review_content'] = reviews['review_content'].str.strip()
reviews['review_content'] = reviews['review_content'].str.lower()
reviews['review_content'] = reviews['review_content'].str.replace(r'http[s]?://\S+', '', regex=True)
reviews['review_content'] = reviews['review_content'].str.replace(r'[\r\n\t]+', ' ', regex=True)
reviews['review_content'] = reviews['review_content'].str.replace(r'\s+', ' ', regex=True)
mask_empty = reviews['review_content'].str.len().fillna(0) == 0
reviews = reviews.loc[~mask_empty]
print("---After text cleaning---")
print(reviews.isna().sum())
print(reviews.shape)

n_before = len(reviews)
print("---After duplicates---")
reviews = reviews.drop_duplicates(
    subset=['rotten_tomatoes_link', 'review_content'],
    keep='first'
)
n_after = len(reviews)
removed = n_before - n_after
print(f"Removed {removed} duplicate rows")
print(reviews.shape)

print("---cleaning movies:---")
movies = movies.drop_duplicates(
    subset=['rotten_tomatoes_link'],
    keep='first'
)
movies['rotten_tomatoes_link'] = movies['rotten_tomatoes_link'].str.strip()
movies['movie_title'] = movies['movie_title'].str.strip()
print(movies.isna().sum())
print(movies.shape)
print(movies['rotten_tomatoes_link'].is_unique)

print("---merged:---")
merged = pd.merge(
    reviews,
    movies,
    how='left',
    on='rotten_tomatoes_link',
    validate='m:1'
)
print(merged.shape)
print(merged[['review_content','movie_title','genres']].head())
print(merged.isna().sum())

export_path = BASE / "export" / "merged_clean.csv"
merged.to_csv(export_path, index=False)
print(f"Saved merged dataset to {export_path}")