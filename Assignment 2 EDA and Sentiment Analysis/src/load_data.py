import pandas as pd
import numpy as np

print("Critic Reviews Data")
reviews = pd.read_csv('data/rotten_tomatoes_critic_reviews.csv')
print(reviews.head())
print(reviews.info())
print("Columns:", reviews.columns.tolist())

print("Movies Data")
movies = pd.read_csv('data/rotten_tomatoes_movies.csv')
print(movies.head())
print(movies.info())
print("Columns:", reviews.columns.tolist())