# import pandas and numpy
import pandas as pd
import numpy as np

# import csv
df = pd.read_csv("data/raw_dataset.csv")

# Exploratory Data Analysis (EDA)
# Shape
print(df.shape); print(df.info()); print(df.dtypes); 
# Missing values & Duplicates
print(df.isnull().sum()); print(df.duplicated().sum()); 
# Summary stats
print( df.describe()); 

# Fill missing values
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0]); 
df['Age'] = df.groupby('Gender')['Age'].transform(lambda x: x.fillna(x.median())); 
df['Sleep_Hours_Per_Night'] = df['Sleep_Hours_Per_Night'].fillna(df['Sleep_Hours_Per_Night'].mean()); 
print(df.isnull().sum())

# Distribution & Outliers
print(df['Gender'].value_counts()); 
print(df['Academic_Level'].value_counts()); 
print(df['Country'].value_counts()); 
print(df['Most_Used_Platform'].value_counts()); 
print(df['Affects_Academic_Performance'].value_counts()); 
print(df['Relationship_Status'].value_counts()); 
df['Age'].hist(); df['Age'].plot.box(); 
df['Avg_Daily_Usage_Hours'].hist(); df['Avg_Daily_Usage_Hours'].plot.box(); 
df['Sleep_Hours_Per_Night'].hist(); df['Sleep_Hours_Per_Night'].plot.box(); 

# Outlier Analysis
print(df[df])
