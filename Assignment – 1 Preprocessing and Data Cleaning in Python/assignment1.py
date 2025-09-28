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
def iqr_outlier_check(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]

    print(f"{column} → Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"Threshold range: [{lower:.2f}, {upper:.2f}]")
    print(f"Number of outliers detected: {outliers.shape[0]}")

    return outliers

for col in ["Age", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Mental_Health_Score"]:
    iqr_outlier_check(df, col)

print(df[df["Conflicts_Over_Social_Media"] > 5])
print(df[(df["Addicted_Score"] < 1) | (df["Addicted_Score"] > 10)])

# Post-Cleaning Validation / Data Quality Check
df = df[(df['Age'] >= 16) & (df['Age'] <= 30)]
df['Avg_Daily_Usage_Hours'] = df['Avg_Daily_Usage_Hours'].clip(upper=24)
df['Mental_Health_Score'] = df['Mental_Health_Score'].clip(lower=0, upper=10)

median_sleep = df['Sleep_Hours_Per_Night'].median()
df.loc[df['Sleep_Hours_Per_Night'] > 12, 'Sleep_Hours_Per_Night'] = median_sleep

print(df['Age'].describe())
print(df['Avg_Daily_Usage_Hours'].describe())
print(df['Sleep_Hours_Per_Night'].describe())
print(df['Mental_Health_Score'].describe())

# save
df.to_csv("data/clean_dataset.csv", index=False)