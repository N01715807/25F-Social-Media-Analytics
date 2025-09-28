## Steps Done

### 1. EDA (Exploratory Data Analysis)
- Checked shape, data types, missing values, duplicates, and summary statistics.

### 2. Missing Value Handling
- **Gender** → filled with mode.  
- **Age** → filled with median per gender group.  
- **Sleep Hours** → filled with mean.  

### 3. Outlier Analysis (IQR Method)
- Checked **Age**, **Avg Daily Usage Hours**, **Sleep Hours**, **Mental Health Score**.  
- Found unrealistic values (e.g., Age = -5 or 120, Usage = 25 hrs, Sleep = 20 hrs).  

### 4. Outlier Handling
- **Age** restricted to 16–30.  
- **Usage Hours** capped at 24.  
- **Mental Health Score** clipped to 0–10.  
- **Sleep Hours** > 12 replaced by median.  

### 5. Validation
- Re-checked `describe()` and confirmed values are within a reasonable range.  
- Final dataset is **clean**, with no missing values and no unrealistic outliers.  

### 6. Save Clean Data
- Exported the cleaned dataset to CSV:  
  ```python
  df.to_csv("data/clean_dataset.csv", index=False)