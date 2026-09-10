"""
Week 1 Task: Data Acquisition, Cleaning, and Preprocessing
Dataset: Titanic Passenger Data (public dataset)
Source: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110

# ---------------------------------------------------------------
# STEP 1: DATA ACQUISITION
# ---------------------------------------------------------------
df = pd.read_csv('titanic_raw.csv')
print("=== STEP 1: DATA ACQUISITION ===")
print(f"Shape: {df.shape}")
print(df.dtypes)
print(df.head())

# ---------------------------------------------------------------
# STEP 2: INITIAL DATA EXPLORATION
# ---------------------------------------------------------------
print("\n=== STEP 2: INITIAL EXPLORATION ===")
print(df.info())
print("\nSummary statistics:")
print(df.describe(include='all'))

missing_summary = df.isnull().sum().sort_values(ascending=False)
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_table = pd.DataFrame({'Missing Count': missing_summary, 'Missing %': missing_pct}).sort_values('Missing Count', ascending=False)
print("\nMissing values:\n", missing_table)

# Plot 1: Missing values heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Value Map (Raw Dataset)')
plt.tight_layout()
plt.savefig('plot1_missing_heatmap.png')
plt.close()

# Plot 2: Missing value bar chart
plt.figure(figsize=(7, 4))
missing_table[missing_table['Missing Count'] > 0]['Missing Count'].plot(kind='bar', color='#e07b39')
plt.title('Count of Missing Values per Column')
plt.ylabel('Missing Count')
plt.tight_layout()
plt.savefig('plot2_missing_bar.png')
plt.close()

# ---------------------------------------------------------------
# STEP 3: IDENTIFY MISSING VALUES, INCONSISTENCIES, OUTLIERS
# ---------------------------------------------------------------
print("\n=== STEP 3: DATA QUALITY ISSUES ===")

# Duplicates
dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes}")

# Inconsistent categorical entries
print("Unique Sex values:", df['Sex'].unique())
print("Unique Embarked values:", df['Embarked'].unique())
print("Unique Pclass values:", sorted(df['Pclass'].unique()))

# Outlier detection in Fare and Age using IQR
def iqr_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series[(series < lower) | (series > upper)], lower, upper

fare_outliers, fare_lo, fare_hi = iqr_outliers(df['Fare'].dropna())
age_outliers, age_lo, age_hi = iqr_outliers(df['Age'].dropna())
print(f"Fare outliers: {len(fare_outliers)} (bounds: {fare_lo:.2f} - {fare_hi:.2f})")
print(f"Age outliers: {len(age_outliers)} (bounds: {age_lo:.2f} - {age_hi:.2f})")

# Plot 3: Boxplots before cleaning
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df['Fare'], ax=axes[0], color='#f4a259')
axes[0].set_title('Fare Distribution (raw) - Outliers visible')
sns.boxplot(y=df['Age'], ax=axes[1], color='#5b8c85')
axes[1].set_title('Age Distribution (raw)')
plt.tight_layout()
plt.savefig('plot3_boxplots_raw.png')
plt.close()

# Erroneous entries: Age = 0 or negative, Fare = 0 with non-null ticket
zero_fare = (df['Fare'] == 0).sum()
print(f"Rows with Fare == 0 (likely crew/erroneous): {zero_fare}")

# ---------------------------------------------------------------
# STEP 4: DATA CLEANING
# ---------------------------------------------------------------
print("\n=== STEP 4: DATA CLEANING ===")
df_clean = df.copy()

# 4a. Drop columns with excessive missingness / low analytical value
df_clean.drop(columns=['Cabin', 'Ticket', 'PassengerId'], inplace=True)
print("Dropped columns: Cabin (77% missing), Ticket (high cardinality, not predictive), PassengerId (identifier)")

# 4b. Handle missing Age -> median imputation grouped by Pclass & Sex (more accurate than global median)
df_clean['Age'] = df_clean.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
print(f"Age missing after grouped-median imputation: {df_clean['Age'].isnull().sum()}")

# 4c. Handle missing Embarked -> mode imputation (only 2 missing)
mode_embarked = df_clean['Embarked'].mode()[0]
df_clean['Embarked'] = df_clean['Embarked'].fillna(mode_embarked)
print(f"Embarked filled with mode: '{mode_embarked}'")

# 4d. Remove duplicate rows if any
before = len(df_clean)
df_clean.drop_duplicates(inplace=True)
print(f"Duplicates removed: {before - len(df_clean)}")

# 4e. Fix data types
df_clean['Survived'] = df_clean['Survived'].astype('category')
df_clean['Pclass'] = df_clean['Pclass'].astype('category')
df_clean['Sex'] = df_clean['Sex'].str.lower().str.strip()
df_clean['Embarked'] = df_clean['Embarked'].str.upper().str.strip()

# 4f. Handle outliers in Fare using capping (winsorization) instead of deletion
# (deletion would remove legitimate first-class high fares; capping preserves sample size)
fare_cap = df_clean['Fare'].quantile(0.99)
n_capped = (df_clean['Fare'] > fare_cap).sum()
df_clean['Fare'] = np.where(df_clean['Fare'] > fare_cap, fare_cap, df_clean['Fare'])
print(f"Capped {n_capped} extreme Fare values at 99th percentile ({fare_cap:.2f})")

# Plot 4: Boxplot after cleaning
plt.figure(figsize=(5, 4))
sns.boxplot(y=df_clean['Fare'], color='#8dbfa8')
plt.title('Fare Distribution After Capping Outliers')
plt.tight_layout()
plt.savefig('plot4_fare_after_capping.png')
plt.close()

# Plot 5: Missing values after cleaning (should be empty)
plt.figure(figsize=(8, 3))
sns.heatmap(df_clean.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Value Map (After Cleaning) - No Gaps')
plt.tight_layout()
plt.savefig('plot5_missing_after.png')
plt.close()

# ---------------------------------------------------------------
# STEP 5: FEATURE ENGINEERING / PREPROCESSING
# ---------------------------------------------------------------
print("\n=== STEP 5: PREPROCESSING ===")

# Extract title from Name (useful engineered feature)
df_clean['Title'] = df['Name'].str.extract(r',\s*([^\.]*)\.')
df_clean['Title'] = df_clean['Title'].replace(
    ['Lady', 'Countess', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df_clean['Title'] = df_clean['Title'].replace(['Mlle', 'Ms'], 'Miss')
df_clean['Title'] = df_clean['Title'].replace('Mme', 'Mrs')

# Family size feature
df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
df_clean['IsAlone'] = (df_clean['FamilySize'] == 1).astype(int)

# Encode categorical variables (one-hot for Embarked, Title; binary map for Sex)
df_clean['Sex_encoded'] = df_clean['Sex'].map({'male': 0, 'female': 1})
df_clean = pd.get_dummies(df_clean, columns=['Embarked', 'Title'], prefix=['Embarked', 'Title'])

# Normalize Fare and Age (Min-Max scaling) into new columns, keep originals for readability
df_clean['Age_scaled'] = (df_clean['Age'] - df_clean['Age'].min()) / (df_clean['Age'].max() - df_clean['Age'].min())
df_clean['Fare_scaled'] = (df_clean['Fare'] - df_clean['Fare'].min()) / (df_clean['Fare'].max() - df_clean['Fare'].min())

print("Final columns:", list(df_clean.columns))
print(f"Final shape: {df_clean.shape}")

df_clean.drop(columns=['Name']).to_csv('titanic_cleaned.csv', index=False)
print("\nSaved cleaned dataset -> titanic_cleaned.csv")

# Plot 6: Age distribution before vs after imputation comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(df['Age'].dropna(), kde=True, ax=axes[0], color='#e07b39')
axes[0].set_title('Age Distribution (Raw, excl. missing)')
sns.histplot(df_clean['Age'], kde=True, ax=axes[1], color='#5b8c85')
axes[1].set_title('Age Distribution (After Imputation)')
plt.tight_layout()
plt.savefig('plot6_age_distribution.png')
plt.close()

# Plot 7: Survival by class and sex (quick insight visual)
plt.figure(figsize=(7, 4))
sns.barplot(data=df, x='Pclass', y='Survived', hue='Sex')
plt.title('Survival Rate by Passenger Class and Sex')
plt.ylabel('Survival Rate')
plt.tight_layout()
plt.savefig('plot7_survival_insight.png')
plt.close()

print("\nAll plots saved. Script complete.")
