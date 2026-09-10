
# Week 2 Task: Exploratory Data Analysis and Visualization
# Dataset: Titanic Cleaned Dataset

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("titanic_cleaned.csv")

print("========== WEEK 2 : EXPLORATORY DATA ANALYSIS ==========")
print("Shape :", df.shape)
print("\nColumns:\n", df.columns.tolist())

print("\nData Types:\n")
print(df.dtypes)

print("\nSummary Statistics:\n")
print(df.describe(include="all"))

print("\nMissing Values:\n")
print(df.isnull().sum())

survival_rate = df["Survived"].mean() * 100
print(f"\nOverall Survival Rate: {survival_rate:.2f}%")

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Survived")
plt.title("Passenger Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("survival_count.png")
plt.close()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.tight_layout()
plt.savefig("gender_survival.png")
plt.close()

plt.figure(figsize=(6,4))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.close()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.tight_layout()
plt.savefig("class_survival.png")
plt.close()

numeric_df = df.select_dtypes(include=["number"])
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="Pclass", y="Fare")
plt.title("Fare Distribution by Passenger Class")
plt.tight_layout()
plt.savefig("fare_boxplot.png")
plt.close()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="Survived", y="Age")
plt.title("Age vs Survival")
plt.tight_layout()
plt.savefig("age_survival_boxplot.png")
plt.close()

print("\n========== KEY INSIGHTS ==========")
print("1. Female passengers survived more than male passengers.")
print("2. First-class passengers had higher survival rates.")
print("3. Most passengers were between 20 and 40 years old.")
print("4. Higher fare passengers generally belonged to higher classes.")
print("5. Correlation heatmap shows relationships among numerical features.")

print("\nEDA Completed Successfully.")
