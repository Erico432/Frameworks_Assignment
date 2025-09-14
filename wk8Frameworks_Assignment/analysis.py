import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("metadata.csv")

# Step 2: Preview first 5 rows
print("🔹 First 5 rows:")
print(df.head())

# Step 3: Dataset shape
print("\n🔹 Dataset shape (rows, columns):")
print(df.shape)

# Step 4: Info about columns
print("\n🔹 Dataset info:")
print(df.info())

# Step 5: Missing values
print("\n🔹 Missing values per column:")
print(df.isnull().sum())

# Step 6: Basic statistics (only for numeric columns)
print("\n🔹 Basic statistics:")
print(df.describe())


# -------------------------------
# PART 2: Data Cleaning & Preparation
# -------------------------------

# Step 1: Keep only useful columns
df_clean = df[["title", "abstract", "publish_time", "authors", "journal", "source_x"]].copy()

print("\n🔹 After selecting useful columns:")
print(df_clean.head())

# Step 2: Handle missing values
# Drop rows missing title or publish_time (important for analysis)
df_clean = df_clean.dropna(subset=["title", "publish_time"])

# Fill missing abstracts with empty text
df_clean["abstract"] = df_clean["abstract"].fillna("")

# Fill missing journals with "Unknown"
df_clean["journal"] = df_clean["journal"].fillna("Unknown")

print("\n🔹 Missing values after cleaning:")
print(df_clean.isnull().sum())

# Step 3: Convert publish_time to datetime
df_clean["publish_time"] = pd.to_datetime(df_clean["publish_time"], errors="coerce")

# Drop rows where publish_time could not be converted
df_clean = df_clean.dropna(subset=["publish_time"])

# Step 4: Extract publication year
df_clean["year"] = df_clean["publish_time"].dt.year

# Step 5 (Optional): Create abstract word count
df_clean["abstract_word_count"] = df_clean["abstract"].apply(lambda x: len(x.split()))

print("\n🔹 Cleaned DataFrame preview:")
print(df_clean.head())


# -------------------------------
# PART 3: analysis & visualization steps
# -------------------------------

# CORD-19 Metadata Analysis 
import pandas as pd
import matplotlib.pyplot as plt

# Try to use df_clean if already in memory, otherwise do a minimal fallback cleaning
try:
    data = df_clean
    print("Using df_clean from memory.")
except NameError:
    print("df_clean not found in memory — loading minimal clean from metadata.csv (fallback).")
    data = pd.read_csv("metadata.csv")
    data = data[["title", "abstract", "publish_time", "authors", "journal", "source_x"]].copy()
    data = data.dropna(subset=["title", "publish_time"])
    data["abstract"] = data["abstract"].fillna("")
    data["journal"] = data["journal"].fillna("Unknown")
    data["publish_time"] = pd.to_datetime(data["publish_time"], errors="coerce")
    data = data.dropna(subset=["publish_time"])
    data["year"] = data["publish_time"].dt.year
    data["abstract_word_count"] = data["abstract"].apply(lambda x: len(x.split()))

# -------------------------------
# STEP 1: Publication trends per year
# -------------------------------
pub_trends = data['year'].value_counts().sort_index()
print("Publication counts per year:")
print(pub_trends.head(10))

plt.figure(figsize=(10, 5))
plt.plot(pub_trends.index, pub_trends.values, marker='o')
plt.title("Publication Trends per Year")
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("pub_trends.png", dpi=150)
plt.show()
print("Saved plot to pub_trends.png")

# -------------------------------
# STEP 2: Top 10 journals
# -------------------------------
top_journals = data['journal'].value_counts().head(10)
print("Top 10 journals:")
print(top_journals)

plt.figure(figsize=(10, 6))
top_journals.plot(kind='bar')
plt.title("Top 10 Journals by Number of Publications")
plt.xlabel("Journal")
plt.ylabel("Number of Papers")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("top_journals.png", dpi=150)
plt.show()
print("Saved plot to top_journals.png")

# -------------------------------
# STEP 3: Abstract word count trend
# -------------------------------
if 'abstract_word_count' not in data.columns:
    data['abstract_word_count'] = data['abstract'].fillna("").apply(lambda x: len(x.split()))

avg_word_count = data.groupby('year')['abstract_word_count'].mean().sort_index()
print("Average abstract word count per year:")
print(avg_word_count.head(10))

plt.figure(figsize=(10, 5))
plt.plot(avg_word_count.index, avg_word_count.values, marker='o', color='orange')
plt.title("Average Abstract Word Count per Year")
plt.xlabel("Year")
plt.ylabel("Average Word Count")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("avg_abstract_wordcount.png", dpi=150)
plt.show()
print("Saved plot to avg_abstract_wordcount.png")
