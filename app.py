import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("metadata.csv")
    df = df[["title", "abstract", "publish_time", "authors", "journal", "source_x"]].copy()
    df = df.dropna(subset=["title", "publish_time"])
    df["abstract"] = df["abstract"].fillna("")
    df["journal"] = df["journal"].fillna("Unknown")
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df = df.dropna(subset=["publish_time"])
    df["year"] = df["publish_time"].dt.year
    df["abstract_word_count"] = df["abstract"].apply(lambda x: len(x.split()))
    return df

df = load_data()

# ---------------------------
# App Layout
# ---------------------------
st.title("📊 CORD-19 Data Explorer")
st.write("Interactive exploration of COVID-19 research papers using the metadata dataset.")

# ---------------------------
# Sidebar Filters
# ---------------------------
year_range = st.slider("Select year range", int(df["year"].min()), int(df["year"].max()), (2015, 2021))

# Filter data
filtered = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

st.write(f"Showing data for **{len(filtered)} papers** between {year_range[0]} and {year_range[1]}.")

# ---------------------------
# Visualization 1: Publications by Year
# ---------------------------
st.subheader("📈 Publications per Year")
pub_trends = filtered['year'].value_counts().sort_index()

fig, ax = plt.subplots()
ax.plot(pub_trends.index, pub_trends.values, marker='o')
ax.set_title("Publication Trends per Year")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Papers")
st.pyplot(fig)

# ---------------------------
# Visualization 2: Top Journals
# ---------------------------
st.subheader("📊 Top 10 Journals")
top_journals = filtered['journal'].value_counts().head(10)

fig, ax = plt.subplots()
top_journals.plot(kind='bar', ax=ax)
ax.set_title("Top 10 Journals by Number of Papers")
ax.set_xlabel("Journal")
ax.set_ylabel("Number of Papers")
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

# ---------------------------
# Visualization 3: Abstract Word Count
# ---------------------------
st.subheader("📝 Average Abstract Word Count per Year")
avg_word_count = filtered.groupby('year')['abstract_word_count'].mean().sort_index()

fig, ax = plt.subplots()
ax.plot(avg_word_count.index, avg_word_count.values, marker='o', color='orange')
ax.set_title("Average Abstract Word Count per Year")
ax.set_xlabel("Year")
ax.set_ylabel("Average Word Count")
st.pyplot(fig)

# ---------------------------
# Show sample data
# ---------------------------
st.subheader("🔍 Sample Data Preview")
st.dataframe(filtered.head(20))
