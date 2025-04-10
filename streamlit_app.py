import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import plotly.express as px
import numpy as np
import matplotlib as mpl

# Configure wide layout
st.set_page_config(page_title="Banking Data Dashboard", layout="wide")

# Load Data
financial_data = pd.read_csv("financial_data.csv")
banking_news = pd.read_csv("banking_news_with_analysis.csv")

# --- Custom Full Page CSS Styling ---
st.markdown("""
    <style>
        html, body, .main {
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            font-size: 2.8rem;
            font-weight: 700;
            text-align: center;
            padding: 0.5rem 0;
        }
        .section-header {
            font-size: 1.4rem;
            font-weight: 600;
            margin-top: 2rem;
            border-left: 5px solid #0099cc;
            padding-left: 10px;
        }
        section[data-testid="stSidebar"] {
            background-color: rgba(230, 242, 255, 0.9);
        }
        @media (prefers-color-scheme: dark) {
            html, body, .main {
                background-color: #0e1117;
                color: #ffffff;
            }
            .title {
                color: #66ccff;
            }
            .section-header {
                color: #80d4ff;
                border-left: 5px solid #00bfff;
            }
            section[data-testid="stSidebar"] {
                background-color: rgba(40, 44, 52, 0.8);
            }
        }
        @media (prefers-color-scheme: light) {
            .title {
                color: #003366;
            }
            .section-header {
                color: #004080;
                border-left: 5px solid #0099cc;
            }
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------
# Theme Detection
# -----------------
theme = st.get_option("theme.backgroundColor")
is_dark = theme in ["#0e1117", "#262730"]
text_color = 'white' if is_dark else 'black'
bg_color = "#0e1117" if is_dark else "white"

# Apply matplotlib theming
plt.rcParams.update({
    'text.color': text_color,
    'axes.labelcolor': text_color,
    'xtick.color': text_color,
    'ytick.color': text_color,
    'axes.edgecolor': text_color,
    'figure.facecolor': bg_color,
    'axes.facecolor': bg_color,
})

# -----------------
# Sidebar
# -----------------
st.sidebar.header("🔍 Filter Options")
selected_bank = st.sidebar.selectbox("Select Bank", ["All"] + list(financial_data["Bank"].unique()))

# -----------------
# Title
# -----------------
st.title("Banking Data Dashboard")

# ================================
# WHEN USER SELECTS ALL BANKS
# ================================
if selected_bank == "All":
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Revenue Breakdown")
        st.bar_chart(financial_data.groupby("Bank")["Total Revenue"].sum())

    with col2:
        combined_revenue = financial_data.groupby("Bank")["Total Revenue"].sum().sort_values(ascending=False).reset_index()
        fig5 = px.pie(
            combined_revenue,
            names="Bank",
            values="Total Revenue",
            title="Market Share by Revenue - All Banks",
            color_discrete_sequence=["#66b3ff", "#ff9999"]
            )
        st.plotly_chart(fig5)


    # 3. Financial Performance
    st.markdown("### Stock Price & Market Cap Trends")
    fig, ax1 = plt.subplots(figsize=(10, 5), facecolor='none')

    sns.lineplot(data=financial_data, x="Bank", y="Stock Price", marker="o", ax=ax1, label="Stock Price", color="blue")
    ax1.set_ylabel("Stock Price", color="blue")
    
    ax1.set_xlabel("Bank", color="white")
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='blue')


    ax2 = ax1.twinx()
    sns.lineplot(data=financial_data, x="Bank", y="Market Cap", marker="s", ax=ax2, label="Market Cap", color="red")
    ax2.set_ylabel("Market Cap", color="red")
    ax2.tick_params(axis='y', colors='red')

    ax1.set_xticks(range(len(financial_data["Bank"])))
    ax1.set_xticklabels(financial_data["Bank"], rotation=45, ha='right', color="white")

    ax1.set_facecolor('none')
    ax2.set_facecolor('none')
    fig.patch.set_alpha(0)

    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    st.pyplot(fig)



    st.markdown("### Market Insights")
    st.dataframe(financial_data[["Bank", "Stock Price", "Market Cap"]])

    st.markdown("### Sentiment Analysis")
    sentiment_group = banking_news.groupby(["Bank", "Sentiment"]).size().unstack().fillna(0)
    st.dataframe(sentiment_group)

    most_positive = sentiment_group["Positive"].idxmax()
    most_neutral = sentiment_group["Neutral"].idxmax()
    most_negative = sentiment_group["Negative"].idxmax()

    st.markdown(f"""
    - 🟢 Most Positive: `{most_positive}` ({int(sentiment_group['Positive'].max())} articles)
    - 🟡 Most Neutral: `{most_neutral}` ({int(sentiment_group['Neutral'].max())} articles)
    - 🔴 Most Negative: `{most_negative}` ({int(sentiment_group['Negative'].max())} articles)
    """)

    st.markdown("### Sentiment Comparison Chart")
    sentiment_group.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="Set3")
    st.pyplot(plt.gcf())


    # Combine all news descriptions into one string
    # Word Cloud with Sentiment Filter
    st.markdown("### ☁️ Word Cloud of Banking News by Sentiment")

    selected_sentiment_wc = st.selectbox("Select Sentiment for Word Cloud", ["Positive", "Neutral", "Negative"], key="wc_sentiment")

    # Filter news descriptions based on sentiment
    sentiment_filtered_text = banking_news[
        banking_news["Sentiment"] == selected_sentiment_wc
    ]["Description"].dropna().astype(str).str.lower().str.cat(sep=" ")

    # Tokenize and clean text (keep only words with at least 3 letters)
    words = re.findall(r'\b[a-z]{3,}\b', sentiment_filtered_text)

    # Remove stopwords
    stopwords_set = set(STOPWORDS)
    filtered_words = [word for word in words if word not in stopwords_set]

    # Count word frequencies
    word_counts = Counter(filtered_words)

    # Remove top 10 most frequent words
    top_10_words = set([word for word, _ in word_counts.most_common(10)])
    final_words = [word for word in filtered_words if word not in top_10_words]
    final_word_counts = Counter(final_words)

    # Generate Word Cloud
    if final_word_counts:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="tab10"
        ).generate_from_frequencies(final_word_counts)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info(f"No sufficient content for sentiment: {selected_sentiment_wc}")


    st.markdown("### 🔐 Cybersecurity Risk Mentions")
    cyber = banking_news[banking_news["Title"].str.contains("cyber", case=False, na=False)]
    st.write(cyber[["Bank", "Title", "Published Date"]].drop_duplicates())

# ================================
# WHEN USER SELECTS A SPECIFIC BANK
# ================================
else:
    st.subheader(f"🏛️ Overview for {selected_bank}")
    bank_df = financial_data[financial_data["Bank"] == selected_bank]
    bank_news = banking_news[banking_news["Bank"] == selected_bank]
    others_df = financial_data[financial_data["Bank"] != selected_bank]

    st.markdown("### 💰 Revenue")
    st.metric(label="Total Revenue", value=f"${bank_df['Total Revenue'].values[0]:,.2f}")

    st.markdown("### 📊 Financial Performance")
    st.metric(label="Stock Price", value=f"${bank_df['Stock Price'].values[0]:,.2f}")
    st.metric(label="Market Cap", value=f"${bank_df['Market Cap'].values[0]:,.2f}")

    perf_df = pd.DataFrame({
        "Metric": ["Stock Price", "Market Cap"],
        "Value": [bank_df["Stock Price"].values[0], bank_df["Market Cap"].values[0]]
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(perf_df["Metric"], perf_df["Value"], color=["skyblue", "lightgreen"])
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f"${yval:,.2f}", ha="center", va="bottom")
    ax.set_ylabel("Value (USD)")
    ax.set_title(f"{selected_bank} - Financial Performance")
    st.pyplot(fig)

    sentiment_group = banking_news.groupby(["Bank", "Sentiment"]).size().unstack().fillna(0)
    if selected_bank in sentiment_group.index:
        selected_sentiment = sentiment_group.loc[[selected_bank]]
        st.markdown("### 🧠 Sentiment Breakdown")
        st.dataframe(selected_sentiment)
        selected_sentiment.T.plot(kind="bar", legend=False, colormap="coolwarm", figsize=(6, 4))
        plt.title(f"{selected_bank} - Sentiment")
        plt.ylabel("Article Count")
        st.pyplot(plt.gcf())
    else:
        st.warning(f"No sentiment data available for `{selected_bank}`.")

    st.markdown("### ⚖️ Competitor Comparison (Stock Price)")
    combined = pd.concat([bank_df, others_df])
    combined["Category"] = combined["Bank"].apply(lambda x: "Selected Bank" if x == selected_bank else "Competitor")
    fig3, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=combined, x="Bank", y="Stock Price", hue="Category",
                palette={"Selected Bank": "dodgerblue", "Competitor": "lightgray"},
                dodge=False, ax=ax1)
    ax1.set_title(f"{selected_bank} vs Competitors")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.markdown("### 🥧 Market Share by Revenue")
    combined_revenue = financial_data.groupby("Bank")["Total Revenue"].sum()
    selected_revenue = combined_revenue[selected_bank]
    others_revenue = combined_revenue.drop(selected_bank).sum()

    pie_df = pd.DataFrame({
        "Category": [selected_bank, "Others"],
        "Revenue": [selected_revenue, others_revenue]
    })

    fig5 = px.pie(
        pie_df,
        names="Category",
        values="Revenue",
        title=f"{selected_bank} vs Others - Revenue Share",
        color_discrete_sequence=["#66b3ff", "#ff9999"]
    )
    fig5.update_traces(textinfo="percent+label", textposition="inside", pull=[0.1, 0])
    st.plotly_chart(fig5)


    # Filter descriptions for the selected bank
    bank_text = " ".join(
        banking_news[banking_news["Bank"] == selected_bank]["Description"]
        .dropna()
        .astype(str)
    ).lower()

    st.markdown(f"### ☁️ Word Cloud for {selected_bank} News by Sentiment")

    selected_sentiment_bank = st.selectbox(
        "Select Sentiment for Word Cloud ({selected_bank})",
        ["Positive", "Neutral", "Negative"],
        key="bank_wc_sentiment"
    )

    # Filter news for this bank and sentiment
    filtered_news = bank_news[
        bank_news["Sentiment"] == selected_sentiment_bank
    ]["Description"].dropna().astype(str).str.lower().str.cat(sep=" ")

    # Tokenize and clean text (keep only words with at least 3 letters)
    words = re.findall(r'\b[a-z]{3,}\b', filtered_news)

# Remove stopwords
    stopwords_set = set(STOPWORDS)
    filtered_words = [word for word in words if word not in stopwords_set]

# Count word frequencies
    word_counts = Counter(filtered_words)

# Remove top 10 frequent words
    top_10_words = set([word for word, _ in word_counts.most_common(10)])
    final_words = [word for word in filtered_words if word not in top_10_words]
    final_word_counts = Counter(final_words)

# Generate Word Cloud
    if final_word_counts:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="tab10"
        ).generate_from_frequencies(final_word_counts)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info(f"No sufficient content found for `{selected_sentiment_bank}` sentiment in {selected_bank}.")


    # Display in Streamlit
    st.markdown(f"### ☁️ Word Cloud for {selected_bank} News")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    st.markdown("### 🔐 Cybersecurity Mentions")
    cyber = bank_news[bank_news["Title"].str.contains("cyber", case=False, na=False)]
    if not cyber.empty:
        st.write(cyber[["Title", "Published Date"]])
        st.bar_chart(cyber["Sentiment"].value_counts())
    else:
        st.info("No cybersecurity-related news found for this bank.")
