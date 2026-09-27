"""
Step 2: Sentiment Analysis (VADER)
Performs sentiment scoring on combined review title + review text using VADER.
Validates alignment against 1-5 star ratings, creates visualization charts,
and exports data/processed/reviews_with_sentiment.csv.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Hardcoded BASE_DIR
BASE_DIR = Path(r"C:\Users\Prashma\Desktop\Resume Projects\Data Analytics\NLP-powered customer review analytics system")

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def run_sentiment_analysis():
    print("--- Step 2: Running VADER Sentiment Analysis ---")
    
    # Load raw review dataset
    raw_path = RAW_DATA_DIR / "amazon_reviews.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw reviews file not found at: {raw_path}")
        
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df):,} reviews from {raw_path.name}")
    
    # Initialize VADER Sentiment Analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Combine title and review text
    df["full_review"] = (df["review_title"].fillna("") + ". " + df["review_text"].fillna("")).str.strip()
    
    # Calculate sentiment scores
    compound_scores = []
    pos_scores = []
    neu_scores = []
    neg_scores = []
    sentiment_labels = []
    
    for text in df["full_review"]:
        vs = analyzer.polarity_scores(text)
        comp = vs["compound"]
        compound_scores.append(comp)
        pos_scores.append(vs["pos"])
        neu_scores.append(vs["neu"])
        neg_scores.append(vs["neg"])
        
        # Standard threshold: >= 0.05 Positive, <= -0.05 Negative, else Neutral
        if comp >= 0.05:
            sentiment_labels.append("positive")
        elif comp <= -0.05:
            sentiment_labels.append("negative")
        else:
            sentiment_labels.append("neutral")
            
    df["sentiment_compound"] = compound_scores
    df["sentiment_pos"] = pos_scores
    df["sentiment_neu"] = neu_scores
    df["sentiment_neg"] = neg_scores
    df["sentiment_label"] = sentiment_labels
    
    # Save processed dataframe
    output_path = PROCESSED_DATA_DIR / "reviews_with_sentiment.csv"
    df.to_csv(output_path, index=False)
    print(f" Saved reviews with sentiment to: {output_path}")
    
    # Validation: Crosstab sentiment_label vs star rating
    print("\n--- Validation: Sentiment Label vs Star Rating (Counts) ---")
    crosstab_counts = pd.crosstab(df["rating"], df["sentiment_label"], margins=True)
    print(crosstab_counts)
    
    print("\n--- Validation: Sentiment Label vs Star Rating (Row Percentages) ---")
    crosstab_pct = pd.crosstab(df["rating"], df["sentiment_label"], normalize="index") * 100
    print(crosstab_pct.round(2).map(lambda x: f"{x:.2f}%"))
    
    # Check targets: 5-star >= 95% positive, 1-star >= 85% negative
    star5_pos_pct = crosstab_pct.loc[5, "positive"] if 5 in crosstab_pct.index and "positive" in crosstab_pct.columns else 0
    star1_neg_pct = crosstab_pct.loc[1, "negative"] if 1 in crosstab_pct.index and "negative" in crosstab_pct.columns else 0
    
    print(f"\nTarget Validation Check:")
    print(f"  5-Star Positive %: {star5_pos_pct:.2f}% (Target: >=95.0%) -> {'PASSED' if star5_pos_pct >= 95 else 'FAILED'}")
    print(f"  1-Star Negative %: {star1_neg_pct:.2f}% (Target: >=85.0%) -> {'PASSED' if star1_neg_pct >= 85 else 'FAILED'}")
    
    # Generate and save charts
    figures_dir = PROCESSED_DATA_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Sentiment Distribution Bar Chart
    plt.figure(figsize=(7, 4))
    palette = {"positive": "#3ECF9E", "neutral": "#A0AEC0", "negative": "#E5484D"}
    sns.countplot(data=df, x="sentiment_label", hue="sentiment_label", order=["positive", "neutral", "negative"], palette=palette, legend=False)
    plt.title("Overall Sentiment Distribution (VADER)", fontsize=13, fontweight="bold")
    plt.xlabel("Sentiment Label")
    plt.ylabel("Review Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "sentiment_distribution.png", dpi=300)
    plt.close()
    
    # 2. Sentiment vs Star Rating Heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(crosstab_pct, annot=True, fmt=".1f", cmap="Blues", cbar=True)
    plt.title("Sentiment Label vs Star Rating (Row %)", fontsize=13, fontweight="bold")
    plt.xlabel("VADER Sentiment Label")
    plt.ylabel("Star Rating")
    plt.tight_layout()
    plt.savefig(figures_dir / "sentiment_vs_rating_heatmap.png", dpi=300)
    plt.close()
    
    # 3. Monthly Sentiment Trend Line
    df_temp = df.copy()
    df_temp["month"] = pd.to_datetime(df_temp["review_date"]).dt.to_period("M").dt.to_timestamp()
    monthly_sentiment = df_temp.groupby(["month", "sentiment_label"]).size().unstack(fill_value=0)
    monthly_sentiment_pct = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(11, 5))
    if "positive" in monthly_sentiment_pct:
        plt.plot(monthly_sentiment_pct.index, monthly_sentiment_pct["positive"], label="Positive %", color="#3ECF9E", marker="o", linewidth=2.5)
    if "negative" in monthly_sentiment_pct:
        plt.plot(monthly_sentiment_pct.index, monthly_sentiment_pct["negative"], label="Negative %", color="#E5484D", marker="s", linewidth=2.5)
    if "neutral" in monthly_sentiment_pct:
        plt.plot(monthly_sentiment_pct.index, monthly_sentiment_pct["neutral"], label="Neutral %", color="#A0AEC0", marker="^", linewidth=2)
    plt.title("Monthly Sentiment Trend (June 2024 - August 2025)", fontsize=13, fontweight="bold")
    plt.ylabel("Percentage of Monthly Reviews (%)")
    plt.xlabel("Month")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "monthly_sentiment_trend.png", dpi=300)
    plt.close()
    
    # 4. Worst 10 Products by Negative Review Percentage
    prod_sentiment = df.groupby(["product_name", "sentiment_label"]).size().unstack(fill_value=0)
    prod_sentiment["total"] = prod_sentiment.sum(axis=1)
    prod_sentiment["neg_pct"] = (prod_sentiment["negative"] / prod_sentiment["total"]) * 100
    worst_10 = prod_sentiment.sort_values("neg_pct", ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=worst_10["neg_pct"], y=worst_10.index, color="#E5484D", edgecolor="black")
    plt.title("Worst 10 Products by Negative Sentiment Rate (%)", fontsize=13, fontweight="bold")
    plt.xlabel("Negative Review Percentage (%)")
    plt.ylabel("Product Name")
    plt.tight_layout()
    plt.savefig(figures_dir / "worst_10_products.png", dpi=300)
    plt.close()
    
    print(f" Generated 4 analysis charts in: {figures_dir}")
    return df

if __name__ == "__main__":
    run_sentiment_analysis()
