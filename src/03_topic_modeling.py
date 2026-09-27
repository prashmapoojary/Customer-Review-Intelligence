"""
Step 3: Topic Modeling (NMF on TF-IDF)
Runs two separate offline NMF topic models:
- Negative reviews (complaint themes, ~10 topics)
- Positive reviews (praise themes, ~8 topics)
Applies context-specific keyword labeling rules, merges duplicates in summary,
saves data/processed/reviews_with_topics.csv and data/processed/topic_summary.csv,
and generates charts.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Hardcoded BASE_DIR
BASE_DIR = Path(r"C:\Users\Prashma\Desktop\Resume Projects\Data Analytics\NLP-powered customer review analytics system")

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = PROCESSED_DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Separate keyword rule sets for Negative vs Positive
NEGATIVE_LABEL_RULES = [
    ("Defective / Hardware Failure", ["defective", "broken", "power", "died", "hardware", "burned", "glitched", "charge", "button", "stopped", "fail", "failed"]),
    ("Durability / Build Quality", ["flimsy", "apart", "cracked", "plastic", "shattered", "snapped", "peeling", "broke", "durability", "junk", "cheaply", "wobbly"]),
    ("Shipping / Packaging Damage", ["shipping", "delivery", "box", "package", "crushed", "delayed", "damaged", "transit", "arrived", "battered", "scratched"]),
    ("Sizing & Dimension Inaccuracy", ["size", "sizing", "dimensions", "fit", "small", "tiny", "gigantic", "measurements", "inaccurate", "clumsy"]),
    ("Overpriced / Poor Value", ["overpriced", "price", "waste", "rip", "money", "expensive", "dollar", "cost", "cash", "worthless"]),
    ("Customer Service & Refund Issues", ["service", "support", "customer", "refund", "warranty", "seller", "emails", "unresponsive", "rude", "incompetent", "runaround"]),
    ("Confusing Manual / Setup Difficulty", ["instructions", "manual", "guide", "steps", "assembly", "setup", "confusing", "translated", "scrambled"]),
    ("Excessive Noise / Chemical Odor", ["noise", "smell", "odor", "loud", "buzzing", "stench", "vibration", "chemical", "screeching", "headache", "clattering"])
]

POSITIVE_LABEL_RULES = [
    ("Superior Build & Durability", ["build", "solid", "quality", "sturdy", "craftsmanship", "robust", "materials", "durable", "heavy", "finish", "textures"]),
    ("Great Value & Affordability", ["value", "price", "affordable", "penny", "bargain", "steal", "money", "worth", "budget", "competitors"]),
    ("Fast Shipping & Secure Packaging", ["shipping", "delivery", "fast", "packaged", "speedy", "cushioning", "pristine", "arrived", "secure", "record"]),
    ("Ease of Use & Simple Setup", ["easy", "intuitive", "simple", "setup", "plug", "effortless", "straightforward", "convenient", "controls", "smart"]),
    ("High Performance & Power", ["performance", "power", "powerful", "flawlessly", "speed", "results", "blazing", "effective", "smooth", "powerhouse"]),
    ("Outstanding Customer Support", ["service", "support", "customer", "courteous", "seller", "responsive", "communication", "helpful", "dedicated", "prompt"]),
    ("Sleek & Modern Design", ["design", "sleek", "modern", "looks", "aesthetic", "stylish", "gorgeous", "beautiful", "chic", "minimalist", "clean"])
]

def auto_label_topic(top_words, rule_set):
    """Matches top words against rule sets or falls back to top 3 words"""
    top_words_lower = [w.lower() for w in top_words]
    for label, keywords in rule_set:
        for kw in keywords:
            if any(kw in word for word in top_words_lower):
                return label
    return " / ".join(top_words[:3]).title()

def fit_nmf_model(texts, n_topics=8, max_features=2500, rule_set=None, polarity="negative"):
    """Fits TF-IDF and NMF on a corpus of text and returns topic assignments and word summaries"""
    if len(texts) == 0:
        return np.array([]), [], []
        
    vectorizer = TfidfVectorizer(
        max_df=0.80,
        min_df=5,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features
    )
    tfidf = vectorizer.fit_transform(texts)
    
    nmf = NMF(
        n_components=n_topics,
        random_state=42,
        init="nndsvda",
        max_iter=400
    )
    doc_topics = nmf.fit_transform(tfidf)
    
    feature_names = vectorizer.get_feature_names_out()
    topic_labels = []
    topic_top_words = []
    
    for topic_idx, topic in enumerate(nmf.components_):
        top_indices = topic.argsort()[:-8:-1]
        top_words = [feature_names[i] for i in top_indices]
        label = auto_label_topic(top_words, rule_set) if rule_set else f"Topic {topic_idx+1}"
        topic_labels.append(label)
        topic_top_words.append(", ".join(top_words[:5]))
        
    dominant_topic_idx = doc_topics.argmax(axis=1)
    assigned_labels = [topic_labels[i] for i in dominant_topic_idx]
    
    return assigned_labels, topic_labels, topic_top_words

def run_topic_modeling():
    print("--- Step 3: Running NMF Topic Modeling (Negative & Positive Sub-Models) ---")
    
    sentiment_path = PROCESSED_DATA_DIR / "reviews_with_sentiment.csv"
    if not sentiment_path.exists():
        raise FileNotFoundError(f"Missing sentiment reviews file at: {sentiment_path}")
        
    df = pd.read_csv(sentiment_path)
    print(f"Loaded {len(df):,} reviews with sentiment.")
    
    # Split into negative and positive subsets
    neg_mask = df["sentiment_label"] == "negative"
    pos_mask = df["sentiment_label"] == "positive"
    neu_mask = df["sentiment_label"] == "neutral"
    
    df["topic_theme"] = "Neutral / General Feedback"
    df["topic_polarity"] = df["sentiment_label"]
    
    # 1. Negative Model (10 topics)
    print("\nFitting Negative Topic Model (Complaint Themes, NMF k=10)...")
    neg_texts = df.loc[neg_mask, "full_review"].tolist()
    neg_assigned, neg_topic_labels, neg_top_words = fit_nmf_model(
        neg_texts, n_topics=10, rule_set=NEGATIVE_LABEL_RULES, polarity="negative"
    )
    df.loc[neg_mask, "topic_theme"] = neg_assigned
    
    print("\nDiscovered Negative Topics:")
    for i, (lab, words) in enumerate(zip(neg_topic_labels, neg_top_words)):
        print(f"  Topic #{i+1:02d} -> Label: '{lab}' | Top words: {words}")
        
    # 2. Positive Model (8 topics)
    print("\nFitting Positive Topic Model (Praise Themes, NMF k=8)...")
    pos_texts = df.loc[pos_mask, "full_review"].tolist()
    pos_assigned, pos_topic_labels, pos_top_words = fit_nmf_model(
        pos_texts, n_topics=8, rule_set=POSITIVE_LABEL_RULES, polarity="positive"
    )
    df.loc[pos_mask, "topic_theme"] = pos_assigned
    
    print("\nDiscovered Positive Topics:")
    for i, (lab, words) in enumerate(zip(pos_topic_labels, pos_top_words)):
        print(f"  Topic #{i+1:02d} -> Label: '{lab}' | Top words: {words}")
        
    # Save enriched reviews
    output_reviews_path = PROCESSED_DATA_DIR / "reviews_with_topics.csv"
    df.to_csv(output_reviews_path, index=False)
    print(f"\n Saved reviews with topic themes to: {output_reviews_path}")
    
    # Create Merged Topic Summary Table
    summary = df.groupby(["topic_polarity", "topic_theme"]).agg(
        review_count=("review_id", "count"),
        avg_star_rating=("rating", "mean"),
        avg_compound_sentiment=("sentiment_compound", "mean")
    ).reset_index()
    
    summary["pct_of_reviews"] = (summary["review_count"] / len(df)) * 100
    summary = summary.sort_values(by=["topic_polarity", "review_count"], ascending=[True, False])
    
    summary_path = PROCESSED_DATA_DIR / "topic_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f" Saved merged topic summary to: {summary_path}")
    
    print("\n--- Summary Table: Merged Topic Themes ---")
    print(summary.to_string(index=False))
    
    # Visualizations
    # 1. Complaint Themes Bar Chart
    neg_themes = summary[summary["topic_polarity"] == "negative"].sort_values("review_count", ascending=True)
    plt.figure(figsize=(10, 5))
    plt.barh(neg_themes["topic_theme"], neg_themes["review_count"], color="#E5484D", edgecolor="black")
    plt.title("Customer Complaint Themes (Negative Reviews)", fontsize=13, fontweight="bold")
    plt.xlabel("Review Volume")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "complaint_themes_distribution.png", dpi=300)
    plt.close()
    
    # 2. Praise Themes Bar Chart
    pos_themes = summary[summary["topic_polarity"] == "positive"].sort_values("review_count", ascending=True)
    plt.figure(figsize=(10, 5))
    plt.barh(pos_themes["topic_theme"], pos_themes["review_count"], color="#3ECF9E", edgecolor="black")
    plt.title("Customer Praise Themes (Positive Reviews)", fontsize=13, fontweight="bold")
    plt.xlabel("Review Volume")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "praise_themes_distribution.png", dpi=300)
    plt.close()
    
    # 3. Top Themes Per Category Breakdown
    cat_theme = df.groupby(["category", "topic_polarity", "topic_theme"]).size().unstack(fill_value=0)
    
    # 4. Top 5 Products Theme Drilldown Chart (Sample)
    top5_neg_prods = df[df["sentiment_label"] == "negative"]["product_name"].value_counts().head(5).index
    drilldown_df = df[df["product_name"].isin(top5_neg_prods) & (df["sentiment_label"] == "negative")]
    
    plt.figure(figsize=(12, 6))
    drilldown_pivot = drilldown_df.groupby(["product_name", "topic_theme"]).size().unstack(fill_value=0)
    drilldown_pivot.plot(kind="barh", stacked=True, colormap="tab10", figsize=(12, 6), edgecolor="black")
    plt.title("Complaint Theme Breakdown for Top 5 Highest-Complaint Products", fontsize=13, fontweight="bold")
    plt.xlabel("Number of Negative Reviews")
    plt.ylabel("Product")
    plt.legend(title="Complaint Theme", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "product_complaint_drilldown.png", dpi=300)
    plt.close()
    
    print(f" Generated topic visualization charts in: {FIGURES_DIR}")
    return df, summary

if __name__ == "__main__":
    run_topic_modeling()
