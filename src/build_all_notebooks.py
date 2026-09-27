"""
Helper script to generate matching, high-quality Jupyter Notebooks for all 5 steps
in the notebooks/ directory.
"""

from pathlib import Path
import nbformat as nbf

BASE_DIR = Path(r"C:\Users\Prashma\Desktop\Resume Projects\Data Analytics\NLP-powered customer review analytics system")
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def create_notebook_01():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Step 1: Synthetic Amazon-Style Review Data Generation
This notebook generates realistic Amazon customer review data across 35 products in 6 categories with hidden quality tiers, realistic timestamps, and distinct sentiment/topic clusters.
"""),
        nbf.v4.new_code_cell("""from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Hardcoded BASE_DIR
BASE_DIR = Path(r"C:\\Users\\Prashma\\Desktop\\Resume Projects\\Data Analytics\\NLP-powered customer review analytics system")
sys.path.append(str(BASE_DIR))

from src.data_generation_step import generate_dataset, RAW_DATA_DIR

# Run generation
reviews_df, catalog_df = generate_dataset()
print(f"Generated {len(reviews_df):,} reviews for {len(catalog_df)} products.")
"""),
        nbf.v4.new_markdown_cell("""### 1.1 Inspect Dataset Schema & First Rows"""),
        nbf.v4.new_code_cell("""print("Dataset Dimensions:", reviews_df.shape)
print("Columns:", reviews_df.columns.tolist())
reviews_df.head()
"""),
        nbf.v4.new_markdown_cell("""### 1.2 Category Breakdown & Star Rating Distribution"""),
        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

reviews_df['category'].value_counts().plot(kind='barh', ax=axes[0], color='#3ECF9E', edgecolor='black')
axes[0].set_title('Review Volume by Category', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Total Reviews')

reviews_df['rating'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='#4A90E2', edgecolor='black')
axes[1].set_title('Star Rating Distribution (1-5 Stars)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Star Rating')
axes[1].set_ylabel('Review Count')

plt.tight_layout()
plt.show()
"""),
        nbf.v4.new_markdown_cell("""### 1.3 Monthly Volume Growth Trajectory"""),
        nbf.v4.new_code_cell("""reviews_df['month'] = pd.to_datetime(reviews_df['review_date']).dt.to_period('M').dt.to_timestamp()
monthly_vol = reviews_df.groupby('month')['review_id'].count()

plt.figure(figsize=(12, 4))
monthly_vol.plot(kind='line', marker='o', color='#E5484D', linewidth=2.5)
plt.title('Monthly Review Volume (June 2024 - August 2025)', fontsize=12, fontweight='bold')
plt.ylabel('Reviews Count')
plt.grid(True, alpha=0.3)
plt.show()
""")
    ]
    
    # Fix import module name to match script
    nb.cells[1].source = nb.cells[1].source.replace("from src.data_generation_step", "import importlib\nfrom src import _01_data_gen as step1\n# Or directly read from raw data path if generated")
    
    # Write notebook
    out_path = NOTEBOOKS_DIR / "01_data_generation.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created: {out_path}")

def create_notebook_02():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Step 2: VADER Sentiment Analysis
Analyzes review title + text sentiment using the offline VADER Sentiment Intensity Analyzer, validates alignment against star ratings, and generates diagnostics.
"""),
        nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Hardcoded BASE_DIR
BASE_DIR = Path(r"C:\\Users\\Prashma\\Desktop\\Resume Projects\\Data Analytics\\NLP-powered customer review analytics system")
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(RAW_DATA_DIR / "amazon_reviews.csv")
print(f"Loaded {len(df):,} reviews.")
"""),
        nbf.v4.new_markdown_cell("""### 2.1 Calculate VADER Sentiment Scores"""),
        nbf.v4.new_code_cell("""analyzer = SentimentIntensityAnalyzer()
df["full_review"] = (df["review_title"].fillna("") + ". " + df["review_text"].fillna("")).str.strip()

compounds = []
labels = []
for text in df["full_review"]:
    vs = analyzer.polarity_scores(text)
    comp = vs["compound"]
    compounds.append(comp)
    if comp >= 0.05:
        labels.append("positive")
    elif comp <= -0.05:
        labels.append("negative")
    else:
        labels.append("neutral")

df["sentiment_compound"] = compounds
df["sentiment_label"] = labels
df.to_csv(PROCESSED_DATA_DIR / "reviews_with_sentiment.csv", index=False)
print("Saved reviews_with_sentiment.csv!")
"""),
        nbf.v4.new_markdown_cell("""### 2.2 Validation: Sentiment vs Star Rating Crosstab"""),
        nbf.v4.new_code_cell("""crosstab_pct = pd.crosstab(df["rating"], df["sentiment_label"], normalize="index") * 100
print(crosstab_pct.round(2).map(lambda x: f"{x:.2f}%"))

plt.figure(figsize=(8, 5))
sns.heatmap(crosstab_pct, annot=True, fmt=".1f", cmap="Blues", cbar=True)
plt.title("Sentiment Label vs Star Rating (Row %)", fontsize=13, fontweight="bold")
plt.xlabel("VADER Sentiment")
plt.ylabel("Star Rating")
plt.show()
"""),
        nbf.v4.new_markdown_cell("""### 2.3 Monthly Sentiment Trajectory"""),
        nbf.v4.new_code_cell("""df["month"] = pd.to_datetime(df["review_date"]).dt.to_period("M").dt.to_timestamp()
monthly_sentiment = df.groupby(["month", "sentiment_label"]).size().unstack(fill_value=0)
monthly_sentiment_pct = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0) * 100

plt.figure(figsize=(11, 5))
plt.plot(monthly_sentiment_pct.index, monthly_sentiment_pct["positive"], label="Positive %", color="#3ECF9E", marker="o", linewidth=2.5)
plt.plot(monthly_sentiment_pct.index, monthly_sentiment_pct["negative"], label="Negative %", color="#E5484D", marker="s", linewidth=2.5)
plt.plot(monthly_sentiment_pct.index, monthly_sentiment_pct["neutral"], label="Neutral %", color="#8B949E", marker="^", linewidth=2)
plt.title("Monthly Sentiment Trend (June 2024 - August 2025)", fontsize=13, fontweight="bold")
plt.ylabel("Percentage (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
""")
    ]
    out_path = NOTEBOOKS_DIR / "02_sentiment_analysis.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created: {out_path}")

def create_notebook_03():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Step 3: NMF Topic Modeling (Dual Model Architecture)
Extracts distinct complaint themes from negative reviews and praise themes from positive reviews using scikit-learn NMF on TF-IDF.
"""),
        nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

BASE_DIR = Path(r"C:\\Users\\Prashma\\Desktop\\Resume Projects\\Data Analytics\\NLP-powered customer review analytics system")
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(PROCESSED_DATA_DIR / "reviews_with_sentiment.csv")
print(f"Loaded {len(df):,} reviews.")
"""),
        nbf.v4.new_markdown_cell("""### 3.1 Run Negative & Positive Topic Models"""),
        nbf.v4.new_code_cell("""# Separate keyword rules
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

def auto_label(top_words, rules):
    for label, kws in rules:
        for kw in kws:
            if any(kw in w.lower() for w in top_words):
                return label
    return " / ".join(top_words[:3]).title()

def fit_nmf(texts, k, rules):
    vec = TfidfVectorizer(max_df=0.8, min_df=5, stop_words="english", ngram_range=(1,2))
    tfidf = vec.fit_transform(texts)
    nmf = NMF(n_components=k, random_state=42, init="nndsvda", max_iter=400)
    doc_topics = nmf.fit_transform(tfidf)
    feats = vec.get_feature_names_out()
    labels = []
    for topic in nmf.components_:
        top_w = [feats[i] for i in topic.argsort()[:-8:-1]]
        labels.append(auto_label(top_w, rules))
    return [labels[i] for i in doc_topics.argmax(axis=1)]

# Assign themes
neg_mask = df["sentiment_label"] == "negative"
pos_mask = df["sentiment_label"] == "positive"

df["topic_theme"] = "Neutral / General Feedback"
df["topic_polarity"] = df["sentiment_label"]

df.loc[neg_mask, "topic_theme"] = fit_nmf(df.loc[neg_mask, "full_review"].tolist(), 10, NEGATIVE_LABEL_RULES)
df.loc[pos_mask, "topic_theme"] = fit_nmf(df.loc[pos_mask, "full_review"].tolist(), 8, POSITIVE_LABEL_RULES)

df.to_csv(PROCESSED_DATA_DIR / "reviews_with_topics.csv", index=False)
print("Saved reviews_with_topics.csv!")
"""),
        nbf.v4.new_markdown_cell("""### 3.2 Visualizing Complaint & Praise Clusters"""),
        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

df[df["sentiment_label"] == "negative"]["topic_theme"].value_counts().plot(kind='barh', ax=axes[0], color='#E5484D', edgecolor='black')
axes[0].set_title('Customer Complaint Themes (Negative Reviews)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Review Count')

df[df["sentiment_label"] == "positive"]["topic_theme"].value_counts().plot(kind='barh', ax=axes[1], color='#3ECF9E', edgecolor='black')
axes[1].set_title('Customer Praise Themes (Positive Reviews)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Review Count')

plt.tight_layout()
plt.show()
""")
    ]
    out_path = NOTEBOOKS_DIR / "03_topic_modeling.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created: {out_path}")

def create_notebook_04():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Step 4: Early-Warning Product Risk Scoring
Builds a composite 0-100 risk score per product combining 90-day deterioration trend (55%), weighted complaint theme severity (45%), and volume confidence dampening.
"""),
        nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(r"C:\\Users\\Prashma\\Desktop\\Resume Projects\\Data Analytics\\NLP-powered customer review analytics system")
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

risk_df = pd.read_csv(PROCESSED_DATA_DIR / "product_risk_scores.csv")
print(f"Loaded {len(risk_df)} product risk records.")
risk_df.head()
"""),
        nbf.v4.new_markdown_cell("""### 4.1 Watchlist of At-Risk Products (Critical & High)"""),
        nbf.v4.new_code_cell("""watchlist = risk_df[risk_df["risk_tier"].isin(["Critical", "High"])][
    ["product_id", "product_name", "risk_score", "risk_tier", "overall_avg_rating", "recent_avg_rating", "recent_negative_rate", "top_complaint_theme"]
]
watchlist
"""),
        nbf.v4.new_markdown_cell("""### 4.2 Lifetime vs Recent Rating Deterioration"""),
        nbf.v4.new_code_cell("""watch_df = watchlist.copy()
x = np.arange(len(watch_df))
width = 0.35

plt.figure(figsize=(12, 5))
plt.bar(x - width/2, watch_df["overall_avg_rating"], width, label="Lifetime Rating", color="#4A90E2", edgecolor="black")
plt.bar(x + width/2, watch_df["recent_avg_rating"], width, label="Recent 90-Day Rating", color="#E5484D", edgecolor="black")

plt.ylabel("Rating (1-5)")
plt.title("Lifetime vs Recent 90-Day Rating Deterioration", fontsize=13, fontweight="bold")
plt.xticks(x, watch_df["product_name"], rotation=35, ha="right", fontsize=9)
plt.ylim(0, 5.5)
plt.legend()
plt.tight_layout()
plt.show()
""")
    ]
    out_path = NOTEBOOKS_DIR / "04_risk_scoring.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created: {out_path}")

def create_notebook_05():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("""# Step 5: Dashboard & Tableau Exports
Verifies the Tableau-ready flat CSV exports and standalone interactive HTML dashboard.
"""),
        nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd

BASE_DIR = Path(r"C:\\Users\\Prashma\\Desktop\\Resume Projects\\Data Analytics\\NLP-powered customer review analytics system")
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
DASHBOARD_PATH = BASE_DIR / "dashboard" / "index.html"

# Verify Tableau CSVs
t1 = pd.read_csv(PROCESSED_DATA_DIR / "monthly_sentiment_trend.csv")
t2 = pd.read_csv(PROCESSED_DATA_DIR / "category_sentiment_breakdown.csv")

print("Tableau Monthly Trend Sample:")
print(t1.head())

print("\nTableau Category Breakdown Sample:")
print(t2.head())

print(f"\nDashboard HTML exists: {DASHBOARD_PATH.exists()} (Size: {DASHBOARD_PATH.stat().st_size / 1024:.2f} KB)")
""")
    ]
    out_path = NOTEBOOKS_DIR / "05_dashboard_and_tableau_exports.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created: {out_path}")

if __name__ == "__main__":
    create_notebook_01()
    create_notebook_02()
    create_notebook_03()
    create_notebook_04()
    create_notebook_05()
    print(" All 5 notebooks successfully generated!")
