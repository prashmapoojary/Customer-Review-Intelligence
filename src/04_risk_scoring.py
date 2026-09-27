"""
Step 4: Composite "At Risk" Product Watchlist & Risk Scoring
Combines:
1. Trend Score (55% weight): 90-day vs prior 90-day negative review rate delta
2. Severity Score (45% weight): Weighted severity of recent complaint themes
3. Volume Confidence: Dampens low-volume false alarms and flags Insufficient Data
Assigns Risk Tiers: Critical (>=60), High (>=35), Watch (>=15), Healthy (<15), Insufficient Data (<8 reviews)
Saves data/processed/product_risk_scores.csv and diagnostic charts.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Hardcoded BASE_DIR
BASE_DIR = Path(r"C:\Users\Prashma\Desktop\Resume Projects\Data Analytics\NLP-powered customer review analytics system")

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = PROCESSED_DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Severity weights for complaint themes (scale 0.0 - 3.0)
THEME_SEVERITY_WEIGHTS = {
    "Defective / Hardware Failure": 3.0,
    "Durability / Build Quality": 3.0,
    "Excessive Noise / Chemical Odor": 2.2,
    "Customer Service & Refund Issues": 2.0,
    "Overpriced / Poor Value": 1.8,
    "Shipping / Packaging Damage": 1.5,
    "Confusing Manual / Setup Difficulty": 1.2,
    "Sizing & Dimension Inaccuracy": 1.2,
    "Average / Product Standard / Standard": 0.5,
    "Neutral / General Feedback": 0.0,
}

MIN_RECENT_REVIEWS = 8  # Below this threshold -> Insufficient Data

def compute_risk_scores():
    print("--- Step 4: Computing Product Risk Scores & Watchlist ---")
    
    input_path = PROCESSED_DATA_DIR / "reviews_with_topics.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing topics file at: {input_path}")
        
    df = pd.read_csv(input_path)
    df["review_date"] = pd.to_datetime(df["review_date"])
    print(f"Loaded {len(df):,} reviews.")
    
    # 90-day time window split
    max_date = df["review_date"].max()
    recent_cutoff = max_date - pd.Timedelta(days=90)
    prior_cutoff = recent_cutoff - pd.Timedelta(days=90)
    
    print(f"Analysis Time Windows (Max Date: {max_date.strftime('%Y-%m-%d')}):")
    print(f"  Recent Period: {recent_cutoff.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} (Last 90 Days)")
    print(f"  Prior Period:  {prior_cutoff.strftime('%Y-%m-%d')} to {recent_cutoff.strftime('%Y-%m-%d')} (Preceding 90 Days)")
    
    recent_df = df[df["review_date"] >= recent_cutoff]
    prior_df = df[(df["review_date"] >= prior_cutoff) & (df["review_date"] < recent_cutoff)]
    
    products = df[["product_id", "product_name", "category"]].drop_duplicates().sort_values("product_id")
    
    records = []
    
    for _, prod in products.iterrows():
        pid = prod["product_id"]
        pname = prod["product_name"]
        cat = prod["category"]
        
        all_p = df[df["product_id"] == pid]
        rec_p = recent_df[recent_df["product_id"] == pid]
        pri_p = prior_df[prior_df["product_id"] == pid]
        
        n_total = len(all_p)
        n_recent = len(rec_p)
        n_prior = len(pri_p)
        
        overall_avg_rating = all_p["rating"].mean() if n_total > 0 else 0.0
        recent_avg_rating = rec_p["rating"].mean() if n_recent > 0 else 0.0
        
        # Negative counts
        rec_neg_df = rec_p[rec_p["sentiment_label"] == "negative"]
        rec_neg_count = len(rec_neg_df)
        pri_neg_count = len(pri_p[pri_p["sentiment_label"] == "negative"])
        
        recent_neg_rate = (rec_neg_count / n_recent) if n_recent > 0 else 0.0
        prior_neg_rate = (pri_neg_count / n_prior) if n_prior > 0 else recent_neg_rate
        
        # 1. Trend Score (55% weight)
        # Rising negative rate is heavily penalized
        delta_neg = max(0.0, recent_neg_rate - prior_neg_rate)
        raw_trend = (recent_neg_rate * 0.50 + delta_neg * 0.50) * 100.0
        trend_score = min(100.0, max(0.0, raw_trend))
        
        # 2. Severity Score (45% weight)
        if rec_neg_count > 0:
            rec_weights = [THEME_SEVERITY_WEIGHTS.get(theme, 1.5) for theme in rec_neg_df["topic_theme"]]
            avg_severity_weight = np.mean(rec_weights)
            # Max possible weight is 3.0
            raw_severity = (avg_severity_weight / 3.0) * recent_neg_rate * 100.0
            severity_score = min(100.0, max(0.0, raw_severity))
            
            # Top complaint theme in recent window
            top_complaint = rec_neg_df["topic_theme"].value_counts().index[0]
        else:
            avg_severity_weight = 0.0
            severity_score = 0.0
            top_complaint = "None (No Recent Complaints)"
            
        # 3. Volume Confidence
        # Full confidence when reviews >= 2 * MIN_RECENT_REVIEWS (16)
        volume_confidence = min(1.0, n_recent / (2.0 * MIN_RECENT_REVIEWS)) if n_recent > 0 else 0.0
        
        # Composite calculation
        raw_composite = (0.55 * trend_score) + (0.45 * severity_score)
        risk_score = round(raw_composite * volume_confidence, 1)
        
        # Risk Tier Assignment
        if n_recent < MIN_RECENT_REVIEWS:
            risk_tier = "Insufficient Data"
        elif risk_score >= 60.0:
            risk_tier = "Critical"
        elif risk_score >= 35.0:
            risk_tier = "High"
        elif risk_score >= 15.0:
            risk_tier = "Watch"
        else:
            risk_tier = "Healthy"
            
        records.append({
            "product_id": pid,
            "product_name": pname,
            "category": cat,
            "overall_avg_rating": round(overall_avg_rating, 2),
            "recent_avg_rating": round(recent_avg_rating, 2),
            "n_reviews_total": n_total,
            "n_reviews_recent": n_recent,
            "n_reviews_prior": n_prior,
            "recent_negative_rate": round(recent_neg_rate, 3),
            "prior_negative_rate": round(prior_neg_rate, 3),
            "trend_score": round(trend_score, 1),
            "severity_score": round(severity_score, 1),
            "volume_confidence": round(volume_confidence, 2),
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "top_complaint_theme": top_complaint
        })
        
    risk_df = pd.DataFrame(records).sort_values("risk_score", ascending=False).reset_index(drop=True)
    
    # Save CSV
    output_path = PROCESSED_DATA_DIR / "product_risk_scores.csv"
    risk_df.to_csv(output_path, index=False)
    print(f" Saved product risk scores to: {output_path}")
    
    # Summary of Tiers
    print("\n--- Risk Tier Distribution ---")
    print(risk_df["risk_tier"].value_counts())
    
    print("\n--- At-Risk Watchlist (Critical & High Tiers) ---")
    watchlist = risk_df[risk_df["risk_tier"].isin(["Critical", "High"])][
        ["product_id", "product_name", "risk_score", "risk_tier", "overall_avg_rating", "recent_avg_rating", "recent_negative_rate", "top_complaint_theme"]
    ]
    print(watchlist.to_string(index=False))
    
    # Charts
    # 1. Risk Score Bar Chart (All products, color-coded by tier)
    plt.figure(figsize=(12, 10))
    tier_colors = {
        "Critical": "#E5484D",
        "High": "#F0913A",
        "Watch": "#F0C531",
        "Healthy": "#3ECF9E",
        "Insufficient Data": "#6E7681"
    }
    colors = [tier_colors.get(t, "#A0AEC0") for t in risk_df["risk_tier"]]
    
    plt.barh(risk_df["product_name"], risk_df["risk_score"], color=colors, edgecolor="black")
    plt.gca().invert_yaxis()
    plt.title("Product Risk Scores (0-100) & Watchlist Status", fontsize=14, fontweight="bold")
    plt.xlabel("Composite Risk Score")
    plt.ylabel("Product")
    
    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor="black", label=t) for t, c in tier_colors.items()]
    plt.legend(handles=legend_elements, loc="lower right", title="Risk Tier")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "product_risk_scores_bar.png", dpi=300)
    plt.close()
    
    # 2. Overall vs Recent Rating Comparison for Watchlist Products
    if len(watchlist) > 0:
        watch_df = risk_df[risk_df["risk_tier"].isin(["Critical", "High"])].copy()
        x = np.arange(len(watch_df))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, watch_df["overall_avg_rating"], width, label="Lifetime Avg Rating", color="#4A90E2", edgecolor="black")
        plt.bar(x + width/2, watch_df["recent_avg_rating"], width, label="Recent 90-Day Avg Rating", color="#E5484D", edgecolor="black")
        
        plt.ylabel("Star Rating (1-5)")
        plt.title("Lifetime vs Recent 90-Day Rating Deterioration (At-Risk Products)", fontsize=13, fontweight="bold")
        plt.xticks(x, watch_df["product_name"], rotation=35, ha="right", fontsize=9)
        plt.ylim(0, 5.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "watchlist_rating_comparison.png", dpi=300)
        plt.close()
        
    # 3. Monthly Negative Rate Trend Line for Top 3 At-Risk Products
    top3_pids = risk_df.head(3)["product_id"].tolist()
    df_temp = df[df["product_id"].isin(top3_pids)].copy()
    df_temp["month"] = df_temp["review_date"].dt.to_period("M").dt.to_timestamp()
    
    top3_trend = df_temp.groupby(["month", "product_name", "sentiment_label"]).size().unstack(fill_value=0)
    top3_trend["total"] = top3_trend.sum(axis=1)
    top3_trend["neg_rate"] = (top3_trend.get("negative", 0) / top3_trend["total"]) * 100
    top3_trend = top3_trend.reset_index()
    
    plt.figure(figsize=(12, 5))
    for pname in df_temp["product_name"].unique():
        sub = top3_trend[top3_trend["product_name"] == pname]
        plt.plot(sub["month"], sub["neg_rate"], marker="o", linewidth=2.5, label=pname)
        
    plt.title("Monthly Negative Review Rate (%) - Top 3 Critical Products", fontsize=13, fontweight="bold")
    plt.ylabel("Negative Review Rate (%)")
    plt.xlabel("Month")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top3_at_risk_trend.png", dpi=300)
    plt.close()
    
    print(f" Generated risk diagnostic charts in: {FIGURES_DIR}")
    return risk_df

if __name__ == "__main__":
    compute_risk_scores()
