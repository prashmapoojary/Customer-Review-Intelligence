"""
Step 5: Interactive Executive Dashboard & Tableau Export Generator
Produces:
1. Tableau-ready flat CSV exports in data/processed/
   - monthly_sentiment_trend.csv
   - category_sentiment_breakdown.csv
2. Standalone dark command-center HTML dashboard at dashboard/index.html
   with Chart.js, tabbed navigation, KPI metrics, watchlist table, and product drill-down.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

# Hardcoded BASE_DIR
BASE_DIR = Path(r"C:\Users\Prashma\Desktop\Resume Projects\Data Analytics\NLP-powered customer review analytics system")

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
DASHBOARD_DIR = BASE_DIR / "dashboard"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

def generate_dashboard_and_tableau_exports():
    print("--- Step 5: Generating Tableau CSVs and HTML Dashboard ---")
    
    # Load processed data
    reviews_path = PROCESSED_DATA_DIR / "reviews_with_topics.csv"
    risk_path = PROCESSED_DATA_DIR / "product_risk_scores.csv"
    summary_path = PROCESSED_DATA_DIR / "topic_summary.csv"
    
    if not reviews_path.exists() or not risk_path.exists():
        raise FileNotFoundError("Required processed files not found. Run Steps 1-4 first.")
        
    df = pd.read_csv(reviews_path)
    risk_df = pd.read_csv(risk_path)
    topic_summary_df = pd.read_csv(summary_path) if summary_path.exists() else None
    
    df["review_date"] = pd.to_datetime(df["review_date"])
    df["review_month"] = df["review_date"].dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m")
    
    # -------------------------------------------------------------
    # 1. Tableau-Ready Flat CSV Exports
    # -------------------------------------------------------------
    # Export 1: Monthly Sentiment Trend
    monthly_tableau = df.groupby(["review_month", "sentiment_label"]).size().reset_index(name="review_count")
    monthly_csv_path = PROCESSED_DATA_DIR / "monthly_sentiment_trend.csv"
    monthly_tableau.to_csv(monthly_csv_path, index=False)
    print(f" Saved Tableau export: {monthly_csv_path}")
    
    # Export 2: Category Sentiment Breakdown
    cat_tableau = df.groupby(["category", "sentiment_label"]).size().reset_index(name="review_count")
    cat_csv_path = PROCESSED_DATA_DIR / "category_sentiment_breakdown.csv"
    cat_tableau.to_csv(cat_csv_path, index=False)
    print(f" Saved Tableau export: {cat_csv_path}")
    
    # -------------------------------------------------------------
    # 2. Pre-Aggregating JSON for HTML Dashboard (<50KB)
    # -------------------------------------------------------------
    total_reviews = int(len(df))
    total_products = int(df["product_id"].nunique())
    avg_rating = round(float(df["rating"].mean()), 2)
    
    pos_count = int((df["sentiment_label"] == "positive").sum())
    neg_count = int((df["sentiment_label"] == "negative").sum())
    neu_count = int((df["sentiment_label"] == "neutral").sum())
    
    pos_pct = round((pos_count / total_reviews) * 100, 1)
    neg_pct = round((neg_count / total_reviews) * 100, 1)
    neu_pct = round((neu_count / total_reviews) * 100, 1)
    
    critical_count = int((risk_df["risk_tier"] == "Critical").sum())
    high_count = int((risk_df["risk_tier"] == "High").sum())
    watch_count = int((risk_df["risk_tier"] == "Watch").sum())
    healthy_count = int((risk_df["risk_tier"] == "Healthy").sum())
    
    # Monthly timeline aggregates
    months_list = sorted(df["review_month"].unique().tolist())
    monthly_pos = []
    monthly_neg = []
    monthly_neu = []
    
    for m in months_list:
        sub = df[df["review_month"] == m]
        m_tot = len(sub)
        m_pos = (sub["sentiment_label"] == "positive").sum()
        m_neg = (sub["sentiment_label"] == "negative").sum()
        m_neu = (sub["sentiment_label"] == "neutral").sum()
        monthly_pos.append(round((m_pos / m_tot) * 100, 1) if m_tot else 0)
        monthly_neg.append(round((m_neg / m_tot) * 100, 1) if m_tot else 0)
        monthly_neu.append(round((m_neu / m_tot) * 100, 1) if m_tot else 0)
        
    # Category sentiment breakdown
    categories_list = sorted(df["category"].unique().tolist())
    cat_pos = []
    cat_neg = []
    cat_neu = []
    for c in categories_list:
        sub = df[df["category"] == c]
        c_tot = len(sub)
        cat_pos.append(int((sub["sentiment_label"] == "positive").sum()))
        cat_neg.append(int((sub["sentiment_label"] == "negative").sum()))
        cat_neu.append(int((sub["sentiment_label"] == "neutral").sum()))
        
    # Complaint & Praise Themes
    neg_themes_agg = df[df["sentiment_label"] == "negative"]["topic_theme"].value_counts().to_dict()
    pos_themes_agg = df[df["sentiment_label"] == "positive"]["topic_theme"].value_counts().to_dict()
    
    # Product Drill-Down Aggregates
    product_drilldown = {}
    for pid in df["product_id"].unique():
        pdf = df[df["product_id"] == pid]
        p_name = pdf["product_name"].iloc[0]
        p_cat = pdf["category"].iloc[0]
        
        # Monthly ratings and negative counts
        p_monthly_agg = pdf.groupby("review_month").agg(
            avg_rating=("rating", "mean"),
            review_count=("review_id", "count"),
            neg_count=("sentiment_label", lambda s: (s == "negative").sum())
        ).reindex(months_list).fillna(0)
        
        p_neg_themes = pdf[pdf["sentiment_label"] == "negative"]["topic_theme"].value_counts().to_dict()
        p_pos_themes = pdf[pdf["sentiment_label"] == "positive"]["topic_theme"].value_counts().to_dict()
        
        # Risk row info
        r_row = risk_df[risk_df["product_id"] == pid]
        if len(r_row) > 0:
            r_info = {
                "risk_score": float(r_row["risk_score"].iloc[0]),
                "risk_tier": str(r_row["risk_tier"].iloc[0]),
                "overall_avg_rating": float(r_row["overall_avg_rating"].iloc[0]),
                "recent_avg_rating": float(r_row["recent_avg_rating"].iloc[0]),
                "recent_neg_rate": float(r_row["recent_negative_rate"].iloc[0]),
                "top_complaint": str(r_row["top_complaint_theme"].iloc[0]),
                "n_reviews_total": int(r_row["n_reviews_total"].iloc[0]),
                "n_reviews_recent": int(r_row["n_reviews_recent"].iloc[0]),
            }
        # Only include active non-zero themes
        active_neg_themes = {k: int(v) for k, v in p_neg_themes.items() if v > 0}
        
        product_drilldown[pid] = {
            "name": p_name,
            "cat": p_cat,
            "tier": str(r_info.get("risk_tier", "N/A")),
            "score": float(r_info.get("risk_score", 0)),
            "rec_r": float(r_info.get("recent_avg_rating", 0)),
            "tot_r": float(r_info.get("overall_avg_rating", 0)),
            "rec_neg": round(float(r_info.get("recent_neg_rate", 0)), 2),
            "top_comp": str(r_info.get("top_complaint", "None")),
            "n_tot": int(r_info.get("n_reviews_total", 0)),
            "n_rec": int(r_info.get("n_reviews_recent", 0)),
            "m_r": [round(float(x), 1) for x in p_monthly_agg["avg_rating"].tolist()],
            "m_neg": [int(x) for x in p_monthly_agg["neg_count"].tolist()],
            "themes": active_neg_themes
        }
        
    # Compact watchlist records
    watchlist_records = []
    for _, row in risk_df.iterrows():
        watchlist_records.append({
            "id": str(row["product_id"]),
            "name": str(row["product_name"]),
            "cat": str(row["category"]),
            "score": round(float(row["risk_score"]), 1),
            "tier": str(row["risk_tier"]),
            "tot_r": round(float(row["overall_avg_rating"]), 1),
            "rec_r": round(float(row["recent_avg_rating"]), 1),
            "rec_neg": round(float(row["recent_negative_rate"]), 2),
            "top_comp": str(row["top_complaint_theme"])
        })
    
    dashboard_data = {
        "kpis": {
            "total_reviews": total_reviews,
            "total_products": total_products,
            "avg_rating": avg_rating,
            "pos_pct": pos_pct,
            "neg_pct": neg_pct,
            "neu_pct": neu_pct,
            "critical_count": critical_count,
            "high_count": high_count,
            "watch_count": watch_count,
            "healthy_count": healthy_count
        },
        "timeline": {
            "months": months_list,
            "pos": monthly_pos,
            "neg": monthly_neg,
            "neu": monthly_neu
        },
        "categories": {
            "names": categories_list,
            "pos": cat_pos,
            "neg": cat_neg,
            "neu": cat_neu
        },
        "themes": {
            "comp": neg_themes_agg,
            "praise": pos_themes_agg
        },
        "watchlist": watchlist_records,
        "drilldown": product_drilldown
    }
    
    data_json_str = json.dumps(dashboard_data, separators=(',', ':'))
    
    # -------------------------------------------------------------
    # 3. Standalone HTML Dashboard Generator
    # -------------------------------------------------------------
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Customer Review Intelligence | NLP Analytics Dashboard</title>
  <meta name="description" content="Executive NLP customer review analytics dashboard featuring sentiment analysis, NMF topic modeling, and early warning risk scoring.">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg-dark: #0D1117;
      --card-bg: #161B22;
      --card-border: #30363D;
      --text-main: #F0F6FC;
      --text-muted: #8B949E;
      --teal-accent: #3ECF9E;
      --red-critical: #E5484D;
      --amber-high: #F0913A;
      --yellow-watch: #F0C531;
      --gray-healthy: #3ECF9E;
      --blue-accent: #58A6FF;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}
    body {{
      background-color: var(--bg-dark);
      color: var(--text-main);
      padding: 24px;
      min-height: 100vh;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .header-title h1 {{
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.5px;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .badge-nlp {{
      background: rgba(62, 207, 158, 0.15);
      color: var(--teal-accent);
      font-size: 12px;
      padding: 4px 10px;
      border-radius: 20px;
      border: 1px solid rgba(62, 207, 158, 0.3);
      text-transform: uppercase;
      font-weight: 600;
    }}
    .header-subtitle {{
      color: var(--text-muted);
      font-size: 13px;
      margin-top: 4px;
    }}
    .nav-tabs {{
      display: flex;
      gap: 8px;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 8px;
      overflow-x: auto;
    }}
    .nav-btn {{
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      padding: 10px 18px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
    }}
    .nav-btn:hover {{
      color: var(--text-main);
      background: rgba(255, 255, 255, 0.05);
    }}
    .nav-btn.active {{
      color: #FFFFFF;
      background: #21262D;
      border-color: var(--card-border);
      box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 16px;
      display: flex;
      flex-direction: column;
    }}
    .kpi-label {{
      color: var(--text-muted);
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 8px;
    }}
    .kpi-value {{
      font-size: 26px;
      font-weight: 700;
      color: var(--text-main);
    }}
    .kpi-sub {{
      font-size: 12px;
      margin-top: 6px;
    }}
    .text-teal {{ color: var(--teal-accent); }}
    .text-red {{ color: var(--red-critical); }}
    .text-amber {{ color: var(--amber-high); }}
    .text-yellow {{ color: var(--yellow-watch); }}
    .text-blue {{ color: var(--blue-accent); }}
    
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    @media (max-width: 900px) {{
      .grid-2 {{ grid-template-columns: 1fr; }}
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 20px;
      position: relative;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }}
    .card-title {{
      font-size: 15px;
      font-weight: 600;
      color: var(--text-main);
    }}
    .chart-box {{
      position: relative;
      height: 320px;
      width: 100%;
    }}
    .tab-pane {{
      display: none;
    }}
    .tab-pane.active {{
      display: block;
    }}
    /* Watchlist Table */
    .table-container {{
      overflow-x: auto;
      border-radius: 8px;
      border: 1px solid var(--card-border);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }}
    th {{
      background: #21262D;
      color: var(--text-muted);
      padding: 12px 16px;
      font-weight: 600;
      border-bottom: 1px solid var(--card-border);
      cursor: pointer;
      user-select: none;
    }}
    th:hover {{
      color: var(--text-main);
    }}
    td {{
      padding: 12px 16px;
      border-bottom: 1px solid rgba(48, 54, 61, 0.5);
      color: var(--text-main);
    }}
    tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
    }}
    .pill {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }}
    .pill-critical {{ background: rgba(229, 72, 77, 0.2); color: #FF6B6B; border: 1px solid rgba(229, 72, 77, 0.4); }}
    .pill-high {{ background: rgba(240, 145, 58, 0.2); color: #FFAA5A; border: 1px solid rgba(240, 145, 58, 0.4); }}
    .pill-watch {{ background: rgba(240, 197, 49, 0.2); color: #FFDD53; border: 1px solid rgba(240, 197, 49, 0.4); }}
    .pill-healthy {{ background: rgba(62, 207, 158, 0.2); color: #3ECF9E; border: 1px solid rgba(62, 207, 158, 0.4); }}
    .pill-insufficient {{ background: rgba(110, 118, 129, 0.2); color: #8B949E; border: 1px solid rgba(110, 118, 129, 0.4); }}
    
    .select-dropdown {{
      background: #21262D;
      border: 1px solid var(--card-border);
      color: var(--text-main);
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 14px;
      width: 100%;
      max-width: 450px;
      margin-bottom: 20px;
      outline: none;
    }}
    .select-dropdown:focus {{
      border-color: var(--teal-accent);
    }}
    .note-box {{
      background: rgba(88, 166, 255, 0.08);
      border: 1px solid rgba(88, 166, 255, 0.25);
      border-radius: 8px;
      padding: 14px 18px;
      font-size: 13px;
      color: #C9D1D9;
      line-height: 1.5;
      margin-top: 20px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-title">
        <h1>Amazon Review Intelligence Center <span class="badge-nlp">VADER + NMF Engine</span></h1>
        <div class="header-subtitle">NLP Sentiment Analysis, Dual-Model Topic Extraction & Predictive Risk Scoring</div>
      </div>
      <div style="font-size: 13px; color: var(--text-muted);">
        Dataset: <strong>6,655 Reviews</strong> | <strong>35 Products</strong> (14-Month Rolling)
      </div>
    </header>

    <!-- Navigation Tabs -->
    <nav class="nav-tabs">
      <button class="nav-btn active" onclick="switchTab('overview')"> Executive Overview</button>
      <button class="nav-btn" onclick="switchTab('themes')"> Complaint & Praise Themes</button>
      <button class="nav-btn" onclick="switchTab('watchlist')"> At-Risk Watchlist</button>
      <button class="nav-btn" onclick="switchTab('drilldown')"> Product Drill-Down</button>
    </nav>

    <!-- TAB 1: EXECUTIVE OVERVIEW -->
    <div id="tab-overview" class="tab-pane active">
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">Total Reviews</div>
          <div class="kpi-value" id="kpi-total-reviews">-</div>
          <div class="kpi-sub text-blue">Across 6 Categories</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Positive Sentiment</div>
          <div class="kpi-value text-teal" id="kpi-pos-pct">-</div>
          <div class="kpi-sub" style="color: var(--text-muted)">VADER Compound &ge; 0.05</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Negative Sentiment</div>
          <div class="kpi-value text-red" id="kpi-neg-pct">-</div>
          <div class="kpi-sub" style="color: var(--text-muted)">VADER Compound &le; -0.05</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Avg Star Rating</div>
          <div class="kpi-value text-yellow" id="kpi-avg-rating">-</div>
          <div class="kpi-sub" style="color: var(--text-muted)">1.0 to 5.0 scale</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Critical / High Risk</div>
          <div class="kpi-value text-red" id="kpi-critical-high">-</div>
          <div class="kpi-sub text-red">Action Required Immediately</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Watch Tier</div>
          <div class="kpi-value text-yellow" id="kpi-watch">-</div>
          <div class="kpi-sub text-yellow">Monitoring Trend Delta</div>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <div class="card-title">14-Month Sentiment Trend Ratio (%)</div>
          </div>
          <div class="chart-box">
            <canvas id="chart-sentiment-trend"></canvas>
          </div>
        </div>
        <div class="card">
          <div class="card-header">
            <div class="card-title">Category Sentiment Distribution</div>
          </div>
          <div class="chart-box">
            <canvas id="chart-category-sentiment"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: COMPLAINT & PRAISE THEMES -->
    <div id="tab-themes" class="tab-pane">
      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <div class="card-title text-red"> Top Customer Complaint Themes (NMF Negative Model)</div>
          </div>
          <div class="chart-box" style="height: 380px;">
            <canvas id="chart-complaint-themes"></canvas>
          </div>
        </div>
        <div class="card">
          <div class="card-header">
            <div class="card-title text-teal"> Top Customer Praise Themes (NMF Positive Model)</div>
          </div>
          <div class="chart-box" style="height: 380px;">
            <canvas id="chart-praise-themes"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: AT-RISK WATCHLIST -->
    <div id="tab-watchlist" class="tab-pane">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Early-Warning Product Risk Watchlist</div>
            <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
              Combines 90-Day Deterioration Trend (55%), Complaint Severity Weights (45%), and Volume Confidence.
            </div>
          </div>
        </div>
        <div class="table-container">
          <table id="watchlist-table">
            <thead>
              <tr>
                <th onclick="sortTable(0)">Product Name</th>
                <th onclick="sortTable(1)">Category</th>
                <th onclick="sortTable(2)">Risk Score</th>
                <th onclick="sortTable(3)">Risk Tier</th>
                <th onclick="sortTable(4)">Lifetime Rating</th>
                <th onclick="sortTable(5)">Recent 90d Rating</th>
                <th onclick="sortTable(6)">Recent Neg %</th>
                <th onclick="sortTable(7)">Top Complaint Theme</th>
              </tr>
            </thead>
            <tbody id="watchlist-tbody">
              <!-- Rendered dynamically -->
            </tbody>
          </table>
        </div>
        <div class="note-box">
          <strong> Tradeoff Note on Early-Warning Detection Systems:</strong> Early-warning risk models are intentionally biased toward higher sensitivity over specificity. False positives (temporary shipping delays or isolated bad batches) are an accepted operational tradeoff to ensure emerging structural defects or manufacturing quality slips are caught before irreversible brand erosion occurs.
        </div>
      </div>
    </div>

    <!-- TAB 4: PRODUCT DRILL-DOWN -->
    <div id="tab-drilldown" class="tab-pane">
      <div class="card">
        <div class="card-header">
          <div class="card-title">Select Product to Inspect</div>
        </div>
        <select id="product-select" class="select-dropdown" onchange="renderProductDrilldown()">
          <!-- Options populated by JS -->
        </select>

        <div class="kpi-grid" style="grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); margin-bottom: 20px;">
          <div class="kpi-card">
            <div class="kpi-label">Risk Status</div>
            <div class="kpi-value" id="pdrill-tier">-</div>
            <div class="kpi-sub" id="pdrill-score">-</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Recent 90d Rating</div>
            <div class="kpi-value" id="pdrill-recent-rating">-</div>
            <div class="kpi-sub" id="pdrill-lifetime-rating">-</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Recent Negativity</div>
            <div class="kpi-value" id="pdrill-recent-neg">-</div>
            <div class="kpi-sub" id="pdrill-reviews-count">-</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Primary Complaint</div>
            <div class="kpi-value" style="font-size: 16px; margin-top: 4px;" id="pdrill-top-complaint">-</div>
          </div>
        </div>

        <div class="grid-2">
          <div class="card" style="background: #11161D;">
            <div class="card-header">
              <div class="card-title">Monthly Average Rating & Negative Review Count</div>
            </div>
            <div class="chart-box">
              <canvas id="chart-product-trend"></canvas>
            </div>
          </div>
          <div class="card" style="background: #11161D;">
            <div class="card-header">
              <div class="card-title">Product Complaint Theme Breakdown</div>
            </div>
            <div class="chart-box">
              <canvas id="chart-product-themes"></canvas>
            </div>
          </div>
        </div>
      </div>
    </div>

  </div>

  <script>
    // Embedded compact aggregated JSON dataset
    const DATA = {data_json_str};

    // Initialize Dashboard
    document.addEventListener("DOMContentLoaded", () => {{
      initKPIs();
      initOverviewCharts();
      initThemeCharts();
      initWatchlistTable();
      initProductDropdown();
    }});

    function switchTab(tabId) {{
      document.querySelectorAll(".tab-pane").forEach(el => el.classList.remove("active"));
      document.querySelectorAll(".nav-btn").forEach(el => el.classList.remove("active"));
      
      document.getElementById("tab-" + tabId).classList.add("active");
      event.currentTarget.classList.add("active");
    }}

    function initKPIs() {{
      const k = DATA.kpis;
      document.getElementById("kpi-total-reviews").innerText = k.total_reviews.toLocaleString();
      document.getElementById("kpi-pos-pct").innerText = k.pos_pct + "%";
      document.getElementById("kpi-neg-pct").innerText = k.neg_pct + "%";
      document.getElementById("kpi-avg-rating").innerText = k.avg_rating + " ★";
      document.getElementById("kpi-critical-high").innerText = (k.critical_count + k.high_count);
      document.getElementById("kpi-watch").innerText = k.watch_count;
    }}

    let trendChart, catChart, compChart, praiseChart, prodTrendChart, prodThemesChart;

    function initOverviewCharts() {{
      // 1. Sentiment Trend Line Chart
      const ctxTrend = document.getElementById("chart-sentiment-trend").getContext("2d");
      trendChart = new Chart(ctxTrend, {{
        type: "line",
        data: {{
          labels: DATA.timeline.months,
          datasets: [
            {{
              label: "Positive %",
              data: DATA.timeline.pos,
              borderColor: "#3ECF9E",
              backgroundColor: "rgba(62, 207, 158, 0.1)",
              fill: true,
              tension: 0.3
            }},
            {{
              label: "Negative %",
              data: DATA.timeline.neg,
              borderColor: "#E5484D",
              backgroundColor: "rgba(229, 72, 77, 0.1)",
              fill: true,
              tension: 0.3
            }},
            {{
              label: "Neutral %",
              data: DATA.timeline.neu,
              borderColor: "#8B949E",
              borderDash: [4, 4],
              tension: 0.3
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color: "#F0F6FC" }} }}
          }},
          scales: {{
            x: {{ ticks: {{ color: "#8B949E" }}, grid: {{ color: "rgba(48, 54, 61, 0.5)" }} }},
            y: {{ ticks: {{ color: "#8B949E" }}, grid: {{ color: "rgba(48, 54, 61, 0.5)" }}, max: 100 }}
          }}
        }}
      }});

      // 2. Category Sentiment Stacked Bar
      const ctxCat = document.getElementById("chart-category-sentiment").getContext("2d");
      catChart = new Chart(ctxCat, {{
        type: "bar",
        data: {{
          labels: DATA.categories.names,
          datasets: [
            {{ label: "Positive", data: DATA.categories.pos, backgroundColor: "#3ECF9E" }},
            {{ label: "Negative", data: DATA.categories.neg, backgroundColor: "#E5484D" }},
            {{ label: "Neutral", data: DATA.categories.neu, backgroundColor: "#8B949E" }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color: "#F0F6FC" }} }}
          }},
          scales: {{
            x: {{ stacked: true, ticks: {{ color: "#8B949E", maxRotation: 20 }}, grid: {{ display: false }} }},
            y: {{ stacked: true, ticks: {{ color: "#8B949E" }}, grid: {{ color: "rgba(48, 54, 61, 0.5)" }} }}
          }}
        }}
      }});
    }}

    function initThemeCharts() {{
      // Complaints Chart
      const compLabels = Object.keys(DATA.themes.comp);
      const compCounts = Object.values(DATA.themes.comp);
      const ctxComp = document.getElementById("chart-complaint-themes").getContext("2d");
      compChart = new Chart(ctxComp, {{
        type: "bar",
        data: {{
          labels: compLabels,
          datasets: [{{ label: "Complaints Count", data: compCounts, backgroundColor: "#E5484D", borderRadius: 4 }}]
        }},
        options: {{
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ ticks: {{ color: "#8B949E" }}, grid: {{ color: "rgba(48, 54, 61, 0.5)" }} }},
            y: {{ ticks: {{ color: "#F0F6FC", font: {{ size: 11 }} }}, grid: {{ display: false }} }}
          }}
        }}
      }});

      // Praise Chart
      const praiseLabels = Object.keys(DATA.themes.praise);
      const praiseCounts = Object.values(DATA.themes.praise);
      const ctxPraise = document.getElementById("chart-praise-themes").getContext("2d");
      praiseChart = new Chart(ctxPraise, {{
        type: "bar",
        data: {{
          labels: praiseLabels,
          datasets: [{{ label: "Praise Count", data: praiseCounts, backgroundColor: "#3ECF9E", borderRadius: 4 }}]
        }},
        options: {{
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ ticks: {{ color: "#8B949E" }}, grid: {{ color: "rgba(48, 54, 61, 0.5)" }} }},
            y: {{ ticks: {{ color: "#F0F6FC", font: {{ size: 11 }} }}, grid: {{ display: false }} }}
          }}
        }}
      }});
    }}

    function initWatchlistTable() {{
      const tbody = document.getElementById("watchlist-tbody");
      tbody.innerHTML = "";
      
      DATA.watchlist.forEach(row => {{
        const tr = document.createElement("tr");
        let pillClass = "pill-healthy";
        if (row.tier === "Critical") pillClass = "pill-critical";
        else if (row.tier === "High") pillClass = "pill-high";
        else if (row.tier === "Watch") pillClass = "pill-watch";
        else if (row.tier === "Insufficient Data") pillClass = "pill-insufficient";

        tr.innerHTML = `
          <td><strong>${{row.name}}</strong> <span style="font-size:11px; color:#8B949E;">(${{row.id}})</span></td>
          <td>${{row.cat}}</td>
          <td><strong>${{row.score}}</strong></td>
          <td><span class="pill ${{pillClass}}">${{row.tier}}</span></td>
          <td>${{row.tot_r.toFixed(2)}} ★</td>
          <td><strong style="color:${{row.rec_r < 2.5 ? '#FF6B6B' : '#3ECF9E'}}">${{row.rec_r.toFixed(2)}} ★</strong></td>
          <td>${{(row.rec_neg * 100).toFixed(1)}}%</td>
          <td style="color:#C9D1D9;">${{row.top_comp}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function initProductDropdown() {{
      const select = document.getElementById("product-select");
      select.innerHTML = "";
      Object.keys(DATA.drilldown).forEach(pid => {{
        const prod = DATA.drilldown[pid];
        const opt = document.createElement("option");
        opt.value = pid;
        opt.innerText = `[${{prod.tier || 'N/A'}}] ${{prod.name}} (${{prod.cat}})`;
        select.appendChild(opt);
      }});
      renderProductDrilldown();
    }}

    function renderProductDrilldown() {{
      const select = document.getElementById("product-select");
      const pid = select.value;
      const prod = DATA.drilldown[pid];
      if (!prod) return;

      document.getElementById("pdrill-tier").innerText = prod.tier || "N/A";
      document.getElementById("pdrill-tier").className = "kpi-value " + (
        prod.tier === "Critical" ? "text-red" :
        prod.tier === "High" ? "text-amber" :
        prod.tier === "Watch" ? "text-yellow" : "text-teal"
      );
      document.getElementById("pdrill-score").innerText = "Composite Score: " + prod.score;
      document.getElementById("pdrill-recent-rating").innerText = prod.rec_r + " ★";
      document.getElementById("pdrill-lifetime-rating").innerText = "Lifetime: " + prod.tot_r + " ★";
      document.getElementById("pdrill-recent-neg").innerText = ((prod.rec_neg || 0) * 100).toFixed(1) + "%";
      document.getElementById("pdrill-reviews-count").innerText = "Recent: " + prod.n_rec + " / Total: " + prod.n_tot;
      document.getElementById("pdrill-top-complaint").innerText = prod.top_comp || "None";

      // Product Trend Chart (Dual Axis)
      if (prodTrendChart) prodTrendChart.destroy();
      const ctxPT = document.getElementById("chart-product-trend").getContext("2d");
      prodTrendChart = new Chart(ctxPT, {{
        type: "line",
        data: {{
          labels: DATA.timeline.months,
          datasets: [
            {{
              label: "Average Star Rating",
              data: prod.m_r,
              borderColor: "#58A6FF",
              backgroundColor: "#58A6FF",
              yAxisID: "yRating",
              tension: 0.3
            }},
            {{
              label: "Negative Reviews Count",
              data: prod.m_neg,
              borderColor: "#E5484D",
              backgroundColor: "rgba(229, 72, 77, 0.2)",
              fill: true,
              yAxisID: "yCount",
              type: "bar"
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ labels: {{ color: "#F0F6FC" }} }} }},
          scales: {{
            x: {{ ticks: {{ color: "#8B949E" }}, grid: {{ color: "rgba(48, 54, 61, 0.4)" }} }},
            yRating: {{
              type: "linear",
              position: "left",
              min: 1,
              max: 5,
              ticks: {{ color: "#58A6FF" }},
              grid: {{ color: "rgba(48, 54, 61, 0.4)" }}
            }},
            yCount: {{
              type: "linear",
              position: "right",
              ticks: {{ color: "#E5484D" }},
              grid: {{ display: false }}
            }}
          }}
        }}
      }});

      // Product Complaint Themes Chart
      if (prodThemesChart) prodThemesChart.destroy();
      const themeLabels = Object.keys(prod.themes);
      const themeCounts = Object.values(prod.themes);
      const ctxPTH = document.getElementById("chart-product-themes").getContext("2d");
      prodThemesChart = new Chart(ctxPTH, {{
        type: "doughnut",
        data: {{
          labels: themeLabels.length ? themeLabels : ["No Complaints"],
          datasets: [{{
            data: themeCounts.length ? themeCounts : [1],
            backgroundColor: themeCounts.length ? ["#E5484D", "#F0913A", "#F0C531", "#58A6FF", "#BC8CFF", "#A371F7", "#7EE787"] : ["#3ECF9E"]
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ position: "right", labels: {{ color: "#F0F6FC", font: {{ size: 11 }} }} }}
          }}
        }}
      }});
    }}

    function sortTable(colIndex) {{
      const table = document.getElementById("watchlist-table");
      let rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
      switching = true;
      dir = "asc";
      while (switching) {{
        switching = false;
        rows = table.rows;
        for (i = 1; i < (rows.length - 1); i++) {{
          shouldSwitch = false;
          x = rows[i].getElementsByTagName("TD")[colIndex];
          y = rows[i + 1].getElementsByTagName("TD")[colIndex];
          
          let valX = isNaN(parseFloat(x.innerText)) ? x.innerText.toLowerCase() : parseFloat(x.innerText);
          let valY = isNaN(parseFloat(y.innerText)) ? y.innerText.toLowerCase() : parseFloat(y.innerText);
          
          if (dir == "asc") {{
            if (valX > valY) {{ shouldSwitch = true; break; }}
          }} else if (dir == "desc") {{
            if (valX < valY) {{ shouldSwitch = true; break; }}
          }}
        }}
        if (shouldSwitch) {{
          rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
          switching = true;
          switchcount++;
        }} else {{
          if (switchcount == 0 && dir == "asc") {{
            dir = "desc";
            switching = true;
          }}
        }}
      }}
    }}
  </script>
</body>
</html>
"""
    
    # Minify HTML and CSS lines to reduce bundle size well below 50KB
    cleaned_lines = [line.strip() for line in html_content.splitlines() if line.strip()]
    minified_html = "\n".join(cleaned_lines)
    # Remove unnecessary spaces around braces in CSS/JS
    minified_html = minified_html.replace(" {", "{").replace(" }", "}").replace(": ", ":").replace("; ", ";")
    
    html_path = DASHBOARD_DIR / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(minified_html)
        
    print(f" Standalone HTML dashboard generated at: {html_path}")
    print(f" Dashboard file size: {html_path.stat().st_size / 1024:.2f} KB (Target: <50KB)")
    return html_path

if __name__ == "__main__":
    generate_dashboard_and_tableau_exports()
