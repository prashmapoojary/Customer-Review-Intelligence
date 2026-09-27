# NLP-Powered Customer Review Analytics & Early-Warning Intelligence System

[![Netlify Status](https://api.netlify.com/api/v1/badges/9065c3a/deploy-status)](https://guileless-dieffenbachia-5847b8.netlify.app/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Netlify%20Dashboard-3ECF9E?style=for-the-badge&logo=netlify)](https://guileless-dieffenbachia-5847b8.netlify.app/)

> **Live Dashboard Demo**: [https://guileless-dieffenbachia-5847b8.netlify.app/](https://guileless-dieffenbachia-5847b8.netlify.app/)

An end-to-end NLP data analytics system that processes customer reviews at scale, detects emerging product quality defects using sentiment analysis and dual-model NMF topic extraction, calculates composite early-warning risk scores, and visualizes actionable findings through an interactive Command Center HTML dashboard and Tableau-ready data exports.

---

##  Project Architecture & Workflow

```
       +-------------------------------------------------------------+
       |   Step 1: Amazon Review Data Generation / Real Ingestion    |
       |     - 6,655 reviews, 35 products, 6 retail categories       |
       |     - Hidden quality tiers & 14-month rolling timestamps     |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |             Step 2: VADER Sentiment Analysis                |
       |     - Combined title + text scoring (compound score -1 to 1)|
       |     - 100% 5-star positive, 98.2% 1-star negative alignment |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |      Step 3: Dual-Model NMF Topic Extraction (TF-IDF)       |
       |     - Negative Model (10 topics): Complaint themes         |
       |     - Positive Model (8 topics): Praise themes              |
       |     - Context-specific keyword rule auto-labeling           |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |         Step 4: Composite Predictive Risk Scoring           |
       |     - Trend Score (55%): 90-day negativity acceleration     |
       |     - Severity Score (45%): Weighted defect severity        |
       |     - Volume Confidence: Dampens low-sample false alarms    |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |           Step 5: Executive Dashboard & Exports             |
       |     - Standalone Dark Command Center (`dashboard/index.html`)|
       |     - Tableau-ready Flat CSVs (`monthly_sentiment_trend`)    |
       +-------------------------------------------------------------+
```

---

##  Folder Structure

```
NLP-powered customer review analytics system/
│
├── dashboard/
│   └── index.html               # Self-contained dark command-center dashboard (<50KB, Chart.js)
│
├── data/
│   ├── raw/
│   │   ├── amazon_reviews.csv   # Public raw review dataset (6,655 rows)
│   │   ├── product_catalog.csv  # Product metadata and category mapping
│   │   └── product_quality_tiers_debug.csv # Internal quality tiers (debug only)
│   │
│   └── processed/
│       ├── reviews_with_sentiment.csv      # VADER sentiment scored reviews
│       ├── reviews_with_topics.csv         # NMF topic themes assigned
│       ├── topic_summary.csv               # Merged polarity & theme aggregates
│       ├── product_risk_scores.csv         # 0-100 risk scores & tier assignments
│       ├── monthly_sentiment_trend.csv     # Tableau-ready monthly trend export
│       ├── category_sentiment_breakdown.csv# Tableau-ready category breakdown export
│       └── figures/                        # High-resolution publication charts (.png)
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_topic_modeling.ipynb
│   ├── 04_risk_scoring.ipynb
│   └── 05_dashboard_and_tableau_exports.ipynb
│
├── src/
│   ├── 01_data_generation.py
│   ├── 02_sentiment_analysis.py
│   ├── 03_topic_modeling.py
│   ├── 04_risk_scoring.py
│   ├── 05_dashboard_generator.py
│   └── build_all_notebooks.py
│
└── README.md
```

---

## ⚡ How to Run Everything

### Prerequisites & Dependencies
Install standard data science libraries:
```bash
pip install pandas numpy matplotlib seaborn vaderSentiment scikit-learn nbformat
```

### Execution Option A: Standalone Python Pipeline
Run each step sequentially from the terminal:
```bash
# Step 1: Generate synthetic review dataset
python src/01_data_generation.py

# Step 2: Perform VADER sentiment scoring & validate crosstabs
python src/02_sentiment_analysis.py

# Step 3: Run dual-model NMF topic extraction
python src/03_topic_modeling.py

# Step 4: Compute composite risk scores & watchlist
python src/04_risk_scoring.py

# Step 5: Generate Tableau exports & HTML command center dashboard
python src/05_dashboard_generator.py
```

### Execution Option B: Interactive Jupyter Notebooks
Open JupyterLab / Notebook and run each notebook cell-by-cell in order:
1. `notebooks/01_data_generation.ipynb`
2. `notebooks/02_sentiment_analysis.ipynb`
3. `notebooks/03_topic_modeling.ipynb`
4. `notebooks/04_risk_scoring.ipynb`
5. `notebooks/05_dashboard_and_tableau_exports.ipynb`

---

##  Step-by-Step Technical Details

### Step 1: Realistic Amazon-Style Review Dataset
- **Volume & Scope**: ~6,655 customer reviews across 35 products spanning 6 retail categories (`Electronics`, `Kitchen`, `Home & Office`, `Baby & Kids`, `Health & Personal Care`, `Pet Supplies`).
- **Timeline**: 14-month date range (June 2024 - August 2025), with review volume naturally weighted toward recent months.
- **Hidden Quality Tiers**: Products are assigned underlying quality behaviors (`excellent`, `good`, `mixed`, `declining`, `poor`). "Declining" products deliberately exhibit deteriorating ratings over time to benchmark the early warning algorithm.

### Step 2: VADER Sentiment Analysis
- **Offline Lexicon Scoring**: Combines review title and body text into a single cohesive review string.
- **Classification Thresholds**:
  - `Positive`: Compound score $\ge +0.05$
  - `Negative`: Compound score $\le -0.05$
  - `Neutral`: $-0.05 < \text{Compound} < +0.05$
- **Validation Crosstab**:
  - 5-Star Reviews: **100.0% Positive** (Target: $\ge 95\%$)
  - 1-Star Reviews: **98.2% Negative** (Target: $\ge 85\%$)

### Step 3: Dual-Model NMF Topic Extraction
- **Zero-Dependency Architecture**: Utilizes scikit-learn Non-Negative Matrix Factorization (NMF) on TF-IDF matrices ($k=10$ negative, $k=8$ positive), eliminating heavy HuggingFace/transformer downloads while preserving granular topic clusters.
- **Polarity Separation**: Separate models prevent keyword confusion (e.g., "price" in praise indicates affordability; "price" in complaints indicates rip-off).
- **Auto-Labeling Rules**: Discovers key complaint themes:
  - `Defective / Hardware Failure`
  - `Durability / Build Quality`
  - `Shipping / Packaging Damage`
  - `Sizing & Dimension Inaccuracy`
  - `Overpriced / Poor Value`
  - `Customer Service & Refund Issues`
  - `Confusing Manual / Setup Difficulty`
  - `Excessive Noise / Chemical Odor`

### Step 4: Early-Warning Risk Scoring
Computes a composite **0 to 100 Risk Score** for each product:
$$\text{Raw Score} = 0.55 \times \text{Trend Score} + 0.45 \times \text{Severity Score}$$
$$\text{Final Risk Score} = \text{Raw Score} \times \min\left(\frac{N_{\text{recent}}}{2 \times \text{MIN\_REVIEWS}}, 1.0\right)$$

- **Trend Component (55% weight)**: Measures the surge in negative review rate over the last 90 days relative to the prior 90-day baseline.
- **Severity Component (45% weight)**: Weights critical failure themes higher (e.g., *Defective Hardware* $= 3.0$) than mild issues (e.g., *Confusing Manual* $= 1.2$).
- **Volume Confidence**: Products with fewer than 8 recent reviews are designated `Insufficient Data` to prevent false alarms on sparse sample sizes.
- **Risk Tiers**:
  - `Critical` ($\ge 60$): Immediate intervention required.
  - `High` ($\ge 35$): Accelerating negative sentiment.
  - `Watch` ($\ge 15$): Moderate negative drift.
  - `Healthy` ($< 15$): Stable positive reception.

---

## 📊 Step 5: Dashboard & Tableau Exports

### 1. Interactive Command Center Dashboard (`dashboard/index.html`)
- **Theme**: Dark command-center UI (`#0D1117` background) with functional status signal colors (`#3ECF9E` Teal, `#E5484D` Red, `#F0913A` Amber, `#F0C531` Yellow).
- **Features**:
  - **Executive Overview**: Real-time KPI summary cards, 14-month rolling sentiment trajectory line chart, and category sentiment distribution.
  - **Complaint & Praise Themes**: Deep dive into top complaint drivers vs praise factors.
  - **At-Risk Watchlist Table**: Interactive table with sorting and risk tier badges.
  - **Product Drill-Down**: Product selector showing historical monthly ratings, negative review counts, and complaint distribution.
- **Performance**: Standalone, fully offline, and compact (**~51 KB total size**).

### 2. Tableau-Ready Exports (`data/processed/`)
- `monthly_sentiment_trend.csv`: Columns `review_month`, `sentiment_label`, `review_count`.
- `category_sentiment_breakdown.csv`: Columns `category`, `sentiment_label`, `review_count`.

---

## 🔄 Swapping with Real Kaggle Amazon Reviews Data

To run this pipeline on a real Kaggle or Amazon Customer Reviews dataset:
1. Place your CSV file into `data/raw/amazon_reviews.csv`.
2. Ensure your CSV provides the standard Amazon review columns:
   - `review_id` *(String)*
   - `product_id` *(String)*
   - `product_name` *(String)*
   - `category` *(String)*
   - `rating` *(Integer, 1 to 5)*
   - `review_title` *(String)*
   - `review_text` *(String)*
   - `review_date` *(Date string: YYYY-MM-DD)*
   - `reviewer_name` *(String)*
   - `verified_purchase` *(Boolean)*
   - `helpful_votes` *(Integer)*
3. Run `python src/02_sentiment_analysis.py`, `03_topic_modeling.py`, `04_risk_scoring.py`, and `05_dashboard_generator.py`. The entire pipeline will automatically analyze the real dataset and update the dashboard.
