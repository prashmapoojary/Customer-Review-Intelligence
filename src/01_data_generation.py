"""
Step 1: Synthetic Amazon-Style Review Data Generation
Generates realistic product reviews across 35 products in 6 categories
with hidden quality tiers (including declining quality) and distinct topic clusters.
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Hardcoded BASE_DIR
BASE_DIR = Path(r"C:\Users\Prashma\Desktop\Resume Projects\Data Analytics\NLP-powered customer review analytics system")

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define 35 products across 6 categories with hidden quality tiers
PRODUCTS_CONFIG = [
    # Electronics (6 products)
    {"product_id": "ELEC-001", "product_name": "SoundPulse Pro Wireless ANC Headphones", "category": "Electronics", "tier": "excellent", "base_price": 149.99},
    {"product_id": "ELEC-002", "product_name": "HyperCharge 65W GaN Fast Wall Charger", "category": "Electronics", "tier": "good", "base_price": 29.99},
    {"product_id": "ELEC-003", "product_name": "AuraGlow RGB Mechanical Gaming Keyboard", "category": "Electronics", "tier": "declining", "base_price": 89.99},
    {"product_id": "ELEC-004", "product_name": "StreamVision 4K Ultra HD Webcam", "category": "Electronics", "tier": "mixed", "base_price": 59.99},
    {"product_id": "ELEC-005", "product_name": "UltraFit GPS Heart Rate Smartwatch", "category": "Electronics", "tier": "declining", "base_price": 119.99},
    {"product_id": "ELEC-006", "product_name": "PocketBass Portable Mini Bluetooth Speaker", "category": "Electronics", "tier": "poor", "base_price": 19.99},

    # Kitchen (6 products)
    {"product_id": "KITCH-001", "product_name": "ChefMaster 8-in-1 Digital Air Fryer XL", "category": "Kitchen", "tier": "excellent", "base_price": 99.99},
    {"product_id": "KITCH-002", "product_name": "BaristaTouch Automatic Espresso Machine", "category": "Kitchen", "tier": "good", "base_price": 199.99},
    {"product_id": "KITCH-003", "product_name": "TitaniumBlade 15-Piece German Knife Set", "category": "Kitchen", "tier": "declining", "base_price": 79.99},
    {"product_id": "KITCH-004", "product_name": "NutriBlend High-Speed Countertop Blender", "category": "Kitchen", "tier": "mixed", "base_price": 69.99},
    {"product_id": "KITCH-005", "product_name": "NonStick Diamond Ceramic Frying Pan 12-inch", "category": "Kitchen", "tier": "declining", "base_price": 39.99},
    {"product_id": "KITCH-006", "product_name": "ThermoPrecision Instant Read Meat Thermometer", "category": "Kitchen", "tier": "good", "base_price": 18.99},

    # Home & Office (6 products)
    {"product_id": "HOME-001", "product_name": "ErgoRest Ergonomic Mesh Executive Office Chair", "category": "Home & Office", "tier": "declining", "base_price": 189.99},
    {"product_id": "HOME-002", "product_name": "LuminaGlow Smart LED Dimmable Desk Lamp", "category": "Home & Office", "tier": "excellent", "base_price": 34.99},
    {"product_id": "HOME-003", "product_name": "PureBreathe True HEPA Air Purifier for Home", "category": "Home & Office", "tier": "good", "base_price": 89.99},
    {"product_id": "HOME-004", "product_name": "AromaMist Ultrasonic Essential Oil Diffuser", "category": "Home & Office", "tier": "mixed", "base_price": 24.99},
    {"product_id": "HOME-005", "product_name": "MemoryCloud Orthopedic Seat Cushion", "category": "Home & Office", "tier": "good", "base_price": 29.99},
    {"product_id": "HOME-006", "product_name": "CableHide Heavy Duty Cable Management Raceway", "category": "Home & Office", "tier": "poor", "base_price": 15.99},

    # Baby & Kids (6 products)
    {"product_id": "BABY-001", "product_name": "SafeSleep Video Baby Monitor with Night Vision", "category": "Baby & Kids", "tier": "declining", "base_price": 129.99},
    {"product_id": "BABY-002", "product_name": "SiliconeSuction Toddler Feeding Plate & Bib Set", "category": "Baby & Kids", "tier": "excellent", "base_price": 22.99},
    {"product_id": "BABY-003", "product_name": "SnuggleWarm Infant Muslin Swaddle Blankets 4-Pack", "category": "Baby & Kids", "tier": "excellent", "base_price": 26.99},
    {"product_id": "BABY-004", "product_name": "WhiteNoise Dream Machine Sound Soother", "category": "Baby & Kids", "tier": "good", "base_price": 29.99},
    {"product_id": "BABY-005", "product_name": "StrollEasy Lightweight Compact Travel Stroller", "category": "Baby & Kids", "tier": "mixed", "base_price": 149.99},
    {"product_id": "BABY-006", "product_name": "TeethRelief Organic Silicone Baby Teething Toys", "category": "Baby & Kids", "tier": "good", "base_price": 12.99},

    # Health & Personal Care (6 products)
    {"product_id": "HLTH-001", "product_name": "SonicClean Pro Electric Sonic Toothbrush", "category": "Health & Personal Care", "tier": "excellent", "base_price": 49.99},
    {"product_id": "HLTH-002", "product_name": "DeepTissue Portable Percussion Massage Gun", "category": "Health & Personal Care", "tier": "declining", "base_price": 69.99},
    {"product_id": "HLTH-003", "product_name": "AquaPulse Cordless Rechargeable Water Flosser", "category": "Health & Personal Care", "tier": "mixed", "base_price": 32.99},
    {"product_id": "HLTH-004", "product_name": "BioGlow Organic Hyaluronic Acid Face Serum", "category": "Health & Personal Care", "tier": "good", "base_price": 19.99},
    {"product_id": "HLTH-005", "product_name": "PostureCorrect Adjustable Spine Support Brace", "category": "Health & Personal Care", "tier": "poor", "base_price": 21.99},
    {"product_id": "HLTH-006", "product_name": "SleepEase 3D Contoured Blackout Eye Mask", "category": "Health & Personal Care", "tier": "excellent", "base_price": 14.99},

    # Pet Supplies (5 products)
    {"product_id": "PETS-001", "product_name": "FreshFlow Stainless Steel Pet Water Fountain", "category": "Pet Supplies", "tier": "declining", "base_price": 39.99},
    {"product_id": "PETS-002", "product_name": "OrthopedicFoam Plush Waterproof Dog Bed Large", "category": "Pet Supplies", "tier": "excellent", "base_price": 64.99},
    {"product_id": "PETS-003", "product_name": "AutoLaser Interactive Cat Toy Rotating Beam", "category": "Pet Supplies", "tier": "mixed", "base_price": 22.99},
    {"product_id": "PETS-004", "product_name": "GroomPro Self-Cleaning Pet Slicker Deshedding Brush", "category": "Pet Supplies", "tier": "good", "base_price": 16.99},
    {"product_id": "PETS-005", "product_name": "NoPull Reflective Heavy Duty Dog Harness", "category": "Pet Supplies", "tier": "good", "base_price": 27.99},
]

# Distinct Complaint Templates by Theme with strong sentiment keywords
COMPLAINT_TEMPLATES = {
    "defective_hardware": [
        ("Defective unit! Stopped working after three days", "Completely defective and useless! The unit stopped working after just three days. It won't power on at all. Terrible build failure, highly frustrated."),
        ("Dead on arrival - completely broken", "Extremely disappointed! It arrived broken and dead on arrival. Pressed the power button and nothing happened. Defective hardware, awful quality control."),
        ("Stopped working suddenly - total waste", "Worked for one week then completely died. Won't turn on, won't charge, zero power. Very angry and disappointed with this defective piece of junk."),
        ("Burned out immediately! Defective device", "Horrible experience! Plugged it in and it immediately burned out and shut down. Defective components, unsafe and totally broken."),
        ("Glitchy and defective hardware", "Terrible product! Constantly glitched and now completely stopped working. A frustrating defective product that failed within a month.")
    ],
    "durability_quality": [
        ("Fell apart completely - cheap flimsy junk", "Horrible quality! The plastic snapped and the entire item fell apart in my hands. Cheap flimsy materials, totally fragile and broke instantly."),
        ("Flimsy construction and broke easily", "Terrible build quality! The parts cracked under normal use. Cheaply made, fragile junk that fell apart within two weeks."),
        ("Disintegrated and broke - awful durability", "Extremely disappointed! The material started peeling and cracking right away. It completely broke and fell apart. Worst durability ever."),
        ("Shattered and snapped! Poor build", "Disaster! It snapped right in half under light pressure. Terrible craftsmanship and cheap materials. Totally broke down."),
        ("Poorly constructed and flimsy", "Awful durability! Very flimsy, wobbly, and cheap plastic that broke after a few uses. Regret buying this junk.")
    ],
    "shipping_packaging": [
        ("Arrived crushed and damaged in transit", "Terrible shipping! The box was completely crushed, smashed, and damaged upon arrival. Horrible packaging, zero protection."),
        ("Extremely late delivery and ruined box", "Awful delivery experience! Took weeks to arrive, way past the promised date. When it finally came, the package was badly damaged."),
        ("Package destroyed and contents scratched", "Horrible condition upon arrival! The cardboard was torn open and the product was severely scratched and dinged. Terrible shipping handling."),
        ("Delayed shipment and battered package", "Frustrating and terrible! Shipping was delayed for two weeks with no updates. Box arrived completely battered and crushed."),
        ("Broken box and missing transit care", "Disgusted by the poor packaging! Box was ripped open and contents were banged up. Unacceptable and careless delivery.")
    ],
    "sizing_fit": [
        ("Completely wrong size - misleading dimensions", "Terrible sizing! The dimensions listed in the product description are totally wrong and misleading. Way too small, completely unusable."),
        ("Awful fit - huge and incorrect sizing", "Huge disappointment! The measurements are completely inaccurate. It is gigantic and does not fit at all. Misleading size chart."),
        ("Way too small - deceptive description", "Horrible fit! Sizing is totally off. Ordered according to specifications and it is ridiculously tiny. Very frustrating and deceptive."),
        ("Inaccurate dimensions and poor fit", "Extremely annoyed! Doesn't fit where it was supposed to. Sizing is completely inaccurate and false."),
        ("Terrible fit - completely unusable", "Disappointed with the sizing! Far too large and clumsy. The product dimensions are totally incorrect.")
    ],
    "value_overpriced": [
        ("Overpriced rip-off - not worth the money", "Total rip-off! Ridiculously overpriced for such cheap and tacky junk. Absolute waste of hard-earned money, do not buy!"),
        ("Extremely expensive for pathetic quality", "Horrible value! Paid premium dollar expecting quality, received garbage. Total waste of cash and completely overpriced."),
        ("Not worth a penny - total scam", "Terrible value! You can find better quality at a dollar store. Highly overpriced and an absolute waste of money."),
        ("Cheap junk at a premium price", "Disgusted by how overpriced this is! Extremely cheap materials with an inflated price tag. Total rip-off."),
        ("Regret this purchase - huge waste of money", "Awful price-to-quality ratio. Overpriced garbage that feels cheap and worthless. Deeply regret spending money on this.")
    ],
    "customer_service": [
        ("Worst customer service - refused refund", "Horrible customer support! Reached out for assistance and the support agent was rude and unresponsive. They refused my refund!"),
        ("Unresponsive support - ignored my emails", "Terrible seller and customer support! Sent multiple emails about the issue and was completely ignored. Terrible service and no help at all."),
        ("Rude customer service and zero help", "Extremely frustrating! Contacted customer service for a replacement and they were dismissive and unhelpful. Horrible support."),
        ("Warranty not honored - scam support", "Awful customer experience! They refused to honor the warranty and gave me the runaround. Completely useless and rude support team."),
        ("Disgraceful support - no resolution", "Horrible experience trying to get a refund. Support is unresponsive, incompetent, and unhelpful. Total nightmare.")
    ],
    "instructions_manual": [
        ("Incomprehensible instructions - impossible setup", "Terrible manual! The instructions are incomprehensible, missing steps, and full of confusing errors. Impossible to assemble, totally frustrating."),
        ("Confusing manual and missing setup details", "Awful setup experience! The included guide makes no sense at all. Poorly translated and utterly confusing instructions."),
        ("Missing assembly steps - horrible guide", "Extremely difficult and frustrating! The manual skips crucial steps and diagrams are illegible. Terrible instructions."),
        ("Vague and useless user guide", "Horrible instructions! Took hours trying to figure out what the manual meant. Inaccurate diagrams and missing guidance."),
        ("Terrible instructions - head scratching nightmare", "Utterly confusing manual! Setup instructions are completely scrambled and unhelpful. Bad user experience.")
    ],
    "noise_smell": [
        ("Obnoxiously loud and horrible buzzing noise", "Terrible annoying noise! Emits a loud, high-pitched screeching and buzzing sound that gives me a headache. Horrible and unbearable."),
        ("Overwhelming noxious chemical smell", "Awful toxic odor! Opened the box and was hit with a disgusting, nauseating chemical smell that won't go away. Horrible!"),
        ("Loud clanking noise - unbearable sound", "Extremely loud and disruptive! Makes an awful clattering and grinding noise during operation. Terrible irritating racket."),
        ("Foul plastic odor and loud vibration", "Horrible burning plastic smell and excessive loud rattling vibration. Completely unbearable and annoying."),
        ("Deafening noise and awful stench", "Terrible! The motor creates a deafening whine, and it smells like burning rubber. Unbearable junk.")
    ]
}

# Distinct Praise Templates by Theme with strong sentiment keywords
PRAISE_TEMPLATES = {
    "durability_quality": [
        ("Outstanding build quality - rock solid!", "Extremely impressed! Exceptional build quality and rock solid materials. Heavy duty, durable, and built to last for years. Love it!"),
        ("Superior quality craftsmanship - fantastic", "Top-notch quality! Premium sturdy construction and excellent durability. Feels robust and high-end. Highly recommend!"),
        ("Incredible durability and premium finish", "Amazing build! It has survived daily heavy use without a scratch. Extremely sturdy, premium quality, and beautifully constructed."),
        ("Solid, sturdy, and built to last", "Wonderful craftsmanship! Very solid feel, premium textures, and robust components. Exceeded my expectations!"),
        ("Excellent premium materials and finish", "Superb durability! High grade materials and immaculate finish. Outstanding quality that shines through.")
    ],
    "value_affordable": [
        ("Incredible value for money - absolute steal!", "Fantastic price for this outstanding quality! Easily rivals brands costing three times as much. Best purchase ever, worth every penny!"),
        ("Great price and high performance", "Amazing value! Very reasonably priced and delivers top-tier performance. A tremendous bargain, highly satisfied!"),
        ("Worth every single penny - great buy", "Exceptional value! High quality at an affordable price point. You definitely get more than what you pay for!"),
        ("Budget friendly with premium features", "Unbeatable value! Premium quality without the hefty price tag. Truly impressed and very happy with this purchase."),
        ("Superb bargain - exceeded expectations", "Outstanding value for money! Affordable, reliable, and beats higher-priced competitors hands down.")
    ],
    "shipping_packaging": [
        ("Lightning fast shipping and perfect packaging", "Arrived super fast in immaculate condition! Beautifully packaged with thick protective cushioning. Flawless delivery!"),
        ("Delivered early in secure protective box", "Incredible delivery speed! Arrived two days ahead of schedule. Packaging was sturdy, clean, and pristine. Very happy!"),
        ("Fast delivery and pristine packaging", "Outstanding shipping experience! Prompt delivery and everything was packaged neatly and safely. Great care taken!"),
        ("Prompt arrival and immaculate condition", "Super fast shipping! Box was in pristine condition, sealed tight and protected. Excellent seller service!"),
        ("Speedy dispatch and flawless packaging", "Arrived in record time! Very well protected inside the box. Five stars for lightning fast delivery!")
    ],
    "ease_of_use": [
        ("Incredibly easy to use - effortless setup", "Brilliant product! Setup took less than two minutes right out of the box. Super intuitive, effortless to use, and user-friendly!"),
        ("Simple, intuitive, and hassle-free", "Love how easy this is to operate! Straightforward controls, zero hassle, and works seamlessly. Highly recommend to everyone!"),
        ("Plug and play perfection - very simple", "Fantastic ease of use! Clear, simple, and straightforward. Was up and running immediately with no hassle whatsoever."),
        ("Extremely user friendly and convenient", "Delightfully easy to use! Very smart intuitive design. Makes daily tasks effortless and enjoyable."),
        ("Effortless operation and smooth setup", "Wonderfully designed! Simple setup, clear controls, and very pleasant to use every single day.")
    ],
    "performance_power": [
        ("Powerful performance - works flawlessly!", "Absolute powerhouse! Delivers blazing fast, powerful performance. Works flawlessly and handles everything with ease. Love it!"),
        ("Top tier speed and exceptional power", "Outstanding performance! Exceeded all my high expectations. Runs smooth, powerful, and whisper quiet. Five stars!"),
        ("Incredible results and flawless operation", "Phenomenal performance! Super effective, fast, and reliable. Performs like an absolute dream!"),
        ("High performance and highly effective", "Impressive power and efficiency! Works brilliantly and delivers top-notch results consistently."),
        ("Fast, powerful, and completely reliable", "Superb performance! Operates smoothly, quickly, and flawlessly. Extremely satisfied with the output.")
    ],
    "customer_service": [
        ("World class customer service - outstanding support", "Incredible customer support! Had a minor question and the seller responded within an hour with friendly helpful guidance. Amazing!"),
        ("Helpful and responsive support team", "Best customer service experience on Amazon! Very courteous, fast, and resolved my inquiry instantly. Truly appreciated!"),
        ("Courteous support and prompt resolution", "Exceptional customer care! Polite, professional, and quick to assist. They stand 100% behind their product!"),
        ("Superb seller communication and care", "Fantastic support! Very responsive, friendly, and went above and beyond to make sure I was happy. Five stars!"),
        ("Friendly, prompt, and dedicated service", "Wonderful seller! Quick replies, very accommodating and helpful. Outstanding customer service experience.")
    ],
    "design_aesthetic": [
        ("Gorgeous sleek modern design - looks amazing", "Stunning aesthetics! Sleek, elegant, and looks gorgeous on my counter. High-end modern look that gets compliments all the time!"),
        ("Beautiful modern styling and compact", "Love the design! Very stylish, clean lines, and compact footprint. Looks beautiful and feels very premium."),
        ("Elegant aesthetics and sleek profile", "Visually stunning! Beautiful modern finish that blends into any room. Gorgeous design and premium feel."),
        ("Chic, attractive, and minimalist design", "Looks incredible! Very sophisticated and minimalist aesthetics. Premium look and feel throughout."),
        ("Stunning look and modern finish", "Delighted by the sleek design! Looks stylish and high-tech. Great attention to visual details.")
    ]
}

REVIEWER_FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Riley", "Casey", "Jamie",
    "Michael", "Sarah", "David", "Emily", "James", "Emma", "Daniel", "Olivia", "Matthew", "Sophia",
    "Andrew", "Isabella", "Joshua", "Mia", "Brian", "Charlotte", "Kevin", "Amelia", "Jason", "Harper",
    "Brandon", "Evelyn", "Justin", "Abigail", "Eric", "Emily", "Tyler", "Elizabeth", "Ryan", "Sofia",
    "Rachel", "Brandon", "Jessica", "Marcus", "Lauren", "Nathan", "Chloe", "Ethan", "Hannah", "Lucas"
]

REVIEWER_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts"
]

def get_random_reviewer():
    return f"{random.choice(REVIEWER_FIRST_NAMES)} {random.choice(REVIEWER_LAST_NAMES)}"

def get_rating_distribution(tier: str, progress_ratio: float):
    """
    Returns probability distribution for star ratings [1, 2, 3, 4, 5]
    progress_ratio goes from 0.0 (oldest, June 2024) to 1.0 (newest, Aug 2025)
    """
    if tier == "excellent":
        return [0.03, 0.04, 0.05, 0.28, 0.60]
    elif tier == "good":
        return [0.06, 0.07, 0.12, 0.35, 0.40]
    elif tier == "mixed":
        return [0.22, 0.18, 0.20, 0.22, 0.18]
    elif tier == "poor":
        return [0.55, 0.25, 0.10, 0.06, 0.04]
    elif tier == "declining":
        # Starts with high ratings (progress 0) and shifts heavily to 1 & 2 stars (progress 1)
        # At progress 0: 75% 4-5 stars. At progress 1: 75% 1-2 stars
        p1 = 0.04 + 0.56 * progress_ratio
        p2 = 0.06 + 0.22 * progress_ratio
        p3 = 0.10 - 0.03 * progress_ratio
        p4 = 0.35 - 0.30 * progress_ratio
        p5 = 0.45 - 0.45 * progress_ratio
        probs = [max(0.01, p) for p in [p1, p2, p3, p4, p5]]
        total = sum(probs)
        return [p / total for p in probs]
    return [0.1, 0.1, 0.2, 0.3, 0.3]

def generate_review_content(rating: int, category: str):
    """Generates strong emotional title and text matched with distinct topic themes"""
    if rating in [1, 2]:
        theme = random.choice(list(COMPLAINT_TEMPLATES.keys()))
        title, text = random.choice(COMPLAINT_TEMPLATES[theme])
        # Sometimes add extra sentence for natural variance
        if random.random() < 0.35:
            text += " I am thoroughly unsatisfied and will never purchase from this brand again."
        elif random.random() < 0.25:
            text += " Total disappointment and huge regret."
        return title, text, theme, "negative"
    elif rating in [4, 5]:
        theme = random.choice(list(PRAISE_TEMPLATES.keys()))
        title, text = random.choice(PRAISE_TEMPLATES[theme])
        if random.random() < 0.35:
            text += " Absolutely wonderful purchase and I highly recommend it to everyone!"
        elif random.random() < 0.25:
            text += " Extremely pleased with this product and fast delivery!"
        return title, text, theme, "positive"
    else: # rating == 3 (neutral/mixed)
        if random.random() < 0.5:
            title = "Standard product - works as expected"
            text = "The product is standard and functional. It has average performance and works as expected for everyday use. Fair build and regular utility."
        else:
            title = "Acceptable and functional item"
            text = "Acceptable performance for regular needs. Fair build and standard functionality. An ordinary item that meets basic daily requirements."
        return title, text, "neutral_mixed", "neutral"

def generate_dataset(target_reviews: int = 6500):
    start_date = datetime(2024, 6, 1)
    end_date = datetime(2025, 8, 31)
    total_days = (end_date - start_date).days

    reviews = []
    
    # Calculate reviews per product with some realistic variance
    base_per_product = target_reviews // len(PRODUCTS_CONFIG)
    
    review_counter = 1
    
    for prod in PRODUCTS_CONFIG:
        prod_reviews_count = int(base_per_product * random.uniform(0.85, 1.25))
        
        for _ in range(prod_reviews_count):
            # Weighted toward recent months: use power distribution for date sampling
            # Beta or quadratic sampling gives realistic growth toward recent dates
            u = np.random.power(1.45) # Skews toward 1.0 (recent)
            day_offset = int(u * total_days)
            review_date = start_date + timedelta(days=day_offset)
            progress_ratio = day_offset / total_days
            
            # Select star rating based on product tier & date progress
            rating_probs = get_rating_distribution(prod["tier"], progress_ratio)
            rating = int(np.random.choice([1, 2, 3, 4, 5], p=rating_probs))
            
            title, text, theme, sentiment_bucket = generate_review_content(rating, prod["category"])
            
            verified = random.random() < 0.88
            
            # Helpful votes distribution (mostly 0, some high)
            if random.random() < 0.70:
                helpful = 0
            elif random.random() < 0.90:
                helpful = random.randint(1, 4)
            else:
                helpful = random.randint(5, 45)
            
            review_id = f"REV-{review_counter:06d}"
            review_counter += 1
            
            reviews.append({
                "review_id": review_id,
                "product_id": prod["product_id"],
                "product_name": prod["product_name"],
                "category": prod["category"],
                "rating": rating,
                "review_title": title,
                "review_text": text,
                "review_date": review_date.strftime("%Y-%m-%d"),
                "reviewer_name": get_random_reviewer(),
                "verified_purchase": verified,
                "helpful_votes": helpful,
                "_internal_theme": theme, # Internal check
                "_internal_tier": prod["tier"] # Internal check
            })
            
    df = pd.DataFrame(reviews)
    
    # Sort chronologically
    df["review_date"] = pd.to_datetime(df["review_date"])
    df = df.sort_values("review_date").reset_index(drop=True)
    df["review_date"] = df["review_date"].dt.strftime("%Y-%m-%d")
    
    # Create product catalog
    catalog_df = pd.DataFrame(PRODUCTS_CONFIG)[["product_id", "product_name", "category", "base_price"]]
    debug_tier_df = pd.DataFrame(PRODUCTS_CONFIG)[["product_id", "product_name", "category", "tier", "base_price"]]
    
    # Public reviews dataset (without internal debug columns)
    public_reviews_df = df.drop(columns=["_internal_theme", "_internal_tier"])
    
    # Save files
    public_reviews_path = RAW_DATA_DIR / "amazon_reviews.csv"
    catalog_path = RAW_DATA_DIR / "product_catalog.csv"
    debug_tier_path = RAW_DATA_DIR / "product_quality_tiers_debug.csv"
    
    public_reviews_df.to_csv(public_reviews_path, index=False)
    catalog_df.to_csv(catalog_path, index=False)
    debug_tier_df.to_csv(debug_tier_path, index=False)
    
    print(f" Generated {len(public_reviews_df):,} reviews across {len(catalog_df)} products.")
    print(f" Saved public reviews to: {public_reviews_path}")
    print(f" Saved product catalog to: {catalog_path}")
    print(f" Saved debug tiers to: {debug_tier_path}")
    
    # Print validation summaries
    print("\n--- Validation: Category Distribution ---")
    print(public_reviews_df["category"].value_counts())
    
    print("\n--- Validation: Star Rating Distribution ---")
    print(public_reviews_df["rating"].value_counts(normalize=True).sort_index().apply(lambda x: f"{x:.1%}"))
    
    print("\n--- Validation: Date Range & Monthly Volume ---")
    df_temp = public_reviews_df.copy()
    df_temp["month"] = pd.to_datetime(df_temp["review_date"]).dt.to_period("M").dt.to_timestamp()
    monthly_counts = df_temp.groupby("month")["review_id"].count()
    for m, c in monthly_counts.items():
        print(f"  {m.strftime('%Y-%m')}: {c:,} reviews")
        
    print("\n--- Validation: Declining Tier Trend Check ---")
    declining_prods = [p["product_id"] for p in PRODUCTS_CONFIG if p["tier"] == "declining"]
    dec_df = df[df["product_id"].isin(declining_prods)].copy()
    dec_df["month"] = pd.to_datetime(dec_df["review_date"]).dt.to_period("M").dt.to_timestamp()
    dec_monthly_avg = dec_df.groupby("month")["rating"].mean()
    print("Average Star Rating for 'Declining' Products Over Time:")
    for m, r in dec_monthly_avg.items():
        print(f"  {m.strftime('%Y-%m')}: {r:.2f} stars")
        
    return public_reviews_df, catalog_df

if __name__ == "__main__":
    generate_dataset()
