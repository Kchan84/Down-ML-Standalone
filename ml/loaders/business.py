import pandas as pd
from pathlib import Path

# ---------- Paths ----------
RAW_BUS = Path("C:/Users/Kevin/Software_Projects/Datasets/YelpData/yelp_academic_dataset_business.json")

def load_business_features(path: Path = RAW_BUS, rows: int | None = None) -> pd.DataFrame:
    """
    Load business.json and return essential features for mood-based recommendations.
    Focuses on quality, atmosphere, and basic business info.
    """
    df = pd.read_json(path, lines=True) if rows is None else \
         next(pd.read_json(path, lines=True, chunksize=rows))

    # --- Only essential columns ---
    df = df[[
        "business_id",
        "name",
        "latitude", "longitude",
        "stars", "review_count",
        "categories", "attributes"
    ]].copy()

    # --- Simple renaming ---
    df = df.rename(columns={
        "stars": "biz_avg_stars",
        "review_count": "biz_review_count"
    })

    # --- Essential attributes only ---
    def safe_get(attrs, key):
        return attrs.get(key) if isinstance(attrs, dict) else None

    # Price bucket - critical for mood/atmosphere
    df["price_bucket"] = df["attributes"].apply(lambda x: safe_get(x, "RestaurantsPriceRange2"))
    df["price_bucket"] = pd.to_numeric(df["price_bucket"], errors="coerce").fillna(-1).astype("int64")

    # --- Categories - critical for mood matching ---
    def primary_category(cats):
        if isinstance(cats, str) and len(cats):
            return cats.split(",")[0].strip()
        return None
    df["primary_category"] = df["categories"].apply(primary_category).astype("category")

    # --- Keep raw hours for time-based filtering at inference ---
    # (don't pre-process, let the model learn time patterns)

    # --- Clean up ---
    df = df.drop(columns=["attributes", "categories"])

    # --- Simple quality filter ---
    df["is_quality"] = (
        (df["biz_avg_stars"] >= 3.0) &  # Reduced from 3.5 to 3.0 for more variety
        (df["biz_review_count"] >= 5)
    ).astype("int64")

    return df