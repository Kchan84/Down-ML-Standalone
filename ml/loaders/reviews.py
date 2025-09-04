import pandas as pd
from pathlib import Path

# ---------- Paths ----------
RAW_REV = Path("C:/Users/Kevin/Software_Projects/Datasets/YelpData/yelp_academic_dataset_review.json")

def load_review_features(path: Path = RAW_REV, rows: int | None = None) -> pd.DataFrame:
    """
    Load review.json and return essential features for mood-based recommendations.
    Focuses on review content, timing, and quality indicators.
    """
    df = next(pd.read_json(path, lines=True, chunksize=rows)) if rows else \
         pd.read_json(path, lines=True)

    # --- Only essential columns ---
    df = df[[
        "review_id", "user_id", "business_id", 
        "stars", "date", "text",
        "useful", "funny", "cool"
    ]].copy()

    # --- Type conversions ---
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce").fillna(0).astype("int64")
    df["useful"] = pd.to_numeric(df["useful"], errors="coerce").fillna(0).astype("int64")
    df["funny"] = pd.to_numeric(df["funny"], errors="coerce").fillna(0).astype("int64")
    df["cool"] = pd.to_numeric(df["cool"], errors="coerce").fillna(0).astype("int64")

    # --- Rename features for clarity ---
    df = df.rename(columns={
        "useful": "review_informative",    # How accurately it describes the place
        "funny": "review_atmospheric",     # How well it captures atmosphere/mood
        "cool": "review_insightful"        # How unique/valuable the insights are
    })

    # --- Date processing ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday, 6=Sunday

    # --- Review quality score ---
    df["review_quality"] = ( # Returns if review is trustworthy and useful
        (df["review_informative"] > 0).astype(int) + 
        (df["review_atmospheric"] > 0).astype(int) + 
        (df["review_insightful"] > 0).astype(int)
    )

    # --- Filter out very short reviews (likely not mood-indicative) ---
    df = df[df["text"].str.len() >= 20]  # At least 20 characters

    return df