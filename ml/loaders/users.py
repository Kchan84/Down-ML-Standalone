import pandas as pd
from pathlib import Path

# ---------- Paths ----------
RAW_USER = Path("C:/Users/Kevin/Software_Projects/Datasets/YelpData/yelp_academic_dataset_user.json")

def load_user_features(path: Path = RAW_USER, rows: int | None = None) -> pd.DataFrame:
    """
    Load user.json and return only essential user_id for joins.
    Removed redundant features since review_quality already captures community validation.
    """
    df = next(pd.read_json(path, lines=True, chunksize=rows)) if rows else \
         pd.read_json(path, lines=True)

    # --- Only keep user_id for joins ---
    df = df[["user_id"]].copy()
    
    return df