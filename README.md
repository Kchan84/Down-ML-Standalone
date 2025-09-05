# Down-ML-Standalone

Showcase of a mood-based recommendation engine built on the Yelp Academic Dataset. It blends review text understanding (TFIDF) with business quality signals to recommend and filter places that match a users vibe (e.g., "romantic dinner", "quiet coffee").

## Highlights
- Mood understanding with TFIDF over review text (unigramstrigrams)
- Business context features: stars, review_count, price_bucket, location
- Quality score: combines stars, review quality, and mood similarity
- Query expansion (e.g., "romantic"  related terms) and light type inference (restaurant/bar/cafe)

## Repo structure
```
ml/
  data_analyze.py         # Main runnable script (prints recommendations)
  train_model.py          # Placeholder training entry point
  requirements.txt        # Python dependencies
  loaders/
    business.py           # Loads business.json (Yelp); feature selection/renames
    reviews.py            # Loads review.json (Yelp); review quality + date parts
    users.py              # (Minimal) user join placeholder
  features/
    availability.py       # Placeholder for hours/availability features
    text.py               # Placeholder for additional text features
```

## Data requirements
This expects the Yelp Academic Dataset JSON files on disk. Paths are currently set as absolute Windows paths inside:
- `ml/loaders/business.py`  `RAW_BUS`
- `ml/loaders/reviews.py`  `RAW_REV`
- `ml/loaders/users.py`  `RAW_USER`

Update those `RAW_*` paths to wherever you stored the Yelp JSON files.

## Setup
```powershell
# From repo root
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r ml/requirements.txt
```

## Run (quick demo)
```powershell
python ml/data_analyze.py
```
Example output:
```
 Created TF-IDF features with 2000 vocabulary terms
============================================================
QUALITY + MOOD-BASED RECOMMENDATIONS
============================================================

 Query: 'romantic dinner'
  1. Example Place - Italian
      4.5 stars |  Price: 2 |  (37.77, -122.42)
      Similarity: 0.412 |  Quality score: 2.983
      Review: great ambiance and cozy, perfect for date night ...
```

## How it works (brief)
1) Load and clean data, engineer review quality and date features
2) Clean and vectorize review text with TFIDF (13 grams, 2,000 max features)
3) Build a combined feature space (business + review quality + text)
4) Compute cosine similarity between the users mood and all businesses
5) Rank by a quality score: 40% stars + 30% review quality + 30% mood similarity
6) Light boosting if the primary category matches the inferred establishment type

## Customize the demo
- Change the sample queries: edit `test_moods` in `ml/data_analyze.py`
- Increase data size: raise `limit_rows` in `create_merged_dataset`
- Try TFIDF only mode: call `recommend_businesses_tfidf_only` (already implemented)

## Tech stack
- Python 3.13, NumPy, Pandas, SciKitLearn, SciPy

## Contact
Questions welcome via GitHub Issues or reach out on LinkedIn.
