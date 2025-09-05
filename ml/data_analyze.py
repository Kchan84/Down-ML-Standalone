import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import re

# Import loader functions from the loaders package, gives cleaned data from both academic data sets
from loaders.business import load_business_features
from loaders.reviews import load_review_features

# Edit query for better TFIDF recommendations.
# Uses regex
def clean_text_for_mood(text):
    """
    Enhanced text cleaning specifically for mood analysis.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove numbers (keep words like "2nd" but remove pure numbers)
    text = re.sub(r'\b\d+\b', '', text)
    
    # Remove punctuation but keep apostrophes for contractions
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def expand_mood_query(user_mood):
    """
    Expand mood queries with related terms for better review-level TF-IDF matching.
    """
    # Define mood expansions:
    # IMPORTANT FOR DEVELOPMENT: Expanding mood_expansions will improve the quality of the recommendations!
    # For now this basic expansions are good enough for the initial development and experimentation.
    mood_expansions = {
        'romantic': ['romantic', 'romance', 'intimate', 'cozy', 'elegant', 'sophisticated', 'charming'],
        'casual': ['casual', 'relaxed', 'laid-back', 'informal', 'comfortable', 'easy-going'],
        'party': ['party', 'lively', 'energetic', 'vibrant', 'fun', 'exciting', 'bustling'],
        'quiet': ['quiet', 'peaceful', 'calm', 'tranquil', 'serene', 'relaxing', 'soothing'],
        'cozy': ['cozy', 'warm', 'intimate', 'comfortable', 'welcoming', 'homey'],
        'lively': ['lively', 'energetic', 'vibrant', 'bustling', 'dynamic', 'exciting'],
        'dinner': ['dinner', 'dining', 'meal', 'restaurant', 'cuisine', 'food'],
        'lunch': ['lunch', 'meal', 'food', 'restaurant', 'dining'],
        'coffee': ['coffee', 'cafe', 'espresso', 'latte', 'cappuccino', 'tea'],
        'bar': ['bar', 'pub', 'tavern', 'cocktail', 'drinks', 'nightlife']
    }
    
    expanded_terms = []
    user_mood_lower = user_mood.lower()
    
    for mood_type, related_terms in mood_expansions.items():
        if mood_type in user_mood_lower:
            expanded_terms.extend(related_terms)
    
    # If no specific mood type found, use the original query
    if not expanded_terms:
        expanded_terms = [user_mood]
    
    # Combine original and expanded terms
    all_terms = [user_mood] + expanded_terms
    return ' '.join(all_terms)

def infer_establishment_types(user_mood):
    """
    Infer likely establishment types from the user's mood query.
    Returns a set like {"restaurant", "bar"} based on simple keyword rules.
    """
    query = str(user_mood).lower()
    type_keywords = {
        'restaurant': ['restaurant', 'dinner', 'lunch', 'brunch', 'eatery', 'dining'],
        'bar': ['bar', 'pub', 'tavern', 'cocktail', 'speakeasy', 'brewery'],
        'cafe': ['cafe', 'coffee', 'espresso', 'latte'],
        'club': ['club', 'nightclub'],
        'bakery': ['bakery', 'pastry'],
    }
    inferred = set()
    for est_type, keywords in type_keywords.items():
        if any(k in query for k in keywords):
            inferred.add(est_type)
    return inferred

def create_merged_dataset(limit_rows=5000):
    """
    Merge business and reviews datasets for mood-based recommendations.
    Creates a comprehensive dataset where each row represents a review with
    business context. Focuses on business features and review content for mood analysis.
    
    Args:
        limit_rows (int): Maximum number of rows to load from each dataset
    """
    # Load limited datasets for faster development
    business_df = load_business_features(rows=limit_rows)
    reviews_df = load_review_features(rows=limit_rows)
    
    # Merge: Business + Reviews (reviews belong to businesses)
    final_df = business_df.merge(
        reviews_df, 
        on="business_id", 
        how="left",
        suffixes=('_biz', '_rev')
    )
    
    # Filter out rows without review text (TF-IDF needs actual text)
    final_df = final_df.dropna(subset=['text'])
    
    # Optional: Filter out very short reviews (less than 20 characters)
    final_df = final_df[final_df['text'].str.len() >= 20]
    
    # Apply enhanced text cleaning
    final_df['text_clean'] = final_df['text'].apply(clean_text_for_mood)
    
    # Filter out reviews that became too short after cleaning
    final_df = final_df[final_df['text_clean'].str.len() >= 15]
    
    return final_df

# Natural Language Processing - TF-IDF for review text
def create_tfidf_features(df):
    """
    Convert review text to TF-IDF numerical features for mood analysis.
    """
    # Initialize TF-IDF vectorizer with better parameters for mood matching
    tfidf = TfidfVectorizer(
        max_features=2000,           # Increased from 1000 for better vocabulary coverage
        stop_words='english',        # Remove common words like 'the', 'and', 'is'
        ngram_range=(1, 3),          # Single words + 2-3 word phrases for better mood capture
        min_df=2,                    # Reduced from 3 - word must appear in at least 2 reviews
        max_df=0.8,                  # Word must not appear in more than 80% of reviews
        lowercase=True,              # Ensure lowercase processing
        strip_accents='unicode'      # Remove accents for better matching
    )
    
    # Convert review text to TF-IDF features
    review_features = tfidf.fit_transform(df['text_clean'])
    
    # Return the TF-IDF features and the vectorizer
    return review_features, tfidf

def prepare_features_for_similarity(df, review_features):
    """
    Prepare all features for similarity-based mood matching.
    Combines business features with TF-IDF features for recommendation engine.
    """
    # Business features (numerical)
    business_features = df[['latitude', 'longitude', 'biz_avg_stars', 'biz_review_count', 'price_bucket']].values
    
    # Review quality features
    quality_features = df[['stars', 'review_quality', 'year', 'month', 'day_of_week']].values
    
    # Combine all features
    X = hstack([business_features, quality_features, review_features])
    
    return X

def recommend_businesses(user_mood, X, tfidf_vectorizer, business_data, top_n=10, min_review_quality=2.0):
    """
    Recommend businesses based on user mood using similarity matching.
    
    Args:
        user_mood (str): User's mood query (e.g., "romantic atmosphere") - in app "What are you down for?"
        X: Feature matrix with business + review + TF-IDF features
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        business_data: Original business data for display
        top_n (int): Number of top recommendations to return
        min_review_quality (float): Minimum review quality score required (default: 2.0)
    
    Returns:
        list: Top business recommendations ranked by quality score (stars + review quality + mood similarity)
    """
    # Preprocess user mood query for better matching
    user_mood_clean = user_mood.lower().strip()
    
    # Convert user mood to TF-IDF features
    user_mood_features = tfidf_vectorizer.transform([user_mood_clean])
    
    # Get the number of TF-IDF features
    tfidf_feature_count = user_mood_features.shape[1]
    
    # Create a zero vector for business and quality features
    business_quality_features = np.zeros((1, X.shape[1] - tfidf_feature_count))
    
    # Combine business/quality features with TF-IDF features
    mood_vector = hstack([business_quality_features, user_mood_features])
    
    # Calculate cosine similarity between mood and all businesses
    similarities = cosine_similarity(mood_vector, X).flatten()
    
    # Get top N most similar businesses (get more candidates for filtering)
    top_indices = similarities.argsort()[-top_n*10:][::-1]  # Increased from 5x to 10x for more variety
    
    # Create recommendations list with quality filtering
    recommendations = []
    inferred_types = infer_establishment_types(user_mood_clean)
    for idx in top_indices:
        business = business_data.iloc[idx]
        
        # Quality filtering: check review quality only
        if business.get('review_quality', 0) >= min_review_quality:
            
            # Calculate quality score (combines stars, review quality, and mood similarity)
            stars = business.get('biz_avg_stars', 0)
            review_quality = business.get('review_quality', 0)
            similarity = similarities[idx]
            
            # Normalize similarity to 0-1 scale (cosine similarity is already -1 to 1, but typically 0 to 1)
            normalized_similarity = max(0, similarity)
            
            # Quality score formula: 40% stars + 30% review quality + 30% mood similarity
            quality_score = (stars * 0.4) + (review_quality * 0.3) + (normalized_similarity * 0.3)

            # Post-boost if primary_category matches inferred establishment type
            primary_category = str(business.get('primary_category', '')).lower()
            if inferred_types and any(t in primary_category for t in inferred_types):
                quality_score *= 1.05  # small boost for matching type
            
            recommendation = {
                'business_id': business['business_id'],
                'name': business.get('name', 'N/A'),
                'category': business.get('primary_category', 'N/A'),
                'stars': business.get('biz_avg_stars', 'N/A'),
                'price': business.get('price_bucket', 'N/A'),
                'location': f"({business.get('latitude', 'N/A')}, {business.get('longitude', 'N/A')})",
                'similarity_score': similarities[idx],
                'review_quality': business.get('review_quality', 'N/A'),
                'quality_score': quality_score,
                'review_text_sample': business.get('text', 'N/A')[:100] + "..." if len(str(business.get('text', ''))) > 100 else business.get('text', 'N/A')
            }
            recommendations.append(recommendation)
    
    # Sort by quality score (highest first) and take top N
    recommendations.sort(key=lambda x: x['quality_score'], reverse=True)
    top_recommendations = recommendations[:top_n]
    
    # Add ranking after sorting
    for i, rec in enumerate(top_recommendations):
        rec['rank'] = i + 1
    
    return top_recommendations

def recommend_businesses_tfidf_only(user_mood, tfidf_vectorizer, business_data, top_n=10, min_review_quality=2.0):
    """
    Recommend businesses based on user mood using ONLY TF-IDF similarity (no business features).
    This approach focuses purely on text content matching for better mood-based recommendations.
    
    Args:
        user_mood (str): User's mood query (e.g., "romantic atmosphere")
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        business_data: Original business data for display
        top_n (int): Number of top recommendations to return
        min_review_quality (float): Minimum review quality score required
    
    Returns:
        list: Top business recommendations ranked by TF-IDF similarity
    """
    # Preprocess user mood query for better matching
    user_mood_clean = clean_text_for_mood(user_mood)
    
    # Expand mood query with related terms
    expanded_mood = expand_mood_query(user_mood_clean)
    
    # Convert expanded mood to TF-IDF features
    mood_features = tfidf_vectorizer.transform([expanded_mood])
    
    # Get all review texts from business data
    review_texts = business_data['text_clean'].fillna('')
    
    # Convert all reviews to TF-IDF features
    all_review_features = tfidf_vectorizer.transform(review_texts)
    
    # Calculate cosine similarity between mood and all reviews
    similarities = cosine_similarity(mood_features, all_review_features).flatten()
    
    # Get top N most similar reviews
    top_indices = similarities.argsort()[-top_n*5:][::-1]  # Get 5x more candidates for filtering
    
    # Create recommendations list with quality filtering
    recommendations = []
    for idx in top_indices:
        business = business_data.iloc[idx]
        
        # Quality filtering: check review quality only
        if business.get('review_quality', 0) >= min_review_quality:
            
            # Calculate quality score (combines stars, review quality, and mood similarity)
            stars = business.get('biz_avg_stars', 0)
            review_quality = business.get('review_quality', 0)
            similarity = similarities[idx]
            
            # Normalize similarity to 0-1 scale
            normalized_similarity = max(0, similarity)
            
            # Quality score formula: 30% stars + 20% review quality + 50% mood similarity
            quality_score = (stars * 0.3) + (review_quality * 0.2) + (normalized_similarity * 0.5)
            
            recommendation = {
                'business_id': business['business_id'],
                'name': business.get('name', 'N/A'),
                'category': business.get('primary_category', 'N/A'),
                'stars': business.get('biz_avg_stars', 'N/A'),
                'price': business.get('price_bucket', 'N/A'),
                'location': f"({business.get('latitude', 'N/A')}, {business.get('longitude', 'N/A')})",
                'similarity_score': similarities[idx],
                'review_quality': business.get('review_quality', 'N/A'),
                'quality_score': quality_score,
                'review_text_sample': business.get('text', 'N/A')[:100] + "..." if len(str(business.get('text', ''))) > 100 else business.get('text', 'N/A'),
                'expanded_query': expanded_mood
            }
            recommendations.append(recommendation)
    
    # Sort by quality score (highest first) and take top N
    recommendations.sort(key=lambda x: x['quality_score'], reverse=True)
    top_recommendations = recommendations[:top_n]
    
    # Add ranking after sorting
    for i, rec in enumerate(top_recommendations):
        rec['rank'] = i + 1
    
    return top_recommendations

def demonstrate_mood_matching(df, tfidf_vectorizer, review_features):
    """
    Demonstrate how mood-based recommendations will work.
    """
    # Example mood queries
    example_moods = [
        "romantic atmosphere",
        "lively party scene", 
        "quiet conversation",
        "cozy intimate setting"
    ]
    
    for mood in example_moods:
        # Convert mood to TF-IDF vector
        mood_vector = tfidf_vectorizer.transform([mood])
        
        # Find top matching words
        feature_names = tfidf_vectorizer.get_feature_names_out()
        mood_scores = mood_vector.toarray()[0]
        
        # Get top 5 matching words
        top_indices = mood_scores.argsort()[-5:][::-1]
        top_words = [(feature_names[i], mood_scores[i]) for i in top_indices if mood_scores[i] > 0]
        
        if top_words:
            pass
        else:
            pass

# ---------- quick peek driver ----------
def main():
    # Set pandas to display all columns and rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    # Create merged dataset with row limits (faster development)
    # Each row is a review with business context
    merged_data = create_merged_dataset(limit_rows=10000)  # Limit to 10000 reviews
    
    # Transform text to TF-IDF features for mood matching
    review_features, tfidf_vectorizer = create_tfidf_features(merged_data)
    
    #Created TF-IDF features with {review_features.shape[1]} vocabulary terms"
    
    # Prepare combined feature matrix (business + quality + text)
    X = prepare_features_for_similarity(merged_data, review_features)
    
    # Test the quality + mood-based recommendation system
    print("\n" + "="*60)
    print("QUALITY + MOOD-BASED RECOMMENDATIONS")
    print("="*60)
    
    # Test with more specific mood queries that should match better
    test_moods = ["romantic dinner", "casual lunch", "bar with music", "quiet coffee", "lively bar"]
    
    for mood in test_moods:
        print(f"\n Query: '{mood}'")
        # Recommend businesses blending mood similarity with business quality
        recommendations = recommend_businesses(mood, X, tfidf_vectorizer, merged_data, top_n=3, min_review_quality=2.0)
        
        if recommendations:
            for rec in recommendations:
                print(f"  {rec['rank']}. {rec['name']} - {rec['category']}")
                print(f"      {rec['stars']} stars |  Price: {rec['price']} |  {rec['location']}")
                print(f"      Similarity: {rec['similarity_score']:.3f} | 📈 Quality score: {rec['quality_score']:.3f}")
                print(f"      Review: {rec['review_text_sample']}")
                print()
        else:
            print(f"   No recommendations found for '{mood}'")

if __name__ == "__main__":
    main()
