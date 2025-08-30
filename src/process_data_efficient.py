#!/usr/bin/env python3
"""
Food Recommendation System - Optimized Data Processing Script
Processes raw data efficiently with sampling for large datasets
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def process_data_efficiently():
    """Process data with memory optimization"""
    print("🍳 Food Recommendation System - Efficient Data Processing")
    print("=" * 60)
    
    data_dir = "data"
    processed_dir = "processed_data"
    
    # Create processed data directory
    os.makedirs(processed_dir, exist_ok=True)
    
    print("📊 Loading raw data...")
    
    # Load recipes data
    recipes_df = pd.read_csv(os.path.join(data_dir, "RAW_recipes.csv"))
    print(f"✅ Loaded {len(recipes_df):,} recipes")
    
    # Load interactions data in chunks and sample
    print("📊 Processing interactions data (sampling for efficiency)...")
    
    # Read interactions in chunks and sample
    chunk_size = 100000
    sampled_interactions = []
    
    for chunk in pd.read_csv(os.path.join(data_dir, "RAW_interactions.csv"), chunksize=chunk_size):
        # Sample 20% of each chunk to reduce memory usage
        sampled_chunk = chunk.sample(frac=0.2, random_state=42)
        sampled_interactions.append(sampled_chunk)
    
    interactions_df = pd.concat(sampled_interactions, ignore_index=True)
    print(f"✅ Sampled {len(interactions_df):,} interactions from original dataset")
    
    # Clean recipes data
    print("🧹 Cleaning recipes data...")
    
    # Remove rows with missing critical information
    recipes_df = recipes_df.dropna(subset=['name', 'id', 'ingredients', 'steps'])
    
    # Parse and clean data
    def parse_list_string(list_str):
        """Parse string representation of list to actual list"""
        try:
            if pd.isna(list_str):
                return []
            list_str = str(list_str).strip('[]')
            items = [item.strip().strip("'\"") for item in list_str.split("',")]
            items = [item for item in items if item and item != '']
            return items
        except:
            return []
    
    def parse_nutrition(nutrition_str):
        """Parse nutrition string to list of floats"""
        try:
            if pd.isna(nutrition_str):
                return [0.0] * 7
            nutrition_str = str(nutrition_str).strip('[]')
            values = [float(x.strip()) for x in nutrition_str.split(',')]
            while len(values) < 7:
                values.append(0.0)
            return values[:7]
        except:
            return [0.0] * 7
    
    # Apply parsing functions
    recipes_df['tags'] = recipes_df['tags'].apply(parse_list_string)
    recipes_df['ingredients'] = recipes_df['ingredients'].apply(parse_list_string)
    recipes_df['steps'] = recipes_df['steps'].apply(parse_list_string)
    recipes_df['nutrition'] = recipes_df['nutrition'].apply(parse_nutrition)
    
    # Extract nutrition features
    nutrition_features = pd.DataFrame(recipes_df['nutrition'].tolist(), 
                                    columns=['calories', 'total_fat', 'sugar', 
                                           'sodium', 'protein', 'saturated_fat', 'carbohydrates'])
    recipes_df = pd.concat([recipes_df, nutrition_features], axis=1)
    
    # Convert minutes to numeric and handle outliers
    recipes_df['minutes'] = pd.to_numeric(recipes_df['minutes'], errors='coerce')
    recipes_df['minutes'] = recipes_df['minutes'].clip(upper=recipes_df['minutes'].quantile(0.99))
    
    # Create difficulty score
    recipes_df['difficulty_score'] = (recipes_df['n_steps'] * 0.3 + recipes_df['minutes'] * 0.01).clip(0, 10)
    
    print(f"✅ Cleaned recipes data: {recipes_df.shape}")
    
    # Clean interactions data
    print("🧹 Cleaning interactions data...")
    
    # Remove rows with missing ratings
    interactions_df = interactions_df.dropna(subset=['rating'])
    interactions_df = interactions_df[(interactions_df['rating'] >= 1) & (interactions_df['rating'] <= 5)]
    
    # Convert date to datetime
    interactions_df['date'] = pd.to_datetime(interactions_df['date'], errors='coerce')
    interactions_df = interactions_df.dropna(subset=['date'])
    
    # Fill missing reviews
    interactions_df['review'] = interactions_df['review'].fillna('')
    
    print(f"✅ Cleaned interactions data: {interactions_df.shape}")
    
    # Filter to keep only recipes and users with sufficient interactions
    print("🔍 Filtering for active users and popular recipes...")
    
    # Keep recipes that appear in interactions
    valid_recipe_ids = set(interactions_df['recipe_id'].unique())
    recipes_df = recipes_df[recipes_df['id'].isin(valid_recipe_ids)]
    
    # Keep users with at least 5 interactions
    user_counts = interactions_df['user_id'].value_counts()
    active_users = user_counts[user_counts >= 5].index
    interactions_df = interactions_df[interactions_df['user_id'].isin(active_users)]
    
    # Keep recipes with at least 5 interactions
    recipe_counts = interactions_df['recipe_id'].value_counts()
    popular_recipes = recipe_counts[recipe_counts >= 5].index
    interactions_df = interactions_df[interactions_df['recipe_id'].isin(popular_recipes)]
    recipes_df = recipes_df[recipes_df['id'].isin(popular_recipes)]
    
    print(f"✅ Filtered to {len(recipes_df):,} recipes and {len(interactions_df):,} interactions")
    
    # Create a smaller user-item matrix by mapping to sequential IDs
    print("📊 Creating user-item matrix...")
    
    # Create mappings for sequential IDs
    unique_users = interactions_df['user_id'].unique()
    unique_recipes = interactions_df['recipe_id'].unique()
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    recipe_to_idx = {recipe_id: idx for idx, recipe_id in enumerate(unique_recipes)}
    
    # Map to sequential IDs
    interactions_df['user_idx'] = interactions_df['user_id'].map(user_to_idx)
    interactions_df['recipe_idx'] = interactions_df['recipe_id'].map(recipe_to_idx)
    
    # Create user-item matrix with sequential indices
    user_item_matrix = interactions_df.pivot_table(
        index='user_idx', 
        columns='recipe_idx', 
        values='rating', 
        fill_value=0
    )
    
    print(f"✅ Created user-item matrix: {user_item_matrix.shape}")
    
    # Create recipe features for content-based filtering
    print("🔧 Extracting recipe features...")
    
    # Get most common tags for feature engineering
    all_tags = []
    for tag_list in recipes_df['tags']:
        if isinstance(tag_list, list):
            all_tags.extend(tag_list)
    
    from collections import Counter
    common_tags = [tag for tag, count in Counter(all_tags).most_common(30)]
    
    # Create binary features for common tags
    for tag in common_tags:
        recipes_df[f'tag_{tag}'] = recipes_df['tags'].apply(
            lambda x: 1 if isinstance(x, list) and tag in x else 0
        )
    
    # Normalize numerical features
    numerical_features = ['minutes', 'n_steps', 'n_ingredients', 'calories', 
                         'total_fat', 'sugar', 'sodium', 'protein', 'difficulty_score']
    
    for feature in numerical_features:
        if feature in recipes_df.columns:
            median_val = recipes_df[feature].median()
            recipes_df[feature] = recipes_df[feature].fillna(median_val)
            std_val = recipes_df[feature].std()
            if std_val > 0:
                recipes_df[feature] = (recipes_df[feature] - recipes_df[feature].mean()) / std_val
    
    recipe_features = recipes_df.copy()
    
    print(f"✅ Created feature matrix: {recipe_features.shape}")
    
    # Save processed data
    print("💾 Saving processed data...")
    
    # Save as pickle files for faster loading
    recipes_df.to_pickle(os.path.join(processed_dir, "recipes_processed.pkl"))
    interactions_df.to_pickle(os.path.join(processed_dir, "interactions_processed.pkl"))
    user_item_matrix.to_pickle(os.path.join(processed_dir, "user_item_matrix.pkl"))
    recipe_features.to_pickle(os.path.join(processed_dir, "recipe_features.pkl"))
    
    # Save mappings
    import pickle
    with open(os.path.join(processed_dir, "user_to_idx.pkl"), 'wb') as f:
        pickle.dump(user_to_idx, f)
    with open(os.path.join(processed_dir, "recipe_to_idx.pkl"), 'wb') as f:
        pickle.dump(recipe_to_idx, f)
    
    # Also save smaller CSV files for inspection
    recipes_df.head(1000).to_csv(os.path.join(processed_dir, "sample_recipes.csv"), index=False)
    interactions_df.head(1000).to_csv(os.path.join(processed_dir, "sample_interactions.csv"), index=False)
    
    print("✅ All data saved successfully!")
    
    # Display final statistics
    print("\n📊 Final Dataset Statistics:")
    print(f"Recipes: {len(recipes_df):,}")
    print(f"Interactions: {len(interactions_df):,}")
    print(f"Users: {len(unique_users):,}")
    print(f"User-Item Matrix: {user_item_matrix.shape}")
    print(f"Feature Matrix: {recipe_features.shape}")
    print(f"Average Rating: {interactions_df['rating'].mean():.2f}")
    print(f"Rating Distribution: {interactions_df['rating'].value_counts().to_dict()}")
    print(f"Average Cooking Time: {recipes_df['minutes'].mean():.1f} minutes")
    print(f"Matrix Sparsity: {(1 - user_item_matrix.astype(bool).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1])) * 100:.2f}%")
    
    return recipes_df, interactions_df, user_item_matrix, recipe_features

def main():
    try:
        recipes, interactions, user_matrix, features = process_data_efficiently()
        print("\n🎉 Data processing completed successfully!")
        print("🚀 You can now run the main application with:")
        print("   python src/ai_recommendation_system.py")
        print("   streamlit run apps/recipe_search_app.py")
        return 0
    except Exception as e:
        print(f"\n❌ Error during data processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
