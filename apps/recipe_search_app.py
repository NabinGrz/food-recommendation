"""
Recipe Search Web Application
A clean, user-friendly interface for finding recipes based on ingredients

Features:
- Semantic ingredient search using TF-IDF vectorization
- Real-time recipe recommendations
- Clean, minimal interface design
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
import time
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(
    page_title="Recipe Search",
    page_icon="🍳",
    layout="centered"
)

# Hide Streamlit's default elements
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}
div[data-testid="stToolbar"] {visibility: hidden;}
div[data-testid="stDecoration"] {visibility: hidden;}
div[data-testid="stStatusWidget"] {visibility: hidden;}
#MainMenu {visibility: hidden;}
button[title="Deploy"] {display: none;}
[data-testid="stToolbar"] {display: none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_data
def load_recipe_data():
    """Load and cache recipe data"""
    try:
        recipes_df = pd.read_csv("processed_data/sample_recipes.csv")
        return recipes_df
    except FileNotFoundError:
        st.error("❌ Recipe data not found. Please ensure 'processed_data/sample_recipes.csv' exists.")
        return None

@st.cache_data
def prepare_search_engine(recipes_df):
    """Prepare the ingredient search engine"""
    if recipes_df is None:
        return None, None
    
    # Extract ingredients text for search
    ingredient_texts = []
    
    for _, recipe in recipes_df.iterrows():
        if pd.notna(recipe.get('ingredients')):
            # Clean the ingredients text
            ingredients_str = str(recipe['ingredients'])
            # Remove brackets and quotes
            ingredients_clean = re.sub(r"[\[\]']", "", ingredients_str)
            ingredient_texts.append(ingredients_clean.lower())
        else:
            ingredient_texts.append("")
    
    # Create TF-IDF vectorizer for ingredients
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ letters
    )
    
    try:
        ingredient_vectors = vectorizer.fit_transform(ingredient_texts)
        return vectorizer, ingredient_vectors
    except ValueError:
        st.error("❌ Error processing ingredient data")
        return None, None

def search_recipes_by_ingredients(user_ingredients, recipes_df, vectorizer, ingredient_vectors, top_n=10):
    """Search for recipes based on user ingredients"""
    if vectorizer is None or ingredient_vectors is None:
        return []
    
    # Transform user input
    user_query = user_ingredients.lower().strip()
    if not user_query:
        return []
    
    try:
        user_vector = vectorizer.transform([user_query])
        
        # Calculate similarities
        similarities = cosine_similarity(user_vector, ingredient_vectors)[0]
        
        # Debug: Print some similarity scores
        max_sim = np.max(similarities)
        non_zero_count = np.sum(similarities > 0)
        
        # Get top matches - lower the threshold to get more results
        top_indices = np.argsort(similarities)[::-1][:top_n * 2]  # Get more candidates
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Lower threshold for testing
                recipe = recipes_df.iloc[idx]
                results.append({
                    'name': recipe.get('name', 'Unknown Recipe'),
                    'similarity': similarities[idx],
                    'ingredients': str(recipe.get('ingredients', 'No ingredients listed'))[:200] + '...',
                    'description': str(recipe.get('description', 'No description'))[:150] + '...',
                    'calories': recipe.get('calories', 'N/A'),
                    'minutes': recipe.get('minutes', 'N/A'),
                    'recipe_id': recipe.get('id', idx)
                })
                
                if len(results) >= top_n:
                    break
        
        return results
    
    except Exception as e:
        st.error(f"❌ Search error: {e}")
        return []

def simple_keyword_search(user_ingredients, recipes_df, top_n=10):
    """Simple keyword-based search as a fallback"""
    user_words = [word.strip().lower() for word in user_ingredients.split(',')]
    
    results = []
    for idx, recipe in recipes_df.iterrows():
        ingredients_str = str(recipe.get('ingredients', '')).lower()
        
        # Count how many user words appear in the recipe ingredients
        matches = sum(1 for word in user_words if word in ingredients_str)
        
        if matches > 0:
            results.append({
                'name': recipe.get('name', 'Unknown Recipe'),
                'similarity': matches / len(user_words),  # Simple scoring
                'ingredients': str(recipe.get('ingredients', 'No ingredients listed'))[:200] + '...',
                'description': str(recipe.get('description', 'No description'))[:150] + '...',
                'calories': recipe.get('calories', 'N/A'),
                'minutes': recipe.get('minutes', 'N/A'),
                'recipe_id': recipe.get('id', idx),
                'matches': matches
            })
    
    # Sort by number of matches and similarity
    results.sort(key=lambda x: (x['matches'], x['similarity']), reverse=True)
    return results[:top_n]

def extract_keywords_from_ingredients(ingredients_text):
    """Extract common ingredient keywords from the ingredients text"""
    if not ingredients_text:
        return []
    
    # Common ingredients that users might search for
    common_ingredients = [
        'chicken', 'beef', 'pork', 'fish', 'salmon', 'shrimp',
        'pasta', 'rice', 'bread', 'cheese', 'milk', 'eggs',
        'tomato', 'onion', 'garlic', 'pepper', 'carrot', 'potato',
        'flour', 'sugar', 'butter', 'oil', 'salt', 'olive',
        'mushroom', 'spinach', 'broccoli', 'corn', 'beans',
        'chocolate', 'vanilla', 'cinnamon', 'herbs', 'basil'
    ]
    
    found_keywords = []
    ingredients_lower = ingredients_text.lower()
    
    for ingredient in common_ingredients:
        if ingredient in ingredients_lower:
            found_keywords.append(ingredient)
    
    return found_keywords[:10]  # Return top 10 matches

def main():
    """Main Streamlit application"""
    
    # Header - Simple and clean
    st.title("Recipe Search")
    
    # Load data
    recipes_df = load_recipe_data()
    
    if recipes_df is None:
        st.stop()
    
    vectorizer, ingredient_vectors = prepare_search_engine(recipes_df)
    
    if vectorizer is None:
        st.stop()
    
    # Simple search interface - single column layout
    st.markdown("### Enter your ingredients:")
    
    # Initialize session state for user input and last search
    if 'search_ingredients' not in st.session_state:
        st.session_state.search_ingredients = ""
    if 'last_search' not in st.session_state:
        st.session_state.last_search = ""
    
    # Search input - use key parameter for better state management
    user_ingredients = st.text_input(
        "Ingredients",
        placeholder="e.g., chicken, tomato, garlic, pasta",
        help="Enter ingredients separated by commas",
        label_visibility="collapsed",
        key="ingredient_input"
    )
    
    # Update session state when input changes
    if user_ingredients != st.session_state.search_ingredients:
        st.session_state.search_ingredients = user_ingredients
        st.session_state.last_search = user_ingredients
    
    # Auto-search when user types (no button needed)
    search_query = user_ingredients.strip() if user_ingredients else ""
    
    if search_query:
        st.markdown(f"**Searching for:** {search_query}")
        
        # Try TF-IDF search first
        results = search_recipes_by_ingredients(
            search_query, recipes_df, vectorizer, ingredient_vectors, 8
        )
        
        # If no results from TF-IDF, try simple keyword search
        if not results:
            results = simple_keyword_search(search_query, recipes_df, 8)
        
        # Show results right below search bar
        if results:
            st.markdown(f"### Found {len(results)} recipes:")
            
            # Display recipes in a simple card format
            for i, result in enumerate(results, 1):
                # Create a simple card-like container
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Recipe name and brief info
                        st.markdown(f"**{i}. {result['name']}**")
                        
                        # Show key ingredients found
                        keywords = extract_keywords_from_ingredients(result['ingredients'])
                        if keywords:
                            st.markdown(f"*Contains: {', '.join(keywords[:5])}*")
                        
                        # Show match info if available
                        if 'matches' in result:
                            st.markdown(f"*({result['matches']} ingredient matches)*")
                    
                    with col2:
                        # Simple metrics
                        if result['minutes'] != 'N/A':
                            st.markdown(f"⏱️ {result['minutes']} min")
                        if result['calories'] != 'N/A':
                            st.markdown(f"🔥 {result['calories']} cal")
                    
                    # Show a brief description if available
                    if result['description'] != 'No description...':
                        desc = str(result['description'])
                        if len(desc) > 100:
                            desc = desc[:100] + "..."
                        st.markdown(f"*{desc}*")
                    
                    st.markdown("---")
        
        else:
            st.warning("🔍 No recipes found. Try common ingredients like chicken, pasta, cheese, etc.")
            st.info("**Tip:** Try single ingredients like 'chicken' or 'pasta' first!")
    
    else:
        # Show quick suggestions when nothing is entered
        st.markdown("### Try these popular ingredients:")
        
        # Simple suggestion buttons in a cleaner layout
        col1, col2, col3 = st.columns(3)
        
        suggestions = [
            "chicken, rice",
            "pasta, tomato", 
            "eggs, flour",
            "beef, potato",
            "chocolate, butter",
            "fish, vegetables"
        ]
        
        for i, suggestion in enumerate(suggestions):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.button(suggestion, key=f"sug_{i}"):
                    st.session_state.search_ingredients = suggestion
                    st.rerun()

if __name__ == "__main__":
    main()
