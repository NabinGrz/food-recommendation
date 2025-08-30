"""
AI Food Recommendation System - Main Prototype
Bachelor Level Implementation

This system demonstrates two integrated AI components:
1. Machine Learning Recommendation Engine (Collaborative + Content-based filtering)
2. AI Search Agent (NLP-powered recipe search)

Author: Nabin Gurung
Course: Bachelor 1st Semester AI Project
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import re
from typing import List, Dict, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class SimpleMLRecommendationEngine:
    """
    Machine Learning Component: 
    Uses collaborative filtering and content-based filtering
    """
    
    def __init__(self):
        self.recipes_df = None
        self.interactions_df = None
        self.user_item_matrix = None
        self.recipe_similarity_matrix = None
        self.user_profiles = {}
        self.recipe_popularity = {}
        
    def load_data(self):
        """Load the processed recipe and interaction data"""
        print("🔄 Loading data...")
        
        # Load sample data (smaller for demo)
        self.recipes_df = pd.read_csv("processed_data/sample_recipes.csv")
        self.interactions_df = pd.read_csv("processed_data/sample_interactions.csv")
        
        print(f"✅ Loaded {len(self.recipes_df)} recipes and {len(self.interactions_df)} interactions")
        
    def prepare_ml_models(self):
        """Prepare machine learning models"""
        print("🤖 Preparing ML models...")
        
        # 1. Create User-Item Matrix for Collaborative Filtering
        self.user_item_matrix = self.interactions_df.pivot_table(
            index='user_id', 
            columns='recipe_id', 
            values='rating',
            fill_value=0
        )
        
        # 2. Calculate Recipe Similarity for Content-Based Filtering
        self._calculate_recipe_similarity()
        
        # 3. Create User Profiles
        self._create_user_profiles()
        
        # 4. Calculate Recipe Popularity
        self._calculate_popularity()
        
        print("✅ ML models prepared")
    
    def _calculate_recipe_similarity(self):
        """Calculate similarity between recipes based on features"""
        # Use numerical features for similarity
        feature_cols = ['calories', 'total_fat', 'protein', 'n_ingredients', 'n_steps']
        available_cols = [col for col in feature_cols if col in self.recipes_df.columns]
        
        if available_cols:
            # Fill missing values and normalize
            feature_matrix = self.recipes_df[available_cols].fillna(0)
            
            # Add tag features (simplified)
            tag_cols = [col for col in self.recipes_df.columns if col.startswith('tag_')]
            if tag_cols:
                tag_matrix = self.recipes_df[tag_cols].fillna(0)
                feature_matrix = pd.concat([feature_matrix, tag_matrix], axis=1)
            
            # Calculate cosine similarity
            self.recipe_similarity_matrix = cosine_similarity(feature_matrix.values)
        else:
            print("⚠️ No recipe features found for content-based filtering")
    
    def _create_user_profiles(self):
        """Create user profiles based on rating behavior"""
        for user_id in self.interactions_df['user_id'].unique():
            user_ratings = self.interactions_df[self.interactions_df['user_id'] == user_id]
            
            profile = {
                'total_ratings': len(user_ratings),
                'avg_rating': user_ratings['rating'].mean(),
                'liked_recipes': user_ratings[user_ratings['rating'] >= 4]['recipe_id'].tolist(),
                'disliked_recipes': user_ratings[user_ratings['rating'] <= 2]['recipe_id'].tolist()
            }
            
            # Classify user type
            if len(user_ratings) < 5:
                profile['type'] = 'new_user'
            elif profile['avg_rating'] > 4.0:
                profile['type'] = 'generous'
            elif profile['avg_rating'] < 3.0:
                profile['type'] = 'critical'
            else:
                profile['type'] = 'average'
                
            self.user_profiles[user_id] = profile
    
    def _calculate_popularity(self):
        """Calculate recipe popularity scores"""
        recipe_stats = self.interactions_df.groupby('recipe_id').agg({
            'rating': ['count', 'mean']
        })
        recipe_stats.columns = ['count', 'avg_rating']
        
        # Popularity score = weighted combination of count and rating
        max_count = recipe_stats['count'].max()
        for recipe_id in recipe_stats.index:
            stats = recipe_stats.loc[recipe_id]
            popularity = (0.7 * (stats['avg_rating'] / 5.0) + 
                         0.3 * (stats['count'] / max_count))
            self.recipe_popularity[recipe_id] = popularity
    
    def get_collaborative_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """Get recommendations using collaborative filtering"""
        if user_id not in self.user_item_matrix.index:
            return self.get_popular_recommendations(n_recommendations)
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Find similar users
        user_similarities = cosine_similarity([user_ratings.values], self.user_item_matrix.values)[0]
        similar_users = np.argsort(user_similarities)[::-1][1:11]  # Top 10 similar users
        
        # Get recommendations from similar users
        recommendations = {}
        for similar_user_idx in similar_users:
            similar_user_id = self.user_item_matrix.index[similar_user_idx]
            similar_user_ratings = self.user_item_matrix.iloc[similar_user_idx]
            
            # Find recipes this similar user liked but current user hasn't tried
            for recipe_id, rating in similar_user_ratings.items():
                if rating >= 4 and user_ratings[recipe_id] == 0:
                    if recipe_id not in recommendations:
                        recommendations[recipe_id] = []
                    recommendations[recipe_id].append(rating * user_similarities[similar_user_idx])
        
        # Calculate average scores
        final_recommendations = []
        for recipe_id, scores in recommendations.items():
            avg_score = np.mean(scores)
            final_recommendations.append((recipe_id, avg_score))
        
        # Sort and return top N
        final_recommendations.sort(key=lambda x: x[1], reverse=True)
        return final_recommendations[:n_recommendations]
    
    def get_content_based_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """Get recommendations using content-based filtering"""
        if user_id not in self.user_profiles:
            return self.get_popular_recommendations(n_recommendations)
        
        user_profile = self.user_profiles[user_id]
        liked_recipes = user_profile['liked_recipes']
        
        if not liked_recipes or self.recipe_similarity_matrix is None:
            return self.get_popular_recommendations(n_recommendations)
        
        # Create user preference profile from liked recipes
        recipe_scores = {}
        
        for liked_recipe in liked_recipes[:5]:  # Use top 5 liked recipes
            try:
                recipe_idx = self.recipes_df[self.recipes_df['id'] == liked_recipe].index[0]
                similarities = self.recipe_similarity_matrix[recipe_idx]
                
                # Find similar recipes
                for i, similarity in enumerate(similarities):
                    recipe_id = self.recipes_df.iloc[i]['id']
                    
                    # Skip if user already rated this recipe
                    user_ratings = self.interactions_df[
                        (self.interactions_df['user_id'] == user_id) & 
                        (self.interactions_df['recipe_id'] == recipe_id)
                    ]
                    
                    if len(user_ratings) == 0 and similarity > 0.1:
                        if recipe_id not in recipe_scores:
                            recipe_scores[recipe_id] = []
                        recipe_scores[recipe_id].append(similarity)
            
            except (IndexError, KeyError):
                continue
        
        # Calculate average similarities
        final_recommendations = []
        for recipe_id, similarities in recipe_scores.items():
            avg_similarity = np.mean(similarities)
            final_recommendations.append((recipe_id, avg_similarity))
        
        # Sort and return top N
        final_recommendations.sort(key=lambda x: x[1], reverse=True)
        return final_recommendations[:n_recommendations]
    
    def get_popular_recommendations(self, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """Get popular recipes (fallback for new users)"""
        popular_recipes = sorted(self.recipe_popularity.items(), 
                               key=lambda x: x[1], reverse=True)
        return popular_recipes[:n_recommendations]
    
    def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Get hybrid recommendations combining multiple methods"""
        user_profile = self.user_profiles.get(user_id, {'type': 'new_user'})
        
        # Adaptive weighting based on user type
        if user_profile['type'] == 'new_user':
            weights = {'popularity': 0.8, 'collaborative': 0.1, 'content': 0.1}
        elif user_profile['type'] == 'generous':
            weights = {'popularity': 0.2, 'collaborative': 0.5, 'content': 0.3}
        else:
            weights = {'popularity': 0.3, 'collaborative': 0.4, 'content': 0.3}
        
        # Get recommendations from each method
        collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)
        content_recs = self.get_content_based_recommendations(user_id, n_recommendations * 2)
        popular_recs = self.get_popular_recommendations(n_recommendations * 2)
        
        # Combine with weights
        final_scores = {}
        
        # Add collaborative recommendations
        for recipe_id, score in collab_recs:
            final_scores[recipe_id] = final_scores.get(recipe_id, 0) + score * weights['collaborative']
        
        # Add content-based recommendations
        for recipe_id, score in content_recs:
            final_scores[recipe_id] = final_scores.get(recipe_id, 0) + score * weights['content']
        
        # Add popular recommendations
        for recipe_id, score in popular_recs:
            final_scores[recipe_id] = final_scores.get(recipe_id, 0) + score * weights['popularity']
        
        # Sort and get top recommendations
        sorted_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create detailed recommendation list
        recommendations = []
        for recipe_id, score in sorted_recommendations[:n_recommendations]:
            recipe_info = self.recipes_df[self.recipes_df['id'] == recipe_id]
            if len(recipe_info) > 0:
                recipe = recipe_info.iloc[0]
                recommendations.append({
                    'recipe_id': recipe_id,
                    'name': recipe['name'],
                    'score': score,
                    'method': 'hybrid',
                    'user_type': user_profile['type'],
                    'calories': recipe.get('calories', 'N/A'),
                    'minutes': recipe.get('minutes', 'N/A')
                })
        
        return recommendations

class SimpleSearchAgent:
    """
    AI Search Agent Component:
    Intelligent recipe search using NLP techniques
    """
    
    def __init__(self):
        self.recipes_df = None
        self.tfidf_vectorizer = None
        self.recipe_vectors = None
        self.ingredient_index = {}
        
    def load_data(self, recipes_df):
        """Load recipe data for search"""
        self.recipes_df = recipes_df.copy()
        print("🔍 Preparing search agent...")
        
        # Prepare text data for search
        self._prepare_search_index()
        print("✅ Search agent ready")
    
    def _prepare_search_index(self):
        """Prepare search index using TF-IDF"""
        # Combine recipe name, description, and ingredients for search
        search_texts = []
        
        for _, recipe in self.recipes_df.iterrows():
            text_parts = []
            
            # Add recipe name
            if pd.notna(recipe['name']):
                text_parts.append(recipe['name'])
            
            # Add description
            if pd.notna(recipe['description']):
                text_parts.append(str(recipe['description'])[:200])  # Limit description length
            
            # Add ingredients
            if pd.notna(recipe['ingredients']):
                ingredients_text = str(recipe['ingredients']).replace('[', '').replace(']', '').replace("'", "")
                text_parts.append(ingredients_text)
            
            search_texts.append(' '.join(text_parts))
        
        # Create TF-IDF vectors
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        self.recipe_vectors = self.tfidf_vectorizer.fit_transform(search_texts)
        
        # Create ingredient index for faster ingredient-based search
        self._create_ingredient_index()
    
    def _create_ingredient_index(self):
        """Create an index of recipes by ingredients"""
        for idx, recipe in self.recipes_df.iterrows():
            if pd.notna(recipe['ingredients']):
                ingredients_text = str(recipe['ingredients']).lower()
                # Extract individual ingredients
                ingredients = re.findall(r"'([^']*)'", ingredients_text)
                
                for ingredient in ingredients:
                    ingredient = ingredient.strip()
                    if ingredient:
                        if ingredient not in self.ingredient_index:
                            self.ingredient_index[ingredient] = []
                        self.ingredient_index[ingredient].append(recipe['id'])
    
    def search_recipes(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Intelligent recipe search using multiple strategies
        """
        print(f"🔍 Searching for: '{query}'")
        
        # Strategy 1: TF-IDF similarity search
        query_vector = self.tfidf_vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vector, self.recipe_vectors)[0]
        
        # Strategy 2: Ingredient matching
        ingredient_matches = self._search_by_ingredients(query)
        
        # Strategy 3: Simple keyword matching
        keyword_matches = self._search_by_keywords(query)
        
        # Combine results with scoring
        recipe_scores = {}
        
        # Add TF-IDF scores
        for idx, similarity in enumerate(similarities):
            recipe_id = self.recipes_df.iloc[idx]['id']
            recipe_scores[recipe_id] = recipe_scores.get(recipe_id, 0) + similarity * 0.5
        
        # Add ingredient match scores
        for recipe_id, score in ingredient_matches.items():
            recipe_scores[recipe_id] = recipe_scores.get(recipe_id, 0) + score * 0.3
        
        # Add keyword match scores
        for recipe_id, score in keyword_matches.items():
            recipe_scores[recipe_id] = recipe_scores.get(recipe_id, 0) + score * 0.2
        
        # Sort and format results
        sorted_results = sorted(recipe_scores.items(), key=lambda x: x[1], reverse=True)
        
        search_results = []
        for recipe_id, score in sorted_results[:n_results]:
            recipe_info = self.recipes_df[self.recipes_df['id'] == recipe_id]
            if len(recipe_info) > 0 and score > 0:
                recipe = recipe_info.iloc[0]
                search_results.append({
                    'recipe_id': recipe_id,
                    'name': recipe['name'],
                    'score': score,
                    'description': str(recipe.get('description', ''))[:150] + '...',
                    'ingredients': str(recipe.get('ingredients', ''))[:100] + '...',
                    'calories': recipe.get('calories', 'N/A'),
                    'minutes': recipe.get('minutes', 'N/A'),
                    'search_query': query
                })
        
        return search_results
    
    def _search_by_ingredients(self, query: str) -> Dict[int, float]:
        """Search recipes by ingredient names"""
        query_words = query.lower().split()
        ingredient_matches = {}
        
        for word in query_words:
            for ingredient, recipe_ids in self.ingredient_index.items():
                if word in ingredient:
                    for recipe_id in recipe_ids:
                        score = len(word) / len(ingredient)  # Partial match scoring
                        ingredient_matches[recipe_id] = ingredient_matches.get(recipe_id, 0) + score
        
        return ingredient_matches
    
    def _search_by_keywords(self, query: str) -> Dict[int, float]:
        """Search recipes by keywords in name and description"""
        query_words = [word.lower() for word in query.split()]
        keyword_matches = {}
        
        for _, recipe in self.recipes_df.iterrows():
            score = 0
            recipe_text = (str(recipe['name']) + ' ' + str(recipe.get('description', ''))).lower()
            
            for word in query_words:
                if word in recipe_text:
                    score += 1
            
            if score > 0:
                keyword_matches[recipe['id']] = score / len(query_words)
        
        return keyword_matches
    
    def get_smart_suggestions(self, query: str) -> List[str]:
        """Get smart search suggestions"""
        suggestions = []
        
        # Popular ingredients
        popular_ingredients = ['chicken', 'beef', 'pasta', 'cheese', 'tomato', 'garlic', 'onion']
        for ingredient in popular_ingredients:
            if ingredient.startswith(query.lower()):
                suggestions.append(f"Recipes with {ingredient}")
        
        # Cooking methods
        cooking_methods = ['baked', 'grilled', 'fried', 'roasted', 'steamed']
        for method in cooking_methods:
            if method.startswith(query.lower()):
                suggestions.append(f"{method.title()} recipes")
        
        # Meal types
        meal_types = ['breakfast', 'lunch', 'dinner', 'dessert', 'snack']
        for meal in meal_types:
            if meal.startswith(query.lower()):
                suggestions.append(f"{meal.title()} recipes")
        
        return suggestions[:5]

class SimpleAIFoodSystem:
    """
    Main AI Food Recommendation System
    Integrating ML Recommendations and Search Agent
    """
    
    def __init__(self):
        self.ml_engine = SimpleMLRecommendationEngine()
        self.search_agent = SimpleSearchAgent()
        self.is_trained = False
    
    def initialize_system(self):
        """Initialize the complete AI system"""
        print("🚀 Initializing AI Food Recommendation System...")
        print("="*50)
        
        # Load data
        self.ml_engine.load_data()
        
        # Prepare ML models
        self.ml_engine.prepare_ml_models()
        
        # Prepare search agent
        self.search_agent.load_data(self.ml_engine.recipes_df)
        
        self.is_trained = True
        print("✅ AI System Ready!")
        print("="*50)
    
    def get_personalized_recommendations(self, user_id: int, n_recommendations: int = 5) -> Dict:
        """Get personalized recommendations using ML"""
        if not self.is_trained:
            raise ValueError("System not initialized. Call initialize_system() first.")
        
        print(f"🤖 Getting ML recommendations for user {user_id}...")
        
        # Get user profile info
        user_profile = self.ml_engine.user_profiles.get(user_id, {'type': 'new_user'})
        
        # Get hybrid recommendations
        recommendations = self.ml_engine.get_hybrid_recommendations(user_id, n_recommendations)
        
        return {
            'user_id': user_id,
            'user_type': user_profile['type'],
            'user_stats': {
                'total_ratings': user_profile.get('total_ratings', 0),
                'avg_rating': user_profile.get('avg_rating', 0)
            },
            'recommendations': recommendations,
            'method': 'Machine Learning (Hybrid Collaborative + Content-Based)',
            'explanation': f"Recommendations adapted for {user_profile['type']} user behavior"
        }
    
    def search_recipes_intelligent(self, query: str, n_results: int = 10) -> Dict:
        """Search recipes using AI search agent"""
        if not self.is_trained:
            raise ValueError("System not initialized. Call initialize_system() first.")
        
        print(f"🔍 Using AI Search Agent for: '{query}'")
        
        # Get search results
        search_results = self.search_agent.search_recipes(query, n_results)
        
        # Get smart suggestions
        suggestions = self.search_agent.get_smart_suggestions(query)
        
        return {
            'query': query,
            'results': search_results,
            'suggestions': suggestions,
            'method': 'AI Search Agent (NLP + TF-IDF + Multi-Strategy)',
            'total_found': len(search_results)
        }
    
    def demo_system(self):
        """Demonstrate the AI system capabilities"""
        print("🎯 AI Food Recommendation System Demo")
        print("="*50)
        
        # Demo 1: ML Recommendations
        print("\n1️⃣ MACHINE LEARNING RECOMMENDATIONS")
        print("-" * 40)
        
        # Test with different user types
        sample_users = list(self.ml_engine.user_profiles.keys())[:3]
        
        for user_id in sample_users:
            user_profile = self.ml_engine.user_profiles[user_id]
            print(f"\n👤 User {user_id} ({user_profile['type']} user)")
            print(f"   📊 {user_profile['total_ratings']} ratings, avg: {user_profile['avg_rating']:.1f}⭐")
            
            ml_results = self.get_personalized_recommendations(user_id, 3)
            
            print("   🤖 ML Recommendations:")
            for i, rec in enumerate(ml_results['recommendations'], 1):
                print(f"      {i}. {rec['name'][:40]}... (Score: {rec['score']:.3f})")
            
            print(f"   💡 {ml_results['explanation']}")
        
        # Demo 2: AI Search Agent
        print("\n\n2️⃣ AI SEARCH AGENT")
        print("-" * 40)
        
        search_queries = ["chicken pasta", "chocolate cake", "healthy vegetarian"]
        
        for query in search_queries:
            print(f"\n🔍 Search: '{query}'")
            search_results = self.search_recipes_intelligent(query, 3)
            
            print("   🎯 AI Search Results:")
            for i, result in enumerate(search_results['results'], 1):
                print(f"      {i}. {result['name'][:40]}... (Relevance: {result['score']:.3f})")
            
            if search_results['suggestions']:
                print(f"   💡 Smart Suggestions: {', '.join(search_results['suggestions'])}")
        
        # Demo 3: System Integration
        print("\n\n3️⃣ INTEGRATED AI SYSTEM")
        print("-" * 40)
        
        print("\n🔄 Combining ML + Search for Complete Experience:")
        
        # Show how both systems work together
        test_user = sample_users[0]
        test_query = "quick dinner"
        
        print(f"\n👤 User {test_user} searches for '{test_query}'")
        
        # Get ML recommendations
        ml_recs = self.get_personalized_recommendations(test_user, 2)
        print("   🤖 Personalized ML suggestions:")
        for rec in ml_recs['recommendations']:
            print(f"      • {rec['name'][:35]}...")
        
        # Get search results
        search_recs = self.search_recipes_intelligent(test_query, 2)
        print("   🔍 Search-based suggestions:")
        for result in search_recs['results']:
            print(f"      • {result['name'][:35]}...")
        
        print("\n✨ This demonstrates how ML personalization + AI search create a complete system!")
        
        return True

def main():
    """Main function to run the complete AI system demo"""
    print("🍳 Simple AI Food Recommendation System")
    print("Bachelor Level Prototype - Integrating ML + Search AI")
    print("="*60)
    
    try:
        # Create and initialize the AI system
        ai_system = SimpleAIFoodSystem()
        ai_system.initialize_system()
        
        # Run the complete demo
        success = ai_system.demo_system()
        
        if success:
            print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("\n📋 System Summary:")
            print("✅ Machine Learning Component:")
            print("   • Collaborative Filtering (user-based recommendations)")
            print("   • Content-Based Filtering (recipe similarity)")
            print("   • Hybrid Approach (adaptive weighting)")
            print("   • User Profiling (behavioral classification)")
            
            print("\n✅ AI Search Agent Component:")
            print("   • Natural Language Processing (TF-IDF)")
            print("   • Multi-Strategy Search (keywords + ingredients)")
            print("   • Smart Suggestions (query completion)")
            print("   • Relevance Scoring (combined metrics)")
            
            print("\n🎯 Integration Benefits:")
            print("   • Personalized recommendations using ML")
            print("   • Intelligent search using NLP")
            print("   • Complete user experience")
            print("   • Scalable and maintainable architecture")
            
            print(f"\n🚀 Ready for submission - meets assignment requirements!")
            
        return True
        
    except Exception as e:
        print(f"❌ Error running system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Change to the correct directory and run
    os.chdir("/Users/nabingurung/Documents/AI Coursework/cw")
    main()
