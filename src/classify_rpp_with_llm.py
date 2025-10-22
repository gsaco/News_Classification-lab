"""
LLM Classification Script for RPP News Articles

This script uses an LLM (e.g., OpenAI's GPT) to classify RPP news articles
into AG News categories. The results are saved as ground truth for comparison
with the trained transformer models.

Requirements:
- OpenAI API key (or other LLM API)
- pip install openai

Usage:
    export OPENAI_API_KEY='your-api-key-here'
    python classify_rpp_with_llm.py
"""

import os
import json
import pandas as pd
from tqdm import tqdm
import time

# Try to import OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("OpenAI library not installed. Install with: pip install openai")
    OPENAI_AVAILABLE = False


CATEGORY_MAPPING = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Technology"
}

CLASSIFICATION_PROMPT = """Classify the following news article into ONE of these categories:
0 - World (international news, politics, global events)
1 - Sports (athletics, games, competitions, teams)
2 - Business (economics, finance, markets, companies)
3 - Science/Technology (scientific discoveries, technological innovations, research)

Article Title: {title}
Article Description: {description}

Important: Respond with ONLY the category number (0, 1, 2, or 3). No explanation needed.

Category:"""


def classify_article_with_llm(client, title, description, model="gpt-4o-mini"):
    """
    Classify a single article using OpenAI API.
    
    Args:
        client: OpenAI client instance
        title: Article title
        description: Article description
        model: Model to use (default: gpt-4o-mini for cost efficiency)
    
    Returns:
        int: Category number (0-3) or None if failed
    """
    prompt = CLASSIFICATION_PROMPT.format(
        title=title,
        description=description if pd.notna(description) else ""
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a news classification expert. Classify articles accurately into the given categories."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # Deterministic output
            max_tokens=5
        )
        
        # Extract and validate response
        category_str = response.choices[0].message.content.strip()
        
        # Try to extract number from response
        category = int(category_str.split()[0])  # Get first token
        
        if category in [0, 1, 2, 3]:
            return category
        else:
            print(f"Warning: Invalid category {category} returned, defaulting to 0")
            return 0
            
    except Exception as e:
        print(f"Error classifying article: {e}")
        return None


def main():
    """Main function to classify all RPP articles."""
    
    # Check if OpenAI is available
    if not OPENAI_AVAILABLE:
        print("Error: OpenAI library not available.")
        print("Install with: pip install openai")
        return
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Load RPP news data
    data_path = '../data/rpp_news_50.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    print("Loading RPP news articles...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} articles\n")
    
    # Classify each article
    classifications = []
    
    print("Classifying articles with LLM...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        category = classify_article_with_llm(
            client,
            row['title'],
            row['description']
        )
        
        if category is not None:
            classifications.append({
                'index': idx,
                'title': row['title'],
                'llm_category': category,
                'llm_category_name': CATEGORY_MAPPING[category]
            })
        else:
            # Default to World (0) if classification failed
            classifications.append({
                'index': idx,
                'title': row['title'],
                'llm_category': 0,
                'llm_category_name': CATEGORY_MAPPING[0]
            })
        
        # Rate limiting: sleep briefly between requests
        time.sleep(0.5)
    
    # Save results
    output_path = '../data/rpp_classified.json'
    
    results = {
        'model_used': 'gpt-4o-mini',
        'num_articles': len(classifications),
        'llm_classifications': [c['llm_category'] for c in classifications],
        'detailed_results': classifications,
        'category_mapping': CATEGORY_MAPPING
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Classifications saved to {output_path}")
    
    # Print summary
    print("\nClassification Summary:")
    category_counts = {}
    for c in classifications:
        cat = c['llm_category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for cat_num in sorted(category_counts.keys()):
        cat_name = CATEGORY_MAPPING[cat_num]
        count = category_counts[cat_num]
        print(f"  {cat_num} ({cat_name}): {count} articles ({count/len(classifications)*100:.1f}%)")
    
    print("\nYou can now run the bonus task section in the notebook!")


if __name__ == "__main__":
    main()
