import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path('../data/raw/complaints.csv')
OUTPUT_PATH = Path('../data/processed/filtered_complaints.csv')
REPORT_DIR = Path('../reports/figures/')

def load_data(file_path):
    """Load and validate complaint dataset"""
    df = pd.read_csv(file_path, low_memory=False)
    required_columns = ['Product', 'Consumer complaint narrative']
    assert all(col in df.columns for col in required_columns), \
        "Missing required columns in dataset"
    return df

def analyze_data(df):
    """Perform EDA and generate visualizations"""
    # Product distribution analysis
    product_dist = df['Product'].value_counts()
    plt.figure(figsize=(10, 6))
    product_dist.plot(kind='bar')
    plt.title('Complaints by Product Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'product_distribution.png')
    
    # Narrative length analysis
    df['narrative_length'] = df['Consumer complaint narrative'].str.split().str.len()
    plt.figure(figsize=(10, 6))
    plt.hist(df['narrative_length'].dropna(), bins=50, range=(0, 500))
    plt.title('Complaint Narrative Length Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.savefig(REPORT_DIR / 'narrative_length.png')
    
    # Missing data report
    missing_narratives = df['Consumer complaint narrative'].isna().sum()
    
    return {
        'product_distribution': product_dist,
        'median_length': df['narrative_length'].median(),
        'missing_narratives': missing_narratives
    }

def clean_text(text):
    """Clean complaint narratives for embedding"""
    if not isinstance(text, str):
        return ""
    
    # Remove boilerplate patterns
    patterns = [
        r"i am writing to file a complaint.*?\.\s*",
        r"please see below my complaint.*?\.\s*",
        r"ccpa.*?:"
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Standard cleaning
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def preprocess_data(df):
    """Filter and clean dataset"""
    target_products = [
        'Credit card', 'Personal loan', 'Payday loan', 
        'Savings account', 'Money transfer'
    ]
    
    # Filter to target products
    filtered = df[df['Product'].isin(target_products)].copy()
    
    # Remove missing narratives
    filtered = filtered.dropna(subset=['Consumer complaint narrative'])
    
    # Clean text
    filtered['clean_narrative'] = filtered['Consumer complaint narrative'].apply(clean_text)
    
    # Remove empty narratives after cleaning
    return filtered[filtered['clean_narrative'].str.len() > 50]

def main():
    # Setup directories
    Path('../data/processed').mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    
    # Load and process data
    df = load_data(DATA_PATH)
    eda_results = analyze_data(df)
    processed_df = preprocess_data(df)
    
    # Save processed data
    processed_df.to_csv(OUTPUT_PATH, index=False)
    
    # Generate EDA report
    report = f"""
    ## Exploratory Data Analysis Summary
    
    **Product Distribution**:
    {eda_results['product_distribution'].to_markdown()}
    
    **Narrative Length**:
    - Median word count: {eda_results['median_length']}
    - Max word count: {processed_df['narrative_length'].max()}
    - Min word count: {processed_df['narrative_length'].min()}
    
    **Data Quality**:
    - Original missing narratives: {eda_results['missing_narratives']}
    - Final complaint count: {len(processed_df)}
    
    ![Product Distribution](figures/product_distribution.png)
    ![Narrative Length](figures/narrative_length.png)
    """
    with open('../reports/interim_report.md', 'a') as f:
        f.write(report)

if __name__ == "__main__":
    main()