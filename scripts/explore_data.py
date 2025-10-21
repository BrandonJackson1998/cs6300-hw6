# scripts/explore_data.py
"""
Exploratory Data Analysis for Fitness Exercise Datasets

This script provides comprehensive analysis of the exercise datasets,
helping understand the structure and quality before building the RAG system.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))


def normalize_name(name: str) -> str:
    """Normalize exercise names for consistent matching"""
    if pd.isna(name) or not isinstance(name, str):
        return ""
    
    # Convert to lowercase, strip whitespace, replace hyphens, consolidate spaces
    normalized = name.lower().strip().replace('-', ' ')
    return ' '.join(normalized.split())


def load_and_process_datasets():
    """Load and process datasets for analysis"""
    data_dir = Path(__file__).parent.parent / 'data'
    
    print("🔄 Loading exercise datasets...")
    exercises_df = pd.read_csv(data_dir / "exercises.csv")
    mega_gym_df = pd.read_csv(data_dir / "megaGymDataset.csv")
    
    print(f"✓ Loaded exercises.csv: {exercises_df.shape}")
    print(f"✓ Loaded megaGymDataset.csv: {mega_gym_df.shape}")
    
    # Apply name normalization
    exercises_df = exercises_df.copy()
    mega_gym_df = mega_gym_df.copy()
    
    exercises_df['normalized_name'] = exercises_df['name'].apply(normalize_name)
    mega_gym_df['normalized_name'] = mega_gym_df['Title'].apply(normalize_name)
    
    # Find overlapping names
    exercises_names = set(exercises_df['normalized_name'].dropna())
    mega_gym_names = set(mega_gym_df['normalized_name'].dropna())
    overlapping_names = exercises_names.intersection(mega_gym_names)
    
    print(f"🎯 Found {len(overlapping_names)} overlapping exercises")
    
    return {
        'exercises_df': exercises_df,
        'mega_gym_df': mega_gym_df,
        'overlapping_names': overlapping_names
    }


def analyze_rating_distribution(mega_gym_df):
    """Analyze the distribution of ratings in the Niharika dataset"""
    print("\n" + "="*60)
    print("RATING DISTRIBUTION ANALYSIS")
    print("="*60)
    
    ratings = mega_gym_df['Rating'].dropna()
    
    print(f"\nRating Statistics:")
    print(f"  Total rated exercises: {len(ratings)}")
    print(f"  Mean rating: {ratings.mean():.2f}")
    print(f"  Median rating: {ratings.median():.2f}")
    print(f"  Rating range: {ratings.min():.1f} - {ratings.max():.1f}")
    
    print(f"\nRating distribution (binned):")
    rating_bins = pd.cut(ratings, bins=[0, 2, 4, 6, 8, 10], labels=['0-2', '2-4', '4-6', '6-8', '8-10'])
    for bin_label, count in rating_bins.value_counts().sort_index().items():
        percentage = (count / len(ratings)) * 100
        print(f"  {bin_label}: {count} exercises ({percentage:.1f}%)")


def analyze_instruction_depth(exercises_df):
    """Analyze instruction completeness in the Omarxadel dataset"""
    print("\n" + "="*60)
    print("INSTRUCTION DEPTH ANALYSIS")
    print("="*60)
    
    instruction_cols = [col for col in exercises_df.columns if col.startswith('instructions')]
    
    instruction_counts = []
    for _, row in exercises_df.iterrows():
        count = sum(1 for col in instruction_cols 
                   if col in row and pd.notna(row[col]) and str(row[col]).strip())
        instruction_counts.append(count)
    
    exercises_df['instruction_count'] = instruction_counts
    
    print(f"\nInstruction Statistics:")
    print(f"  Average instructions per exercise: {pd.Series(instruction_counts).mean():.1f}")
    print(f"  Median instructions per exercise: {pd.Series(instruction_counts).median():.1f}")
    print(f"  Max instructions: {max(instruction_counts)}")
    
    print(f"\nInstruction count distribution:")
    instruction_dist = pd.Series(instruction_counts).value_counts().sort_index()
    for count, exercises in instruction_dist.items():
        percentage = (exercises / len(exercises_df)) * 100
        print(f"  {count} instructions: {exercises} exercises ({percentage:.1f}%)")


def analyze_equipment_usage(exercises_df, mega_gym_df):
    """Analyze equipment distribution across both datasets"""
    print("\n" + "="*60)
    print("EQUIPMENT USAGE ANALYSIS")
    print("="*60)
    
    print("\nOmarxadel (exercises.csv) equipment:")
    omar_equipment = exercises_df['equipment'].value_counts().head(10)
    for equipment, count in omar_equipment.items():
        percentage = (count / len(exercises_df)) * 100
        print(f"  {equipment}: {count} exercises ({percentage:.1f}%)")
    
    print("\nNiharika (megaGymDataset.csv) equipment:")
    niharika_equipment = mega_gym_df['Equipment'].value_counts().head(10)
    for equipment, count in niharika_equipment.items():
        percentage = (count / len(mega_gym_df)) * 100
        print(f"  {equipment}: {count} exercises ({percentage:.1f}%)")


def analyze_body_parts(exercises_df, mega_gym_df):
    """Analyze body part distribution"""
    print("\n" + "="*60)
    print("BODY PART ANALYSIS")
    print("="*60)
    
    print("\nOmarxadel body parts:")
    omar_bodyparts = exercises_df['bodyPart'].value_counts()
    for bodypart, count in omar_bodyparts.items():
        percentage = (count / len(exercises_df)) * 100
        print(f"  {bodypart}: {count} exercises ({percentage:.1f}%)")
    
    print("\nNiharika body parts:")
    niharika_bodyparts = mega_gym_df['BodyPart'].value_counts()
    for bodypart, count in niharika_bodyparts.items():
        percentage = (count / len(mega_gym_df)) * 100
        print(f"  {bodypart}: {count} exercises ({percentage:.1f}%)")


def generate_insights(overlapping_names, exercises_df, mega_gym_df):
    """Generate key insights for RAG system design"""
    print("\n" + "="*60)
    print("KEY INSIGHTS FOR RAG SYSTEM")
    print("="*60)
    
    total_exercises = len(exercises_df) + len(mega_gym_df) - len(overlapping_names)
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Total unique exercises: {total_exercises}")
    print(f"  Overlapping exercises: {len(overlapping_names)}")
    print(f"  Coverage: {(len(overlapping_names) / min(len(exercises_df), len(mega_gym_df))) * 100:.1f}% overlap")
    
    print(f"\n🎯 RAG System Advantages:")
    print(f"  • Premium documents: {len(overlapping_names)} exercises with ratings AND instructions")
    print(f"  • Comprehensive coverage: {total_exercises} total exercises")
    print(f"  • Multi-modal data: ratings, instructions, equipment, body parts")
    print(f"  • Quality diversity: rated exercises + detailed instructions")
    
    print(f"\n🔧 Indexing Strategy:")
    print(f"  • Merge overlaps for premium documents (ratings + instructions)")
    print(f"  • Keep Niharika-only for rating-based recommendations")  
    print(f"  • Keep Omarxadel-only for instruction-heavy exercises")
    print(f"  • Use semantic search on exercise descriptions")
    print(f"  • Enable metadata filtering (equipment, body part, ratings)")


def main():
    print("\n" + "="*60)
    print("COMPREHENSIVE EXERCISE DATASET ANALYSIS")
    print("="*60)
    
    # Load and process datasets
    results = load_and_process_datasets()
    
    # Extract results
    exercises_df = results['exercises_df']
    mega_gym_df = results['mega_gym_df']
    overlapping_names = results['overlapping_names']
    
    # Extended analysis
    analyze_rating_distribution(mega_gym_df)
    analyze_instruction_depth(exercises_df)
    analyze_equipment_usage(exercises_df, mega_gym_df)
    analyze_body_parts(exercises_df, mega_gym_df)
    generate_insights(overlapping_names, exercises_df, mega_gym_df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("✓ Ready to proceed with indexing pipeline")
    print("✓ Run 'make index' to build the vector database")


if __name__ == "__main__":
    main()