#!/usr/bin/env python3
"""
Simple Query Generator for RAG Testing
Generates diverse fitness queries and saves them to a file for evaluation
"""

import json
import random
from typing import List
from pathlib import Path


def generate_test_queries(num_queries: int = 50) -> List[str]:
    """Generate diverse fitness queries for testing"""
    
    # Core components for query generation
    body_parts = [
        "chest", "back", "shoulders", "arms", "legs", "abs", "core",
        "biceps", "triceps", "quads", "hamstrings", "calves", "glutes"
    ]
    
    equipment = [
        "dumbbells", "barbells", "kettlebells", "cables", "machines",
        "bodyweight", "resistance bands", "exercise ball"
    ]
    
    difficulty = ["beginner", "intermediate", "advanced", "easy", "challenging"]
    quality = ["best", "top rated", "highest rated", "detailed", "premium"]
    locations = ["at home", "in gym", "outdoors", "with limited space"]
    goals = ["muscle building", "strength", "fat loss", "toning", "endurance"]
    
    # Query templates
    templates = [
        # Simple queries
        "{body_part} exercises",
        "{equipment} workout",
        "{body_part} training",
        
        # Equipment + body part
        "{body_part} exercises with {equipment}",
        "{equipment} {body_part} workout",
        
        # Quality focused
        "{quality} {body_part} exercises",
        "{quality} {equipment} workout",
        
        # Location specific
        "{body_part} exercises {location}",
        "{equipment} workout {location}",
        
        # Goal oriented
        "{body_part} exercises for {goal}",
        "{equipment} workout for {goal}",
        
        # Difficulty based
        "{difficulty} {body_part} exercises",
        "{difficulty} {equipment} workout",
        
        # Complex combinations
        "{quality} {body_part} exercises with {equipment}",
        "{difficulty} {body_part} workout {location}",
        "{body_part} exercises with {equipment} for {goal}",
        
        # Boolean logic queries
        "{body_part1} or {body_part2} exercises",
        "{body_part} and {body_part2} workout",
        "{equipment1} or {equipment2} exercises",
        
        # Natural requests
        "I need {body_part} exercises",
        "What are good {body_part} exercises?",
        "Show me {equipment} exercises",
        "Find {body_part} workout",
        
        # Instructional
        "How to do {body_part} exercises",
        "Step by step {body_part} workout",
        "Guide for {equipment} exercises",
    ]
    
    queries = []
    
    # Generate random queries
    for _ in range(num_queries):
        template = random.choice(templates)
        
        query = template.format(
            body_part=random.choice(body_parts),
            body_part1=random.choice(body_parts),
            body_part2=random.choice(body_parts),
            equipment=random.choice(equipment),
            equipment1=random.choice(equipment),
            equipment2=random.choice(equipment),
            difficulty=random.choice(difficulty),
            quality=random.choice(quality),
            location=random.choice(locations),
            goal=random.choice(goals)
        )
        
        queries.append(query)
    
    # Add essential test cases
    essential_queries = [
        # Basic body parts
        "chest exercises",
        "back workout",
        "leg training",
        "shoulder exercises",
        "abs workout",
        "arm exercises",
        
        # Equipment specific
        "dumbbell exercises",
        "barbell workout",
        "bodyweight exercises",
        "cable training",
        "kettlebell routine",
        
        # Boolean logic
        "chest or shoulder exercises",
        "back and bicep workout",
        "legs or glutes training",
        "dumbbells or barbells exercises",
        
        # Quality focused
        "best chest exercises",
        "highest rated back workout",
        "exercises with detailed instructions",
        "premium quality workouts",
        
        # Complex combinations
        "chest exercises with dumbbells for muscle building",
        "beginner back workout at home",
        "advanced leg training with barbells",
        "best rated shoulder exercises with cables",
        
        # Edge cases
        "exercise",
        "workout",
        "fitness",
        "training",
        "",  # Empty query
        "a",  # Single character
        "best exercises ever",
        "workout routine for everything",
    ]
    
    # Combine and remove duplicates
    all_queries = list(set(queries + essential_queries))
    
    return all_queries


def save_queries_to_file(queries: List[str], filename: str = "test_queries.json"):
    """Save generated queries to a JSON file"""
    
    # Create query data with metadata
    query_data = {
        "metadata": {
            "total_queries": len(queries),
            "generated_timestamp": "2025-10-21",
            "description": "Generated test queries for RAG evaluation"
        },
        "queries": [
            {
                "id": i + 1,
                "query": query,
                "length": len(query),
                "complexity": assess_complexity(query)
            }
            for i, query in enumerate(queries)
        ]
    }
    
    # Save to file
    filepath = Path(filename)
    with open(filepath, 'w') as f:
        json.dump(query_data, f, indent=2)
    
    print(f"✅ Saved {len(queries)} queries to {filepath}")
    return filepath


def assess_complexity(query: str) -> str:
    """Assess query complexity"""
    if len(query) == 0:
        return "empty"
    elif len(query) < 10:
        return "simple"
    elif "or" in query.lower() or "and" in query.lower():
        return "boolean"
    elif len(query.split()) > 8:
        return "complex"
    elif any(word in query.lower() for word in ["best", "rated", "detailed", "premium"]):
        return "quality_focused"
    else:
        return "moderate"


def main():
    """Generate and save test queries"""
    print("🔍 Generating Test Queries for RAG Evaluation")
    print("="*50)
    
    # Generate queries
    print("📝 Generating diverse queries...")
    queries = generate_test_queries(num_queries=75)
    
    print(f"✓ Generated {len(queries)} unique queries")
    
    # Show some examples
    print("\n📋 Sample queries:")
    for i, query in enumerate(queries[:10], 1):
        complexity = assess_complexity(query)
        print(f"  {i:2d}. [{complexity:>12}] '{query}'")
    
    if len(queries) > 10:
        print(f"       ... and {len(queries) - 10} more")
    
    # Save to file
    print(f"\n💾 Saving queries...")
    filepath = save_queries_to_file(queries, "data/test_queries.json")
    
    # Print complexity distribution
    complexity_counts = {}
    for query in queries:
        complexity = assess_complexity(query)
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
    
    print(f"\n📊 Complexity Distribution:")
    for complexity, count in sorted(complexity_counts.items()):
        percentage = (count / len(queries)) * 100
        print(f"  • {complexity:>12}: {count:3d} queries ({percentage:5.1f}%)")
    
    print(f"\n🎯 Ready for evaluation!")
    print(f"   Use: make evaluate")
    print(f"   File: {filepath}")
    
    return str(filepath)


if __name__ == "__main__":
    main()