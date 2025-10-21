"""
Tool functions for ReAct Agent integration
Provides clean interfaces to underlying RAG and fitness systems
"""

from .rag_agent import FitnessRAG
from typing import Optional
import traceback


def exercise_lookup(query: str) -> str:
    """
    Search for exercises based on natural language query
    
    Purpose: Provides exercise recommendations using semantic search and intelligent filtering.
    Supports complex queries with boolean logic (AND/OR), equipment filters, body part targeting,
    and quality indicators.
    
    Args:
        query (str): Natural language fitness query. Examples:
            - "chest exercises with dumbbells"
            - "best back exercises for strength training"  
            - "shoulder or chest workouts with high ratings"
            - "beginner leg exercises at home"
            - "core strengthening with detailed instructions"
    
    Returns:
        str: Formatted exercise recommendations with details including:
            - Exercise names and ratings
            - Equipment requirements
            - Target muscles and body parts
            - Instruction previews for premium exercises
            - Organized by quality (Premium > Rated > Detailed Instructions)
    
    Examples:
        >>> exercise_lookup("chest exercises with dumbbells")
        Returns chest-focused dumbbell exercises with ratings and instructions
        
        >>> exercise_lookup("beginner back exercises")
        Returns beginner-friendly back exercises from the database
        
        >>> exercise_lookup("shoulder or chest workouts with high ratings")
        Returns highly-rated exercises targeting shoulders OR chest
    """
    try:
        # Initialize RAG agent
        rag = FitnessRAG(persist_dir="chroma_db", top_k=10)
        
        # Execute query with verbose output disabled for clean tool interface
        result = rag.query(query, verbose=False)
        
        # Return formatted recommendations
        return result['recommendations']
        
    except FileNotFoundError:
        return ("Error: Vector database not found. Please run 'make index' first to build "
                "the exercise database before using the exercise lookup tool.")
    
    except Exception as e:
        error_msg = f"Error in exercise lookup: {str(e)}"
        # Log full traceback for debugging while returning clean error to user
        print(f"DEBUG - Full error trace:\n{traceback.format_exc()}")
        return error_msg


def get_available_tools() -> dict:
    """
    Returns a dictionary of available tools for the ReAct agent
    
    Returns:
        dict: Tool registry with function references and descriptions
    """
    return {
        "exercise_lookup": {
            "function": exercise_lookup,
            "description": "Search for exercises based on natural language query",
            "parameters": {
                "query": "Natural language fitness query (string)"
            },
            "examples": [
                "chest exercises with dumbbells",
                "best back exercises for strength training",
                "shoulder or chest workouts with high ratings"
            ]
        }
    }
