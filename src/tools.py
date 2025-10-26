"""
Tool functions for ReAct Agent integration
Provides clean interfaces to underlying RAG and fitness systems
"""

from .rag_agent import FitnessRAG
from typing import Optional, List
import traceback
from datetime import datetime


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


def validate_rest_day(schedule: str) -> str:
    """
    Validate if there are too many rest days in a weekly schedule
    
    Args:
        schedule (str): Weekly workout schedule. Can be a list of days with activities/exercises,
                       or a description of the weekly schedule. Days without exercises are 
                       counted as rest days.
    
    Returns:
        str: Validation message indicating if schedule has too many rest days
    
    Examples:
        >>> validate_rest_day("Monday: Chest, Tuesday: Rest, Wednesday: Back, Thursday: Rest, Friday: Legs, Saturday: Rest, Sunday: Rest")
        "Warning: Too many rest days! You have 4 rest days in the week. Maximum recommended is 2."
        
        >>> validate_rest_day("Mon: Chest, Tue: Back, Wed: Legs, Thu: Shoulders, Fri: Arms, Sat: Rest, Sun: Rest")
        "Valid schedule. You have 2 rest days in the week."
    """
    try:
        if not schedule or not isinstance(schedule, str):
            return "Error: Invalid schedule format. Expected a string describing the weekly schedule."
        
        schedule_lower = schedule.lower()
        days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        days_abbrev = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        
        rest_count = 0
        workout_count = 0
        
        for day_full, day_abbr in zip(days_of_week, days_abbrev):
            day_found = False
            
            if day_full in schedule_lower or day_abbr in schedule_lower:
                day_found = True
                if "rest" in schedule_lower:
                    lines = schedule_lower.split('\n') if '\n' in schedule_lower else schedule_lower.split(',')
                    for line in lines:
                        if (day_full in line or day_abbr in line):
                            if "rest" in line and not any(term in line for term in ["chest", "back", "legs", "arms", "shoulders", "core", "cardio", "workout", "exercise"]):
                                rest_count += 1
                            else:
                                workout_count += 1
                            break
                else:
                    workout_count += 1
        
        if workout_count == 0 and rest_count == 0:
            parts = [p.strip() for p in schedule.replace('\n', ',').split(',') if p.strip()]
            total_days = len(parts)
            
            if total_days > 7:
                total_days = 7
            
            rest_count = sum(1 for p in parts if "rest" in p.lower() and not any(term in p.lower() for term in ["chest", "back", "legs", "arms", "shoulders", "core", "cardio", "workout", "exercise"]))
            workout_count = total_days - rest_count
        
        if rest_count > 2:
            return f"Warning: Too many rest days! You have {rest_count} rest days in the week. Maximum recommended is 2."
        else:
            return f"Valid schedule. You have {rest_count} rest days in the week."
    
    except Exception as e:
        return f"Error validating rest days: {str(e)}"


def get_current_day(dummy: str = "") -> str:
    """
    Get the current day of the week
    
    Args:
        dummy (str): Unused parameter (LangChain requires tools to accept input)
    
    Returns:
        str: Current day of the week (e.g., "Monday", "Tuesday")
    
    Examples:
        >>> get_current_day()
        "Wednesday"
    """
    try:
        return datetime.now().strftime("%A")
    except Exception as e:
        return f"Error getting current day: {str(e)}"


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
        },
        "validate_rest_day": {
            "function": validate_rest_day,
            "description": "Validate if there are too many rest days in a weekly schedule",
            "parameters": {
                "schedule": "Weekly workout schedule as a string describing each day"
            },
            "examples": [
                "Monday: Chest, Tuesday: Rest, Wednesday: Back, Thursday: Legs, Friday: Rest, Saturday: Arms, Sunday: Rest",
                "Mon: Chest, Tue: Back, Wed: Legs, Thu: Shoulders, Fri: Arms, Sat: Rest, Sun: Rest"
            ]
        },
        "get_current_day": {
            "function": get_current_day,
            "description": "Get the current day of the week",
            "parameters": {},
            "examples": []
        }
    }
