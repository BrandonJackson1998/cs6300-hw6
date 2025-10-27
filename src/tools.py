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


def estimate_workout_duration(workout_description: str) -> str:
    """
    Estimate the total duration of a workout based on exercises and structure
    
    Purpose: Calculates estimated workout time including exercise sets, rest periods,
    warm-up, and cool-down. Helps users plan their training sessions realistically.
    
    Args:
        workout_description (str): Description of the workout including exercises,
                                  sets, reps, or list of exercise names. Can be:
            - List of exercises (e.g., "Bench Press, Squats, Deadlifts")
            - Detailed workout (e.g., "3 sets of Bench Press, 4 sets of Squats")
            - Exercise count (e.g., "5 exercises for chest and back")
    
    Returns:
        str: Detailed duration estimate with breakdown of:
            - Warm-up time (5-10 minutes)
            - Exercise time (based on sets/reps/exercises)
            - Rest periods between sets
            - Cool-down time (5 minutes)
            - Total estimated duration
    
    Estimation Guidelines:
        - Each exercise set: ~1 minute (includes execution time)
        - Rest between sets: 1-2 minutes (shorter for isolation, longer for compounds)
        - Default sets per exercise: 3-4 sets
        - Warm-up: 5-10 minutes
        - Cool-down: 5 minutes
        - Compound exercises (squat, deadlift, bench): +30 seconds per set
    
    Examples:
        >>> estimate_workout_duration("Bench Press, Squats, Deadlifts")
        "Estimated workout duration: 45-55 minutes\\n\\nBreakdown:\\n- Warm-up: 5-10 min\\n- 3 exercises × 3-4 sets: 25-35 min\\n- Rest periods: 10-15 min\\n- Cool-down: 5 min"
        
        >>> estimate_workout_duration("3 sets of Bench Press (8 reps), 4 sets of Squats (10 reps), 3 sets of Rows (12 reps)")
        "Estimated workout duration: 50-60 minutes\\n\\nBreakdown:\\n- Warm-up: 5-10 min\\n- 10 total sets: 30-35 min\\n- Rest periods: 12-18 min\\n- Cool-down: 5 min"
        
        >>> estimate_workout_duration("5 exercises for upper body")
        "Estimated workout duration: 55-70 minutes\\n\\nBreakdown:\\n- Warm-up: 5-10 min\\n- 5 exercises × 3-4 sets: 35-45 min\\n- Rest periods: 15-20 min\\n- Cool-down: 5 min"
    """
    try:
        if not workout_description or not isinstance(workout_description, str):
            return "Error: Invalid workout description. Please provide a string describing the workout."
        
        workout_lower = workout_description.lower()
        
        # Parse workout to estimate number of exercises and sets
        num_exercises = 0
        total_sets = 0
        
        # Try to extract explicit set counts (e.g., "3 sets of", "4 sets")
        import re
        set_patterns = re.findall(r'(\d+)\s*sets?\s*(?:of|x)?', workout_lower)
        if set_patterns:
            total_sets = sum(int(s) for s in set_patterns)
        
        # Count exercises by common delimiters or keywords
        exercise_indicators = [',', '\n', ' and ', ';']
        exercise_parts = [workout_description]
        for delimiter in exercise_indicators:
            temp_parts = []
            for part in exercise_parts:
                temp_parts.extend(part.split(delimiter))
            exercise_parts = temp_parts
        
        # Filter out empty parts and count meaningful exercise mentions
        exercise_parts = [p.strip() for p in exercise_parts if p.strip() and len(p.strip()) > 3]
        num_exercises = len(exercise_parts)
        
        # Check for explicit exercise count mentions (e.g., "5 exercises", "3 chest exercises")
        exercise_count_patterns = re.findall(r'(\d+)\s*exercises?', workout_lower)
        if exercise_count_patterns:
            num_exercises = max(num_exercises, int(exercise_count_patterns[0]))
        
        # If we couldn't detect exercises, assume a minimal workout
        if num_exercises == 0:
            num_exercises = 3  # Default assumption
        
        # If sets weren't explicitly mentioned, estimate based on standard practice
        if total_sets == 0:
            sets_per_exercise = 3.5  # Average of 3-4 sets
            total_sets = int(num_exercises * sets_per_exercise)
        
        # Estimate compound vs isolation exercises
        compound_keywords = ['squat', 'deadlift', 'bench', 'press', 'row', 'pull-up', 'chin-up']
        compound_count = sum(1 for keyword in compound_keywords if keyword in workout_lower)
        
        # Calculate time components
        warmup_min = 5
        warmup_max = 10
        
        # Exercise execution time (1-1.5 min per set, more for compounds)
        avg_time_per_set = 1.0  # minutes
        if compound_count > 0:
            avg_time_per_set = 1.25  # Add time for compound movements
        
        exercise_time_min = int(total_sets * avg_time_per_set)
        exercise_time_max = int(total_sets * (avg_time_per_set + 0.25))
        
        # Rest periods (1-2 min between sets)
        # Subtract 1 from total_sets because no rest after last set
        rest_periods = max(0, total_sets - num_exercises)  # Rest between sets within each exercise
        rest_time_min = int(rest_periods * 1.0)
        rest_time_max = int(rest_periods * 1.5)
        
        cooldown = 5
        
        # Calculate totals
        total_min = warmup_min + exercise_time_min + rest_time_min + cooldown
        total_max = warmup_max + exercise_time_max + rest_time_max + cooldown
        
        # Format output
        breakdown = (
            f"Estimated workout duration: {total_min}-{total_max} minutes\n\n"
            f"Breakdown:\n"
            f"- Warm-up: {warmup_min}-{warmup_max} min\n"
            f"- Exercise execution ({total_sets} sets across {num_exercises} exercises): {exercise_time_min}-{exercise_time_max} min\n"
            f"- Rest periods: {rest_time_min}-{rest_time_max} min\n"
            f"- Cool-down: {cooldown} min\n\n"
            f"Note: Actual time may vary based on exercise complexity, rest needs, and fitness level."
        )
        
        return breakdown
        
    except Exception as e:
        error_msg = f"Error estimating workout duration: {str(e)}"
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
        },
        "estimate_workout_duration": {
            "function": estimate_workout_duration,
            "description": "Estimate the total duration of a workout based on exercises and structure",
            "parameters": {
                "workout_description": "Description of workout with exercises, sets, or exercise count (string)"
            },
            "examples": [
                "Bench Press, Squats, Deadlifts",
                "3 sets of Bench Press, 4 sets of Squats, 3 sets of Rows",
                "5 exercises for upper body"
            ]
        }
    }
