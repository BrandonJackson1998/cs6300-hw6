# CS6300 Homework 6: Fitness & Training Coach Agent

A comprehensive AI fitness coaching system that provides personalized workout recommendations, exercise guidance, and training assistance.

## Overview

This project implements a **Fitness & Training Coach Agent** with multiple specialized capabilities:

- **Exercise Recommendation Engine**: RAG-powered exercise search with 4,125+ fitness exercises
- **Conversational Fitness Coach**: ReAct agent with natural language interaction and tool use
- **Workout Planning Tools**: Schedule validation, duration estimation, and rest day analysis
- **Automated Evaluation**: LLM-as-a-judge with comprehensive quality metrics

## 🚀 Quick Start

## Prerequisites

- **Python 3.12+**  
  Make sure you are using Python 3.12 or later.

```bash
python3 --version
```

- **API Keys (stored in `.env`)**  
  You will need API keys for the LLM services:

1. **Gemini API key** (Required for ReAct agent)
   - Create via [Google AI Studio](https://aistudio.google.com/) → "Get API key"  
   - Save in `.env` as:  
     ```
     GEMINI_API_KEY=your_gemini_api_key
     ```

2. **LangSmith API Key** (Required for ReAct agent tracing)
   - Create via [LangSmith](https://smith.langchain.com/)  
   - Save in `.env` as:  
     ```
     LANGCHAIN_API_KEY=your_langsmith_api_key
     ```

**Note**: The RAG agent (`make query`) works without these keys. The ReAct conversational agent (`make react`) requires both API keys.

## Setup

1. **Create a virtual environment**

Mac:
```bash
make .virtual_environment
source .virtual_environment/bin/activate
```

2. **Install dependencies**

On Mac:
```bash
make install-mac
```

On Linux (teacher's original setup):
```bash
make install
```

---

### Build Vector Database
```bash
make index          # Processes 4,125+ exercise documents
```

### Run the RAG Agent
```bash
make query          # Start exercise recommendation interface
```

### Run the ReAct Agent (Conversational Coach)
```bash
make react          # Interactive AI fitness coach with tool use
```

The ReAct agent provides:
- Natural conversation about fitness and exercises
- Automatic tool selection (exercise search, schedule validation, duration estimation)
- Workout planning with intelligent recommendations
- Rest day validation and time management

### Setup Evaluation (Optional)
```bash
make ollama-setup
```

This installs Ollama and Llama 3.2 for automated quality assessment with detailed scoring.

### Development Commands

```bash
make explore          # Analyze exercise datasets
make test-tools       # Test tool wrapper functions
make clean           # Clean generated files
```

## Usage Examples

### RAG Exercise Search (`make query`)

Direct exercise search with semantic matching:

```bash
# After running 'make query', try:
"chest or shoulder exercises with dumbbells"
"best back exercises for strength training"
"Show me shoulder workouts with dumbbells"
"I need back exercises for strength training"
"What leg exercises can I do at home?"
"Upper body exercises with high ratings"
"Core strengthening exercises with detailed instructions"
```

### ReAct Conversational Coach (`make react`)

Natural conversation with intelligent tool use:

```bash
# After running 'make react', try:
"I want to build a chest and back workout plan"
"Create a 5-day workout schedule for me"
"How long will a workout with bench press, squats, and deadlifts take?"
"Is my schedule good? Monday: Chest, Tue: Rest, Wed: Back, Thu: Rest, Fri: Legs, Sat: Rest, Sun: Rest"
"Show me shoulder exercises and estimate the workout time"
"What day is it today and what should I train?"
```

The ReAct agent automatically:
- Searches for exercises when you ask about workouts
- Validates schedules for proper rest day distribution (max 2 rest days recommended)
- Estimates workout duration including warm-up, exercises, rest, and cool-down
- Gets current day when planning schedules

## Available Make Commands

```bash
# Environment Setup
make install-mac        # Install on macOS with Homebrew
make install           # Install on Linux/Ubuntu with apt

# Data & Indexing
make explore           # Explore the fitness exercise datasets
make index             # Build vector database from exercise data

# Running the System
make query             # Interactive fitness exercise recommendation agent
make react             # Interactive ReAct agent (conversational coach)
make test-tools        # Test tool wrapper functions

# Evaluation Setup (Optional)
make ollama-setup      # Complete Ollama + Llama 3.2 setup
make ollama-install    # Install Ollama only
make ollama-start      # Start Ollama service
make ollama-stop       # Stop Ollama service
make ollama-status     # Check Ollama status

# Evaluation Commands
make generate-queries  # Generate test queries for evaluation
make evaluate          # Run evaluation with detailed explanations
make evaluate-quiet    # Run evaluation without displaying explanations

# Maintenance
make clean             # Clean generated files
make clean-index       # Remove vector database only
make clean-all         # Deep clean including virtual environment
```

System configuration is handled directly in the code where needed.

## Fitness & Training Coach Agent Architecture

### Core Components

1. **RAG Search Engine** ✅
   - **Semantic Search**: Vector similarity matching with sentence-transformers/all-MiniLM-L6-v2 embeddings
   - **Boolean Query Logic**: Advanced AND/OR operations for complex exercise filtering
   - **Intelligent Query Processing**: Natural language understanding with auto-extraction of:
     - Equipment filters (dumbbells, barbells, bodyweight, etc.)
     - Body part targeting (chest, shoulders, back, legs, etc.)
     - Quality indicators (best, top-rated, beginner-friendly)
     - Numeric constraints (ratings, difficulty levels)
   - **Comprehensive Database**: 4,125+ exercises from merged fitness datasets
   - **Smart Filtering**: ChromaDB metadata filtering with normalized query sanitization
   - **Multi-Source Integration**: Combined exercises.csv (1,324) + megaGymDataset.csv (2,918)

2. **ReAct Conversational Agent** ✅
   - **LangChain Integration**: ReAct (Reasoning + Acting) framework with Gemini 2.5 Flash
   - **Tool Orchestration**: Automatic tool selection and chaining for complex queries
   - **Natural Dialogue**: Conversational interface with context understanding
   - **LangSmith Tracing**: Full observability of agent reasoning and tool calls
   - **Error Handling**: Robust parsing and recovery from errors

3. **Agent Tools** ✅
   - **`exercise_lookup`**: Natural language exercise search using RAG system
     - Supports complex queries with boolean logic
     - Returns formatted recommendations with ratings and instructions
   - **`validate_rest_day`**: Weekly schedule analysis
     - Validates rest day count (max 2 recommended)
     - Parses various schedule formats
     - Provides warnings for overtraining risk
   - **`get_current_day`**: Current day of week retrieval
     - Used for schedule planning from today
   - **`estimate_workout_duration`**: Workout time estimation
     - Calculates warm-up, exercise, rest, and cool-down time
     - Accounts for compound vs isolation exercises
     - Provides detailed breakdowns (e.g., "45-55 minutes")

**Tool Quick Reference:**

| Tool | Purpose | Example Input | Example Output |
|------|---------|--------------|----------------|
| `exercise_lookup` | Search exercises via RAG | "chest exercises with dumbbells" | Formatted list with ratings, equipment, targets |
| `validate_rest_day` | Check rest day count | "Mon: Chest, Tue: Rest, Wed: Back..." | "Valid schedule. 2 rest days." |
| `get_current_day` | Get current weekday | (no input needed) | "Monday" |
| `estimate_workout_duration` | Calculate workout time | "Bench Press, Squats, Deadlifts" | "45-55 minutes (breakdown included)" |


## Project Structure

```
cs6300-hw6/
├── src/
│   ├── indexing.py          # Vector database creation & data processing
│   ├── rag_agent.py         # Exercise recommendation interface (RAG)
│   ├── re_agent.py          # ReAct agent implementation (conversational coach)
│   ├── tools.py             # Agent tools (exercise_lookup, validate_rest_day, etc.)
│   └── evaluation.py        # LLM-based quality evaluation framework
├── scripts/
│   ├── explore_data.py      # Dataset analysis utilities
│   └── generate_test_queries.py  # Test query generation for evaluation
├── data/
│   ├── exercises.csv        # Exercise database (1,324 exercises)
│   ├── megaGymDataset.csv   # Exercise database (2,918 exercises)
│   └── test_queries.json    # Generated test queries for evaluation (gitignored)
├── logs/                    # Evaluation results with timestamps
│   └── rag_evaluation_*.json # Detailed evaluation logs
├── chroma_db/               # Generated vector store (4,125+ documents)
├── requirements.txt         # Python dependencies
├── Makefile                 # Build and run commands
├── README.md                # Project documentation
└── .env                     # API keys (GEMINI_API_KEY, LANGSMITH_API_KEY)
```

## Technical Implementation

### RAG Exercise Search
- **Vector DB**: ChromaDB with sentence-transformers/all-MiniLM-L6-v2 embeddings
- **Query Processing**: Boolean logic support (AND/OR operations)
- **Embeddings**: Local sentence-transformers model (all-MiniLM-L6-v2)
- **Results**: Top-10 semantic similarity with intelligent metadata filtering

### ReAct Agent
- **Framework**: LangChain ReAct (Reasoning + Acting)
- **LLM**: Google Gemini 2.5 Flash (temperature 0.7)
- **Tools**: 4 specialized fitness tools with automatic selection
- **Tracing**: LangSmith integration for full observability
- **Prompt**: Custom fitness coach prompt with tool usage guidelines

### Agent Tools
1. **exercise_lookup**: Wrapper around RAG system for exercise search
2. **validate_rest_day**: Schedule parser with rest day validation logic
3. **get_current_day**: Python datetime integration
4. **estimate_workout_duration**: Workout time calculator with component breakdown

### Evaluation Framework ✅

The evaluation system provides comprehensive assessment of RAG response quality with beautiful visualizations and detailed feedback.

#### Features
- **Automated LLM Evaluation**: Llama 3.2 judges final RAG responses (not just retrieved documents)
- **Beautiful Progress Bars**: Visual scores like `████████░░ 8.0/10`
- **Detailed Explanations**: Comprehensive reasoning for each score
- **Timestamped Logs**: All results saved to `logs/` folder with timestamps
- **Flexible Display**: Show/hide explanations with `--no-explanations` flag
- **Multiple Metrics**: Retrieval metrics + LLM quality assessment

#### Evaluation Workflow
```bash
# 1. Generate diverse test queries
make generate-queries

# 2. Run evaluation (shows detailed explanations)
make evaluate

# 3. Or run quietly (explanations saved to logs but not displayed)
make evaluate-quiet
```

#### Sample Evaluation Output
```
================================================================================
EVALUATION (LLM-as-a-judge with Llama 3.2)
================================================================================

Scores:
  Retrieval Relevance:   ████████░░ 8.0/10
  Answer Accuracy:       █████████░ 9.0/10
  Answer Completeness:   ███████░░░ 7.0/10
  Citation Quality:      ████████░░ 8.0/10
  ──────────────────────────────────────────────────
  Overall Score:         ███████░░░ 7.9/10

Detailed Feedback:
RETRIEVAL_RELEVANCE: 8/10
REASON: The response provides relevant exercise recommendations that match the query criteria, with appropriate filtering and targeting.

ANSWER_ACCURACY: 9/10
REASON: Exercise information is factually correct with accurate equipment, body parts, and safety considerations.

ANSWER_COMPLETENESS: 7/10
REASON: Good coverage of exercises but could benefit from more detailed instructions and safety notes.

CITATION_QUALITY: 8/10
REASON: Well-organized structure with clear formatting, though some additional metadata would be helpful.

OVERALL: 7.9/10
REASON: Strong performance across all criteria with room for minor improvements in completeness and detail.
================================================================================
```

#### Metrics Tracked
- **Retrieval Metrics**: Precision@K, Hit Rate, Metadata Accuracy
- **LLM Judge Scores**: Relevance, Accuracy, Completeness, Citation Quality
- **Summary Statistics**: Averages, ranges, standard deviations
- **Performance Analysis**: Best/worst queries, recommendations

#### Log Files
- **Location**: `logs/rag_evaluation_YYYYMMDD_HHMMSS.json`
- **Contents**: Full evaluation data, formatted reports, explanations
- **Always Saved**: Even with `--no-explanations`, full details preserved
- **Results**: Top-10 semantic similarity with intelligent metadata filtering
- **Data**: 4,125+ exercises from merged fitness datasets

### Dependencies
- **LangChain**: Agent framework and tool orchestration
- **Google Gemini**: LLM for conversational agent (gemini-2.5-flash)
- **ChromaDB**: Vector database for semantic search
- **Sentence Transformers**: Local embedding model
- **Ollama + Llama 3.2**: Local LLM for evaluation
- **LangSmith**: Agent tracing and observability
- **Pandas/NumPy**: Data processing
- **DuckDuckGo Search**: Web search capability (available but not yet utilized)

## ReAct Agent Workflow

The ReAct agent follows a **Thought → Action → Observation** loop:

### Example: Creating a Workout Plan

**User**: "Create a 5-day workout schedule for me"

**Agent Reasoning**:
1. **Thought**: User wants a weekly workout plan. I should create a schedule and then validate it for rest days.
2. **Action**: `get_current_day()` to start from today
3. **Observation**: "Monday"
4. **Thought**: Now I'll create a balanced 5-day split and validate rest days
5. **Action**: Create schedule with appropriate muscle groups
6. **Action**: `validate_rest_day("Monday: Chest, Tuesday: Back, Wednesday: Rest, Thursday: Legs, Friday: Shoulders, Saturday: Arms, Sunday: Rest")`
7. **Observation**: "Valid schedule. You have 2 rest days in the week."
8. **Thought**: Should estimate workout duration for each day
9. **Action**: `estimate_workout_duration("Chest exercises: Bench Press, Incline Press, Flyes")`
10. **Observation**: "Estimated workout duration: 45-55 minutes..."
11. **Final Answer**: Complete workout plan with validation and time estimates

### Tool Usage Guidelines (from Agent Prompt)

The agent is instructed to:
- **Always validate** any weekly schedule with `validate_rest_day`
- **Always estimate** workout duration when creating workout plans
- **Use `exercise_lookup`** for all exercise-related queries
- **Break down complex requests** into multiple tool calls
- **Provide context** and explanations with recommendations

### LangSmith Tracing

Every agent interaction is traced in LangSmith (project: "fitness-react-coach"):
- Full reasoning chain (Thought/Action/Observation loops)
- Tool inputs and outputs
- Token usage and latency
- Error handling and retries
- Metadata: input length, model name, timestamp

To enable tracing, ensure `LANGSMITH_API_KEY` is set in `.env`.

## Cleaning Up
```bash
# Remove vector database only
make clean-index

# Cleanup
make clean
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Google Gemini**: LLM powering the ReAct conversational agent (Gemini 2.5 Flash)
- **LangChain**: ReAct agent framework and tool orchestration
- **LangSmith**: Agent tracing and observability platform
- **HuggingFace**: Local embedding models (all-MiniLM-L6-v2)
- **ChromaDB**: Efficient vector storage and similarity search
- **Ollama**: Local LLM hosting for automated evaluation (Llama 3.2)