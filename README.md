# CS6300 Homework 6: Fitness & Training Coach Agent

A comprehensive AI fitness coaching system that provides personalized workout recommendations, exercise guidance, and training assistance.

## Overview

This project implements a **Fitness & Training Coach Agent** with multiple specialized capabilities:

- **Exercise Recommendation Engine**: RAG-powered exercise search with 4,125+ fitness exercises
- **[TODO]**: 
- **[TODO]**: 
- **[TODO]**: 
- **[TODO]**: 
- **[TODO]**: 

## 🚀 Quick Start

## Prerequisites

- **Python 3.12+**  
  Make sure you are using Python 3.12 or later.

```bash
python3 --version
```

- **API Keys (stored in `.env`)**  
  You will need API keys for the LLM services:

1. **Gemini API key**  
   - Create via [Google AI Studio](https://aistudio.google.com/) → "Get API key"  
   - Save in `.env` as:  
     ```
     GEMINI_API_KEY=your_gemini_api_key
     ```

2. **LangSmith API Key**  
   - Create via [LangSmith](https://smith.langchain.com/)  
   - Save in `.env` as:  
     ```
     LANGCHAIN_API_KEY=your_langsmith_api_key
     ```

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

### Setup Evaluation (Optional)
```bash
make ollama-setup
```

This installs Ollama and Llama 3.2 for automated quality assessment with detailed scoring.

### Development Commands

```bash
make explore          # Analyze exercise datasets
make clean           # Clean generated files  
make ollama-setup    # [TODO] Setup local LLM for coaching
```

## RAG Exercise Search Examples

The current exercise recommendation system supports intelligent natural language queries:

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
make bulk-evaluate     # Run bulk evaluation (100+ queries)
make quick-test        # Quick stress test (25 essential queries)

# Maintenance
make clean             # Clean generated files
make clean-index       # Remove vector database only
make clean-all         # Deep clean including virtual environment
```

System configuration is handled directly in the code where needed.

## Fitness & Training Coach Agent (Planned)

### Tool Specifications

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

2. **[TODO]**
   - 

3. **[TODO]**
   - 

4. **[TODO]**
   - 

5. **[TODO]**
   - 

## Project Structure

```
cs6300-hw6/
├── src/
│   ├── indexing.py          # Vector database creation & data processing
│   ├── rag_agent.py         # Exercise recommendation interface
│   ├── re_agent.py          # ReAct agent implementation
│   ├── tools.py             # Agent tools and utilities
│   └── evaluation.py        # LLM-based quality evaluation framework
├── scripts/
│   ├── explore_data.py      # Dataset analysis utilities
│   ├── generate_test_queries.py  # Test query generation for evaluation
│   ├── bulk_query_evaluator.py  # Comprehensive stress testing (100+ queries)
│   └── quick_stress_test.py # Fast performance validation (24 queries)
├── data/
│   ├── exercises.csv        # Exercise database (1,324 exercises)
│   ├── megaGymDataset.csv   # Exercise database (2,918 exercises)
│   └── test_queries.json    # Generated test queries for evaluation
├── logs/                    # Evaluation results with timestamps
│   └── rag_evaluation_*.json # Detailed evaluation logs
├── chroma_db/               # Generated vector store (4,125+ documents)
├── requirements.txt         # Python dependencies
├── Makefile                 # Build and run commands
├── README.md                # Project documentation
└── .env                     # API keys (create from template)
```

## Technical Implementation

### RAG Exercise Search (Current)
- **Vector DB**: ChromaDB with sentence-transformers/all-MiniLM-L6-v2 embeddings
- **Query Processing**: Boolean logic support (AND/OR operations)

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

### Future Components
- **[TODO] LLM Integration**: Local Llama 3.2 for conversational coaching
- **[TODO] Workout Planning**: Goal-based routine generation
- **[TODO] Progress Analytics**: Metrics tracking and trend analysis

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

- **Google Gemini**: Advanced AI capabilities for query processing and response generation
- **HuggingFace**: Local embedding models (all-MiniLM-L6-v2)
- **ChromaDB**: Efficient vector storage and similarity search
- **Ollama**: Local LLM hosting for automated evaluation