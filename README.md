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
│   └── tools.py             # Agent tools and utilities
├── scripts/
│   └── explore_data.py      # Dataset analysis utilities
├── data/
│   ├── exercises.csv        # Exercise database (1,324 exercises)
│   └── megaGymDataset.csv   # Exercise database (2,918 exercises)
├── chroma_db/               # Generated vector store (4,125+ documents)
├── requirements.txt         # Python dependencies
├── Makefile                # Build and run commands
├── README.md               # Project documentation
└── .env                    # API keys (create from template)
```

## Technical Implementation

### RAG Exercise Search (Current)
- **Vector DB**: ChromaDB with sentence-transformers/all-MiniLM-L6-v2 embeddings
- **Query Processing**: Boolean logic support (AND/OR operations)
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