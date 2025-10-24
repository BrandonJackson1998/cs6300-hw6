# AGENTS.md - Coding Guidelines for CS6300 HW6

## Build/Run Commands
- **Setup**: `make install-mac` (macOS) or `make install` (Linux)
- **Build index**: `make index` (builds ChromaDB vector database)
- **Run RAG agent**: `make query` (interactive exercise search)
- **Run ReAct agent**: `make react` (conversational fitness coach with reasoning + LangSmith tracing)
- **Generate test queries**: `make generate-queries`
- **Run evaluation**: `make evaluate` (with explanations) or `make evaluate-quiet` (silent)
- **Explore data**: `make explore`
- **Run tools test**: `make test-tools`
- **Clean**: `make clean` (remove generated files), `make clean-index` (remove DB only)

## LangSmith Tracing
- **Enable**: Add `LANGCHAIN_API_KEY=lsv2_xxx` to `.env` (get from https://smith.langchain.com/)
- **Project**: Traces appear in "fitness-react-coach" project
- **Metadata**: Each run logs input length, model name, timestamp
- **Disable**: Set `enable_tracing=False` in FitnessReActAgent constructor

## Code Style
- **Imports**: Standard lib first, then third-party (pandas, chromadb, langchain), then local (from .rag_agent)
- **Formatting**: Use Black (installed as dev tool); 4-space indentation, max line ~100 chars
- **Types**: Use type hints (from typing import List, Dict, Any, Tuple, Optional)
- **Naming**: snake_case for functions/variables, PascalCase for classes (e.g., FitnessRAG, QueryAnalyzer)
- **Docstrings**: Use triple-quoted strings with Args/Returns sections
- **Error handling**: Wrap file/API operations in try-except; return clean error messages
- **Logging**: Use print() for user-facing output; suppress library logs (chromadb.setLevel(ERROR))

## Project Structure
- `src/` - Core modules (indexing.py, rag_agent.py, re_agent.py, tools.py, evaluation.py)
- `data/` - CSV datasets (exercises.csv, megaGymDataset.csv, test_queries.json)
- `chroma_db/` - Vector database (generated)
- `logs/` - Evaluation results (timestamped JSON files)
