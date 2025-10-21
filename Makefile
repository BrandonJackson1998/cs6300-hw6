VENV := .virtual_environment

all: help

help:
	@echo
	@echo "Fitness & Trainer Coach RAG System - Available Commands:"
	@echo "======================================================"
	@echo "Setup:"
	@echo "  install                   - Install Python dependencies"
	@echo
	@echo "Data & Indexing:"
	@echo "  explore                   - Explore the fitness exercise datasets"
	@echo "  index                     - Build vector database from exercise data"
	@echo "  query                     - Interactive RAG query interface"
	@echo
	@echo "Ollama Integration:"
	@echo "  ollama-install            - Install Ollama (macOS only)"
	@echo "  ollama-start              - Start Ollama service"
	@echo "  ollama-stop               - Stop Ollama service"
	@echo "  ollama-pull-model         - Download Llama 3.2 model"
	@echo "  ollama-status             - Check Ollama service status"
	@echo "  ollama-setup              - Complete Ollama setup (install + start + model)"
	@echo
	@echo "Utilities:"
	@echo "  pipeline                  - Run complete workflow (explore + index)"
	@echo "  clean                     - Clean generated files and cache"
	@echo "  clean-all                 - Clean everything including virtual environment"
	@echo

$(VENV):
	python3 -m venv $(VENV)

install: $(VENV)
	@echo "📦 Installing Python dependencies..."
	source $(VENV)/bin/activate; pip install --upgrade pip
	source $(VENV)/bin/activate; pip install -r requirements.txt
	@echo "✅ Installation complete!"

# Core workflow commands
explore:
	@echo "🔍 Exploring fitness exercise datasets..."
	source $(VENV)/bin/activate; python -m scripts.explore_data

index:
	@echo "🔧 Building vector database..."
	source $(VENV)/bin/activate; python -m src.indexing

query:
	@echo "💬 Starting RAG query interface..."
	source $(VENV)/bin/activate; python -m src.rag_agent

pipeline: explore index
	@echo "🎯 Complete fitness coach pipeline finished!"

# Cleanup commands
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf chroma_db/
	rm -rf __pycache__ src/__pycache__ scripts/__pycache__
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete
	@echo "✅ Cleanup complete!"

clean-all: clean
	@echo "🗑️ Removing virtual environment..."
	rm -rf $(VENV)
	@echo "✅ Full cleanup complete!"

# Ollama management commands
ollama-install:
	@echo "Installing Ollama (macOS only)..."
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		brew list --versions ollama > /dev/null 2>&1 || brew install ollama; \
		echo "✓ Ollama installed"; \
	else \
		echo "❌ This command is for macOS only. For Linux, run:"; \
		echo "   curl -fsSL https://ollama.ai/install.sh | sh"; \
	fi

ollama-start:
	@echo "Starting Ollama service..."
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		brew services start ollama; \
		echo "✓ Ollama service started"; \
	else \
		echo "Starting Ollama manually..."; \
		ollama serve & \
		echo "✓ Ollama service started in background"; \
	fi

ollama-stop:
	@echo "Stopping Ollama service..."
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		brew services stop ollama; \
		echo "✓ Ollama service stopped"; \
	else \
		pkill -f "ollama serve" || echo "Ollama not running"; \
		echo "✓ Ollama service stopped"; \
	fi

ollama-pull-model:
	@echo "Downloading Llama 3.2 model (this may take a few minutes)..."
	ollama pull llama3.2
	@echo "✓ Llama 3.2 model downloaded"

ollama-status:
	@echo "Checking Ollama status..."
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		brew services list | grep ollama || echo "Ollama not managed by brew services"; \
	fi
	@ollama list 2>/dev/null || echo "❌ Ollama not running or not installed"

ollama-setup: ollama-install ollama-start ollama-pull-model
	@echo "✅ Ollama setup complete!"
	@echo "   • Ollama installed and running"
	@echo "   • Llama 3.2 model downloaded"
	@echo "   • Ready for RAG evaluation"

.PHONY: all help install explore index query pipeline clean clean-all 
.PHONY: ollama-install ollama-start ollama-stop ollama-pull-model ollama-status ollama-setup