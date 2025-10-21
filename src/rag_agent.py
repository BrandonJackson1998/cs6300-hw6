"""
Fitness Exercise RAG Agent with ChromaDB Integration and LLM Query Analysis
"""

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re


class QueryAnalyzer:
    """Analyzes user queries to extract structured filters and sanitized semantic queries"""
    
    def __init__(self):
        # Define available filter categories based on our metadata (include both singular/plural)
        self.body_parts = {
            'chest', 'shoulder', 'shoulders', 'back', 'arm', 'arms', 'leg', 'legs', 
            'core', 'abs', 'waist', 'upper arms', 'lower arms', 'upper legs', 'lower legs', 
            'cardio', 'bicep', 'biceps', 'tricep', 'triceps', 'quad', 'quads'
        }
        
        self.equipment_types = {
            'barbell', 'dumbbell', 'kettlebell', 'cable', 'machine', 'body weight', 
            'resistance band', 'exercise ball', 'stability ball', 'weighted', 'lever',
            'smith machine', 'ez barbell', 'rope'
        }
        
        self.difficulty_indicators = {
            'beginner': ['easy', 'beginner', 'simple', 'basic', 'starter'],
            'intermediate': ['intermediate', 'moderate', 'standard'],
            'advanced': ['hard', 'advanced', 'difficult', 'challenging', 'expert']
        }
        
        self.quality_indicators = {
            'high_rating': ['best', 'top', 'highest rated', 'excellent', 'great'],
            'has_instructions': ['detailed', 'with instructions', 'step by step', 'guided'],
            'premium': ['premium', 'complete', 'comprehensive', 'merged']
        }
        
        # Mapping from detected terms to database values (match actual DB values)
        self.body_part_mapping = {
            'shoulder': 'Shoulders', 'shoulders': 'Shoulders',
            'chest': 'Chest',
            'back': 'Lats', 'lats': 'Lats', 'lat': 'Lats',
            'middle back': 'Middle Back', 'lower back': 'Lower Back',
            'arm': 'Biceps', 'arms': 'Biceps', 
            'bicep': 'Biceps', 'biceps': 'Biceps',
            'tricep': 'Triceps', 'triceps': 'Triceps',
            'leg': 'Quadriceps', 'legs': 'Quadriceps',
            'quad': 'Quadriceps', 'quads': 'Quadriceps',
            'hamstring': 'Hamstrings', 'hamstrings': 'Hamstrings',
            'glute': 'Glutes', 'glutes': 'Glutes',
            'abs': 'Abdominals', 'core': 'Abdominals', 'abdominals': 'Abdominals',
            'calf': 'Calves', 'calves': 'Calves',
            'trap': 'Traps', 'traps': 'Traps',
            'forearm': 'Forearms', 'forearms': 'Forearms',
            'abductor': 'Abductors', 'abductors': 'Abductors'
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to extract structured filters and sanitized semantic query
        Handles both AND and OR logic for body parts and equipment
        
        Returns:
            {
                'filters': Dict with ChromaDB-compatible filters,
                'sanitized_query': str with semantic keywords only,
                'original_query': str,
                'has_or_logic': bool
            }
        """
        query_lower = query.lower().strip()
        filters = {}
        semantic_keywords = []
        has_or_logic = False
        
        # Check for OR logic patterns
        or_patterns = [' or ', ' OR ']
        for pattern in or_patterns:
            if pattern in query_lower:
                has_or_logic = True
                break
        
        # Extract body part filters (with OR support)
        body_part_matches = []
        for body_part in self.body_parts:
            if body_part in query_lower:
                body_part_matches.append(body_part)
                # Remove from semantic query
                query_lower = query_lower.replace(body_part, ' ')
        
        if body_part_matches:
            # Normalize body parts to database values
            normalized_parts = []
            for part in body_part_matches:
                normalized_part = self.body_part_mapping.get(part, part)
                if normalized_part not in normalized_parts:
                    normalized_parts.append(normalized_part)
            
            if len(normalized_parts) > 1 and has_or_logic:
                # Multiple body parts with OR logic
                filters['body_part'] = {'$in': normalized_parts}
            else:
                # Single body part or multiple with AND logic (use first/most specific)
                filters['body_part'] = normalized_parts[0]
        
        # Extract equipment filters (with OR support)
        equipment_matches = []
        for equipment in self.equipment_types:
            if equipment in query_lower:
                equipment_matches.append(equipment)
                # Remove from semantic query
                query_lower = query_lower.replace(equipment, ' ')
        
        if equipment_matches:
            if len(equipment_matches) > 1 and has_or_logic:
                # Multiple equipment types with OR logic
                filters['equipment'] = {'$in': equipment_matches}
            else:
                # Single equipment or multiple with AND logic
                filters['equipment'] = equipment_matches[0]
        
        # Extract quality/source filters
        if any(indicator in query_lower for indicator in self.quality_indicators['premium']):
            filters['source'] = 'merged'
            for indicator in self.quality_indicators['premium']:
                query_lower = query_lower.replace(indicator, ' ')
        
        if any(indicator in query_lower for indicator in self.quality_indicators['high_rating']):
            filters['has_rating'] = True
            for indicator in self.quality_indicators['high_rating']:
                query_lower = query_lower.replace(indicator, ' ')
        
        if any(indicator in query_lower for indicator in self.quality_indicators['has_instructions']):
            filters['has_instructions'] = True
            for indicator in self.quality_indicators['has_instructions']:
                query_lower = query_lower.replace(indicator, ' ')
        
        # Extract numeric constraints (ratings)
        rating_pattern = r'(?:rating|rated)\s*(?:above|over|>=?)\s*(\d+(?:\.\d+)?)'
        rating_match = re.search(rating_pattern, query_lower)
        if rating_match:
            min_rating = float(rating_match.group(1))
            filters['rating'] = {'$gte': min_rating}
            query_lower = re.sub(rating_pattern, ' ', query_lower)
        
        # Clean up and extract semantic keywords
        # Remove common stop words but keep fitness-related terms
        stop_words = {
            'the', 'a', 'an', 'and', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'about', 'into', 'through', 'during', 
            'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 
            'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
            'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
            'should', 'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
            'need', 'want', 'show', 'find', 'get', 'give'
        }
        
        # Keep important fitness terms that might be filtered out
        fitness_terms = {
            'exercises', 'workout', 'workouts', 'training', 'routine',
            'strength', 'cardio', 'endurance', 'power', 'flexibility',
            'outdoors', 'indoor', 'home', 'gym', 'advanced', 'beginner',
            'bodyweight', 'weighted', 'explosive', 'slow', 'controlled'
        }
        
        # Only remove 'or' if it was used for filter logic
        if has_or_logic:
            stop_words.add('or')
        
        # Tokenize and filter - keep fitness terms even if they seem like stop words
        words = re.findall(r'\b\w+\b', query_lower)
        semantic_keywords = []
        for word in words:
            if word in fitness_terms or (word not in stop_words and len(word) > 2):
                semantic_keywords.append(word)
        
        # Create sanitized query - if no meaningful keywords, use body-part specific terms
        if semantic_keywords:
            sanitized_query = ' '.join(semantic_keywords)
        else:
            # Fallback to simple terms for semantic search
            sanitized_query = 'exercise workout'
        
        # Special handling: enhance with body-part specific terms for better results
        should_enhance = False
        
        # Always enhance if only generic terms
        generic_only_terms = {'training', 'workout', 'exercises', 'exercise', 'routine'}
        if set(semantic_keywords).issubset(generic_only_terms) and body_part_matches:
            should_enhance = True
        
        # Special case: if "outdoors" or similar environmental terms, prioritize bodyweight
        environmental_terms = {'outdoors', 'outdoor', 'home', 'bodyweight', 'no'}
        if any(term in semantic_keywords for term in environmental_terms) and body_part_matches:
            should_enhance = True
            
        if should_enhance:
            # Enhance semantic search with body-part specific exercise terms
            body_part_enhancements = {
                'legs': 'squat lunge deadlift leg press',
                'quads': 'squat lunge leg press extension',
                'hamstring': 'deadlift curl stiff leg',
                'glutes': 'squat deadlift lunge hip thrust',
                'chest': 'press pushup fly bench',
                'back': 'row pullup pulldown deadlift',
                'shoulders': 'press raise lateral overhead',
                'biceps': 'curl chin hammer preacher',
                'triceps': 'press dip extension overhead',
                'abs': 'crunch plank sit up bodyweight',
                'core': 'plank crunch deadbug mountain'
            }
            
            # Special case: if "outdoors" is mentioned, prioritize bodyweight exercises
            if any(term in semantic_keywords for term in environmental_terms):
                if any(bp.lower() in ['abs', 'core'] for bp in body_part_matches):
                    sanitized_query = f"bodyweight crunch plank sit mountain climber"
                else:
                    sanitized_query = f"{sanitized_query} bodyweight".strip()
            else:
                # Regular enhancement
                for detected_part in body_part_matches:
                    if detected_part.lower() in body_part_enhancements:
                        enhancement = body_part_enhancements[detected_part.lower()]
                        sanitized_query = f"{sanitized_query} {enhancement}".strip()
                        break
        
        return {
            'filters': filters if filters else None,
            'sanitized_query': sanitized_query if sanitized_query else query,
            'original_query': query,
            'has_or_logic': has_or_logic
        }


class FitnessRAG:
    """RAG Agent for Fitness Exercise queries with intelligent query analysis"""
    
    def __init__(self, persist_dir: str = "chroma_db", top_k: int = 10):
        self.persist_dir = Path(persist_dir)
        self.top_k = top_k
        self.query_analyzer = QueryAnalyzer()
        
        print("Initializing Fitness Exercise RAG Agent...")
        print(f"✓ ChromaDB path: {self.persist_dir}")
        
        # Load persisted ChromaDB
        print("Loading persisted vector store...")
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        # Get the existing collection
        try:
            self.collection = self.client.get_collection(
                name="fitness_exercises",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            )
            print(f"✓ Loaded collection with {self.collection.count()} documents")
        except Exception as e:
            raise ValueError(f"Could not load vector store. Please run indexing first. Error: {e}")
        
        print("✓ RAG Agent initialized")
    
    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None, allow_fallback: bool = True, silent: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve relevant exercises using semantic search with intelligent fallback
        
        Args:
            query: Natural language query
            filters: Optional metadata filters
            allow_fallback: Whether to try fallback searches if no results found
            silent: Whether to suppress all output
            
        Returns:
            List of retrieved documents with metadata
        """
        if not silent:
            print(f"🔍 Searching for: '{query}'")
            if filters:
                print(f"📋 Filters: {filters}")
        
        # First attempt: Query with filters
        results = self.collection.query(
            query_texts=[query],
            n_results=self.top_k,
            where=filters
        )
        
        # Check if we got results
        documents = self._format_results(results)
        
        # If no results and fallback enabled, try progressive fallback
        if not documents and allow_fallback and filters:
            if not silent:
                print("🔄 No results found, trying fallback strategies...")
            
            # Strategy 1: Remove semantic query, just use filters (common case: "outdoors", "training", etc.)
            if query.strip() and len(query.split()) <= 2:
                if not silent:
                    print(f"   Fallback 1: Filter-only search")
                # Use body part filter only with empty/generic query
                fallback_results = self.collection.query(
                    query_texts=["exercise workout training"],  # Generic terms
                    n_results=self.top_k,
                    where=filters
                )
                documents = self._format_results(fallback_results)
                
            # Strategy 2: If still no results, expand body part filters
            if not documents and 'body_part' in filters:
                body_part = filters['body_part']
                if not silent:
                    print(f"   Fallback 2: Expanding {body_part} search")
                
                # Expand certain body parts to include related ones
                expanded_filters = filters.copy()
                if body_part == 'Quadriceps':
                    # Include all leg muscles
                    expanded_filters['body_part'] = {'$in': ['Quadriceps', 'Hamstrings', 'Glutes', 'Calves']}
                elif body_part == 'Biceps':
                    # Include all arm muscles  
                    expanded_filters['body_part'] = {'$in': ['Biceps', 'Triceps', 'Forearms']}
                elif body_part == 'Lats':
                    # Include all back muscles
                    expanded_filters['body_part'] = {'$in': ['Lats', 'Middle Back', 'Lower Back', 'Traps']}
                
                fallback_results = self.collection.query(
                    query_texts=["exercise"],
                    n_results=self.top_k,
                    where=expanded_filters
                )
                documents = self._format_results(fallback_results)
                
            # Strategy 3: Remove all filters, pure semantic search
            if not documents:
                if not silent:
                    print(f"   Fallback 3: Pure semantic search")
                fallback_results = self.collection.query(
                    query_texts=[query],
                    n_results=self.top_k
                )
                documents = self._format_results(fallback_results)
        
        # Log the exercise names only if not silent
        if not silent:
            exercise_names = [doc['metadata']['name'] for doc in documents]
            print(f"✓ Found {len(documents)} relevant exercises")
            for i, name in enumerate(exercise_names, 1):
                print(f"  {i}. {name}")
        return documents
    
    def _format_results(self, results) -> List[Dict[str, Any]]:
        """Format ChromaDB results into standard document format"""
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, (doc_id, document, metadata) in enumerate(zip(
                results['ids'][0], 
                results['documents'][0], 
                results['metadatas'][0]
            )):
                documents.append({
                    'id': doc_id,
                    'content': document,
                    'metadata': metadata,
                    'rank': i + 1
                })
        return documents
    
    def format_recommendations(self, documents: List[Dict[str, Any]], query: str) -> str:
        """
        Format retrieved documents into structured recommendations
        
        Args:
            documents: Retrieved documents
            query: Original query
            
        Returns:
            Formatted recommendation string
        """
        if not documents:
            return "No exercises found matching your criteria. Try a broader search."
        
        recommendations = []
        recommendations.append(f"# Exercise Recommendations for: '{query}'\n")
        
        # Group by source for better organization
        merged_docs = [doc for doc in documents if doc['metadata'].get('source') == 'merged']
        niharika_docs = [doc for doc in documents if doc['metadata'].get('source') == 'niharika']
        omarxadel_docs = [doc for doc in documents if doc['metadata'].get('source') == 'omarxadel']
        
        # Premium recommendations first (merged documents with ratings AND instructions)
        if merged_docs:
            recommendations.append("## 🌟 Premium Recommendations (Rated + Instructions)")
            for doc in merged_docs[:5]:  # Top 5 premium
                meta = doc['metadata']
                recommendations.append(f"\n### **{meta['name']}**")
                recommendations.append(f"- **Rating:** {meta['rating']}/10")
                recommendations.append(f"- **Equipment:** {meta['equipment']}")
                recommendations.append(f"- **Target:** {meta['target_muscle']} (Primary)")
                recommendations.append(f"- **Body Part:** {meta['body_part']}")
                
                # Show instructions preview
                content_lines = doc['content'].split('\n')
                instructions_start = -1
                for i, line in enumerate(content_lines):
                    if "Step-by-step Instructions:" in line:
                        instructions_start = i + 1
                        break
                
                if instructions_start > 0 and instructions_start < len(content_lines):
                    recommendations.append("- **Instructions Preview:**")
                    # Show first 2 instruction steps
                    for j in range(instructions_start, min(instructions_start + 2, len(content_lines))):
                        if content_lines[j].strip():
                            recommendations.append(f"  {content_lines[j].strip()}")
        
        # Additional recommendations with ratings
        rated_docs = [doc for doc in niharika_docs if doc['metadata'].get('has_rating', False)][:3]
        if rated_docs:
            recommendations.append("\n## ⭐ Additional Rated Exercises")
            for doc in rated_docs:
                meta = doc['metadata']
                recommendations.append(f"\n### **{meta['name']}**")
                recommendations.append(f"- **Rating:** {meta['rating']}/10")
                recommendations.append(f"- **Equipment:** {meta['equipment']}")
                recommendations.append(f"- **Body Part:** {meta['body_part']}")
        
        # Detailed instruction exercises
        instruction_docs = [doc for doc in omarxadel_docs if doc['metadata'].get('has_instructions', False)][:3]
        if instruction_docs:
            recommendations.append("\n## 📋 Exercises with Detailed Instructions")
            for doc in instruction_docs:
                meta = doc['metadata']
                recommendations.append(f"\n### **{meta['name']}**")
                recommendations.append(f"- **Equipment:** {meta['equipment']}")
                recommendations.append(f"- **Target:** {meta['target_muscle']} (Primary)")
                recommendations.append(f"- **Body Part:** {meta['body_part']}")
        
        recommendations.append(f"\n---\n*Found {len(documents)} total matches. Showing top recommendations organized by quality and completeness.*")
        
        return '\n'.join(recommendations)
    
    def query(self, question: str, filters: Optional[Dict[str, Any]] = None, verbose: bool = True, silent: bool = False) -> Dict[str, Any]:
        """
        Main RAG pipeline: retrieve + format recommendations
        
        Args:
            question: User question/query
            filters: Optional metadata filters  
            verbose: Whether to print intermediate steps
            silent: Whether to suppress all output (overrides verbose)
            
        Returns:
            Dictionary with formatted recommendations and metadata
        """
        if verbose and not silent:
            print(f"\n{'='*60}")
            print(f"FITNESS EXERCISE QUERY")
            print(f"{'='*60}")
        
        # Step 1: Intelligent query analysis
        if filters is None:  # Only analyze if no manual filters provided
            analysis = self.query_analyzer.analyze_query(question)
            extracted_filters = analysis['filters']
            sanitized_query = analysis['sanitized_query']
            
            if verbose and not silent and extracted_filters:
                print(f"🧠 Extracted filters: {extracted_filters}")
                print(f"🔍 Semantic query: '{sanitized_query}'")
        else:
            extracted_filters = filters
            sanitized_query = question
            
        # Step 2: Format filters for ChromaDB
        if extracted_filters:
            if len(extracted_filters) > 1:
                # Multiple filters always need $and wrapper
                chroma_filters = {"$and": []}
                for key, value in extracted_filters.items():
                    chroma_filters["$and"].append({key: value})
            else:
                # Single filter - use directly (can be simple or with $in for OR)
                chroma_filters = extracted_filters
        else:
            chroma_filters = extracted_filters
            
        # Step 3: Retrieve relevant exercises
        documents = self.retrieve(sanitized_query, chroma_filters, silent=silent)
        
        # Step 4: Format recommendations
        if verbose and not silent:
            print("📝 Formatting recommendations...")
        
        recommendations = self.format_recommendations(documents, question)
        
        if verbose and not silent:
            print("✅ Query complete!")
            print("\n" + recommendations)
        
        return {
            'question': question,
            'filters': filters,
            'retrieved_count': len(documents),
            'recommendations': recommendations,
            'raw_documents': documents
        }
    
    def test_basic_queries(self):
        """Test basic retrieval capabilities"""
        print("\n" + "=" * 60)
        print("TESTING BASIC QUERIES")
        print("=" * 60)
        
        test_queries = [
            "chest exercises",
            "shoulder workout", 
            "leg exercises",
            "back exercises"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing: '{query}'")
            result = self.query(query, verbose=False)
            print(f"Found {result['retrieved_count']} matches")
        
        # Test with filters
        print(f"\n🔍 Testing with filters: 'strength training' + source='merged'")
        result = self.query("strength training", filters={"source": "merged"}, verbose=False)
        print(f"Found {result['retrieved_count']} premium matches")
        
        print("\n✅ Basic query testing complete!")


def main():
    """Interactive query interface"""
    print("="*60)
    print("FITNESS & TRAINER COACH RAG AGENT")
    print("="*60)
    
    # Initialize RAG agent
    try:
        rag = FitnessRAG(top_k=10)
    except ValueError as e:
        print(f"❌ {e}")
        print("Please run 'make index' first to build the vector database.")
        return
    
    # Example queries
    example_queries = [
        "What are good chest exercises?",
        "Show me shoulder workouts with dumbbells",
        "I need back exercises for strength training",
        "What leg exercises can I do at home?",
        "Upper body exercises with high ratings",
        "Core strengthening exercises with detailed instructions"
    ]
    
    print("Example queries you can try:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q}")
    
    print("\n" + "="*60)
    print("Enter your question (or 'quit' to exit)")
    print("="*60)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            rag.query(question, verbose=True)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()