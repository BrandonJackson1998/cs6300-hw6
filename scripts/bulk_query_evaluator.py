#!/usr/bin/env python3
"""
Comprehensive RAG Query Generator and Bulk Evaluation Script
Generates diverse fitness queries and evaluates RAG performance at scale
"""

import json
import time
import random
from typing import List, Dict, Any
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_agent import FitnessRAG
from src.tools import exercise_lookup
import statistics


class QueryGenerator:
    """Generates diverse fitness queries for comprehensive RAG testing"""
    
    def __init__(self):
        # Core exercise components for query generation
        self.body_parts = [
            "chest", "back", "shoulders", "arms", "legs", "abs", "core",
            "biceps", "triceps", "quads", "hamstrings", "calves", "glutes",
            "upper body", "lower body", "full body"
        ]
        
        self.equipment = [
            "dumbbells", "barbells", "kettlebells", "cables", "machines",
            "bodyweight", "resistance bands", "exercise ball", "smith machine",
            "ez bar", "rope", "weighted", "lever"
        ]
        
        self.workout_types = [
            "strength training", "hypertrophy", "endurance", "power",
            "conditioning", "functional", "isolation", "compound",
            "circuit training", "superset", "drop set"
        ]
        
        self.difficulty_levels = [
            "beginner", "intermediate", "advanced", "easy", "hard",
            "challenging", "simple", "basic", "expert level"
        ]
        
        self.quality_indicators = [
            "best", "top rated", "highest rated", "excellent", "premium",
            "with detailed instructions", "step by step", "comprehensive",
            "complete", "effective", "proven"
        ]
        
        self.locations = [
            "at home", "in gym", "outdoors", "at office", "in hotel",
            "with limited space", "in small room"
        ]
        
        self.goals = [
            "muscle building", "fat loss", "strength gain", "toning",
            "rehabilitation", "injury prevention", "flexibility",
            "mobility", "endurance", "power development"
        ]
        
        # Query templates for systematic generation
        self.query_templates = [
            # Simple queries
            "{body_part} exercises",
            "{equipment} workout",
            "{body_part} training",
            
            # Equipment + body part
            "{body_part} exercises with {equipment}",
            "{equipment} {body_part} workout",
            "{body_part} training using {equipment}",
            
            # Quality focused
            "{quality} {body_part} exercises",
            "{quality} {equipment} workout",
            "Show me {quality} {body_part} exercises",
            
            # Location specific
            "{body_part} exercises {location}",
            "{equipment} workout {location}",
            "{body_part} training {location}",
            
            # Goal oriented
            "{body_part} exercises for {goal}",
            "{equipment} workout for {goal}",
            "Best exercises for {goal}",
            
            # Difficulty based
            "{difficulty} {body_part} exercises",
            "{difficulty} {equipment} workout",
            "{difficulty} exercises for {body_part}",
            
            # Complex combinations
            "{quality} {body_part} exercises with {equipment} for {goal}",
            "{difficulty} {body_part} workout {location}",
            "{quality} {equipment} exercises for {body_part} training",
            
            # Boolean logic queries
            "{body_part1} or {body_part2} exercises",
            "{body_part} and {body_part2} workout",
            "{equipment1} or {equipment2} exercises",
            "{body_part} exercises with {equipment1} or {equipment2}",
            
            # Specific requests
            "I need {body_part} exercises",
            "What are good {body_part} exercises?",
            "Find me {equipment} exercises",
            "Looking for {body_part} workout",
            "Can you suggest {body_part} exercises?",
            
            # Instructional queries
            "How to do {body_part} exercises",
            "Step by step {body_part} workout",
            "Detailed {equipment} exercise instructions",
            "Guide for {body_part} training",
            
            # Rating and quality queries
            "Highest rated {body_part} exercises",
            "Best {equipment} exercises with high ratings",
            "Top quality {body_part} workout",
            "Premium {equipment} exercises",
            
            # Workout style queries
            "{workout_type} for {body_part}",
            "{workout_type} using {equipment}",
            "{body_part} {workout_type} routine",
            
            # Time and intensity
            "Quick {body_part} workout",
            "Intensive {body_part} training",
            "Short {equipment} exercises",
            "Fast {body_part} routine"
        ]
    
    def generate_queries(self, num_queries: int = 100) -> List[str]:
        """Generate diverse queries for testing"""
        queries = []
        
        for _ in range(num_queries):
            template = random.choice(self.query_templates)
            
            # Fill template with random components
            query = template.format(
                body_part=random.choice(self.body_parts),
                body_part1=random.choice(self.body_parts),
                body_part2=random.choice(self.body_parts),
                equipment=random.choice(self.equipment),
                equipment1=random.choice(self.equipment),
                equipment2=random.choice(self.equipment),
                workout_type=random.choice(self.workout_types),
                difficulty=random.choice(self.difficulty_levels),
                quality=random.choice(self.quality_indicators),
                location=random.choice(self.locations),
                goal=random.choice(self.goals)
            )
            
            queries.append(query)
        
        # Add some manually crafted edge cases
        edge_cases = [
            "exercise",  # Very vague
            "workout routine",  # Generic
            "fitness",  # Broad
            "training plan",  # Non-specific
            "gym exercises",  # Location only
            "muscle building workout",  # Goal only
            "strength training routine",  # Type only
            "best exercises ever",  # Superlative
            "exercises with high ratings above 8",  # Numeric constraint
            "detailed step by step instructions",  # Instruction focus
            "premium quality exercises",  # Quality focus
            "beginner friendly workouts",  # Difficulty + tone
            "advanced challenging exercises",  # Multiple difficulty terms
            "chest or shoulder or back exercises",  # Multiple OR
            "legs and glutes and core workout",  # Multiple AND
            "dumbbell and barbell and bodyweight exercises",  # Equipment AND
            "best rated chest exercises with dumbbells at home for muscle building",  # Everything
        ]
        
        queries.extend(edge_cases)
        
        # Remove duplicates and return
        return list(set(queries))
    
    def generate_targeted_queries(self) -> Dict[str, List[str]]:
        """Generate queries targeting specific RAG capabilities"""
        targeted_queries = {
            "semantic_search": [
                "muscle building chest workout",
                "strength training for back",
                "hypertrophy leg routine",
                "power exercises for shoulders",
                "functional training movements",
                "isolation exercises for arms",
                "compound movements for full body",
                "conditioning workout routine"
            ],
            
            "boolean_logic": [
                "chest or shoulder exercises",
                "legs and glutes workout",
                "dumbbells or barbells exercises",
                "back and biceps training",
                "abs or core strengthening",
                "upper body or lower body routine",
                "chest or back or shoulder exercises",
                "dumbbells and cables and bodyweight"
            ],
            
            "metadata_filtering": [
                "chest exercises with dumbbells",
                "back workout using cables",
                "leg training with barbells",
                "shoulder exercises bodyweight only",
                "core workout with exercise ball",
                "arm exercises using resistance bands",
                "full body kettlebell workout",
                "upper body machine exercises"
            ],
            
            "quality_filtering": [
                "best rated chest exercises",
                "highest rated back workout",
                "top quality shoulder exercises",
                "premium leg training",
                "excellent core workout",
                "exercises with detailed instructions",
                "step by step workout guide",
                "comprehensive training routine"
            ],
            
            "difficulty_assessment": [
                "beginner chest exercises",
                "advanced back workout",
                "intermediate leg training",
                "easy shoulder exercises",
                "challenging core workout",
                "expert level arm training",
                "simple bodyweight exercises",
                "difficult barbell movements"
            ],
            
            "edge_cases": [
                "",  # Empty query
                "a",  # Single character
                "the best ever",  # No specific exercise terms
                "12345",  # Numbers only
                "!@#$%",  # Special characters
                "very very very long query with many words that might test the system limits",
                "CHEST EXERCISES",  # All caps
                "chest exercises " * 10,  # Repetitive
                "exercise exercise exercise"  # Repeated words
            ]
        }
        
        return targeted_queries


class BulkRAGEvaluator:
    """Runs bulk queries against RAG system and collects performance metrics"""
    
    def __init__(self, rag_agent: FitnessRAG):
        self.rag = rag_agent
        self.results = []
        
    def evaluate_query_batch(self, queries: List[str], use_tool_wrapper: bool = False) -> List[Dict[str, Any]]:
        """Evaluate a batch of queries and collect metrics"""
        print(f"🔍 Evaluating {len(queries)} queries...")
        
        batch_results = []
        
        for i, query in enumerate(queries, 1):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(queries)} queries processed")
            
            start_time = time.time()
            
            try:
                if use_tool_wrapper:
                    # Test tool wrapper
                    response = exercise_lookup(query)
                    retrieved_count = response.count("###") if "###" in response else 0
                    raw_docs = []
                else:
                    # Test RAG agent directly
                    result = self.rag.query(query, verbose=False)
                    response = result['recommendations']
                    retrieved_count = result['retrieved_count']
                    raw_docs = result['raw_documents']
                
                end_time = time.time()
                query_time = end_time - start_time
                
                # Calculate basic metrics
                response_length = len(response)
                has_exercises = "exercise" in response.lower() or "workout" in response.lower()
                has_instructions = "step" in response.lower() or "instruction" in response.lower()
                has_ratings = "rating" in response.lower() or "/10" in response
                
                # Analyze query complexity
                query_complexity = self._assess_query_complexity(query)
                
                result = {
                    "query": query,
                    "query_length": len(query),
                    "query_complexity": query_complexity,
                    "response_length": response_length,
                    "retrieved_count": retrieved_count,
                    "query_time_seconds": query_time,
                    "has_exercises": has_exercises,
                    "has_instructions": has_instructions,
                    "has_ratings": has_ratings,
                    "success": True,
                    "error": None,
                    "tool_wrapper_used": use_tool_wrapper
                }
                
                # Add document quality metrics if available
                if raw_docs:
                    result.update(self._analyze_document_quality(raw_docs))
                
            except Exception as e:
                end_time = time.time()
                query_time = end_time - start_time
                
                result = {
                    "query": query,
                    "query_length": len(query),
                    "query_complexity": self._assess_query_complexity(query),
                    "response_length": 0,
                    "retrieved_count": 0,
                    "query_time_seconds": query_time,
                    "has_exercises": False,
                    "has_instructions": False,
                    "has_ratings": False,
                    "success": False,
                    "error": str(e),
                    "tool_wrapper_used": use_tool_wrapper
                }
            
            batch_results.append(result)
        
        return batch_results
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess query complexity based on content"""
        if len(query) == 0:
            return "empty"
        elif len(query) < 10:
            return "simple"
        elif "or" in query.lower() or "and" in query.lower():
            return "boolean"
        elif len(query.split()) > 8:
            return "complex"
        elif any(word in query.lower() for word in ["best", "rated", "detailed", "step"]):
            return "quality_focused"
        else:
            return "moderate"
    
    def _analyze_document_quality(self, docs: List[Dict]) -> Dict[str, Any]:
        """Analyze quality of retrieved documents"""
        if not docs:
            return {"avg_doc_rating": 0, "docs_with_instructions": 0, "doc_sources": []}
        
        ratings = []
        instruction_count = 0
        sources = []
        
        for doc in docs:
            metadata = doc.get('metadata', {})
            
            # Collect ratings
            if 'rating' in metadata and metadata['rating']:
                try:
                    ratings.append(float(metadata['rating']))
                except (ValueError, TypeError):
                    pass
            
            # Count instructions
            if metadata.get('has_instructions', False):
                instruction_count += 1
            
            # Collect sources
            source = metadata.get('source', 'unknown')
            if source not in sources:
                sources.append(source)
        
        return {
            "avg_doc_rating": statistics.mean(ratings) if ratings else 0,
            "docs_with_instructions": instruction_count,
            "doc_sources": sources,
            "rating_coverage": len(ratings) / len(docs) if docs else 0
        }
    
    def generate_performance_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        total_queries = len(results)
        successful_queries = [r for r in results if r['success']]
        failed_queries = [r for r in results if not r['success']]
        
        if not results:
            return {"error": "No results to analyze"}
        
        # Basic statistics
        success_rate = len(successful_queries) / total_queries
        avg_query_time = statistics.mean([r['query_time_seconds'] for r in results])
        avg_response_length = statistics.mean([r['response_length'] for r in successful_queries]) if successful_queries else 0
        avg_retrieved_count = statistics.mean([r['retrieved_count'] for r in successful_queries]) if successful_queries else 0
        
        # Query complexity analysis
        complexity_counts = {}
        for result in results:
            complexity = result['query_complexity']
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        # Performance by complexity
        complexity_performance = {}
        for complexity in complexity_counts:
            complexity_results = [r for r in results if r['query_complexity'] == complexity]
            if complexity_results:
                complexity_performance[complexity] = {
                    "count": len(complexity_results),
                    "success_rate": len([r for r in complexity_results if r['success']]) / len(complexity_results),
                    "avg_time": statistics.mean([r['query_time_seconds'] for r in complexity_results])
                }
        
        # Content quality metrics
        exercise_coverage = len([r for r in successful_queries if r['has_exercises']]) / len(successful_queries) if successful_queries else 0
        instruction_coverage = len([r for r in successful_queries if r['has_instructions']]) / len(successful_queries) if successful_queries else 0
        rating_coverage = len([r for r in successful_queries if r['has_ratings']]) / len(successful_queries) if successful_queries else 0
        
        # Speed percentiles
        query_times = [r['query_time_seconds'] for r in results]
        query_times.sort()
        
        report = {
            "summary": {
                "total_queries": total_queries,
                "successful_queries": len(successful_queries),
                "failed_queries": len(failed_queries),
                "success_rate": success_rate,
                "avg_query_time_seconds": avg_query_time,
                "avg_response_length": avg_response_length,
                "avg_retrieved_count": avg_retrieved_count
            },
            "performance_metrics": {
                "query_time_percentiles": {
                    "p50": query_times[len(query_times)//2] if query_times else 0,
                    "p90": query_times[int(len(query_times)*0.9)] if query_times else 0,
                    "p95": query_times[int(len(query_times)*0.95)] if query_times else 0,
                    "p99": query_times[int(len(query_times)*0.99)] if query_times else 0
                },
                "slowest_queries": sorted(results, key=lambda x: x['query_time_seconds'], reverse=True)[:5],
                "fastest_queries": sorted(results, key=lambda x: x['query_time_seconds'])[:5]
            },
            "query_complexity_analysis": {
                "complexity_distribution": complexity_counts,
                "performance_by_complexity": complexity_performance
            },
            "content_quality": {
                "exercise_coverage": exercise_coverage,
                "instruction_coverage": instruction_coverage,
                "rating_coverage": rating_coverage
            },
            "error_analysis": {
                "error_count": len(failed_queries),
                "error_types": list(set([r['error'] for r in failed_queries if r['error']])),
                "failed_queries": [r['query'] for r in failed_queries]
            }
        }
        
        return report


def main():
    """Run comprehensive bulk query evaluation"""
    print("🚀 Bulk RAG Query Evaluation")
    print("="*60)
    
    # Initialize components
    try:
        rag = FitnessRAG(persist_dir="chroma_db", top_k=10)
        print("✓ RAG agent initialized")
    except Exception as e:
        print(f"❌ Failed to initialize RAG agent: {e}")
        return 1
    
    generator = QueryGenerator()
    evaluator = BulkRAGEvaluator(rag)
    
    # Generate queries
    print("\n📝 Generating test queries...")
    
    # Generate different types of queries
    random_queries = generator.generate_queries(num_queries=50)
    targeted_queries = generator.generate_targeted_queries()
    
    # Flatten targeted queries
    all_targeted = []
    for category, queries in targeted_queries.items():
        all_targeted.extend(queries)
    
    # Combine all queries
    all_queries = random_queries + all_targeted
    all_queries = list(set(all_queries))  # Remove duplicates
    
    print(f"✓ Generated {len(all_queries)} unique queries")
    print(f"  - Random queries: {len(random_queries)}")
    print(f"  - Targeted queries: {len(all_targeted)}")
    
    # Run evaluation
    print(f"\n🔍 Running bulk evaluation...")
    start_time = time.time()
    
    # Test both RAG agent and tool wrapper
    print("\n1. Testing RAG Agent directly...")
    rag_results = evaluator.evaluate_query_batch(all_queries, use_tool_wrapper=False)
    
    print("\n2. Testing Tool Wrapper...")
    tool_results = evaluator.evaluate_query_batch(all_queries[:20], use_tool_wrapper=True)  # Smaller sample for tool
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n✅ Evaluation complete! Total time: {total_time:.2f} seconds")
    
    # Generate reports
    print("\n📊 Generating performance reports...")
    
    rag_report = evaluator.generate_performance_report(rag_results)
    tool_report = evaluator.generate_performance_report(tool_results)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    with open(f"bulk_evaluation_rag_{timestamp}.json", "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "total_queries": len(all_queries),
                "evaluation_time_seconds": total_time
            },
            "raw_results": rag_results,
            "performance_report": rag_report
        }, f, indent=2)
    
    with open(f"bulk_evaluation_tools_{timestamp}.json", "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "total_queries": len(tool_results),
                "evaluation_time_seconds": total_time
            },
            "raw_results": tool_results,
            "performance_report": tool_report
        }, f, indent=2)
    
    # Print summary report
    print("\n" + "="*80)
    print("📈 BULK EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\n🎯 RAG AGENT PERFORMANCE:")
    print(f"  • Total Queries:       {rag_report['summary']['total_queries']}")
    print(f"  • Success Rate:        {rag_report['summary']['success_rate']:.1%}")
    print(f"  • Avg Query Time:      {rag_report['summary']['avg_query_time_seconds']:.3f}s")
    print(f"  • Avg Response Length: {rag_report['summary']['avg_response_length']:.0f} chars")
    print(f"  • Avg Retrieved Docs:  {rag_report['summary']['avg_retrieved_count']:.1f}")
    
    print(f"\n🔧 TOOL WRAPPER PERFORMANCE:")
    print(f"  • Total Queries:       {tool_report['summary']['total_queries']}")
    print(f"  • Success Rate:        {tool_report['summary']['success_rate']:.1%}")
    print(f"  • Avg Query Time:      {tool_report['summary']['avg_query_time_seconds']:.3f}s")
    print(f"  • Avg Response Length: {tool_report['summary']['avg_response_length']:.0f} chars")
    
    print(f"\n⚡ PERFORMANCE PERCENTILES (RAG Agent):")
    percentiles = rag_report['performance_metrics']['query_time_percentiles']
    print(f"  • 50th percentile:     {percentiles['p50']:.3f}s")
    print(f"  • 90th percentile:     {percentiles['p90']:.3f}s")
    print(f"  • 95th percentile:     {percentiles['p95']:.3f}s")
    print(f"  • 99th percentile:     {percentiles['p99']:.3f}s")
    
    print(f"\n📋 CONTENT QUALITY:")
    quality = rag_report['content_quality']
    print(f"  • Exercise Coverage:   {quality['exercise_coverage']:.1%}")
    print(f"  • Instruction Coverage: {quality['instruction_coverage']:.1%}")
    print(f"  • Rating Coverage:     {quality['rating_coverage']:.1%}")
    
    print(f"\n🏆 TOP PERFORMING QUERIES:")
    fastest = rag_report['performance_metrics']['fastest_queries']
    for i, query in enumerate(fastest[:3], 1):
        print(f"  {i}. '{query['query'][:50]}...' ({query['query_time_seconds']:.3f}s)")
    
    print(f"\n⚠️ SLOWEST QUERIES:")
    slowest = rag_report['performance_metrics']['slowest_queries']
    for i, query in enumerate(slowest[:3], 1):
        print(f"  {i}. '{query['query'][:50]}...' ({query['query_time_seconds']:.3f}s)")
    
    if rag_report['error_analysis']['error_count'] > 0:
        print(f"\n❌ ERROR ANALYSIS:")
        print(f"  • Error Count: {rag_report['error_analysis']['error_count']}")
        print(f"  • Error Types: {rag_report['error_analysis']['error_types']}")
    
    print(f"\n💾 Results saved to:")
    print(f"  • bulk_evaluation_rag_{timestamp}.json")
    print(f"  • bulk_evaluation_tools_{timestamp}.json")
    
    print("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())